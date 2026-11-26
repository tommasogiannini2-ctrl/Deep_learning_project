"""Componenti comuni condivisi da tutti i modelli di src/models/.

Ogni architettura (baseline, GRU, Transformer, ...) deve risolvere gli stessi
tre problemi. Li risolviamo qui una volta sola:

1. `build_step_features`: rendere gli input "consapevoli" dei dati mancanti.
   Nel batch i NaN sono già stati sostituiti con 0.0, che dopo la
   normalizzazione coincide con la media della feature. Senza un'informazione
   esplicita la rete non distinguerebbe "valore nella media" da "dato mancante":
   per questo affianchiamo ai valori i flag di presenza e il tempo relativo.

2. `ContinuousTimeEncoding`: codificare il tempo relativo `delta_t` (giorni
   rispetto a t0) in un vettore, in modo che funzioni con sequenze di qualsiasi
   lunghezza. Nel train le sequenze arrivano a 70 giorni, in ood-s a 126: un
   positional embedding appreso per indice non generalizzerebbe.

3. `MonotonicQuantileHead`: trasformare la rappresentazione di ciascuno dei 3
   step futuri nei 3 quantili dei 3 indici, garantendo per costruzione
   q0.1 <= q0.5 <= q0.9 (nessun "quantile crossing") e, opzionalmente,
   prevedendo la variazione rispetto all'ultima osservazione (`last_obs`).

In più `gather_time_steps` estrae da una sequenza giornaliera i passi indicati
da `future_target_positions` (i giorni dei 3 target): serve a qualsiasi modello
che produce una rappresentazione per ogni giorno di `future`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.contract import N_QUANTILES, N_TARGETS

# Scala di default per normalizzare delta_t (giorni). Con 30 giorni (circa un mese)
# gli orizzonti tipici dei target (5-20 giorni) diventano valori tra 0.17 e 0.67,
# e anche i casi estremi di ood-s (circa 126 giorni) restano dell'ordine di 4.
DEFAULT_DELTA_T_SCALE: float = 30.0


# =============================================================================
# 1. Input mask-aware
# =============================================================================


def step_feature_dim(num_features: int) -> int:
    """Numero di canali per timestep prodotti da `build_step_features`.

    Per ogni giorno: F valori + F flag di presenza + 1 canale di tempo = 2F + 1.
    Con le 21 feature di default sono 43 canali; con un'ablation che rimuove
    una feature (F = 20) diventano 41. I modelli usano questa funzione per
    dimensionare il primo layer, invece di scrivere il numero a mano.
    """
    return 2 * num_features + 1


def build_step_features(
    values: torch.Tensor,
    mask: torch.Tensor,
    delta_t: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    delta_t_scale: float = DEFAULT_DELTA_T_SCALE,
) -> torch.Tensor:
    """Costruisce l'input per timestep: [valori, flag di presenza, delta_t scalato].

    Args:
        values: (B, T, F) feature normalizzate (history o future del batch).
        mask: (B, T, F) bool, True = mancante o padding (convenzione del contratto).
        delta_t: (B, T) giorni rispetto a t0 (negativi in history, positivi in future).
        padding_mask: (B, T) bool, True = timestep di padding. Facoltativa: se data,
            il canale del tempo viene azzerato sul padding per sicurezza.
        delta_t_scale: divisore per portare delta_t su valori dell'ordine di 1.

    Returns:
        Tensore (B, T, 2F + 1) senza NaN, pronto per un layer lineare, una GRU o
        un Transformer.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} e mask {tuple(mask.shape)} devono avere la stessa shape.")
    if delta_t.shape != values.shape[:2]:
        raise ValueError(f"delta_t {tuple(delta_t.shape)} deve avere shape (B, T) = {tuple(values.shape[:2])}.")
    if delta_t_scale <= 0:
        raise ValueError("delta_t_scale deve essere positivo.")

    # Valori: le posizioni mancanti sono già 0.0 per contratto; le riazzeriamo
    # comunque, così il modello resta corretto anche con input non conformi.
    clean_values = values.masked_fill(mask, 0.0)

    # Flag di presenza: 1.0 = valore osservato, 0.0 = mancante o padding.
    # È ciò che permette alla rete di distinguere uno 0.0 "vero" da uno 0.0 di riempimento.
    presence = (~mask).to(values.dtype)

    # Tempo relativo scalato, come canale aggiuntivo (B, T, 1).
    time_channel = delta_t / delta_t_scale
    if padding_mask is not None:
        time_channel = time_channel.masked_fill(padding_mask, 0.0)
    time_channel = time_channel.unsqueeze(-1).to(values.dtype)

    return torch.cat([clean_values, presence, time_channel], dim=-1)


# =============================================================================
# 2. Codifica continua del tempo
# =============================================================================


class ContinuousTimeEncoding(nn.Module):
    """Codifica sinusoidale di un tempo continuo (delta_t in giorni).

    Ogni valore di delta_t viene trasformato in un vettore di dimensione d_model:

        [sin(w_1 * dt), ..., sin(w_k * dt), cos(w_1 * dt), ..., cos(w_k * dt)]

    con k = d_model / 2 frequenze w_i = 2*pi / periodo_i, e periodi in
    progressione geometrica tra `min_period` e `max_period` giorni:
    - i periodi brevi distinguono giorni vicini (ad esempio +5 da +7);
    - i periodi lunghi distinguono distanze grandi (ad esempio +20 da +120).

    Perché non un positional embedding appreso:
    - non ci sono parametri da imparare, quindi nessun rischio di adattarsi solo
      al range di giorni visto nel train (massimo 70);
    - funziona per qualsiasi delta_t, anche negativo (history) o oltre 150 giorni;
    - lavora sui giorni reali, quindi rispetta le distanze irregolari tra i
      passaggi clear-sky.

    Nota: non sostituisce le feature di calendario del dataset (giorno dell'anno
    e sue armoniche di Fourier), che descrivono la stagione. Qui codifichiamo la
    distanza temporale da t0.
    """

    def __init__(self, d_model: int, min_period: float = 4.0, max_period: float = 1000.0) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model deve essere un intero pari e positivo, trovato {d_model}.")
        # min_period >= 4: con delta_t interi, periodi di 1 o 2 giorni darebbero
        # sin(2*pi*k) = 0 o sin(pi*k) = 0 per ogni k, cioè canali inutili.
        if not 0 < min_period < max_period:
            raise ValueError("Serve 0 < min_period < max_period.")

        self.d_model = d_model
        n_freq = d_model // 2

        # Periodi geometrici da min_period a max_period. max_period = 1000 giorni è
        # ben oltre la massima distanza nei dati (circa 250 giorni tra inizio della
        # history e ultimo target), quindi la codifica non "si ripete" sul range utile.
        if n_freq == 1:
            periods = torch.tensor([max_period])
        else:
            periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freq)
        frequencies = 2.0 * math.pi / periods  # (n_freq,)

        # Buffer (non parametro): segue il modello su GPU ma non viene addestrato.
        self.register_buffer("frequencies", frequencies.to(torch.float32))

    def forward(self, delta_t: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Codifica delta_t.

        Args:
            delta_t: tensore di qualsiasi forma (...), ad esempio (B, T) o (B, 3).
            padding_mask: stessa forma di delta_t, True = padding. Se data, la
                codifica viene azzerata sulle posizioni di padding.

        Returns:
            Tensore (..., d_model).
        """
        # (..., 1) * (n_freq,) -> (..., n_freq): ogni giorno moltiplicato per ogni frequenza.
        angles = delta_t.unsqueeze(-1).to(self.frequencies.dtype) * self.frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., d_model)

        if padding_mask is not None:
            encoding = encoding.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return encoding


# =============================================================================
# 3. Testa multi-quantile monotona (con previsione residua opzionale)
# =============================================================================


def _inverse_softplus(y: float) -> float:
    """Restituisce x tale che softplus(x) = y (con y > 0). Serve per l'inizializzazione."""
    return math.log(math.expm1(y))


class MonotonicQuantileHead(nn.Module):
    """Testa finale comune a tutti i modelli: rappresentazioni dei 3 step -> 3 quantili.

    Input: un vettore di dimensione D per ciascuno dei 3 step futuri, (B, H, D).
    La stessa testa (stessi pesi) è applicata a tutti gli step: l'informazione su
    "quale step è" deve arrivare dentro la rappresentazione, ad esempio dalla
    codifica di `target_delta_t`. Così la testa funziona anche con orizzonti
    irregolari (5, 10, 20 giorni ma anche 100+ in ood-s).

    Per ogni step e indice il layer lineare produce 3 numeri grezzi:
        raw_med, raw_low, raw_high
    da cui:
        q0.5 = raw_med (+ last_obs se residual=True)
        q0.1 = q0.5 - softplus(raw_low)
        q0.9 = q0.5 + softplus(raw_high)
    softplus(x) = log(1 + e^x) è sempre >= 0 e liscia, quindi
    q0.1 <= q0.5 <= q0.9 vale per qualsiasi input, anche fuori distribuzione.

    Previsione residua (residual=True, default): la rete prevede di quanto
    cambiano gli indici rispetto all'ultima osservazione clear-sky (`last_obs`),
    invece del loro valore assoluto. Rende il modello più robusto quando il
    livello medio degli indici cambia tra regioni (shift spaziale di ood-s).
    Con residual=False la testa prevede direttamente il valore (utile per l'ablation).

    Inizializzazione: i pesi partono a zero, quindi alla prima iterazione
    - con residual=True il modello coincide con la baseline di persistenza
      (q0.5 = last_obs);
    - con residual=False q0.5 = 0, cioè la media del train nello spazio normalizzato;
    - gli intervalli partono con semi-ampiezza `init_spread` (nello spazio del target).
    È un punto di partenza sensato, da cui il training migliora.
    """

    def __init__(
        self,
        in_features: int,
        n_targets: int = N_TARGETS,
        residual: bool = True,
        init_spread: float = 0.5,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features deve essere positivo.")
        if init_spread <= 0:
            raise ValueError("init_spread deve essere positivo.")

        self.n_targets = n_targets
        self.residual = residual

        # Un solo layer lineare: D -> n_targets * 3 (mediana, scarto basso, scarto alto).
        self.proj = nn.Linear(in_features, n_targets * N_QUANTILES)

        # Inizializzazione descritta nella docstring della classe.
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            bias = self.proj.bias.view(n_targets, N_QUANTILES)
            bias[:, 0] = 0.0                                # raw_med
            bias[:, 1] = _inverse_softplus(init_spread)     # raw_low  -> softplus = init_spread
            bias[:, 2] = _inverse_softplus(init_spread)     # raw_high -> softplus = init_spread

    def forward(self, x: torch.Tensor, last_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Calcola i quantili.

        Args:
            x: (B, H, D) rappresentazione di ciascuno step futuro.
            last_obs: (B, C) ultima osservazione di [NDVI, GNDVI, EVI], nella stessa
                scala del target. Obbligatoria se residual=True.

        Returns:
            preds: (B, H, C, 3) con quantili ordinati [q0.1, q0.5, q0.9].
        """
        if x.ndim != 3:
            raise ValueError(f"x deve avere shape (B, H, D), trovato {tuple(x.shape)}.")
        B, H, _ = x.shape

        # (B, H, D) -> (B, H, C * 3) -> (B, H, C, 3)
        raw = self.proj(x).view(B, H, self.n_targets, N_QUANTILES)
        raw_med, raw_low, raw_high = raw.unbind(dim=-1)  # ciascuno (B, H, C)

        # Mediana: valore assoluto oppure variazione rispetto a last_obs.
        q50 = raw_med
        if self.residual:
            if last_obs is None:
                raise ValueError("residual=True richiede last_obs (B, C).")
            if tuple(last_obs.shape) != (B, self.n_targets):
                raise ValueError(f"last_obs deve avere shape {(B, self.n_targets)}, trovato {tuple(last_obs.shape)}.")
            # (B, C) -> (B, 1, C): la stessa ultima osservazione vale per tutti gli step.
            q50 = q50 + last_obs.unsqueeze(1)

        # Scarti sempre >= 0 grazie a softplus: garantiscono l'ordine dei quantili.
        q10 = q50 - F.softplus(raw_low)
        q90 = q50 + F.softplus(raw_high)

        # Ordine dell'ultimo asse come nel contratto: [q0.1, q0.5, q0.9].
        return torch.stack([q10, q50, q90], dim=-1)


# =============================================================================
# 4. Lettura dei passi target da una sequenza
# =============================================================================


def gather_time_steps(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Estrae da una sequenza i timestep indicati, separatamente per ogni campione.

    Esempio: `x` contiene un vettore per ogni giorno di `future` e `positions`
    contiene i giorni dei 3 target (`batch["future_target_positions"]`, 0-based):
    il risultato contiene solo i 3 vettori di quei giorni.

    Args:
        x: (B, T, D) sequenza di vettori.
        positions: (B, K) indici interi in [0, T-1].

    Returns:
        (B, K, D) con out[b, k] = x[b, positions[b, k]].
    """
    if x.ndim != 3 or positions.ndim != 2 or positions.shape[0] != x.shape[0]:
        raise ValueError(
            f"Attesi x (B, T, D) e positions (B, K), trovati {tuple(x.shape)} e {tuple(positions.shape)}."
        )
    # positions (B, K) -> (B, K, D): stesso indice temporale per tutti i D canali.
    index = positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(dim=1, index=index)
