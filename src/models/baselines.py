"""Baseline del progetto: persistenza e MLP residuo.

Una baseline è un modello semplice che fa da metro di paragone: un modello più
complesso è utile solo se la batte chiaramente. Qui ce ne sono due:

1. `PersistenceModel` (zero parametri): prevede che gli indici restino uguali
   all'ultima osservazione clear-sky. È il riferimento minimo da battere.

2. `ResidualMLPBaseline`: la prima rete neurale. Non legge i giorni uno per uno:
   riassume history e future in un vettore di dimensione fissa per ciascuno dei
   3 step futuri e lo passa in un piccolo MLP. È veloce e, rispetto alla
   persistenza, usa il meteo; rispetto ai modelli sequenziali della Fase 2
   (GRU/TCN, Transformer) non sfrutta l'ordine dei giorni. Il confronto tra
   queste famiglie mostra quanto vale modellare davvero la sequenza.

Entrambi rispettano il contratto dei modelli (docs/batch_contract.md):
    preds = model(batch)    # (B, 3, 3, 3) = (batch, step, indice, quantile)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.contract import N_FUTURE, N_QUANTILES
from src.models.common import (
    DEFAULT_DELTA_T_SCALE,
    ContinuousTimeEncoding,
    MonotonicQuantileHead,
    build_step_features,
    gather_time_steps,
    step_feature_dim,
)


# =============================================================================
# 1. Persistenza
# =============================================================================


class PersistenceModel(nn.Module):
    """Prevede per tutti e 3 gli step l'ultima osservazione clear-sky (`last_obs`).

    I tre quantili coincidono (intervallo di ampiezza zero): la persistenza non
    ha una stima dell'incertezza. Serve come riferimento per le metriche
    puntuali (RMSE, MAE, WMAPE, MASE) e per la Pinball Loss.

    È comunque un nn.Module con la stessa firma degli altri modelli, così
    training ed eval lo trattano come qualsiasi rete. Non ha parametri: il
    training loop deve saltare l'ottimizzazione per questo modello.
    """

    def __init__(self, num_features: int | None = None) -> None:
        super().__init__()
        # num_features è accettato solo per avere la stessa firma di tutti i
        # modelli (build_model lo passa sempre); qui non serve.
        self.num_features = num_features

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        last_obs = batch["last_obs"]  # (B, C), stessa scala di batch["target"]
        B, C = last_obs.shape
        # (B, C) -> (B, 1, C, 1) -> (B, 3, C, 3): stesso valore per ogni step e quantile.
        return last_obs[:, None, :, None].expand(B, N_FUTURE, C, N_QUANTILES).contiguous()


# =============================================================================
# 2. MLP residuo
# =============================================================================


def _summarize_history(
    h_feats: torch.Tensor,
    num_features: int,
    history_len: torch.Tensor,
    delta_t_scale: float,
) -> torch.Tensor:
    """Riassume tutta la history in un vettore di dimensione fissa, (B, 2F + 1).

    Contiene, per ogni feature:
    - la media sui soli giorni in cui è osservata (i mancanti non contano);
    - la frazione di giorni in cui è osservata;
    più la lunghezza della history (in giorni, scalata).

    h_feats è l'output di build_step_features: i primi F canali sono i valori
    (0 dove mancanti o padding), i successivi F i flag di presenza.
    """
    F = num_features
    values = h_feats[..., :F]           # (B, T_h, F)
    presence = h_feats[..., F : 2 * F]  # (B, T_h, F)

    n_observed = presence.sum(dim=1)                      # (B, F) giorni osservati
    mean_values = values.sum(dim=1) / n_observed.clamp(min=1.0)  # clamp: evita 0/0
    length = history_len.to(h_feats.dtype).unsqueeze(-1)  # (B, 1)
    observed_fraction = n_observed / length

    return torch.cat([mean_values, observed_fraction, length / delta_t_scale], dim=-1)


def _summarize_future_windows(
    f_feats: torch.Tensor,
    num_features: int,
    target_positions: torch.Tensor,
    delta_t_scale: float,
) -> torch.Tensor:
    """Riassunto causale del futuro: per ogni step, solo i giorni da t0+1 al suo target.

    Per lo step h con target al giorno pos[h] (0-based in future) usa i giorni
    0..pos[h] inclusi: lo step 1 non vede mai il meteo successivo al suo target.

    Per calcolare in un colpo solo le medie su finestre di lunghezza diversa
    usiamo le somme cumulative: la somma dei giorni 0..pos[h] è il valore della
    somma cumulativa in posizione pos[h].

    Returns:
        (B, 3, 2F + 1): media delle feature osservate nella finestra, frazione di
        giorni osservati, lunghezza della finestra (giorni, scalata).
    """
    F = num_features
    values = f_feats[..., :F]           # (B, T_f, F)
    presence = f_feats[..., F : 2 * F]  # (B, T_f, F)

    # Somme cumulative lungo il tempo, poi lettura nei giorni dei 3 target.
    window_sum = gather_time_steps(values.cumsum(dim=1), target_positions)          # (B, 3, F)
    window_observed = gather_time_steps(presence.cumsum(dim=1), target_positions)   # (B, 3, F)
    window_len = (target_positions + 1).to(f_feats.dtype).unsqueeze(-1)            # (B, 3, 1)

    mean_values = window_sum / window_observed.clamp(min=1.0)
    observed_fraction = window_observed / window_len

    return torch.cat([mean_values, observed_fraction, window_len / delta_t_scale], dim=-1)


class ResidualMLPBaseline(nn.Module):
    """Baseline neurale: riassunto causale delle sequenze + MLP + testa dei quantili.

    Per ciascuno dei 3 step futuri costruisce un vettore con 5 blocchi:
      1. riassunto della history (medie delle feature osservate, frazioni osservate);
      2. il giorno t0 (ultimo giorno di history: stato attuale di vegetazione e meteo);
      3. riassunto causale del futuro, solo dai giorni t0+1 fino al target dello step;
      4. il giorno esatto del target (meteo previsto e calendario di quel giorno);
      5. la codifica continua dell'orizzonte (target_delta_t, in giorni).
    I blocchi 1-4 sono costruiti da build_step_features, quindi includono i flag
    di presenza: la rete distingue i dati mancanti dai valori nella media.

    Il vettore passa in un MLP (Linear -> LayerNorm -> GELU -> Dropout, ripetuto
    `num_layers` volte), con gli stessi pesi per tutti e 3 gli step, e infine
    nella MonotonicQuantileHead (quantili ordinati, previsione residua su last_obs).

    Scelte motivate:
    - LayerNorm invece di BatchNorm: normalizza ogni campione da solo, senza
      statistiche accumulate sul train, quindi è più robusta agli shift di ood-s e ood-t;
    - GELU: attivazione liscia, gradienti più regolari di ReLU;
    - Dropout: riduce l'overfitting sul train.

    Limite (voluto, è una baseline): le medie perdono l'ordine dei giorni
    (pioggia al giorno +1 o al giorno +10 pesano uguale).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 16,
        residual: bool = True,
        init_spread: float = 0.5,
        delta_t_scale: float = DEFAULT_DELTA_T_SCALE,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features deve essere positivo.")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim e num_layers devono essere positivi.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve essere in [0, 1).")

        self.num_features = num_features
        self.delta_t_scale = delta_t_scale

        # Codifica sinusoidale dell'orizzonte di ciascun target (blocco 5).
        self.time_encoding = ContinuousTimeEncoding(time_dim)

        # Dimensione del vettore per step: 4 blocchi da (2F + 1) + codifica temporale.
        in_dim = 4 * step_feature_dim(num_features) + time_dim

        # MLP: num_layers blocchi Linear -> LayerNorm -> GELU -> Dropout.
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Testa finale: 3 quantili ordinati per ciascuno dei 3 indici.
        self.head = MonotonicQuantileHead(hidden_dim, residual=residual, init_spread=init_spread)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        history = batch["history"]
        F = history.shape[-1]
        if F != self.num_features:
            raise ValueError(
                f"Il batch ha {F} feature ma il modello è stato creato con num_features={self.num_features}."
            )
        positions = batch["future_target_positions"]  # (B, 3)

        # --- Input mask-aware per ogni giorno: [valori, flag di presenza, delta_t scalato].
        h_feats = build_step_features(
            history, batch["history_mask"], batch["history_delta_t"],
            batch["history_padding_mask"], self.delta_t_scale,
        )  # (B, T_h, 2F + 1)
        f_feats = build_step_features(
            batch["future"], batch["future_mask"], batch["future_delta_t"],
            batch["future_padding_mask"], self.delta_t_scale,
        )  # (B, T_f, 2F + 1)

        # --- Blocchi 1-2: uguali per i 3 step (dipendono solo dal passato).
        history_summary = _summarize_history(h_feats, F, batch["history_len"], self.delta_t_scale)
        t0_index = (batch["history_len"] - 1).unsqueeze(1)                 # (B, 1)
        t0_features = gather_time_steps(h_feats, t0_index).squeeze(1)      # (B, 2F + 1)
        past = torch.cat([history_summary, t0_features], dim=-1)          # (B, 2(2F + 1))
        past = past.unsqueeze(1).expand(-1, N_FUTURE, -1)                  # (B, 3, 2(2F + 1))

        # --- Blocchi 3-5: specifici di ciascuno step.
        future_windows = _summarize_future_windows(f_feats, F, positions, self.delta_t_scale)  # (B, 3, 2F + 1)
        target_days = gather_time_steps(f_feats, positions)                                    # (B, 3, 2F + 1)
        horizon = self.time_encoding(batch["target_delta_t"])                                  # (B, 3, time_dim)

        x = torch.cat([past, future_windows, target_days, horizon], dim=-1)  # (B, 3, in_dim)

        # --- MLP (stessi pesi per i 3 step) e testa dei quantili.
        return self.head(self.mlp(x), batch["last_obs"])  # (B, 3, 3, 3)
