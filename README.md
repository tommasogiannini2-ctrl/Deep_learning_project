# Deep_learning_project

Progetto del corso di **Deep Learning** – Laurea Magistrale in Ingegneria dei Sistemi Intelligenti, Università Campus Bio-Medico di Roma.

**Autori:** Tommaso Giannini, Flavia Romito, Erika Oliveri

---

## Indice

1. [La traccia del progetto](#1-la-traccia-del-progetto)
   - [1.1 Obiettivo](#11-obiettivo)
   - [1.2 Indici di vegetazione da prevedere](#12-indici-di-vegetazione-da-prevedere)
   - [1.3 Dataset](#13-dataset)
   - [1.4 Nuvole e campionamento irregolare](#14-nuvole-e-campionamento-irregolare)
   - [1.5 Split dei dati e generalizzazione](#15-split-dei-dati-e-generalizzazione)
   - [1.6 Parametri fissati e scelte libere](#16-parametri-fissati-e-scelte-libere)
   - [1.7 Output probabilistico e funzione di perdita](#17-output-probabilistico-e-funzione-di-perdita)
   - [1.8 Metriche di valutazione](#18-metriche-di-valutazione)
   - [1.9 Consegna](#19-consegna)
2. [Configurazione e iperparametri](#2-configurazione-e-iperparametri)
3. [Setup](#3-setup)
4. [Struttura del repository e stato dei lavori](#4-struttura-del-repository-e-stato-dei-lavori)
5. [Pipeline dati e contratto del batch](#5-pipeline-dati-e-contratto-del-batch)
6. [Risultati dell'EDA](#6-risultati-delleda)
7. [Addestramento e valutazione](#7-addestramento-e-valutazione)
8. [Risultati](#8-risultati)
9. [Punti aperti](#9-punti-aperti)

---

## 1. La traccia del progetto

### 1.1 Obiettivo

Il progetto riguarda la **previsione dello stato di salute della vegetazione** in Europa a partire da serie temporali satellitari e da dati meteorologici.

Bisogna prevedere **tre indici di vegetazione** (NDVI, GNDVI, EVI) **circa 14 giorni in avanti**, cioè per le **prossime 3 acquisizioni a cielo sereno** del satellite. Come input si usano le osservazioni passate a cielo sereno e le covariate meteo giornaliere (passate e future).

```
  storia (3 acquisizioni clear-sky passate) + meteo giornaliero
                          │
                          ▼
                    [ nostro modello ]
                          │
                          ▼
  3 indici × 3 passi futuri × 3 quantili (0.1, 0.5, 0.9)
```

### 1.2 Indici di vegetazione da prevedere

Gli indici si calcolano dalle bande spettrali di Sentinel-2 (Blue, Green, Red, Near-Infrared):

| Indice | Formula | Significato |
|---|---|---|
| **NDVI** – Normalized Difference Vegetation Index | $\dfrac{NIR - Red}{NIR + Red}$ | Indicatore standard del verde della chioma (*canopy greenness*) |
| **GNDVI** – Green NDVI | $\dfrac{NIR - Green}{NIR + Green}$ | Più sensibile al contenuto di clorofilla |
| **EVI** – Enhanced Vegetation Index | $2.5 \cdot \dfrac{NIR - Red}{NIR + 6\,Red - 7.5\,Blue + 1}$ | Satura meno sulla vegetazione densa e riduce le interferenze atmosferiche |

I valori stanno nell'intervallo $[-1, 1]$: valori vicini a $-1$ indicano suolo nudo, acqua, rocce o vegetazione morta, mentre valori che crescono verso $1$ indicano vegetazione viva, sana e in crescita.

### 1.3 Dataset

I dati provengono da **GreenEarthNet**:

- **Minicubi Sentinel-2**: 128 × 128 pixel a 20 m di risoluzione (circa 2.56 km di lato), con bande Blue, Green, Red e NIR, anni **2017–2022**.
- **Target clear-sky**: gli indici sono mediati sui pixel di coltura validi. Esistono solo per le date senza nuvole, quindi la serie è campionata in modo irregolare.
- **Meteo giornaliero (E-OBS)**: vento, umidità, radiazione solare, pioggia, pressione al livello del mare, temperatura (minima, media, massima).
- **Area geografica**: tutta Europa, dai climi mediterranei a quelli continentali e boreali (classi di Köppen-Geiger diverse). Prevalgono frumento, mais e altri cereali.

Le **21 feature** di input comprendono: meteo (8), copertura nuvolosa (3), calendario/termini di Fourier del giorno dell'anno (7) e i valori clear-sky di NDVI, GNDVI ed EVI (3).

### 1.4 Nuvole e campionamento irregolare

Quando passa sopra un'area nuvolosa il satellite non vede il suolo, e gli indici calcolati vengono sbagliati (vicini a $-1$). Se la copertura è parziale, la media sui pixel risulta falsata verso il basso. Per questo:

- **Vanno usati solo i target clear-sky**, sia in addestramento sia in valutazione. Nel dataset ci sono anche target calcolati su tutte le date (es. `NDVI all`), ma **non vanno usati**.
- Sentinel-2 passa nominalmente ogni 5 giorni, ma nella pratica ogni 3–7 giorni. Togliendo le date nuvolose, la serie dei target diventa **sparsa e irregolare**. Il meteo invece è **giornaliero e regolare**.

**Vincoli sul preprocessing:**

1. **Niente interpolazione**: i buchi restano `NaN`. Il modello li deve gestire con un **masking** o con un'architettura che accetta valori mancanti.
2. **Finestre già tagliate**: 3 acquisizioni passate (*lookback* = 3) → 3 acquisizioni future (*horizon* = 3, circa 14 giorni), con **stride 4**.
3. **Meteo futuro rumoroso**: alle covariate meteo future è già stato aggiunto rumore, per simulare una vera previsione meteo.
4. **Normalizzazione calcolata solo sul train**: viene fornito uno `scaler.json` (media, varianza, trasformazione `arcsinh`) stimato sul train ma **non ancora applicato**. Possiamo usarlo oppure definirne uno nostro, ma va sempre stimato solo sul train e poi applicato senza modifiche agli altri split.

### 1.5 Split dei dati e generalizzazione

Si addestra **una sola volta** e poi si misura quanto il modello generalizza nello **spazio** e nel **tempo**.

| Split | Ruolo | Dove | Quando | Campioni (traccia) | Campioni (nostra EDA su `DATA/`) |
|---|---|---|---|---|---|
| `train` | Addestramento | Regioni di training | 2017–2020 (fino al 30/05/2020) | 55.338 *(refuso)* | **104.699** (84 aree) |
| `id` | Test in-distribution | Vicino alle località di training | 2020 | 3.336 | 3.336 |
| `ood-s` | Test con spostamento spaziale | **Regioni mai viste** | 2017–2019 | 25.619 | 25.576 |
| `ood-t` | Test con spostamento temporale | Stesse località di `id` | **2021–2022** | 6.512 | 6.511 |
| *hidden* | Valutazione finale | Tenuto dai docenti, mai rilasciato | – | – | – |

> ℹ️ **Nota sugli split**: i numeri riportati nella traccia per il `train` (55.338 campioni, 2017–2019) contengono un **refuso confermato**. Vanno utilizzati **tutti i dati presenti nei vari split** in `DATA/` così come forniti (104.699 campioni per `train`, 3.336 per `id`, 25.576 per `ood-s`, 6.511 per `ood-t`), senza applicare filtri temporali o sottocampionamenti.

Regole:
- Si addestra **solo sullo split `train`**.
- `id`, `ood-s` e `ood-t` servono **solo per la valutazione**.
- Gli iperparametri si scelgono su un **sottoinsieme di validazione ricavato dal `train`**.

> Conta soprattutto **quanto il modello generalizza a regioni e anni mai visti**.

### 1.6 Parametri fissati e scelte libere

| Fisso per tutti i gruppi | A nostra scelta |
|---|---|
| Finestre: 3 passi passati → 3 passi futuri clear-sky, stride 4 | **Normalizzazione**: lo scaler fornito oppure uno nostro |
| Target: NDVI, GNDVI ed EVI, tutti e tre | **Valori mancanti**: masking, o modelli che accettano `NaN` |
| 21 feature (meteo, nuvole, Fourier del giorno dell'anno, indici clear-sky) | **Feature**: selezione, embedding, feature derivate |
| Buchi lasciati come `NaN` con maschere, senza interpolazione | **Architettura**: statistica, ricorrente (RNN/LSTM), convoluzionale, Transformer, foundation model, … |
| Meteo futuro già perturbato con rumore | **Loss e training**: qualsiasi obiettivo che produca i tre quantili |
| `scaler.json` stimato sul train, non applicato | |
| Addestramento solo sullo split `train` | |
| Output: quantili 0.1, 0.5, 0.9 per ogni indice e ogni passo | |
| Valutazione su `id`, `ood-s`, `ood-t` e split nascosto | |

### 1.7 Output probabilistico e funzione di perdita

Il modello deve fare una **previsione probabilistica**: non basta una stima puntuale, servono i **quantili 0.1, 0.5 (mediana) e 0.9** per ognuno dei 3 indici e dei 3 passi futuri, quindi **3 × 3 × 3 = 27 valori** per campione.

Una scelta naturale per la loss è la **pinball loss** (o *quantile loss*), che per un quantile $\tau \in \{0.1, 0.5, 0.9\}$ vale:

$$
L_\tau(y, \hat{q}_\tau) = \max\big(\tau\,(y - \hat{q}_\tau),\; (\tau - 1)\,(y - \hat{q}_\tau)\big)
$$

In alternativa si può usare il **CRPS** (*Continuous Ranked Probability Score*).

### 1.8 Metriche di valutazione

Il modello viene confrontato con il **modello di riferimento dei docenti**, usando gli stessi split, lo stesso orizzonte e metriche calcolate sulla **scala originale degli indici** (cioè dopo aver annullato la normalizzazione).

- **Accuratezza puntuale** sulla mediana ($q = 0.5$), per ogni indice e ogni split:
  - RMSE – *Root Mean Squared Error*
  - MAE – *Mean Absolute Error*
  - WMAPE – *Weighted Mean Absolute Percentage Error*
  - MASE – *Mean Absolute Scaled Error*
- **Incertezza** sui quantili 0.1, 0.5, 0.9:
  - CRPS
  - Pinball loss media
  - **Copertura** empirica dell'intervallo all'80% $[\hat{q}_{0.1}, \hat{q}_{0.9}]$

### 1.9 Consegna

- Il progetto vale il **25% del voto finale**.
- **Presentazione**: 12 minuti esatti, seguiti da domande. I gruppi sono gli stessi della presentazione del paper.
- **Repository del codice**:
  - codice pulito, commentato ed eseguibile dall'inizio alla fine;
  - `README.md` con setup, istruzioni per addestramento e valutazione, e risultati;
  - `requirements.txt` con versioni fissate.
- **Slide**: il file `.pptx` usato nella presentazione.
- **Scadenza**: da comunicare.
- Nella valutazione contano le metriche sugli split OOD, ma anche il **rigore metodologico**, la **pulizia del codice** e la **motivazione teorica delle scelte architetturali**.

---

## 2. Configurazione e iperparametri

Tutti gli iperparametri e i percorsi del progetto sono centralizzati nel file [`config.yaml`](config.yaml) nella cartella principale. Per cambiare un parametro o fare un nuovo esperimento basta modificare quel file, senza toccare il codice Python.

### Sezioni principali di `config.yaml`

- **`seed`**: seme per la riproducibilità degli esperimenti.
- **`data`**: percorso della cartella `DATA/`, nomi degli split (`train`, `id`, `ood_s`, `ood_t`), file dello scaler del train (`scaler_file`), frazione e criterio di validazione (`val_fraction`, `val_split_by`, `val_balance_climate`), `num_workers`.
- **`preprocessing`**: normalizzazione di feature e target, riempimento dei NaN, convenzione delle maschere, padding dinamico, `use_future_meteo`, `drop_features` (per ablation), `add_delta_t`. Il significato di ogni parametro è in [`docs/batch_contract.md`](docs/batch_contract.md), sezione 5.
- **`task`**: vincoli fissati dalla traccia (`targets`, `n_past`, `n_future`, `quantiles`) — **da non modificare**.
- **`model`**: modello da usare (`name`: `persistence` oppure `residual_mlp`) e parametri di ciascun modello nel blocco `params.<nome>`. Viene letto solo il blocco del modello scelto, quindi per cambiare modello basta cambiare `name`.
- **`training`**: epoche, `batch_size`, `learning_rate`, `weight_decay`, `optimizer`, `scheduler`, `gradient_clip`, `early_stopping_patience`, `loss` (funzione di perdita, default `pinball`) e `device`.
- **`output`**: cartella di salvataggio (`dir`) e nome dell'esperimento (`experiment_name`).

### Lettura dal codice

Il modulo [`src/config.py`](src/config.py) espone `load_config()`, che legge `config.yaml` e lo restituisce come dizionario Python:

```python
from src.config import load_config

cfg = load_config()
lr = cfg["training"]["learning_rate"]
batch_size = cfg["training"]["batch_size"]
```

---

## 3. Setup

```bash
# creare e attivare l'ambiente virtuale (testato con Python 3.14)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# installare le dipendenze
pip install -r requirements.txt
```

**Dati.** La cartella `DATA/` non è versionata (vedi `.gitignore`). Va copiata nella cartella principale del progetto con la struttura fornita dal professore, cioè una sottocartella per split (`train_multi_output_clear_sky`, `val_chopped_multi_output_clear_sky`, `ood-s_chopped_multi_output_clear_sky`, `ood-t_chopped_multi_output_clear_sky`) contenente i file `.npz` e `.json` per area, più `scaler.json` nella cartella del train. I nomi delle cartelle sono in `config.yaml`, sezione `data.splits`.

**Test.**

```bash
python -m unittest discover tests
```

---

## 4. Struttura del repository e stato dei lavori

```
.
├── config.yaml                 # configurazione unica (iperparametri e percorsi)         [fatto]
├── requirements.txt            # dipendenze con versioni fissate                         [fatto]
├── main.py                     # punto di ingresso                                       [da fare]
├── docs/
│   └── batch_contract.md       # specifica del batch e dell'interfaccia modello–pipeline [fatto]
├── src/
│   ├── config.py               # load_config()                                           [fatto]
│   ├── contract.py             # costanti, tipo Batch, validatori, dummy batch           [fatto]
│   ├── eda.py                  # analisi esplorativa dei 4 split                         [fatto]
│   ├── preprocessing.py        # Dataset, normalizzazione, masking, DataLoader           [da fare]
│   ├── metrics.py              # RMSE, MAE, WMAPE, MASE, CRPS, pinball, coverage         [da fare]
│   ├── losses.py               # Pinball Loss e build_loss()                             [fatto]
│   ├── train.py                # ciclo di addestramento                                  [da fare]
│   ├── eval.py                 # valutazione su id / ood-s / ood-t                       [da fare]
│   └── models/                 # architetture (baseline, RNN, Transformer, ...)          [in corso]
│       ├── __init__.py         # registro dei modelli e build_model(cfg)                 [fatto]
│       ├── baselines.py        # persistenza e ResidualMLPBaseline                       [fatto]
│       └── common.py           # testa dei quantili, input mask-aware, codifica tempo    [fatto]
└── tests/
    ├── test_contract.py        # test sul contratto del batch                            [fatto]
    ├── test_losses_and_common.py  # test su loss e componenti comuni dei modelli         [fatto]
    └── test_models.py          # test su baseline e build_model                          [fatto]
```

L'EDA si lancia con `python -m src.eda` e salva grafici e `eda_summary.json` in `runs/eda/`.

---

## 5. Pipeline dati e contratto del batch

Tutto il team lavora sullo stesso contratto, descritto per intero in [`docs/batch_contract.md`](docs/batch_contract.md) e verificato da [`src/contract.py`](src/contract.py). Questo permette di sviluppare pipeline, modelli, loss e metriche **in parallelo**, usando `make_dummy_batch()` al posto dei file `.npz`.

**Regole principali (invarianti):**

1. **Maschere**: `True` = mancante o padding, `False` = valido. È la stessa convenzione di `np.isnan` e di `key_padding_mask` di PyTorch.
2. **Nessuna interpolazione**: dopo la normalizzazione `standardization_arcsinh` le posizioni mancanti vengono riempite con `0.0`, **sempre accompagnate dalle maschere**. Nessun tensore float del batch contiene `NaN` o `Inf`. Poiché `0.0` coincide con la media standardizzata, **ogni modello deve usare le maschere come input** (ad esempio concatenando `(~mask).float()` alle feature), altrimenti non distingue "mancante" da "valore medio".
3. **Padding a destra e dinamico**: `history` e `future` hanno lunghezze variabili. I modelli non devono assumere lunghezze fisse né usare embedding posizionali appresi; devono supportare almeno 150 giorni (in `ood-s` si arriva a 126, nel train a 70). Per il tempo si usano i `delta_t` continui.
4. **Zero leakage**: lo scaler è stimato solo sul train; `future_noise` non entra mai nel batch; le colonne degli indici in `future` sono sempre mascherate; `climate` **non è mai un input** (vale `"unknown"` nel 100% di `ood-s`) ed è esposto solo per diagnostica.
5. **Output del modello**: `preds = model(batch)` con shape `(B, 3, 3, 3)` = (batch, step futuro, indice, quantile), nella stessa scala di `batch["target"]`, con quantili monotoni per costruzione.

**Ordine delle dimensioni:** step `[1, 2, 3]`, indici `[NDVI, GNDVI, EVI]`, quantili `[0.1, 0.5, 0.9]`.

**Valutazione:** le metriche si calcolano sempre sulla scala originale, applicando l'inversa dello scaler del train alle predizioni e confrontandole con `batch["target_raw"]`.

---

## 6. Risultati dell'EDA

Riassunto dell'analisi eseguita con `python -m src.eda` su tutti i campioni dei 4 split. I controlli di integrità sono passati senza errori (0 mismatch tra maschere e `NaN`).

| Split | Aree | Campioni | Intervallo date | `hist_len` mediana (max) | `fut_len` mediana (max) |
|---|---|---|---|---|---|
| `train` | 84 | 104.699 | 2017-05-14 … 2020-05-30 | 11 (66) | 20 (70) |
| `id` | 80 | 3.336 | 2020-01-11 … 2020-12-09 | 16 (71) | 20 (105) |
| `ood_s` | 50 | 25.576 | 2017-01-10 … 2019-12-09 | 16 (126) | 20 (115) |
| `ood_t` | 80 | 6.511 | 2021-01-10 … 2022-12-09 | 11 (71) | 20 (90) |

**Struttura dei dati**

- Ogni `history` contiene esattamente 3 osservazioni clear-sky, la prima al giorno 0 e la terza all'ultimo giorno; le sequenze sono giornaliere e contigue.
- L'orizzonte effettivo dei 3 target ha mediana 5, 10 e 20 giorni (step 1, 2, 3) nel train, con code molto lunghe fuori distribuzione (fino a 105–115 giorni). I 14 giorni "nominali" sono solo un'approssimazione.
- Gli split `id` e `ood_t` condividono 80 aree, tutte presenti nel train. `train` e `ood_s` non hanno aree in comune (`ood_s` è raggruppato per fasce di latitudine, 36.25–60.75).
- Le aree nel train sono molto sbilanciate: `29SND` da sola ha 9.467 campioni (9%), mentre `30UWD` ne ha 3 e `30UUA` 4.

**Target**

- Tutti i valori sono in $[-1, 1]$, senza `NaN`.
- Correlazione molto forte tra gli indici nel train: NDVI–GNDVI 0.949, NDVI–EVI 0.885, GNDVI–EVI 0.797. Questo giustifica un modello **multi-task condiviso** che preveda i tre indici insieme.
- Shift spaziale evidente in `ood_s` (NDVI medio 0.489 contro 0.446 del train; EVI medio 0.380 contro 0.322).

**Insidie individuate**

1. **`climate` è `"unknown"` nel 100% di `ood_s`** (nel train è lo 0.5%): non va usata come input, solo per diagnostica e per stratificare la validazione.
2. **Anomalia sul vento**: nel train il 52.3% dei giorni ha `wind == 0` (probabilmente un valore mancante codificato come 0), contro il 3–4.5% degli altri split. Da trattare con attenzione (ablation, oppure come mancante con maschera).
3. **Più `NaN` nel meteo fuori dal train**: in `id` e `ood_t` la quota di `NaN` su radiazione solare, pressione, umidità, pioggia e temperatura è 3–6 volte quella del train. Il modello deve ricevere le maschere delle feature meteo e probabilmente beneficia di *feature-masking augmentation* in training.
4. **Rumore del meteo futuro crescente con l'orizzonte**: circa 5–10% della deviazione standard della feature al giorno +1, fino a circa 10–21% al giorno +25.

**Baseline di persistenza** (previsione = ultima osservazione clear-sky, per tutti e 3 gli step): è il riferimento minimo da battere. L'errore cresce con lo step ed è più alto fuori distribuzione, in particolare su `ood_s`. Valori sulla scala originale (dettaglio completo in `runs/eda/eda_summary.json`):

**MAE (media sui 3 step, e per step 1 / 2 / 3)**

| Split | NDVI | GNDVI | EVI |
|---|---|---|---|
| `train` | 0.061 (0.048 / 0.061 / 0.074) | 0.050 (0.043 / 0.049 / 0.057) | 0.056 (0.045 / 0.055 / 0.067) |
| `id` | 0.074 (0.060 / 0.074 / 0.088) | 0.061 (0.054 / 0.062 / 0.069) | 0.072 (0.060 / 0.072 / 0.086) |
| `ood_s` | 0.095 (0.077 / 0.096 / 0.113) | 0.081 (0.070 / 0.081 / 0.092) | 0.100 (0.081 / 0.100 / 0.119) |
| `ood_t` | 0.077 (0.062 / 0.076 / 0.093) | 0.065 (0.057 / 0.064 / 0.074) | 0.077 (0.063 / 0.076 / 0.091) |

**RMSE (media sui 3 step)**

| Split | NDVI | GNDVI | EVI |
|---|---|---|---|
| `train` | 0.101 | 0.089 | 0.095 |
| `id` | 0.120 | 0.108 | 0.125 |
| `ood_s` | 0.152 | 0.138 | 0.162 |
| `ood_t` | 0.125 | 0.114 | 0.127 |

Un modello utile deve scendere chiaramente sotto questi valori, soprattutto su `ood_s` e agli step 2–3, dove la persistenza peggiora di più.

---

## 7. Addestramento e valutazione

*Da completare quando `train.py`, `eval.py` e `metrics.py` saranno pronti.* Struttura prevista:

- **Addestramento**: `python main.py train` (o `python -m src.train`), con parametri letti da `config.yaml`.
- **Valutazione**: calcolo di RMSE, MAE, WMAPE, MASE, CRPS, pinball loss e copertura all'80% su `id`, `ood_s` e `ood_t`, sulla scala originale.
- **Validazione**: sottoinsieme del train separato **per area** (`val_split_by: area`), bilanciato per numero di campioni e classi di Köppen (`val_balance_climate: true`).
- **Riferimenti**: baseline di persistenza e modello dei docenti.

### Loss e componenti comuni dei modelli

Questi blocchi sono condivisi da tutte le architetture e sono già implementati e testati (`tests/test_losses_and_common.py`).

- **Loss** ([`src/losses.py`](src/losses.py)): **Pinball Loss** media sui quantili 0.1, 0.5, 0.9, che esclude i target con `target_mask == True`. Si sceglie con `training.loss` in `config.yaml` tramite `build_loss(cfg)`.
- **Testa dei quantili** (`MonotonicQuantileHead` in [`src/models/common.py`](src/models/common.py)): per ogni step e indice prevede la mediana e due scarti positivi (`softplus`), quindi $\hat{q}_{0.1} \le \hat{q}_{0.5} \le \hat{q}_{0.9}$ vale sempre, senza *quantile crossing*. Con `residual=True` (default) prevede la variazione rispetto all'ultima osservazione clear-sky (`last_obs`); all'inizio del training il modello coincide con la baseline di persistenza.
- **Input mask-aware** (`build_step_features`): per ogni giorno affianca ai valori i flag di presenza `(~mask)` e il tempo relativo `delta_t / 30`, così il modello distingue un dato mancante (riempito con `0.0`) da un valore nella media. Il numero di feature è letto dai dati, quindi funziona anche con `drop_features`.
- **Codifica del tempo** (`ContinuousTimeEncoding`): codifica sinusoidale di `delta_t` (periodi da 4 a 1000 giorni), senza parametri appresi, valida per sequenze di qualsiasi lunghezza (anche oltre i 126 giorni di `ood_s`).

### Baseline

Le baseline sono in [`src/models/baselines.py`](src/models/baselines.py) e si scelgono con `model.name` in `config.yaml` (test in `tests/test_models.py`).

- **Persistenza** (`persistence`): prevede per tutti e 3 gli step l'ultima osservazione clear-sky (`last_obs`), con i tre quantili coincidenti. Non ha parametri ed è il riferimento minimo da battere.
- **MLP residuo** (`residual_mlp`): per ciascuno dei 3 step costruisce un vettore di dimensione fissa con la media della history e il giorno $t_0$, la media del futuro **solo fino al giorno del target** (riassunto causale: lo step 1 non vede il meteo successivo) e il giorno del target, più la codifica dell'orizzonte `target_delta_t`. Il vettore passa in un MLP (`Linear → LayerNorm → GELU → Dropout`, 2 strati da 128 neuroni) e nella testa dei quantili con previsione residua. Le medie perdono l'ordine dei giorni: è il limite che i modelli sequenziali dovranno superare.

---

## 8. Risultati

*Da completare.* Verrà inserita una tabella con le metriche per ogni split (`id`, `ood_s`, `ood_t`), per indice e per modello, confrontate con la baseline di persistenza.

---

## 9. Punti aperti

- [x] **Dimensione e periodo del train**: chiarito che il valore nella traccia (55.338 campioni, 2017–2019) è un refuso; vanno usati tutti i dati presenti nei vari split di `DATA/` (104.699 campioni nel `train`) senza filtri.
- [ ] **Trattamento di `wind == 0`**: rimuovere la feature, oppure trattare gli zeri come mancanti con maschera.
- [ ] **Qualità dello split di validazione per area**: verificare le proporzioni dato lo sbilanciamento delle aree.
- [ ] **Scelta dell'architettura** e compilazione della sezione `model` di `config.yaml`.
- [ ] **Definizione di MASE** (scala di normalizzazione, ad esempio errore della persistenza sul train) da fissare e documentare.
- [ ] **Versioni di Python e delle librerie**: verificare che quelle in `requirements.txt` corrispondano a quelle realmente usate dal team.
- [ ] Scadenza della consegna e organizzazione della presentazione (12 minuti).
