# Coffee_Supplier_Selection

Modello di ottimizzazione bi-obiettivo per la selezione dei fornitori e l'allocazione degli ordini di caffè verde, sviluppato per il corso di Ottimizzazione (CdLM Data Science per le Strategie Aziendali, Università della Calabria) — **Gruppo Simplex**.

Il progetto costruisce un sistema di supporto alle decisioni per una torrefazione che acquista caffè verde da cinque fornitori (Colombia, Etiopia, Brasile, Vietnam, Indonesia) su un orizzonte di 4 trimestri, bilanciando **costo di approvvigionamento** e **qualità/affidabilità** dei fornitori tramite programmazione lineare intera mista (MILP) e generazione della frontiera di Pareto.

## Autori

- Cristian Tedesco
- Maria Grazia Tassone
- Nabil Larhram

## Struttura del repository

    data/                              dataset di input (prezzi storici ICO, produzione, LPI)
    outputs/                           risultati del modello di ottimizzazione (CSV, grafici, log)
    outputs_prophet/                   output del forecasting Prophet sui prezzi (grafici, metriche)
    data_pipeline.py                   preprocessing dati + forecasting prezzi (Prophet) -> prezzi_forecast_cit.csv
    simplex_supplier_selection.py      costruzione del modello Pyomo, frontiera di Pareto, sensitivity analysis
    README.md

`data_pipeline.py` prepara i parametri di input (in particolare i prezzi previsti `c_it`); `simplex_supplier_selection.py` costruisce il modello di ottimizzazione (`build_model()`), calcola la matrice dei payoff, genera la frontiera di Pareto e le analisi di sensitività, salvando tutto in `outputs/`.

## Sintesi logica del modello

**Decisioni:** quantità ordinata `x_it` (fornitore i, periodo t), selezione del fornitore `y_i` (binaria), attivazione del fornitore nel periodo `z_it` (binaria).

**Obiettivi in conflitto:**
- minimizzare il costo totale: costi variabili `c_it · x_it` + costi fissi d'ordine `f_i · y_i`
- massimizzare il punteggio complessivo di qualità/affidabilità: `Σ r_i · y_i`

**Vincoli principali:** soddisfacimento della domanda totale `d`; capacità minima/massima per fornitore-periodo (lotto minimo `o_it`, capacità `O_it`); budget massimo `b`; tolleranza al ritardo complessivo `α` (basata sui tassi di ritardo `α_i` per fornitore); numero massimo di fornitori attivabili `n`; vincolo di collegamento `z_it ≤ y_i`.

Non esiste una soluzione che ottimizzi entrambi gli obiettivi simultaneamente: il modello genera quindi l'insieme delle soluzioni Pareto-efficienti (frontiera), tra cui la torrefazione può scegliere il compromesso più adatto.

**Metodo di risoluzione:** matrice dei payoff (ottimizzazione dei due obiettivi singolarmente) seguita da generazione della frontiera con **epsilon-constraint classico**, confrontato con la variante **AUGMECON** (che evita soluzioni debolmente dominate tramite variabile di scarto in obiettivo). Solver: **CBC**.

## Parametri scelti

| Parametro | Descrizione | Valore |
|---|---|---|
| `c_it` | Prezzo unitario, da forecast Prophet su serie storiche ICO (1990–2018) convertite in EUR/kg | vedi tabella fornitori sotto |
| `r_i` | Punteggio qualità (0–100), da reputazione delle origini nel cupping | 90 / 88 / 80 / 68 / 65 |
| `f_i` | Costo fisso d'ordine (EUR) | 1200 / 1000 / 1500 / 800 / 900 |
| `α_i` | Tasso di ritardo, da Logistics Performance Index 2023 (Banca Mondiale) | 0.073 / 0.100 / 0.029 / 0.020 / 0.047 |
| `O_it` | Capacità produttiva, da quota produzione ICO (`total-production.csv`), scala 1.5·d | 1800 / 930 / 7210 / 3620 / 1430 (quintali) |
| `d` | Domanda complessiva (4 trimestri) | 10 000 quintali |
| `o_it` | Lotto minimo d'ordine (uniforme) | 300 quintali |
| `b` | Budget massimo (1.4 · d · prezzo medio) | 6 865 000 EUR |
| `n` | Numero massimo di fornitori attivabili | 3 |
| `α` | Soglia massima di ritardo medio complessivo | 0.06 |

**Fornitori:** S1 Colombia, S2 Etiopia, S3 Brasile, S4 Vietnam, S5 Indonesia — mappati sui quattro gruppi ICO (Colombian Milds, Other Milds, Brazilian Naturals, Robustas). I prezzi `c_it` sono previsti con Prophet (`changepoint_prior_scale=0.1`, stagionalità annuale attiva, MAPE medio di validazione ~9.7%). Qualità e prezzo sono allineati (i fornitori più pregiati sono anche i più costosi), mentre qualità e affidabilità di consegna sono disallineate (le arabica migliori vengono da paesi con logistica meno performante).

## Risultati principali

**Matrice dei payoff** (estremi della frontiera):

| Scenario | Costo (EUR) | Punteggio |
|---|---|---|
| Minimizzazione costo | 4 296 362 | 148 |
| Massimizzazione punteggio | 4 807 815 | 258 |

**Frontiera di Pareto** (5 punti non dominati, metodo AUGMECON = metodo classico, 0 soluzioni debolmente dominate su 21 sottoproblemi):

| Punto | Costo (EUR) | Punteggio | Fornitori attivi |
|---|---|---|---|
| P1 | 4 296 362 | 148 | S3+S4 |
| P2 | 4 307 684 | 213 | S3+S4+S5 |
| P3 | 4 311 824 | 238 | S1+S3+S4 |
| P4 | 4 694 132 | 246 | S1+S2+S4 |
| P5 | 4 789 845 | 258 | S1+S2+S3 |

La frontiera mostra un "ginocchio" marcato: da P1 a P3 un incremento di costo minimo (+0.4%) porta a un forte guadagno di punteggio (148→238); da P3 a P5 servono invece incrementi di costo molto più consistenti (+11%) per guadagnare solo 20 punti addizionali — rendimenti marginali decrescenti. Gli ordini si concentrano tipicamente su T4 nei punti a basso costo; nei punti ad alto punteggio (P4, P5) i fornitori premium S1/S2 vengono introdotti anche scaglionando gli ordini su più periodi.

**Analisi di sensitività:**
- **α (tolleranza ritardo):** poco informativa nello scenario base — il fornitore più economico (Vietnam) è anche il più affidabile, quindi il vincolo non è quasi mai stringente.
- **n (numero massimo fornitori):** il costo minimo resta invariato (~2 fornitori usati comunque); il punteggio massimo raggiungibile cresce con n (da 170 con n=2 fino a 391 con n=5).
- **b (budget):** il piano di costo minimo non cambia fino a budget molto ristretti; solo sotto ~4.46M EUR il punteggio massimo raggiungibile scende (da 258 a 238).
- **d (domanda):** sia il costo minimo sia il costo del punto a punteggio massimo crescono in modo pressoché lineare con la domanda.
- **Tempi di calcolo:** 96 sotto-problemi risolti con CBC in 26.3s totali (~0.27s medi a sotto-problema).