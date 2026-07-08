# -*- coding: utf-8 -*-
"""
Scenario A - Analisi esplorativa e previsione dei prezzi indicatori ICO
(dataset Kaggle 'indicator-prices.csv', fonte originale ICO, mensile 1990-2018).

Colonne: ICO composite + 4 gruppi (Colombian Milds, Other Milds,
Brazilian Naturals, Robustas). Valori convertiti da $/lb a cents/lb (x100).

Pipeline:
  1) EDA: statistiche, profilo lineare nel tempo, boxplot di distribuzione;
  2) previsione con Prophet (Meta), con holdout degli ultimi 24 mesi;
  3) metriche di errore sul test: MAE, RMSE, MAPE.

Dipendenze: pip install prophet pandas numpy matplotlib
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from prophet import Prophet

CSV_PATH = "data/indicator-prices.csv"      # serie storica ICO
GROUPS = ["Colombian Milds", "Other Milds", "Brazilian Naturals", "Robustas"]
TEST_MONTHS = 24                       # orizzonte di validazione (2 anni)
FORECAST_QUARTERS = 4                  # trimestri da prevedere per i c_it del modello
EUR_USD = 1.10                         # ipotesi di cambio per conversione in EUR/kg
FORECAST_CSV = "data/prezzi_forecast_cit.csv"   # output letto dal modello
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outputs_prophet")


# ----------------------------------------------------------------------------
# 1. Caricamento e pulizia
# ----------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["ds"] = pd.to_datetime(df["months"], format="%m/%Y")
    df = df.sort_values("ds").reset_index(drop=True)
    price_cols = ["ICO composite indicator"] + GROUPS
    df[price_cols] = df[price_cols] * 100        # $/lb -> cents/lb
    return df, price_cols


# ----------------------------------------------------------------------------
# 2. Analisi esplorativa
# ----------------------------------------------------------------------------

def eda(df, price_cols):
    desc = df[price_cols].describe().T
    desc["cv_%"] = (df[price_cols].std() / df[price_cols].mean() * 100).round(1)
    desc = desc.round(2)
    print("=" * 70)
    print("STATISTICHE DESCRITTIVE (cents/lb, 1990-2018)")
    print("=" * 70)
    print(desc.to_string())
    desc.to_csv(os.path.join(OUTPUT_DIR, "statistiche.csv"))

    # profilo lineare nel tempo
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colori = {"Colombian Milds": "#1f4e79", "Other Milds": "#2e8b57",
              "Brazilian Naturals": "#c0504d", "Robustas": "#8064a2"}
    for g in GROUPS:
        ax.plot(df["ds"], df[g], label=g, color=colori[g], lw=1.3)
    ax.plot(df["ds"], df["ICO composite indicator"], label="I-CIP composito",
            color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_title("Prezzi indicatori ICO per gruppo - profilo mensile (1990-2018)")
    ax.set_xlabel("Anno"); ax.set_ylabel("US cents/lb")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_profilo_lineare.png"), dpi=150)
    plt.close(fig)

    # boxplot di distribuzione
    fig, ax = plt.subplots(figsize=(8, 5))
    df[GROUPS].plot.box(ax=ax, color={"medians": "black"})
    ax.set_title("Distribuzione dei prezzi per gruppo (cents/lb)")
    ax.set_ylabel("US cents/lb"); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_boxplot.png"), dpi=150)
    plt.close(fig)
    return desc


# ----------------------------------------------------------------------------
# 3. Previsione con Prophet + metriche di errore
# ----------------------------------------------------------------------------

def metriche(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def forecast_prophet(df, gruppo):
    serie = df[["ds", gruppo]].rename(columns={gruppo: "y"})
    train = serie.iloc[:-TEST_MONTHS]
    test = serie.iloc[-TEST_MONTHS:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.1)
    m.fit(train)
    future = m.make_future_dataframe(periods=TEST_MONTHS, freq="MS")
    fc = m.predict(future)

    yhat_test = fc.set_index("ds").loc[test["ds"], "yhat"].values
    mae, rmse, mape = metriche(test["y"].values, yhat_test)

    # grafico: storico + previsione + intervallo + reale sul test
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train["ds"], train["y"], color="#1f4e79", lw=1.0, label="train")
    ax.plot(test["ds"], test["y"], color="black", lw=1.8, label="reale (test)")
    fc_test = fc[fc["ds"] >= test["ds"].iloc[0]]
    ax.plot(fc_test["ds"], fc_test["yhat"], color="#c0504d", lw=1.8,
            label="previsione Prophet")
    ax.fill_between(fc_test["ds"], fc_test["yhat_lower"], fc_test["yhat_upper"],
                    color="#c0504d", alpha=0.2, label="intervallo 80%")
    ax.set_title(f"Prophet - {gruppo}  (MAE={mae:.1f}, RMSE={rmse:.1f}, "
                 f"MAPE={mape:.1f}%)")
    ax.set_xlabel("Anno"); ax.set_ylabel("cents/lb")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    nome = gruppo.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"03_prophet_{nome}.png"), dpi=150)
    plt.close(fig)

    # componenti (trend + stagionalita')
    fig2 = m.plot_components(fc)
    fig2.savefig(os.path.join(OUTPUT_DIR, f"04_componenti_{nome}.png"), dpi=130)
    plt.close(fig2)

    return {"gruppo": gruppo, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "MAPE_%": round(mape, 2)}


def salva_forecast_trimestrale(df):
    """Riaddestra Prophet sull'INTERA serie di ogni gruppo, prevede i prossimi
    trimestri e salva un CSV (in EUR/kg) usato dal modello come c_it."""
    n_mesi = FORECAST_QUARTERS * 3
    ultimo = df["ds"].max()
    prezzi_q = {}
    for g in GROUPS:
        serie = df[["ds", g]].rename(columns={g: "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.1)
        m.fit(serie)
        fc = m.predict(m.make_future_dataframe(periods=n_mesi, freq="MS"))
        fut = fc[fc["ds"] > ultimo].set_index("ds")["yhat"]      # solo mesi futuri
        q = fut.resample("QE").mean().head(FORECAST_QUARTERS)    # media trimestrale
        # cents/lb -> EUR/kg
        prezzi_q[g] = (q / 100 / 0.453592 / EUR_USD).round(3).values

    out = pd.DataFrame(prezzi_q, index=[f"T{i+1}" for i in range(FORECAST_QUARTERS)])
    out.index.name = "periodo"
    out.to_csv(FORECAST_CSV)
    print("\n" + "=" * 70)
    print(f"PREVISIONE PROPHET dei prossimi {FORECAST_QUARTERS} trimestri (EUR/kg)")
    print("  -> salvata in", FORECAST_CSV)
    print("=" * 70)
    print(out.to_string())
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, price_cols = load_data()
    print(f"[dati] {len(df)} mesi, da {df['ds'].min():%m/%Y} a {df['ds'].max():%m/%Y}")

    eda(df, price_cols)

    print("\n" + "=" * 70)
    print(f"PREVISIONE PROPHET - holdout ultimi {TEST_MONTHS} mesi (validazione)")
    print("=" * 70)
    risultati = [forecast_prophet(df, g) for g in GROUPS]
    met = pd.DataFrame(risultati)
    print(met.to_string(index=False))
    met.to_csv(os.path.join(OUTPUT_DIR, "metriche_errore.csv"), index=False)
    print(f"\nMAPE medio sui 4 gruppi: {met['MAPE_%'].mean():.2f}%")

    # riaddestra su tutta la serie e salva i prezzi attesi per il modello
    salva_forecast_trimestrale(df)
    print(f"\n[output] grafici e tabelle in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()