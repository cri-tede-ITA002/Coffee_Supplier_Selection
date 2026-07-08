# -*- coding: utf-8 -*-
"""
Progetto di Ottimizzazione - Modello bi-obiettivo di selezione dei fornitori
Gruppo Simplex

Modello matematico (1)-(12) della traccia in Pyomo, risolto con CBC/CPLEX.
Frontiera di Pareto generata con il metodo AUGMECON (epsilon-constraint
augmentato): min costi (obiettivo 1), max punteggio (obiettivo 2).

Lo script include anche:
- confronto tra epsilon-constraint classico e AUGMECON;
- piano di approvvigionamento dettagliato per i punti di Pareto;
- analisi di sensitivita' ceteris paribus su alpha, n, b e D.

Interpretazione dei parametri nella sensitivita':
- n = numero massimo di fornitori selezionabili (vincolo 8). Con n >= T il
      vincolo (9) si disattiva (nessun limite sui periodi per fornitore),
      quindi per la sensitivity su n si usa un parametro p separato, fissato
      a 3, per non confondere l'effetto dei due vincoli;
- b = budget massimo disponibile (vincolo 6);
- D = domanda totale da soddisfare (vincolo 3, implementata come d nel
      dataset).
"""

import os
import sys
import logging
import traceback
import time
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt

# Silenzia i warning di Pyomo
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

TEMPI_SOLVE = []
AUG_DELTA = 1e-3          # coefficiente di augmentation di AUGMECON
GRID_POINTS = 20          # n. di punti della griglia epsilon

# Cartella di output.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Imposta a mano un solver (licenza free:"cbc", licenza ibm:"cplex") per forzarlo,
# oppure lascia None per la scelta automatica, ma richiede che almeno uno sia disponibile.
FORCE_SOLVER = None
SOLVER_CANDIDATES = ["cplex", "cbc"]

# Nome del solver effettivamente usato (impostato in main).
SOLVER_NAME = None


# =============================================================================
# 0. GESTIONE DEL SOLVER (istanza nuova a ogni solve)
# =============================================================================

def _is_available(solver):
    try:
        return bool(solver.available(exception_flag=False))   # legacy
    except TypeError:
        try:
            return bool(solver.available())                   # APPSI
        except Exception:
            return False
    except Exception:
        return False


def resolve_solver_name():
    """Trova il primo solver disponibile e ne restituisce il nome."""
    if FORCE_SOLVER:
        return FORCE_SOLVER
    for name in SOLVER_CANDIDATES:
        try:
            s = pyo.SolverFactory(name)
        except Exception:
            continue
        if s is not None and _is_available(s):
            return name
    raise RuntimeError(
        "Nessun solver disponibile.\n"
        "Installa CBC: conda install -c conda-forge coincbc"
    )


def solve_model(m):
    """Crea un'istanza NUOVA del solver e risolve (evita crash APPSI su Windows)."""
    solver = pyo.SolverFactory(SOLVER_NAME)
    t0 = time.perf_counter()
    res = solver.solve(m)
    TEMPI_SOLVE.append(time.perf_counter() - t0)
    return res


def _feasible(results):
    tc = getattr(results, "termination_condition", None)
    if tc is not None:  # APPSI
        return str(tc).lower().endswith("optimal")
    try:  # legacy
        return (results.solver.termination_condition ==
                pyo.TerminationCondition.optimal)
    except Exception:
        return True


# =============================================================================
# 1. DATASET PSEUDO-REALE
# =============================================================================

def _prezzi_forecast(path="data/prezzi_forecast_cit.csv"):
    """Legge i prezzi attesi per gruppo (EUR/kg) prodotti da test.py (Prophet).
    Se il file manca, usa valori di ripiego e avvisa."""
    try:
        df = pd.read_csv(path, index_col="periodo")
        return df, True
    except Exception:
        print(
            f"[dataset] '{path}' non trovato: uso prezzi di ripiego. "
            "Esegui prima test.py per i prezzi reali previsti da Prophet."
        )
        fallback = pd.DataFrame(
            {
                "Colombian Milds": [5.80, 5.85, 5.78, 5.90],
                "Other Milds": [5.60, 5.66, 5.59, 5.72],
                "Brazilian Naturals": [5.10, 5.16, 5.08, 5.20],
                "Robustas": [3.45, 3.50, 3.42, 3.55],
            },
            index=["T1", "T2", "T3", "T4"],
        )
        return fallback, False


def _capacita_da_produzione(paese_di, periods, cap_tot,
                            path="data/total-production.csv", n_anni=5):
    """Deriva le capacita' O_i ripartendo cap_tot in proporzione
    alla produzione media recente."""
    prod_share = {}
    try:
        df = pd.read_csv(path)
        anni = [c for c in df.columns if c.isdigit()][-n_anni:]
        prod = {}
        for sup, paese in paese_di.items():
            riga = df[df["total_production"].str.strip().str.lower()
                      == paese.lower()]
            prod[sup] = float(riga[anni].mean(axis=1).iloc[0]) if len(riga) else 1.0
        tot = sum(prod.values())
        prod_share = {s: prod[s] / tot for s in paese_di}
    except Exception:
        print(f"[dataset] '{path}' non trovato: capacita' uniformi di ripiego.")
        prod_share = {s: 1.0 / len(paese_di) for s in paese_di}

    O = {}
    for s in paese_di:
        cap_periodo = max(500, round(prod_share[s] * cap_tot, -1))  # arrotonda a 10
        for t in periods:
            O[(s, t)] = cap_periodo
    return O, prod_share


def build_dataset():
    periods = ["T1", "T2", "T3", "T4"]
    suppliers = ["S1", "S2", "S3", "S4", "S5"]

    # mappatura fornitore -> Paese / gruppo ICO
    paese_di = {"S1": "Colombia", "S2": "Ethiopia", "S3": "Brazil",
                "S4": "Viet Nam", "S5": "Indonesia"}
    gruppo_di = {"S1": "Colombian Milds", "S2": "Other Milds",
                 "S3": "Brazilian Naturals", "S4": "Robustas", "S5": "Robustas"}
    spread = {"S1": 1.00, "S2": 1.00, "S3": 1.00, "S4": 1.00, "S5": 1.03}

    # punteggio r_i - criterio qualita'
    r = {"S1": 90, "S2": 88, "S3": 80, "S4": 68, "S5": 65}
    # % media di consegna in ritardo
    alpha_i = {"S1": 0.04, "S2": 0.06, "S3": 0.03, "S4": 0.08, "S5": 0.10}
    # costo fisso di ordine
    f = {"S1": 1200, "S2": 1000, "S3": 1500, "S4": 800, "S5": 900}

    # costi c_it dai prezzi attesi
    prezzi, reali = _prezzi_forecast()
    c = {}
    for i in suppliers:
        for t in periods:
            eur_kg = float(prezzi.loc[t, gruppo_di[i]]) * spread[i]
            c[(i, t)] = round(eur_kg * 100, 2)  # EUR/quintale

    # domanda, lotto minimo, capacita'
    d = 10000
    o = {(i, t): 300 for i in suppliers for t in periods}
    O, quote = _capacita_da_produzione(paese_di, periods, cap_tot=1.5 * d)

    # budget coerente con la scala dei prezzi
    prezzo_medio = sum(c.values()) / len(c)
    b = round(1.4 * d * prezzo_medio, -3)

    if reali:
        print("[dataset] prezzi c_it dai forecast Prophet; "
              "capacita' O_i proporzionali alla produzione ICO.")

    return {
        "I": suppliers, "T": periods,
        "r": r, "f": f, "alpha_i": alpha_i,
        "c": c, "O": O, "o": o,
        "d": d, "b": b,
        "alpha": 0.06,
        "n": 3,
    }


# =============================================================================
# 2. COSTRUZIONE DEL MODELLO (vincoli 3-12, due obiettivi come Expression)
# =============================================================================

def build_model(data):
    m = pyo.ConcreteModel(name="SupplierSelection")
    m.I = pyo.Set(initialize=data["I"])
    m.T = pyo.Set(initialize=data["T"])

    m.r = pyo.Param(m.I, initialize=data["r"])
    m.f = pyo.Param(m.I, initialize=data["f"])
    m.alpha_i = pyo.Param(m.I, initialize=data["alpha_i"])
    m.c = pyo.Param(m.I, m.T, initialize=data["c"])
    m.O = pyo.Param(m.I, m.T, initialize=data["O"])
    m.o = pyo.Param(m.I, m.T, initialize=data["o"])
    m.d = pyo.Param(initialize=data["d"])
    m.b = pyo.Param(initialize=data["b"])
    m.alpha = pyo.Param(initialize=data["alpha"])
    m.n = pyo.Param(initialize=data["n"])
    m.p = pyo.Param(initialize=data.get("p", data["n"]))

    m.x = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.z = pyo.Var(m.I, m.T, domain=pyo.Binary)

    # obiettivi come Expression
    m.cost = pyo.Expression(
        expr=sum(m.c[i, t] * m.x[i, t] for i in m.I for t in m.T) +
             sum(m.f[i] * m.y[i] for i in m.I)
    )
    m.score = pyo.Expression(expr=sum(m.r[i] * m.y[i] for i in m.I))

    # vincoli (3)-(10) + must_use (aggiunto per evitare fornitori "fantasma")
    m.demand = pyo.Constraint(
        expr=sum(m.x[i, t] for i in m.I for t in m.T) == m.d
    )
    m.cap_max = pyo.Constraint(
        m.I, m.T, rule=lambda m, i, t: m.x[i, t] <= m.O[i, t] * m.z[i, t]
    )
    m.cap_min = pyo.Constraint(
        m.I, m.T, rule=lambda m, i, t: m.o[i, t] * m.z[i, t] <= m.x[i, t]
    )
    m.budget = pyo.Constraint(expr=m.cost <= m.b)
    m.late = pyo.Constraint(
        expr=sum(m.alpha_i[i] * sum(m.x[i, t] for t in m.T) for i in m.I)
        <= m.alpha * m.d
    )
    m.max_suppliers = pyo.Constraint(
        expr=sum(m.y[i] for i in m.I) <= m.n
    )
    m.max_periods = pyo.Constraint(
        m.I, rule=lambda m, i: sum(m.z[i, t] for t in m.T) <= m.p   # usa p, non n
    )
    m.link = pyo.Constraint(
        m.I, m.T, rule=lambda m, i, t: m.z[i, t] <= m.y[i]
    )
    m.must_use = pyo.Constraint(
        m.I, rule=lambda m, i: sum(m.z[i, t]for t in m.T) >= m.y[i]
    )
    return m


# =============================================================================
# 3. TABELLA DEI PAYOFF
# =============================================================================

def payoff_table(data):
    # min costo
    m1 = build_model(data)
    m1.obj = pyo.Objective(expr=m1.cost, sense=pyo.minimize)
    solve_model(m1)
    cost_min = pyo.value(m1.cost)
    score_lo = pyo.value(m1.score)

    # max score
    m2 = build_model(data)
    m2.obj = pyo.Objective(expr=m2.score, sense=pyo.maximize)
    solve_model(m2)
    score_max = pyo.value(m2.score)
    cost_hi = pyo.value(m2.cost)
    table = pd.DataFrame(
        {"costo": [cost_min, cost_hi],
         "punteggio": [score_lo, score_max]},
        index=["min costo", "max punteggio"],
    )
    return table, score_lo, score_max

# =============================================================================
# 4. METODO AUGMECON: generazione della frontiera di Pareto
# =============================================================================

def augmecon(data, score_lo, score_max, grid_points=GRID_POINTS, delta=AUG_DELTA):
    rng = score_max - score_lo
    if rng <= 1e-9:
        return pd.DataFrame([{"costo": None, "punteggio": score_max}]), {}

    results = []
    alloc = {}
    n_iter = grid_points + 1
    print(f"Risoluzione di {n_iter} sottoproblemi epsilon-constraint...", flush=True)

    for k in range(n_iter):
        eps = score_lo + k * rng / grid_points
        m = build_model(data)
        m.s = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, rng))
        m.eps_con = pyo.Constraint(expr=m.score - m.s == eps)
        m.obj = pyo.Objective(expr=m.cost - delta * (m.s / rng),
                              sense=pyo.minimize)
        res = solve_model(m)

        if _feasible(res):
            eps_key = round(eps, 3)
            row = {
                "eps": eps_key,
                "costo": round(pyo.value(m.cost), 2),
                "punteggio": round(pyo.value(m.score), 3),
                "fornitori": "+".join(i for i in m.I if pyo.value(m.y[i]) > 0.5),
            }
            results.append(row)
            # piano di approvvigionamento
            alloc[eps_key] = [
                {
                    "fornitore": i,
                    "periodo": t,
                    "quantita": round(pyo.value(m.x[i, t]), 2),
                }
                for i in m.I for t in m.T if pyo.value(m.x[i, t]) > 1e-6
            ]
            print(
                f" [{k+1:>2}/{n_iter}] eps={row['eps']:>7.1f} "
                f"costo={row['costo']:>10.2f} punteggio={row['punteggio']:>6.1f} "
                f"({row['fornitori']})",
                flush=True,
            )
        else:
            print(f" [{k+1:>2}/{n_iter}] eps={eps:>7.1f} -> non ammissibile",
                  flush=True)
    print(flush=True)
    return pd.DataFrame(results), alloc


def pareto_filter(df):
    pts = df.drop_duplicates(subset=["costo", "punteggio"]).copy()
    keep = []
    for idx, row in pts.iterrows():
        dominated = any(
            (o["costo"] <= row["costo"] and o["punteggio"] >= row["punteggio"] and
             (o["costo"] < row["costo"] or o["punteggio"] > row["punteggio"]))
            for _, o in pts.iterrows()
        )
        if not dominated:
            keep.append(idx)
    return pts.loc[keep].sort_values("costo").reset_index(drop=True)


# =============================================================================
# 4-bis. CONFRONTO: epsilon-constraint CLASSICO vs AUGMECON
# =============================================================================

def epsilon_classic(data, score_lo, score_max, grid_points=GRID_POINTS):
    """Metodo epsilon-constraint CLASSICO (Haimes 1971)."""
    rng = score_max - score_lo
    if rng <= 1e-9:
        return pd.DataFrame([{
            "eps": round(score_max, 3),
            "costo": None,
            "punteggio": score_max,
            "fornitori": ""
        }])

    results = []
    n_iter = grid_points + 1
    print("Metodo CLASSICO: risoluzione di "
          f"{n_iter} sottoproblemi (min costo, score >= eps)...",
          flush=True)

    for k in range(n_iter):
        eps = score_lo + k * rng / grid_points
        m = build_model(data)
        m.eps_con = pyo.Constraint(expr=m.score >= eps)
        m.obj = pyo.Objective(expr=m.cost, sense=pyo.minimize)
        res = solve_model(m)
        if _feasible(res):
            results.append({
                "eps": round(eps, 3),
                "costo": round(pyo.value(m.cost), 2),
                "punteggio": round(pyo.value(m.score), 3),
                "fornitori": "+".join(i for i in m.I if pyo.value(m.y[i]) > 0.5),
            })
    print(flush=True)
    return pd.DataFrame(results)


def compare_methods(classic_df, aug_df):
    """Allinea i due metodi per valore di eps e segnala i punti del metodo
    classico debolmente dominati."""
    merged = pd.merge(
        classic_df[["eps", "costo", "punteggio"]].rename(
            columns={"costo": "costo_clas", "punteggio": "punt_clas"}
        ),
        aug_df[["eps", "costo", "punteggio"]].rename(
            columns={"costo": "costo_aug", "punteggio": "punt_aug"}
        ),
        on="eps", how="outer"
    ).sort_values("eps").reset_index(drop=True)

    def _nota(r):
        if pd.isna(r["punt_clas"]) or pd.isna(r["punt_aug"]):
            return ""
        if (abs(r["costo_clas"] - r["costo_aug"]) < 1e-3 and
                r["punt_aug"] > r["punt_clas"] + 1e-6):
            return "classico DEBOLMENTE DOMINATO"
        return "coincidono"

    merged["nota"] = merged.apply(_nota, axis=1)
    n_weak = int((merged["nota"] == "classico DEBOLMENTE DOMINATO").sum())
    return merged, n_weak


def build_detailed_plan(front, alloc, periods):
    """Costruisce il piano di approvvigionamento x_it per ogni punto della
    frontiera, in formato lungo e tabellare."""
    long_rows = []
    for p, (_, row) in enumerate(front.iterrows(), start=1):
        punto = f"P{p}"
        for rec in alloc.get(row["eps"], []):
            long_rows.append({
                "punto": punto,
                "costo": row["costo"],
                "punteggio": row["punteggio"],
                "fornitore": rec["fornitore"],
                "periodo": rec["periodo"],
                "quantita": rec["quantita"],
            })

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    wide = (
        long_df
        .pivot_table(index=["punto", "costo", "punteggio", "fornitore"],
                     columns="periodo", values="quantita",
                     aggfunc="sum", fill_value=0)
        .reindex(columns=periods, fill_value=0)
    )
    wide["Totale"] = wide.sum(axis=1)
    wide = wide.reset_index()
    return long_df, wide


# =============================================================================
# 5. ANALISI DI SENSITIVITA'
# =============================================================================

def _payoff_detail(data):
    """Risolve min-costo e max-punteggio e restituisce tutte le metriche."""
    # min costo
    m1 = build_model(data)
    m1.obj = pyo.Objective(expr=m1.cost, sense=pyo.minimize)
    res1 = solve_model(m1)
    if not _feasible(res1):
        raise RuntimeError("modello (min costo) non ammissibile")
    costo_min = pyo.value(m1.cost)
    punteggio_costo_min = pyo.value(m1.score)
    fornitori_mincost = int(sum(pyo.value(m1.y[i]) > 0.5 for i in m1.I))

    # max punteggio
    m2 = build_model(data)
    m2.obj = pyo.Objective(expr=m2.score, sense=pyo.maximize)
    res2 = solve_model(m2)
    if not _feasible(res2):
        raise RuntimeError("modello (max punteggio) non ammissibile")
    punteggio_max = pyo.value(m2.score)
    costo_punteggio_max = pyo.value(m2.cost)
    fornitori_maxptg = int(sum(pyo.value(m2.y[i]) > 0.5 for i in m2.I))

    return {
        "costo_min": round(costo_min, 2),
        "punteggio_costo_min": round(punteggio_costo_min, 3),
        "fornitori_attivi_mincost": fornitori_mincost,
        "punteggio_max": round(punteggio_max, 3),
        "costo_punteggio_max": round(costo_punteggio_max, 2),
        "fornitori_attivi_maxptg": fornitori_maxptg,
    }


_EMPTY_DETAIL = {
    "costo_min": None,
    "punteggio_costo_min": None,
    "fornitori_attivi_mincost": None,
    "punteggio_max": None,
    "costo_punteggio_max": None,
    "fornitori_attivi_maxptg": None,
}


def sensitivity_alpha(base_data, alpha_values):
    rows = []
    for a in alpha_values:
        data = dict(base_data)
        data["alpha"] = a
        try:
            detail = _payoff_detail(data)
            rows.append({"alpha": a, **detail})
        except Exception:
            rows.append({"alpha": a, **_EMPTY_DETAIL})
    return pd.DataFrame(rows)



def sensitivity_n(base_data, n_values, p_fisso=3):
    rows = []
    for n_val in n_values:
        data = dict(base_data)
        data["n"] = n_val
        data["p"] = p_fisso  # vincolo (9) tenuto fisso: confronto ceteris paribus
        try:
            detail = _payoff_detail(data)
            rows.append({"n": n_val, **detail})
        except Exception:
            rows.append({"n": n_val, **_EMPTY_DETAIL})
    return pd.DataFrame(rows)


def sensitivity_b(base_data, b_values):
    rows = []
    for b_val in b_values:
        data = dict(base_data)
        data["b"] = round(b_val, 2)
        try:
            detail = _payoff_detail(data)
            rows.append({"b": data["b"], **detail})
        except Exception:
            rows.append({"b": data["b"], **_EMPTY_DETAIL})
    return pd.DataFrame(rows)    


def sensitivity_D(base_data, d_values):
    rows = []
    for d_val in d_values:
        data = dict(base_data)
        data["d"] = round(d_val, 2)
        try:
            detail = _payoff_detail(data)
            rows.append({"D": data["d"], **detail})
        except Exception:
            rows.append({"D": data["d"], **_EMPTY_DETAIL})
    return pd.DataFrame(rows)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    global SOLVER_NAME
    SOLVER_NAME = resolve_solver_name()
    print(f"[solver] uso: {SOLVER_NAME}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[output] cartella: {OUTPUT_DIR}", flush=True)

    data = build_dataset()

    print("\n" + "=" * 64)
    print("TABELLA DEI PAYOFF (estremi della frontiera)")
    print("=" * 64, flush=True)
    table, score_lo, score_max = payoff_table(data)
    print(table.to_string())
    print(f"\nRange del punteggio: [{score_lo:.1f}, {score_max:.1f}]", flush=True)

    print("\n" + "=" * 64)
    print("FRONTIERA DI PARETO (AUGMECON)")
    print("=" * 64, flush=True)
    raw, alloc = augmecon(data, score_lo, score_max)
    front = pareto_filter(raw)
    print(front.to_string(index=False))
    front.to_csv(os.path.join(OUTPUT_DIR, "frontiera_pareto.csv"), index=False)

    plt.figure(figsize=(8, 5.5))
    plt.plot(front["costo"], front["punteggio"],
             "o--", color="#1f4e79", markersize=8, linewidth=1.4)
    for _, row in front.iterrows():
        plt.annotate(row["fornitori"], (row["costo"], row["punteggio"]),
                     textcoords="offset points", xytext=(8, -4), fontsize=8)
    plt.xlabel("Costo totale (euro) - obiettivo 1 (min)")
    plt.ylabel("Punteggio fornitori - obiettivo 2 (max)")
    plt.title("Frontiera di Pareto - Selezione dei fornitori (AUGMECON)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "frontiera_pareto.png"), dpi=150)
    print("\n[output] salvati: frontiera_pareto.csv, frontiera_pareto.png", flush=True)

    # ---- confronto: epsilon-constraint classico vs AUGMECON ----
    print("\n" + "=" * 64)
    print("CONFRONTO: epsilon-constraint CLASSICO vs AUGMECON")
    print("=" * 64, flush=True)
    classic_raw = epsilon_classic(data, score_lo, score_max)
    cmp_df, n_weak = compare_methods(classic_raw, raw)
    print(cmp_df.to_string(index=False))
    print(f"\nPunti del metodo classico debolmente dominati: {n_weak}", flush=True)
    cmp_df.to_csv(os.path.join(OUTPUT_DIR, "confronto_metodi.csv"), index=False)

    front_classic = pareto_filter(classic_raw)
    plt.figure(figsize=(8, 5.5))
    plt.plot(front_classic["costo"], front_classic["punteggio"], "s-",
             color="#c0504d", markersize=11, linewidth=1.2, alpha=0.65,
             label="epsilon-constraint classico")
    plt.plot(front["costo"], front["punteggio"], "o--",
             color="#1f4e79", markersize=7, linewidth=1.2, label="AUGMECON")
    plt.xlabel("Costo totale (euro) - obiettivo 1 (min)")
    plt.ylabel("Punteggio fornitori - obiettivo 2 (max)")
    plt.title("Confronto metodi: epsilon-constraint classico vs AUGMECON")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confronto_metodi.png"), dpi=150)
    print("[output] salvati: confronto_metodi.csv, confronto_metodi.png", flush=True)

    # ---- piano di approvvigionamento ----
    print("\n" + "=" * 64)
    print("PIANO DI APPROVVIGIONAMENTO per punto della frontiera (x_it)")
    print("=" * 64, flush=True)
    long_df, wide_df = build_detailed_plan(front, alloc, list(data["T"]))
    print(wide_df.to_string(index=False))
    long_df.to_csv(os.path.join(OUTPUT_DIR, "piano_approvvigionamento.csv"),
                   index=False)
    wide_df.to_csv(os.path.join(OUTPUT_DIR, "piano_approvvigionamento_tab.csv"),
                   index=False)
    print("\n[output] salvati: piano_approvvigionamento.csv, "
          "piano_approvvigionamento_tab.csv", flush=True)

    # ---- sensitivita' su alpha ----
    print("\n" + "=" * 64)
    print("ANALISI DI SENSITIVITA' sulla tolleranza al ritardo (alpha)")
    print("=" * 64, flush=True)
    alpha_values = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    sens_alpha = sensitivity_alpha(data, alpha_values)
    print(sens_alpha.to_string(index=False))
    sens_alpha.to_csv(os.path.join(OUTPUT_DIR, "sensitivita_alpha.csv"),
                      index=False)

    # ---- sensitivita' su n ----
    print("\n" + "=" * 64)
    print("ANALISI DI SENSITIVITA' su n (ceteris paribus)")
    print("=" * 64, flush=True)
    n_values = [2, 3, 4, 5]
    sens_n = sensitivity_n(data, n_values)
    print(sens_n.to_string(index=False))
    sens_n.to_csv(os.path.join(OUTPUT_DIR, "sensitivita_n.csv"), index=False)

    # ---- sensitivita' su b (decrescente) ----
    print("\n" + "=" * 64)
    print("ANALISI DI SENSITIVITA' su b decrescente (ceteris paribus)")
    print("=" * 64, flush=True)
    b0 = data["b"]
    # riduzioni moderate del budget: 0%, 3%, 6%, 9%, 12%, 15%, 25%, 30%, 35%
    b_values = [round(b0 * k, -3) for k in [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.75, 0.70, 0.65]]
    sens_b = sensitivity_b(data, b_values)
    print(sens_b.to_string(index=False))
    sens_b.to_csv(os.path.join(OUTPUT_DIR, "sensitivita_b.csv"), index=False)

    # ---- sensitivita' su D (crescente) ----
    print("\n" + "=" * 64)
    print("ANALISI DI SENSITIVITA' su D crescente (ceteris paribus)")
    print("=" * 64, flush=True)
    d0 = data["d"]
    # aumento graduale della domanda: +0%, +5%, +10%, +15%, +20%, +25%, +30%
    D_values = [round(d0 * k, 0) for k in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]]
    sens_D = sensitivity_D(data, D_values)
    print(sens_D.to_string(index=False))
    sens_D.to_csv(os.path.join(OUTPUT_DIR, "sensitivita_D.csv"), index=False)


    # ---- tempi di calcolo ----
    n_solves = len(TEMPI_SOLVE)
    tot = sum(TEMPI_SOLVE)
    print("\n" + "=" * 64)
    print(f"TEMPI DI CALCOLO - solver: {SOLVER_NAME}")
    print("=" * 64)
    print(f" n. di sotto-problemi risolti : {n_solves}")
    print(f" tempo totale di solve       : {tot:.3f} s")
    print(f" tempo medio per solve      : {tot / n_solves:.4f} s")
    print(f" solve piu' lento           : {max(TEMPI_SOLVE):.4f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERRORE] lo script si e' interrotto:", flush=True)
        traceback.print_exc()
        sys.exit(1)