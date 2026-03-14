"""
run_pipeline.py  —  Economic Stress Index v5
============================================
10-step pipeline implementing the full central-bank-grade ESI framework.

Steps
-----
  1. Download FRED data (18 series including VIX, TED spread, Retail Sales)
  2. Feature engineering (stress-oriented; higher = more stress)
  3. Build Custom FSI (KCFSI-style: PCA of credit_spread + VIX + TED + yield_spread)
  4. Exploratory Data Analysis
  5. Economic Engine Scoring (5 engines, expanding percentile rank → 0-100)
  6. Stress Index charts + annual heatmap
  7. Dynamic Factor Model (Federal Reserve / IMF methodology; Kalman Filter)
  8. Stress Regime Detection (Gaussian Mixture Model, 6 regimes)
  9. ML Validation (logistic + RF + GB on engine scores; current state dashboard)
  10. Rolling OOS Backtest (1996→present) + Crisis Autopsies

Usage
-----
  python run_pipeline.py            # all steps
  python run_pipeline.py 1 2 3      # specific steps

Requirements
-----------
  pip install fredapi numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
"""

import os, sys, time, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

for d in ["data/raw","data/processed","results/figures","results/tables","notebooks"]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)


def banner(step, title, total=10):
    print(f"\n{'='*64}")
    print(f"  STEP {step}/{total}  —  {title}")
    print(f"{'='*64}\n")


def run_step(step_num, title, func, steps_to_run, total=10):
    if steps_to_run and step_num not in steps_to_run:
        print(f"  [SKIP] Step {step_num}: {title}")
        return True
    banner(step_num, title, total)
    t0 = time.time()
    try:
        func()
        print(f"\n  ✓  Done in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ✗  FAILED: {e}")
        traceback.print_exc()
        return False


def step1_download():
    from src.data.download_fred import main
    main()

def step2_features():
    from src.features.engineer_features import main
    main()

def step3_custom_fsi():
    from src.features.build_custom_fsi import main
    main()

def step4_eda():
    import pandas as pd
    from src.analysis.eda import run_eda
    from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB
    feat = pd.read_csv(os.path.join(DATA_PROC, "fred_features.csv"),
                       index_col=0, parse_dates=True)
    run_eda(feat, RESULTS_FIG, RESULTS_TAB)

def step5_engines():
    from src.analysis.engines import main
    main()

def step6_stress_index():
    from src.analysis.stress_index import main
    main()

def step7_dfm():
    from src.analysis.dfm import main
    main()

def step8_regimes():
    from src.analysis.regime_detection import main
    main()

def step9_ml():
    from src.models.ml_validation import main
    main()

def step10_backtest_autopsies():
    import pandas as pd
    from src.models.backtest import main as backtest_main
    from src.visualization.crisis_autopsy import plot_crisis_autopsies, plot_annual_heatmap
    from config import DATA_PROC, RESULTS_FIG

    backtest_main()

    feat = pd.read_csv(os.path.join(DATA_PROC, "fred_features.csv"),
                       index_col=0, parse_dates=True)
    eng  = pd.read_csv(os.path.join(DATA_PROC, "engine_scores.csv"),
                       index_col=0, parse_dates=True)
    esi  = eng["ESI"] if "ESI" in eng.columns else eng[[c for c in eng.columns
                                                         if c not in ("RECESSION",)]].mean(axis=1)
    print("\n============================================================")
    print("  Generating Crisis Autopsies")
    print("============================================================\n")
    plot_crisis_autopsies(feat, esi, RESULTS_FIG)
    print("\n  Autopsy visualisations complete.\n")


def step11_download_oecd():
    """
    Download OECD multi-country panel (7 countries, 6 series each).
    No API key required — OECD API is publicly accessible.
    Requires internet access.
    """
    from src.data.download_oecd import main
    main()


def step12_panel_ml():
    """
    Build international panel features and run leave-one-country-out ML.
    Trains on 6 countries, evaluates on each held-out country.
    ~45 recession episodes vs 6 US-only.
    Requires step 11 to have run successfully first.
    """
    import os
    from config import DATA_RAW, DATA_PROC
    oecd_panel = os.path.join(DATA_RAW, "oecd_panel_raw.csv")
    if os.path.exists(oecd_panel):
        from src.features.engineer_panel_features import main as eng_main
        eng_main()
    from src.models.panel_ml import main
    main()


def step13_engine_momentum():
    """
    Engine momentum + causal transmission analysis.
    Estimates empirical lags: Monetary→Financial→Real→Labour.
    Adds forward transmission score as a new early warning signal.
    """
    from src.analysis.engine_momentum import main
    main()


def step14_regime_conditioned_ml():
    """
    Regime-conditioned ML — now possible with 45 international episodes.
    Trains separate models per regime type (Financial Crisis / Demand Shock /
    Inflation Shock / Monetary Squeeze). Each regime uses weights appropriate
    to THAT type of stress, not an average across all types.
    """
    from src.models.regime_conditioned_ml import main
    main()


if __name__ == "__main__":
    # Parse step args
    try:
        steps_to_run = [int(s) for s in sys.argv[1:]]
    except ValueError:
        steps_to_run = []

    all_steps = [
        (1,  "Download FRED Data",                   step1_download),
        (2,  "Feature Engineering",                  step2_features),
        (3,  "Build Custom FSI (KCFSI-style)",       step3_custom_fsi),
        (4,  "Exploratory Data Analysis",            step4_eda),
        (5,  "Economic Engine Scoring (v5)",         step5_engines),
        (6,  "Stress Index Charts",                  step6_stress_index),
        (7,  "Dynamic Factor Model (DFM)",           step7_dfm),
        (8,  "Stress Regime Detection (GMM — fixed)",step8_regimes),
        (9,  "ML Validation + Current Dashboard",    step9_ml),
        (10, "Backtest (expanding) + Autopsies",     step10_backtest_autopsies),
        (11, "Download OECD Panel (7 countries)",    step11_download_oecd),
        (12, "Panel ML — International Training",    step12_panel_ml),
        (13, "Engine Momentum + Transmission Lags",  step13_engine_momentum),
        (14, "Regime-Conditioned ML (45 episodes)",  step14_regime_conditioned_ml),
    ]

    print(f"\n{'='*64}")
    print(f"  Economic Stress Index — Pipeline v5")
    print(f"  Steps: {steps_to_run if steps_to_run else 'all'}")
    print(f"{'='*64}")

    t_total = time.time()
    results = []
    for step_num, title, func in all_steps:
        ok = run_step(step_num, title, func, steps_to_run, total=10)
        results.append((step_num, title, ok))

    elapsed = time.time() - t_total
    print(f"\n{'='*64}")
    ok_count  = sum(1 for _,_,ok in results if ok)
    skip_count = sum(1 for s,_,_ in results if steps_to_run and s not in steps_to_run)
    fail_count = len(results) - ok_count - skip_count
    if fail_count == 0:
        print(f"  ✓  Pipeline complete in {elapsed:.0f}s  "
              f"({ok_count} steps run, {skip_count} skipped)")
    else:
        print(f"  ✗  Pipeline finished with {fail_count} error(s) in {elapsed:.0f}s")
        for s, t, ok in results:
            if not ok and (not steps_to_run or s in steps_to_run):
                print(f"     FAILED: Step {s} — {t}")

    # Summary of outputs
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    tab_dir = os.path.join(BASE_DIR, "results", "tables")
    figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
    tabs = sorted([f for f in os.listdir(tab_dir) if f.endswith(".csv")])
    print(f"\n  Results saved to:")
    print(f"    data/processed/   — datasets (raw features, engine scores, DFM factors, regimes, backtest)")
    print(f"    results/figures/  — {len(figs)} charts")
    print(f"    results/tables/   — {len(tabs)} CSV tables")
    if figs:
        print(f"\n  Figures ({len(figs)}):")
        for f in figs: print(f"    {f}")
    if tabs:
        print(f"\n  Tables ({len(tabs)}):")
        for f in tabs: print(f"    {f}")
    print(f"{'='*64}\n")
