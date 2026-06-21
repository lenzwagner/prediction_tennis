"""
Training Script — run once per week locally
============================================
Loads historical match data, builds features, trains the
ensemble model (LightGBM + XGBoost) and saves tennis_model.pkl.

After: git add tennis_model.pkl && git push
"""

import tennis_predictor as tp
from odds_integrator import load_odds_data, add_odds_features
import pickle, os, time

MODEL_FILE = tp.MODEL_FILE

def main():
    print("=== TENNIS MODEL TRAINING ===\n")

    t0 = time.time()

    print("[1/5] Loading historical matches (5 years)...")
    df = tp.collect_historical_matches(years_back=5)
    print(f"      {len(df)} matches loaded ({time.time()-t0:.0f}s)\n")

    print("[2/5] Building features (ELO, form, H2H, home, injury, defending)...")
    t1 = time.time()
    X, y, elo, form, h2h, ts_map, player_countries = tp.build_features(df)
    print(f"      Feature matrix: {X.shape}  ({time.time()-t1:.0f}s)\n")

    print("[3/5] Loading odds data (ATP + WTA)...")
    t2 = time.time()
    import pandas as pd
    odds_df = load_odds_data()
    df_sorted = df.copy()
    df_sorted["event_date"] = pd.to_datetime(df_sorted["event_date"])
    df_sorted = df_sorted.sort_values("event_date").reset_index(drop=True)
    X = add_odds_features(X, y, df_sorted, odds_df)
    print(f"      Final feature matrix: {X.shape}  ({time.time()-t2:.0f}s)\n")

    print("[4/5] Training ensemble (LightGBM + XGBoost) with walk-forward CV...")
    t3 = time.time()
    lgbm_model, xgb_model = tp.train_model(X, y)
    feature_cols = [c for c in X.columns if c != "_year"]
    print(f"      Training done ({time.time()-t3:.0f}s)\n")

    print("[5/5] Saving model bundle to tennis_model.pkl...")
    bundle = {
        "lgbm_model":       lgbm_model,
        "xgb_model":        xgb_model,
        "feature_cols":     feature_cols,
        "elo":              elo,
        "form":             form,
        "h2h":              h2h,
        "ts_map":           ts_map,
        "player_countries": player_countries,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)

    size_mb = os.path.getsize(MODEL_FILE) / 1024 / 1024
    total = time.time() - t0
    print(f"      Saved: {MODEL_FILE} ({size_mb:.1f} MB)")
    print(f"\n=== DONE ({total:.0f}s total) ===")
    print("\nNext steps:")
    print("  git add tennis_model.pkl && git commit -m 'Retrain' && git push")

if __name__ == "__main__":
    main()
