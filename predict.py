"""
Prediction Script — täglich morgens ausführen
==============================================
Lädt das trainierte Modell (tennis_model.pkl), holt die heutigen Matches
von der API und schreibt predictions_latest.json.

Danach: git add predictions_latest.json && git push
Oder als Mac-Cron (crontab -e):
  0 7 * * * cd /Users/lenz/Documents/Tennis_Pred && python3 predict.py && git add predictions_latest.json && git commit -m "Predictions $(date +%F)" && git push
"""

import os, pickle
import tennis_predictor as tp
from odds_integrator import load_odds_data, build_odds_lookup

MODEL_FILE = tp.MODEL_FILE

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"FEHLER: {MODEL_FILE} nicht gefunden. Erst train.py ausführen.")
        return

    print("Lade Modell...")
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    lgbm_model      = bundle["lgbm_model"]
    xgb_model       = bundle["xgb_model"]
    feature_cols    = bundle["feature_cols"]
    elo             = bundle["elo"]
    form            = bundle["form"]
    h2h             = bundle["h2h"]
    ts_map          = bundle["ts_map"]
    player_countries = bundle["player_countries"]

    print("Lade aktuelle Odds...")
    odds_df     = load_odds_data()
    odds_lookup = build_odds_lookup(odds_df)

    tp.predict_today(lgbm_model, xgb_model, elo, form, h2h,
                     ts_map, player_countries, feature_cols, odds_lookup)

if __name__ == "__main__":
    main()
