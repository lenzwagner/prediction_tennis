"""
Odds Integrator — tennis-data.co.uk
=====================================
Lädt historische ATP-Match-Odds (Bet365, Pinnacle, Max, Avg) von tennis-data.co.uk,
matched sie gegen unsere Match-Daten via Spielername + Datum + Turnier,
und fügt implied-probability Features hinzu.

Features:
  - market_implied_p1: implied probability aus AvgW / (1/AvgW + 1/AvgL)
  - market_margin:     Buchmacher-Margin (Overround)
  - odds_p1, odds_p2:  Durchschnittliche Quoten

Match-Rate typischerweise 60-75% (Namensvarianten, fehlende Turniere).
"""

import os, re, time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from difflib import SequenceMatcher
from functools import lru_cache

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "odds")
os.makedirs(CACHE_DIR, exist_ok=True)

ATP_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
WTA_URL = "http://www.tennis-data.co.uk/{year}w/{year}.xlsx"

YEARS = list(range(2021, 2027))


def _download_year(year: int, tour: str = "atp") -> pd.DataFrame | None:
    cache_path = os.path.join(CACHE_DIR, f"{tour}_{year}.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    url = (ATP_URL if tour == "atp" else WTA_URL).format(year=year)
    print(f"  Lade {url} ...")
    try:
        from io import BytesIO
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content), engine="openpyxl")
        df["_tour"] = tour
        df.to_pickle(cache_path)
        time.sleep(0.3)
        return df
    except Exception as e:
        print(f"  Fehler bei {tour} {year}: {e}")
        return None


def _normalize_name(name: str) -> str:
    """Normalisiert Spielernamen für fuzzy-matching."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"['\-\.]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _canonical_name(name: str) -> str:
    """
    Vereinheitlicht Namen auf "nachname initial"-Format.
    Unser Format: "C. Zhao"  → "zhao c"
    Odds-Format:  "Zhao C."  → "zhao c"
    """
    n = _normalize_name(name)
    parts = n.split()
    if not parts:
        return ""
    # Erkenne Initial (ein Buchstabe, evtl. mit Punkt)
    if len(parts) >= 2 and len(parts[0]) == 1:
        # Format "C Zhao" → initial first → "zhao c"
        initial = parts[0]
        last = " ".join(parts[1:])
        return f"{last} {initial}"
    # Format "Zhao C" oder "Zhao C A" — letzter 1-buchstabiger Teil = initial
    if len(parts[-1]) == 1:
        return " ".join(parts[:-1]) + " " + parts[-1]
    return n


def _name_similarity(a: str, b: str) -> float:
    ca, cb = _canonical_name(a), _canonical_name(b)
    if ca and cb and ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


def load_odds_data(years: list[int] = YEARS, tours: list[str] = ("atp", "wta")) -> pd.DataFrame:
    """Lädt und kombiniert Odds-Daten für ATP und WTA."""
    dfs = []
    for tour in tours:
        for y in years:
            df = _download_year(y, tour)
            if df is not None and len(df) > 0:
                df["_source_year"] = y
                dfs.append(df)
    if not dfs:
        print("  Keine Odds-Daten geladen.")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  {len(combined)} Odds-Zeilen geladen (ATP + WTA).")
    return combined


def _standardize_odds_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vereinheitlicht Spaltennamen über verschiedene Jahre."""
    col_map = {}
    cols = {c.lower(): c for c in df.columns}

    for canon, variants in [
        ("winner",   ["winner", "w"]),
        ("loser",    ["loser",  "l"]),
        ("date",     ["date"]),
        ("surface",  ["surface", "court"]),
        ("avgw",     ["avgw", "avg w", "b365w"]),
        ("avgl",     ["avgl", "avg l", "b365l"]),
        ("maxw",     ["maxw", "max w"]),
        ("maxl",     ["maxl", "max l"]),
        ("psw",      ["psw"]),
        ("psl",      ["psl"]),
        ("tournament", ["tournament"]),
        ("round",    ["round"]),
        ("series",   ["series"]),
    ]:
        for v in variants:
            if v in cols:
                col_map[cols[v]] = canon
                break

    df = df.rename(columns=col_map)
    return df


def build_odds_lookup(odds_df: pd.DataFrame) -> dict:
    """
    Baut ein Lookup-Dict: (date_str, winner_name_norm, loser_name_norm) → odds_dict
    """
    if odds_df.empty:
        return {}

    df = _standardize_odds_df(odds_df)
    required = {"winner", "loser", "date"}
    if not required.issubset(df.columns):
        print(f"  Fehlende Spalten in Odds-Daten: {required - set(df.columns)}")
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "winner", "loser"])

    lookup = {}
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        w = _canonical_name(str(row["winner"]))
        l = _canonical_name(str(row["loser"]))

        avgw = row.get("avgw", np.nan)
        avgl = row.get("avgl", np.nan)
        maxw = row.get("maxw", np.nan)
        maxl = row.get("maxl", np.nan)

        entry = {
            "avgw": avgw, "avgl": avgl,
            "maxw": maxw, "maxl": maxl,
        }
        key = (date_str, w, l)
        lookup[key] = entry

    return lookup


def _find_odds(lookup: dict, date_str: str, p1_name: str, p2_name: str,
               p1_won: bool, tol_days: int = 1) -> dict | None:
    """
    Sucht Odds für ein Match. Probiert exaktes Datum ± tol_days,
    dann fuzzy-Namensmatching.
    Returns: dict mit avgw/avgl aus Sicht des Gewinners, oder None.
    """
    from datetime import datetime, timedelta

    p1n = _canonical_name(p1_name)
    p2n = _canonical_name(p2_name)
    wn  = p1n if p1_won else p2n
    ln  = p2n if p1_won else p1n

    cd = datetime.strptime(date_str, "%Y-%m-%d")

    for delta in range(-tol_days, tol_days + 1):
        d = (cd + timedelta(days=delta)).strftime("%Y-%m-%d")
        key = (d, wn, ln)
        if key in lookup:
            return lookup[key]

    # Fuzzy-Suche: gleicher Tag ± 1, beste Namensähnlichkeit
    best_sim, best_entry = 0.0, None
    for delta in range(-tol_days, tol_days + 1):
        d = (cd + timedelta(days=delta)).strftime("%Y-%m-%d")
        for (kd, kw, kl), entry in lookup.items():
            if kd != d:
                continue
            sim = (SequenceMatcher(None, wn, kw).ratio() + SequenceMatcher(None, ln, kl).ratio()) / 2
            if sim > best_sim and sim >= 0.75:
                best_sim = sim
                best_entry = entry

    return best_entry


def add_odds_features(X: pd.DataFrame, y: np.ndarray,
                      df_matches: pd.DataFrame,
                      odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt `market_implied_p1`, `market_margin`, `odds_p1`, `odds_p2`
    zu Feature-Matrix X hinzu. Fehlende Werte: NaN (→ später median-imputed).
    """
    lookup = build_odds_lookup(odds_df)
    print(f"  Odds-Lookup: {len(lookup)} Einträge")

    market_implied, market_margin, odds_p1_col, odds_p2_col = [], [], [], []
    matched = 0

    for i, (_, row) in enumerate(df_matches.iterrows()):
        date_str = row["event_date"].strftime("%Y-%m-%d") if hasattr(row["event_date"], "strftime") else str(row["event_date"])[:10]
        p1_name  = str(row.get("event_first_player",  row.get("first_player_key",  "")))
        p2_name  = str(row.get("event_second_player", row.get("second_player_key", "")))
        p1_won   = row["event_winner"] == "First Player"

        entry = _find_odds(lookup, date_str, p1_name, p2_name, p1_won)

        if entry and not (pd.isna(entry.get("avgw")) or pd.isna(entry.get("avgl"))):
            aw = float(entry["avgw"])
            al = float(entry["avgl"])
            imp_w = 1 / aw
            imp_l = 1 / al
            margin = imp_w + imp_l
            imp_p1 = (imp_w if p1_won else imp_l) / margin  # normalized
            odds_p1 = aw if p1_won else al
            odds_p2 = al if p1_won else aw

            market_implied.append(imp_p1)
            market_margin.append(margin)
            odds_p1_col.append(odds_p1)
            odds_p2_col.append(odds_p2)
            matched += 1
        else:
            market_implied.append(np.nan)
            market_margin.append(np.nan)
            odds_p1_col.append(np.nan)
            odds_p2_col.append(np.nan)

    match_rate = matched / len(df_matches) * 100
    print(f"  Match-Rate: {matched}/{len(df_matches)} ({match_rate:.1f}%)")

    X = X.copy()
    X["market_implied_p1"] = market_implied
    X["market_margin"]     = market_margin
    X["odds_p1"]           = odds_p1_col
    X["odds_p2"]           = odds_p2_col

    # Median-Imputation für fehlende Werte
    for col in ["market_implied_p1", "market_margin", "odds_p1", "odds_p2"]:
        med = X[col].median()
        X[col] = X[col].fillna(med)

    return X


def main():
    """Testet den Odds-Download und zeigt Beispielzeilen."""
    print("Lade Odds-Daten...")
    odds_df = load_odds_data(years=[2024, 2025])
    if odds_df.empty:
        print("Keine Daten.")
        return

    df_std = _standardize_odds_df(odds_df)
    print(f"\nSpalten: {list(df_std.columns)}")
    print(f"\nBeispiel-Zeilen:")
    show_cols = [c for c in ["date", "tournament", "winner", "loser", "avgw", "avgl", "maxw", "maxl"] if c in df_std.columns]
    print(df_std[show_cols].dropna(subset=["winner"]).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
