"""
Tennis Match Win Probability Predictor
=======================================
Features:
  - ELO-Rating (gesamt + pro Untergrund: Clay/Hard/Grass)
  - Aktuelle Form (letzte 5 / 10 Spiele, inkl. untergrundspezifisch)
  - Head-to-Head (gesamt + Untergrund)
  - Grand Slam / Pre-Grand-Slam Flag
  - Qualifying Flag
  - Heimvorteil (Spieler-Nationalität == Turnierland)
  - Verletzungsrisiko (Retirement in letzten 30 Tagen oder ungewöhnliche Pause)
  - Zu verteidigende Punkte (Rundenabschnitt im selben Turnier Vorjahr)
  - Tage seit letztem Match (Ermüdung/Frische)
  - Turnierrunde (Tiefe)
"""

import os, json, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("TENNIS_API_KEY") or "83f1ca0d81403233614126fb77fe4d1dd9f7e28c878e6d64e700e6e4a9d38202"
BASE_URL = "https://api.api-tennis.com/tennis/"
CACHE_DIR = "cache"
DATA_FILE = "matches_raw.pkl"
MODEL_FILE = "tennis_model.pkl"
TODAY = datetime.today().strftime("%Y-%m-%d")

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Grand Slam detection ───────────────────────────────────────────────────────
GRAND_SLAMS = {"Australian Open", "Roland Garros", "Wimbledon", "US Open"}
GS_DATES = {
    2021: [("Australian Open","2021-02-08"),("Roland Garros","2021-05-24"),
           ("Wimbledon","2021-06-28"),("US Open","2021-08-30")],
    2022: [("Australian Open","2022-01-17"),("Roland Garros","2022-05-22"),
           ("Wimbledon","2022-06-27"),("US Open","2022-08-29")],
    2023: [("Australian Open","2023-01-16"),("Roland Garros","2023-05-28"),
           ("Wimbledon","2023-07-03"),("US Open","2023-08-28")],
    2024: [("Australian Open","2024-01-14"),("Roland Garros","2024-05-26"),
           ("Wimbledon","2024-07-01"),("US Open","2024-08-26")],
    2025: [("Australian Open","2025-01-12"),("Roland Garros","2025-05-25"),
           ("Wimbledon","2025-06-30"),("US Open","2025-08-25")],
    2026: [("Australian Open","2026-01-18"),("Roland Garros","2026-05-24"),
           ("Wimbledon","2026-06-29"),("US Open","2026-08-31")],
}

def _build_pre_gs():
    pre = set()
    for year, events in GS_DATES.items():
        for name, start in events:
            dt = datetime.strptime(start, "%Y-%m-%d")
            for i in range(1, 15):
                pre.add((dt - timedelta(days=i)).strftime("%Y-%m-%d"))
    return pre

PRE_GS_DATES = _build_pre_gs()

def is_grand_slam(name: str) -> bool:
    return any(gs.lower() in name.lower() for gs in GRAND_SLAMS)

def is_pre_grand_slam(date: str) -> bool:
    return date in PRE_GS_DATES

# ── Tournament country mapping (Heimvorteil) ───────────────────────────────────
TOURNAMENT_COUNTRY = {
    "australian open": "Australia", "sydney": "Australia", "brisbane": "Australia",
    "adelaide": "Australia", "melbourne": "Australia",
    "roland garros": "France", "paris": "France", "lyon": "France",
    "strasbourg": "France", "rouen": "France", "marseille": "France",
    "montpellier": "France", "metz": "France", "bordeaux": "France",
    "wimbledon": "United Kingdom", "eastbourne": "United Kingdom",
    "nottingham": "United Kingdom", "birmingham": "United Kingdom",
    "london": "United Kingdom", "queens": "United Kingdom",
    "us open": "USA", "indian wells": "USA", "miami": "USA",
    "cincinnati": "USA", "washington": "USA", "atlanta": "USA",
    "dallas": "USA", "houston": "USA", "san jose": "USA",
    "stanford": "USA", "chicago": "USA", "charleston": "USA",
    "san diego": "USA", "cleveland": "USA", "new york": "USA",
    "los angeles": "USA",
    "halle": "Germany", "hamburg": "Germany", "munich": "Germany",
    "stuttgart": "Germany", "berlin": "Germany", "bad homburg": "Germany",
    "cologne": "Germany", "dusseldorf": "Germany", "düsseldorf": "Germany",
    "nuremberg": "Germany", "nürnberg": "Germany",
    "madrid": "Spain", "barcelona": "Spain", "mallorca": "Spain",
    "monte carlo": "Monaco",
    "rome": "Italy", "milan": "Italy", "parma": "Italy",
    "florence": "Italy", "palermo": "Italy", "naples": "Italy",
    "toronto": "Canada", "montreal": "Canada",
    "tokyo": "Japan",
    "shanghai": "China", "beijing": "China", "wuhan": "China",
    "shenzhen": "China", "guangzhou": "China",
    "auckland": "New Zealand",
    "basel": "Switzerland", "geneva": "Switzerland",
    "gstaad": "Switzerland", "lausanne": "Switzerland",
    "vienna": "Austria", "kitzbuhel": "Austria", "kitzbühel": "Austria",
    "rotterdam": "Netherlands", "hertogenbosch": "Netherlands",
    "moscow": "Russia", "st. petersburg": "Russia",
    "prague": "Czech Republic", "ostrava": "Czech Republic",
    "budapest": "Hungary",
    "buenos aires": "Argentina", "cordoba": "Argentina",
    "rio": "Brazil",
    "bogota": "Colombia",
    "acapulco": "Mexico", "guadalajara": "Mexico",
    "seoul": "South Korea",
    "doha": "Qatar",
    "dubai": "UAE", "abu dhabi": "UAE",
    "marrakech": "Morocco", "casablanca": "Morocco",
    "warsaw": "Poland", "gdynia": "Poland",
    "bucharest": "Romania",
    "bastad": "Sweden", "båstad": "Sweden",
    "umag": "Croatia",
    "portoroz": "Slovenia", "portorož": "Slovenia",
    "belgrade": "Serbia",
    "sofia": "Bulgaria",
    "nur-sultan": "Kazakhstan", "astana": "Kazakhstan",
    "tashkent": "Uzbekistan",
    "santiago": "Chile",
}

def tournament_country(name: str) -> str:
    n = name.lower().strip()
    for key, country in TOURNAMENT_COUNTRY.items():
        if key in n:
            return country
    return "Unknown"

# ── API helpers ────────────────────────────────────────────────────────────────
def api_get(method: str, params: dict, cache_key: str = None, retries: int = 3) -> dict:
    if cache_key:
        path = os.path.join(CACHE_DIR, cache_key + ".json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    p = {"method": method, "APIkey": API_KEY}
    p.update(params)
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=p, timeout=30)
            if not resp.text.strip():
                time.sleep(2 ** attempt)
                continue
            data = resp.json()
            if cache_key and data.get("success"):
                with open(os.path.join(CACHE_DIR, cache_key + ".json"), "w") as f:
                    json.dump(data, f)
            return data
        except (requests.exceptions.JSONDecodeError, ValueError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"success": 0, "result": []}

def fetch_matches_range(date_start: str, date_stop: str) -> list:
    key = f"fixtures_{date_start}_{date_stop}"
    data = api_get("get_fixtures", {"date_start": date_start, "date_stop": date_stop}, key)
    return data.get("result", [])

def fetch_standings(league: str = "ATP") -> list:
    data = api_get("get_standings", {"event_type": league}, f"standings_{league}")
    return data.get("result", [])

def fetch_h2h(p1_key: int, p2_key: int) -> dict:
    key = f"h2h_{min(p1_key,p2_key)}_{max(p1_key,p2_key)}"
    data = api_get("get_H2H", {"first_player_key": p1_key, "second_player_key": p2_key}, key)
    return data.get("result", {})

# ── Data collection ────────────────────────────────────────────────────────────
# Akzeptierte Match-Typen für Training: alles außer Junioren (für ELO-History)
def is_valid_singles(m: dict) -> bool:
    t = m.get("event_type_type", "")
    return "Singles" in t and ("Boys" not in t) and ("Girls" not in t)

# Akzeptierte Match-Typen für Vorhersagen: nur ATP/WTA/Challenger (kein ITF)
PREDICT_TYPES = {
    "Atp Singles", "Atp - Singles",
    "Wta Singles", "Wta - Singles",
    "Challenger Men Singles", "Challenger Men - Singles",
    "Challenger Women Singles", "Challenger Women - Singles",
}

def is_relevant_singles(m: dict) -> bool:
    return m.get("event_type_type", "") in PREDICT_TYPES

def collect_historical_matches(years_back: int = 5) -> pd.DataFrame:
    start_year = datetime.today().year - years_back
    current = datetime(start_year, 1, 1)
    end = datetime.today()
    all_matches = []

    while current < end:
        chunk_end = min(current + timedelta(days=13), end)
        d_start = current.strftime("%Y-%m-%d")
        d_end   = chunk_end.strftime("%Y-%m-%d")
        cache_key  = f"fixtures_{d_start}_{d_end}"
        cache_path = os.path.join(CACHE_DIR, cache_key + ".json")
        fresh = not os.path.exists(cache_path)
        if fresh:
            print(f"  Fetching {d_start} → {d_end}...", end=" ", flush=True)
        else:
            print(f"  Loading  {d_start} → {d_end}...", end=" ", flush=True)

        matches = fetch_matches_range(d_start, d_end)
        singles = [m for m in matches
                   if m.get("event_status") == "Finished"
                   and is_valid_singles(m)
                   and m.get("event_winner") in ("First Player", "Second Player")]
        print(f"{len(singles)} matches")
        all_matches.extend(singles)
        current = chunk_end + timedelta(days=1)
        if fresh:
            time.sleep(0.25)

    df = pd.DataFrame(all_matches)
    df.to_pickle(DATA_FILE)
    print(f"Gesamt {len(df)} Matches gespeichert.")
    return df

# ── Helpers ────────────────────────────────────────────────────────────────────
SURFACE_MAP = {"clay": "Clay", "hard": "Hard", "grass": "Grass", "carpet": "Hard"}

def normalize_surface(s) -> str:
    if not s:
        return "Hard"
    return SURFACE_MAP.get(str(s).lower().strip(), "Hard")

ROUND_KEYWORDS = [
    ("final", 7), ("semi", 6), ("quarter", 5),
    ("1/8", 4), ("round of 16", 4),
    ("1/16", 3), ("round of 32", 3),
    ("1/32", 2), ("round of 64", 2),
    ("1/64", 1), ("round of 128", 1),
    ("qualifying", 0), ("qual", 0),
]

def round_depth(round_str) -> int:
    s = (round_str or "").lower()
    for kw, depth in ROUND_KEYWORDS:
        if kw in s:
            return depth
    return 3  # default mid-round

def detect_qualifying(event_qualification, tournament_round) -> int:
    """1 wenn Qualifying, 0 sonst, -1 wenn unbekannt (None/fehlend)."""
    if str(event_qualification).strip().lower() == "true":
        return 1
    round_str = (tournament_round or "").lower()
    if "qual" in round_str:
        return 1
    if event_qualification is None and not tournament_round:
        return -1  # unbekannt
    return 0

def is_retirement(event_status, event_final_result) -> bool:
    return (str(event_status).lower() == "retired"
            or "ret" in str(event_final_result).lower()
            or "w/o" in str(event_final_result).lower())

def load_tournament_surfaces() -> dict:
    path = os.path.join(CACHE_DIR, "all_tournaments.json")
    if not os.path.exists(path):
        data = api_get("get_tournaments", {}, "all_tournaments")
    else:
        with open(path) as f:
            data = json.load(f)
    return {t["tournament_key"]: t.get("tournament_sourface") for t in data.get("result", [])}

# ── ELO Engine ────────────────────────────────────────────────────────────────
class EloEngine:
    def __init__(self, k_base: float = 32.0, initial: float = 1500.0):
        self.k_base   = k_base
        self.initial  = initial
        self._ratings: dict = {}   # {pk: {surface: float}}
        self._matches: dict = {}   # {pk: {surface: int}}
        self.history:  dict = defaultdict(list)
        # history für ELO-Trend: {player_key: [(date_str, elo_overall)]}
        self.history: dict = defaultdict(list)

    def _r(self, pk, surf):
        return self._ratings.get(pk, {}).get(surf, self.initial)

    def _m(self, pk, surf):
        return self._matches.get(pk, {}).get(surf, 0)

    def _set_r(self, pk, surf, val):
        if pk not in self._ratings:
            self._ratings[pk] = {}
        self._ratings[pk][surf] = val

    def _set_m(self, pk, surf, val):
        if pk not in self._matches:
            self._matches[pk] = {}
        self._matches[pk][surf] = val

    def _k(self, pk, surface):
        n = self._m(pk, surface)
        if n < 10:  return self.k_base * 2
        if n < 30:  return self.k_base * 1.5
        return self.k_base

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, winner, loser, surface, date: str = None):
        for surf in [surface, "overall"]:
            ra = self._r(winner, surf)
            rb = self._r(loser,  surf)
            ea = self.expected(ra, rb)
            self._set_r(winner, surf, ra + self._k(winner, surf) * (1 - ea))
            self._set_r(loser,  surf, rb + self._k(loser,  surf) * (0 - (1 - ea)))
            self._set_m(winner, surf, self._m(winner, surf) + 1)
            self._set_m(loser,  surf, self._m(loser,  surf) + 1)
        if date:
            self.history[winner].append((date, self._r(winner, "overall")))
            self.history[loser].append((date,  self._r(loser,  "overall")))

    def get(self, pk, surface):
        return self._r(pk, surface)

    def trend(self, pk, current_date: str, days: int = 90) -> float:
        h = self.history[pk]
        if not h:
            return 0.0
        cd = datetime.strptime(current_date, "%Y-%m-%d")
        cutoff = (cd - timedelta(days=days)).strftime("%Y-%m-%d")
        past = [elo for d, elo in h if d <= cutoff]
        elo_past = past[-1] if past else self.initial
        return self._r(pk, "overall") - elo_past

# ── Form Tracker ───────────────────────────────────────────────────────────────
def _parse_sets(event_final_result) -> int:
    """Parst '2 - 1' → 3 Sätze gespielt. Fallback = 2."""
    try:
        parts = str(event_final_result).split("-")
        return sum(int(p.strip()) for p in parts if p.strip().isdigit())
    except Exception:
        return 2

class FormTracker:
    def __init__(self):
        self.history: dict = defaultdict(list)       # [(date, surface, won)]
        self.last_date: dict = {}
        self.recent_retired: dict = defaultdict(list)
        self.sets_log: dict = defaultdict(list)      # [(date, sets_played)]

    def record(self, pk, date, surface, won, loser_retired=False, sets_played=2):
        self.history[pk].append((date, surface, won))
        self.last_date[pk] = date
        self.sets_log[pk].append((date, sets_played))
        if loser_retired and not won:
            self.recent_retired[pk].append(date)

    def recent_win_rate(self, pk, n=10, surface=None):
        h = self.history[pk]
        if surface:
            h = [(d, s, w) for d, s, w in h if s == surface]
        recent = h[-n:]
        if not recent: return 0.5
        return sum(w for _, _, w in recent) / len(recent)

    def days_since_last(self, pk, current_date):
        last = self.last_date.get(pk)
        if not last: return 30.0
        return max((datetime.strptime(current_date, "%Y-%m-%d")
                    - datetime.strptime(last, "%Y-%m-%d")).days, 0)

    def _median_gap(self, pk) -> float:
        """Mittlere Pausenlänge ZWISCHEN Turnieren (nur Gaps > 3 Tage, um Intra-Turnier-Tage herauszufiltern)."""
        h = self.history[pk]
        if len(h) < 4:
            return 10.0
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _, _ in h[-52:]]
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)
                if (dates[i+1] - dates[i]).days > 3]
        return float(np.median(gaps)) if gaps else 10.0

    def injury_risk(self, pk, current_date):
        """1 wenn Retirement in letzten 30 Tagen ODER Pause > 2.5x persönlicher Median."""
        cd = datetime.strptime(current_date, "%Y-%m-%d")
        # Signal 1: Retirement in letzten 30 Tagen
        for rd in self.recent_retired.get(pk, []):
            days = (cd - datetime.strptime(rd, "%Y-%m-%d")).days
            if 0 <= days <= 30:
                return 1
        # Signal 2: Pause ungewöhnlich lang für diesen Spieler (individueller Schwellwert)
        gap = self.days_since_last(pk, current_date)
        median = self._median_gap(pk)
        if gap > max(median * 2.5, 21):  # mind. 21 Tage absolut
            return 1
        return 0

    def sets_last_n_days(self, pk, current_date: str, days: int = 7) -> int:
        """Anzahl Sätze gespielt in den letzten `days` Tagen (Müdigkeits-Feature)."""
        cd = datetime.strptime(current_date, "%Y-%m-%d")
        cutoff = (cd - timedelta(days=days)).strftime("%Y-%m-%d")
        return sum(s for d, s in self.sets_log[pk] if d >= cutoff)

# ── H2H Tracker ───────────────────────────────────────────────────────────────
class H2HTracker:
    def __init__(self):
        self.records: dict = {}   # {pair: {surface: [wins, losses]}}

    def record(self, winner, loser, surface):
        pair = (min(winner, loser), max(winner, loser))
        for surf in [surface, "overall"]:
            if pair not in self.records:
                self.records[pair] = {}
            if surf not in self.records[pair]:
                self.records[pair][surf] = [0, 0]
            idx = 0 if winner == pair[0] else 1
            self.records[pair][surf][idx] += 1

    def h2h_win_rate(self, pk, opp, surface="overall"):
        pair = (min(pk, opp), max(pk, opp))
        wins = self.records.get(pair, {}).get(surface, [0, 0])
        total = wins[0] + wins[1]
        if total == 0: return 0.5
        return wins[0 if pk == pair[0] else 1] / total

# ── Defending Points ───────────────────────────────────────────────────────────
def build_defending_lookup(df: pd.DataFrame) -> dict:
    """Gibt {(player_key, tournament_key, year): best_round_depth} zurück."""
    lookup = {}
    for _, row in df.iterrows():
        year = pd.to_datetime(row["event_date"]).year
        t_key = row.get("tournament_key")
        rd = round_depth(row.get("tournament_round"))
        for pkey in [int(row["first_player_key"]), int(row["second_player_key"])]:
            k = (pkey, t_key, year)
            lookup[k] = max(lookup.get(k, 0), rd)
    return lookup

# ── Player country lookup ──────────────────────────────────────────────────────
def build_player_countries() -> dict:
    """Gibt {player_key: country} zurück."""
    countries = {}
    for league in ["ATP", "WTA"]:
        for entry in fetch_standings(league):
            pk = entry.get("player_key")
            c  = entry.get("country", "")
            if pk:
                countries[pk] = c
    return countries

# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    ts_map = load_tournament_surfaces()
    defending_lookup = build_defending_lookup(df)
    player_countries = build_player_countries()

    elo  = EloEngine()
    form = FormTracker()
    h2h  = H2HTracker()

    rows = []
    for _, row in df.iterrows():
        p1   = int(row["first_player_key"])
        p2   = int(row["second_player_key"])
        date = row["event_date"].strftime("%Y-%m-%d")
        year = row["event_date"].year
        t_key   = row.get("tournament_key")
        t_name  = row.get("tournament_name", "")
        t_round = row.get("tournament_round", "")
        t_type  = row.get("event_type_type", "")
        e_qual  = row.get("event_qualification")
        e_status = row.get("event_status", "")
        e_result = row.get("event_final_result", "")
        winner = row["event_winner"]
        p1_won = 1 if winner == "First Player" else 0

        surface  = normalize_surface(ts_map.get(t_key))
        surf_enc = {"Hard": 0, "Clay": 1, "Grass": 2}.get(surface, 0)

        # ── Snapshot vor Update ────────────────────────────────────────────
        elo_p1_s = elo.get(p1, surface);  elo_p2_s = elo.get(p2, surface)
        elo_p1_o = elo.get(p1, "overall"); elo_p2_o = elo.get(p2, "overall")

        f10_p1 = form.recent_win_rate(p1, 10); f10_p2 = form.recent_win_rate(p2, 10)
        f5_p1  = form.recent_win_rate(p1, 5);  f5_p2  = form.recent_win_rate(p2, 5)
        fs_p1  = form.recent_win_rate(p1, 5, surface)
        fs_p2  = form.recent_win_rate(p2, 5, surface)

        h_p1   = h2h.h2h_win_rate(p1, p2, "overall")
        hs_p1  = h2h.h2h_win_rate(p1, p2, surface)

        days_p1   = form.days_since_last(p1, date)
        days_p2   = form.days_since_last(p2, date)
        inj_p1    = form.injury_risk(p1, date)
        inj_p2    = form.injury_risk(p2, date)
        sets7_p1  = form.sets_last_n_days(p1, date, 7)
        sets7_p2  = form.sets_last_n_days(p2, date, 7)
        trend_p1  = elo.trend(p1, date, 90)
        trend_p2  = elo.trend(p2, date, 90)

        # Defending points: bester Rundenstand im selben Turnier Vorjahr
        def_p1 = defending_lookup.get((p1, t_key, year - 1), 0)
        def_p2 = defending_lookup.get((p2, t_key, year - 1), 0)

        # Heimvorteil
        tc = tournament_country(t_name)
        home_p1 = 1 if tc != "Unknown" and player_countries.get(p1, "") == tc else 0
        home_p2 = 1 if tc != "Unknown" and player_countries.get(p2, "") == tc else 0

        is_gs   = 1 if is_grand_slam(t_name) else 0
        is_pre  = 1 if is_pre_grand_slam(date) else 0
        rd_dep  = round_depth(t_round)
        is_atp  = 1 if ("Atp" in t_type or "Challenger Men" in t_type or "Itf Men" in t_type) else 0
        is_qual = 1 if detect_qualifying(e_qual, t_round) == 1 else 0

        feat = {
            "elo_diff_surf":      elo_p1_s - elo_p2_s,
            "elo_diff_overall":   elo_p1_o - elo_p2_o,
            "elo_p1_surf":        elo_p1_s,
            "elo_p2_surf":        elo_p2_s,
            "form_p1_10":         f10_p1,
            "form_p2_10":         f10_p2,
            "form_diff_10":       f10_p1 - f10_p2,
            "form_p1_5":          f5_p1,
            "form_p2_5":          f5_p2,
            "form_diff_5":        f5_p1 - f5_p2,
            "form_p1_surf":       fs_p1,
            "form_p2_surf":       fs_p2,
            "form_diff_surf":     fs_p1 - fs_p2,
            "h2h_p1":             h_p1,
            "h2h_p1_surf":        hs_p1,
            "days_p1":            days_p1,
            "days_p2":            days_p2,
            "days_rest_diff":     days_p2 - days_p1,
            "injury_p1":          inj_p1,
            "injury_p2":          inj_p2,
            "injury_diff":        inj_p2 - inj_p1,
            "sets7_p1":           sets7_p1,
            "sets7_p2":           sets7_p2,
            "sets7_diff":         sets7_p2 - sets7_p1,
            "elo_trend_p1":       trend_p1,
            "elo_trend_p2":       trend_p2,
            "elo_trend_diff":     trend_p1 - trend_p2,
            "defending_p1":       def_p1,
            "defending_p2":       def_p2,
            "defending_diff":     def_p1 - def_p2,
            "home_p1":            home_p1,
            "home_p2":            home_p2,
            "home_diff":          home_p1 - home_p2,
            "is_grand_slam":      is_gs,
            "is_pre_gs":          is_pre,
            "is_qualifying":      is_qual,
            "surface":            surf_enc,
            "round_depth":        rd_dep,
            "is_atp":             is_atp,
        }
        rows.append((feat, p1_won))

        # ── Update nach Snapshot ───────────────────────────────────────────
        winner_key    = p1 if p1_won else p2
        loser_key     = p2 if p1_won else p1
        loser_retired = is_retirement(e_status, e_result)
        sets_played   = _parse_sets(e_result)

        elo.update(winner_key, loser_key, surface, date)
        form.record(p1, date, surface, bool(p1_won), loser_retired and not p1_won, sets_played)
        form.record(p2, date, surface, not bool(p1_won), loser_retired and p1_won, sets_played)
        h2h.record(winner_key, loser_key, surface)

    X = pd.DataFrame([r for r, _ in rows])
    y = np.array([l for _, l in rows])
    # Jahresspalte für Walk-Forward-CV (wird vor Training entfernt)
    X["_year"] = df["event_date"].dt.year.values
    return X, y, elo, form, h2h, ts_map, player_countries

# ── Model training ─────────────────────────────────────────────────────────────
def train_model(X: pd.DataFrame, y: np.ndarray):
    import lightgbm as lgb
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    lgbm_base = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        num_leaves=40, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    xgb_base = XGBClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )

    print(f"\nTraining Ensemble (LightGBM + XGBoost) auf {len(X)} Matches...")

    # Walk-Forward CV
    years = sorted(X["_year"].unique()) if "_year" in X.columns else []
    wf_scores = []
    if len(years) >= 3:
        for i in range(2, len(years)):
            train_mask = X["_year"] < years[i]
            val_mask   = X["_year"] == years[i]
            if val_mask.sum() < 100:
                continue
            X_tr = X[train_mask].drop(columns=["_year"])
            X_va = X[val_mask].drop(columns=["_year"])
            y_tr = y[train_mask.values]
            y_va = y[val_mask.values]
            lgbm_base.fit(X_tr, y_tr)
            xgb_base.fit(X_tr, y_tr)
            p = (lgbm_base.predict_proba(X_va)[:, 1] + xgb_base.predict_proba(X_va)[:, 1]) / 2
            wf_scores.append(roc_auc_score(y_va, p))
        print(f"Walk-Forward AUC (Ensemble): {np.mean(wf_scores):.4f} ± {np.std(wf_scores):.4f}  "
              f"(Jahre: {[years[i] for i in range(2, len(years))]})")

    X_fit = X.drop(columns=["_year"], errors="ignore")
    lgbm_cal = CalibratedClassifierCV(lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        num_leaves=40, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    ), cv=5, method="isotonic")
    xgb_cal = CalibratedClassifierCV(XGBClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    ), cv=5, method="isotonic")
    lgbm_cal.fit(X_fit, y)
    xgb_cal.fit(X_fit, y)

    try:
        lgbm_base.fit(X_fit, y)
        imp = pd.Series(lgbm_base.feature_importances_, index=X_fit.columns)
        print("\nTop Features (LightGBM):")
        print(imp.sort_values(ascending=False).head(15).to_string())
    except Exception:
        pass

    return lgbm_cal, xgb_cal

# ── Rankings for prediction ────────────────────────────────────────────────────
def build_ranking_lookup() -> dict:
    rankings = {}
    for league in ["ATP", "WTA"]:
        for e in fetch_standings(league):
            name = e.get("player", "").lower().strip()
            rankings[name] = {"rank": int(e.get("place", 999)), "points": int(e.get("points", 0))}
    return rankings

# ── Predictions ────────────────────────────────────────────────────────────────
def _load_player_names() -> dict:
    path = os.path.join(os.path.dirname(__file__), "cache", "player_names.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def predict_today(lgbm_model, xgb_model, elo: EloEngine, form: FormTracker,
                  h2h: H2HTracker, ts_map: dict, player_countries: dict, feature_cols: list,
                  odds_lookup: dict = None):
    print(f"\n{'='*65}")
    print(f"  VORHERSAGEN FÜR {TODAY}")
    print(f"{'='*65}")

    player_names = _load_player_names()

    matches = fetch_matches_range(TODAY, TODAY)
    singles = [m for m in matches if is_relevant_singles(m)]

    if not singles:
        print("Keine ATP/WTA/Challenger Singles heute gefunden.")
        return

    results = []
    for m in singles:
        p1_key  = int(m["first_player_key"])
        p2_key  = int(m["second_player_key"])
        p1_name = m["event_first_player"]
        p2_name = m["event_second_player"]
        t_name  = m.get("tournament_name", "")
        t_key   = m.get("tournament_key")
        t_round = m.get("tournament_round", "")
        t_type  = m.get("event_type_type", "")
        date    = m.get("event_date", TODAY)
        time_   = m.get("event_time", "")
        e_qual  = m.get("event_qualification")
        year    = int(date[:4])

        surface  = normalize_surface(ts_map.get(t_key))
        surf_enc = {"Hard": 0, "Clay": 1, "Grass": 2}.get(surface, 0)

        tc = tournament_country(t_name)
        home_p1 = 1 if tc != "Unknown" and player_countries.get(p1_key, "") == tc else 0
        home_p2 = 1 if tc != "Unknown" and player_countries.get(p2_key, "") == tc else 0

        def_p1 = 0  # defending lookup not accessible here; use 0 for today
        def_p2 = 0

        qual_raw = detect_qualifying(e_qual, t_round)
        is_qual  = 1 if qual_raw == 1 else 0
        qual_display = "1" if qual_raw == 1 else ("?" if qual_raw == -1 else "0")

        # Odds-Feature: aus Live-Quoten falls vorhanden, sonst Median-Imputation
        market_implied_p1 = np.nan
        market_margin     = np.nan
        odds_p1_val       = np.nan
        odds_p2_val       = np.nan
        if odds_lookup:
            from odds_integrator import _find_odds
            entry = _find_odds(odds_lookup, date, p1_name, p2_name, p1_won=True)
            if entry and not (pd.isna(entry.get("avgw")) or pd.isna(entry.get("avgl"))):
                aw, al = float(entry["avgw"]), float(entry["avgl"])
                margin = 1/aw + 1/al
                market_implied_p1 = (1/aw) / margin
                market_margin     = margin
                odds_p1_val, odds_p2_val = aw, al

        feat = {
            "elo_diff_surf":    elo.get(p1_key, surface) - elo.get(p2_key, surface),
            "elo_diff_overall": elo.get(p1_key, "overall") - elo.get(p2_key, "overall"),
            "elo_p1_surf":      elo.get(p1_key, surface),
            "elo_p2_surf":      elo.get(p2_key, surface),
            "form_p1_10":       form.recent_win_rate(p1_key, 10),
            "form_p2_10":       form.recent_win_rate(p2_key, 10),
            "form_diff_10":     form.recent_win_rate(p1_key, 10) - form.recent_win_rate(p2_key, 10),
            "form_p1_5":        form.recent_win_rate(p1_key, 5),
            "form_p2_5":        form.recent_win_rate(p2_key, 5),
            "form_diff_5":      form.recent_win_rate(p1_key, 5) - form.recent_win_rate(p2_key, 5),
            "form_p1_surf":     form.recent_win_rate(p1_key, 5, surface),
            "form_p2_surf":     form.recent_win_rate(p2_key, 5, surface),
            "form_diff_surf":   form.recent_win_rate(p1_key, 5, surface) - form.recent_win_rate(p2_key, 5, surface),
            "h2h_p1":           h2h.h2h_win_rate(p1_key, p2_key, "overall"),
            "h2h_p1_surf":      h2h.h2h_win_rate(p1_key, p2_key, surface),
            "days_p1":          form.days_since_last(p1_key, date),
            "days_p2":          form.days_since_last(p2_key, date),
            "days_rest_diff":   form.days_since_last(p2_key, date) - form.days_since_last(p1_key, date),
            "injury_p1":        form.injury_risk(p1_key, date),
            "injury_p2":        form.injury_risk(p2_key, date),
            "injury_diff":      form.injury_risk(p2_key, date) - form.injury_risk(p1_key, date),
            "sets7_p1":         form.sets_last_n_days(p1_key, date, 7),
            "sets7_p2":         form.sets_last_n_days(p2_key, date, 7),
            "sets7_diff":       form.sets_last_n_days(p2_key, date, 7) - form.sets_last_n_days(p1_key, date, 7),
            "elo_trend_p1":     elo.trend(p1_key, date, 90),
            "elo_trend_p2":     elo.trend(p2_key, date, 90),
            "elo_trend_diff":   elo.trend(p1_key, date, 90) - elo.trend(p2_key, date, 90),
            "defending_p1":     def_p1,
            "defending_p2":     def_p2,
            "defending_diff":   def_p1 - def_p2,
            "home_p1":          home_p1,
            "home_p2":          home_p2,
            "home_diff":        home_p1 - home_p2,
            "is_grand_slam":    1 if is_grand_slam(t_name) else 0,
            "is_pre_gs":        1 if is_pre_grand_slam(date) else 0,
            "is_qualifying":    is_qual,
            "surface":          surf_enc,
            "round_depth":      round_depth(t_round),
            "is_atp":           1 if ("Atp" in t_type or "Challenger Men" in t_type or "Itf Men" in t_type) else 0,
            "market_implied_p1": market_implied_p1,
            "market_margin":     market_margin,
            "odds_p1":           odds_p1_val,
            "odds_p2":           odds_p2_val,
        }

        X_pred = pd.DataFrame([feat])
        # Fülle fehlende Odds-Werte mit 0.5 / 1.05 / 2.0 (Median aus Trainingsdaten)
        X_pred["market_implied_p1"] = X_pred["market_implied_p1"].fillna(0.5)
        X_pred["market_margin"]     = X_pred["market_margin"].fillna(1.05)
        X_pred["odds_p1"]           = X_pred["odds_p1"].fillna(2.0)
        X_pred["odds_p2"]           = X_pred["odds_p2"].fillna(2.0)
        X_pred = X_pred[feature_cols]

        p_lgbm = lgbm_model.predict_proba(X_pred)[0]
        p_xgb  = xgb_model.predict_proba(X_pred)[0]
        proba  = (p_lgbm + p_xgb) / 2
        p1_prob, p2_prob = proba[1], proba[0]

        results.append({
            "time":        time_,
            "tournament":  t_name,
            "type":        t_type,
            "surface":     surface,
            "round":       t_round or "?",
            "p1":          p1_name,
            "p2":          p2_name,
            "p1_fullname": player_names.get(p1_key, p1_name),
            "p2_fullname": player_names.get(p2_key, p2_name),
            "p1_prob":     p1_prob,
            "p2_prob":     p2_prob,
        })

    results.sort(key=lambda r: abs(r["p1_prob"] - 0.5), reverse=True)

    HDR = (f"{'Turnier':<22} {'Typ':<22} {'Surf':<6} "
           f"{'Spieler 1':<24} {'Win%':>6}  "
           f"{'Spieler 2':<24} {'Win%':>6}")
    print(f"\n{HDR}")
    print("-" * len(HDR))

    for r in results:
        p1_pct = f"{r['p1_prob']*100:.1f}%"
        p2_pct = f"{r['p2_prob']*100:.1f}%"
        fav1   = "◀" if r["p1_prob"] > 0.5 else " "
        fav2   = "◀" if r["p2_prob"] > 0.5 else " "
        print(
            f"{r['tournament']:<22} {r['type']:<22} {r['surface']:<6} "
            f"{r['p1']:<24} {p1_pct:>6}{fav1}  "
            f"{r['p2']:<24} {p2_pct:>6}{fav2}"
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(f"predictions_{TODAY}.csv", index=False)

    # JSON für Android App
    import json
    json_data = {
        "date": TODAY,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matches": [
            {
                "time":       r["time"],
                "tournament": r["tournament"],
                "type":       r["type"],
                "surface":    r["surface"],
                "round":      r["round"],
                "p1":          r["p1"],
                "p1_fullname": r["p1_fullname"],
                "p2":          r["p2"],
                "p2_fullname": r["p2_fullname"],
                "p1_prob":    round(float(r["p1_prob"]), 4),
                "p2_prob":    round(float(r["p2_prob"]), 4),
                "favorite":   r["p1_fullname"] if r["p1_prob"] > r["p2_prob"] else r["p2_fullname"],
                "confidence": round(float(abs(r["p1_prob"] - 0.5) * 2), 4),
            }
            for r in results
        ]
    }
    with open("predictions_latest.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Gespeichert: predictions_{TODAY}.csv + predictions_latest.json ({len(results)} Matches)")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = collect_historical_matches(years_back=5)
    print(f"\nDatensatz: {len(df)} Matches")

    print("Baue Features (ELO, Form, H2H, Heimvorteil, Verletzung, Defending)...")
    X, y, elo, form, h2h, ts_map, player_countries = build_features(df)

    # Odds-Features hinzufügen
    print("Lade Odds-Daten (ATP + WTA)...")
    from odds_integrator import load_odds_data, add_odds_features, build_odds_lookup
    odds_df = load_odds_data()
    df_sorted = df.copy()
    df_sorted["event_date"] = pd.to_datetime(df_sorted["event_date"])
    df_sorted = df_sorted.sort_values("event_date").reset_index(drop=True)
    X = add_odds_features(X, y, df_sorted, odds_df)
    print(f"Feature-Matrix: {X.shape}")

    if os.path.exists(MODEL_FILE):
        print(f"\nLade bestehendes Modell aus {MODEL_FILE}...")
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        lgbm_model   = bundle["lgbm_model"]
        xgb_model    = bundle["xgb_model"]
        feature_cols = bundle["feature_cols"]
    else:
        lgbm_model, xgb_model = train_model(X, y)
        feature_cols = [c for c in X.columns if c != "_year"]
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"lgbm_model": lgbm_model, "xgb_model": xgb_model,
                         "feature_cols": feature_cols}, f)
        print(f"Modell gespeichert: {MODEL_FILE}")

    odds_lookup = build_odds_lookup(odds_df)
    predict_today(lgbm_model, xgb_model, elo, form, h2h, ts_map, player_countries,
                  feature_cols, odds_lookup)


if __name__ == "__main__":
    main()
