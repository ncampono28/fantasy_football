"""
Weighted Projection Model — Steps 1 & 2
==========================================
Improves on the base model by adding:
  1. Weighted recent seasons (2025 counts more than 2023)
  2. Age curve adjustments by position

Run after 01_pull_data.py:
    python 04_weighted_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")
GAMES = 17

def safe_round(x, n=0):
    return round(float(x), n) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SEASON WEIGHTS
# Recent seasons matter more than old ones
# ─────────────────────────────────────────────────────────────────────────────
SEASON_WEIGHTS = {
    2025: 0.55,
    2024: 0.25,
    2023: 0.12,
    2022: 0.05,
    2021: 0.03,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — AGE CURVES BY POSITION
# Multiplier applied to base projection based on player age
# Sources: historical NFL aging curve research
# ─────────────────────────────────────────────────────────────────────────────
AGE_CURVES = {
    "WR": {
        # WRs peak 26-28, decline sharply after 30
        22: 0.82, 23: 0.90, 24: 0.96, 25: 0.99,
        26: 1.02, 27: 1.03, 28: 1.02, 29: 0.98,
        30: 0.93, 31: 0.87, 32: 0.80, 33: 0.72,
        34: 0.63, 35: 0.55,
    },
    "RB": {
        # RBs peak early (24-26), steepest decline of any position
        22: 0.88, 23: 0.95, 24: 1.01, 25: 1.02,
        26: 1.00, 27: 0.94, 28: 0.86, 29: 0.76,
        30: 0.65, 31: 0.54, 32: 0.44,
    },
    "TE": {
        # TEs develop slowly, peak 27-30, age gracefully
        22: 0.72, 23: 0.80, 24: 0.88, 25: 0.94,
        26: 0.98, 27: 1.01, 28: 1.02, 29: 1.02,
        30: 1.00, 31: 0.97, 32: 0.92, 33: 0.86,
        34: 0.79, 35: 0.71,
    },
    "QB": {
        # QBs peak late (28-33), longest career arc
        22: 0.80, 23: 0.86, 24: 0.91, 25: 0.95,
        26: 0.98, 27: 1.00, 28: 1.02, 29: 1.03,
        30: 1.03, 31: 1.02, 32: 1.01, 33: 1.00,
        34: 0.97, 35: 0.93, 36: 0.88, 37: 0.82,
        38: 0.75, 39: 0.67, 40: 0.59,
    },
}

def get_age_multiplier(position, age):
    """Get age curve multiplier. Defaults to 1.0 if age unknown."""
    if pd.isna(age) or age is None:
        return 1.0
    age = int(age)
    curve = AGE_CURVES.get(position, {})
    if age in curve:
        return curve[age]
    # Extrapolate: if older than curve, use the last value
    if age > max(curve.keys()):
        return min(curve.values())
    if age < min(curve.keys()):
        return curve[min(curve.keys())]
    return 1.0

# ─────────────────────────────────────────────────────────────────────────────
# SCORING FORMATS
# ─────────────────────────────────────────────────────────────────────────────
SCORING = {
    "standard":   {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 0,   "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "half_ppr":   {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "ppr":        {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "te_premium": {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0.5},
}

def calc_fpts(proj, fmt, position):
    s = SCORING[fmt]
    pts  = proj.get("passing_yards", 0)   * s["pass_yd"]
    pts += proj.get("passing_tds", 0)     * s["pass_td"]
    pts += proj.get("interceptions", 0)   * s["int"]
    pts += proj.get("rushing_yards", 0)   * s["rush_yd"]
    pts += proj.get("rushing_tds", 0)     * s["rush_td"]
    te_b = s["te_bonus"] if position == "TE" else 0
    pts += proj.get("receptions", 0)      * (s["rec"] + te_b)
    pts += proj.get("receiving_yards", 0) * s["rec_yd"]
    pts += proj.get("receiving_tds", 0)   * s["rec_td"]
    return safe_round(pts, 1)

# ─────────────────────────────────────────────────────────────────────────────
# TEAM PASSING BUDGET — QB Age-Adjusted Receiving Volume
# Scales each team's projected receiving totals so they match a QB-age-
# adjusted passing budget derived from 2025 actual team passing yards.
# ─────────────────────────────────────────────────────────────────────────────

# QB passing output curve: peak at 28-29 = 1.0, anchors at 38 = 0.62, 39 = 0.54
QB_AGE_FACTORS = {
    22: 0.78, 23: 0.83, 24: 0.88, 25: 0.92, 26: 0.96, 27: 0.99,
    28: 1.00, 29: 1.00,
    30: 0.99, 31: 0.97, 32: 0.94, 33: 0.91, 34: 0.87,
    35: 0.82, 36: 0.77, 37: 0.70, 38: 0.62, 39: 0.54,
    40: 0.46,
}

# Projected 2026 starting QB per team + their age at start of 2026 season.
# age_2025 is derived as age_2026 - 1 (same QB aging one year).
# Update this dict each offseason when rosters are known.
TEAM_QB_2026 = {
    "ARI": ("Kyler Murray",        29),  # returning from 2025 injury
    "ATL": ("Michael Penix Jr",    26),  # took over mid-2024
    "BAL": ("Lamar Jackson",       30),
    "BUF": ("Josh Allen",          30),
    "CAR": ("Bryce Young",         25),
    "CHI": ("Caleb Williams",      25),
    "CIN": ("Joe Burrow",          30),
    "CLE": ("Shedeur Sanders",     24),  # 2025 draft pick
    "DAL": ("Dak Prescott",        33),
    "DEN": ("Bo Nix",              27),
    "DET": ("Jared Goff",          32),
    "GB":  ("Jordan Love",         28),
    "HOU": ("C.J. Stroud",         25),
    "IND": ("Anthony Richardson",  24),
    "JAX": ("Trevor Lawrence",     27),
    "KC":  ("Patrick Mahomes",     31),
    "LA":  ("Matthew Stafford",    38),
    "LAC": ("Justin Herbert",      29),
    "LV":  ("Aidan O'Connell",     27),
    "MIA": ("Tua Tagovailoa",      28),
    "MIN": ("J.J. McCarthy",       23),
    "NE":  ("Drake Maye",          24),
    "NO":  ("Derek Carr",          36),
    "NYG": ("Jaxson Dart",         24),
    "NYJ": ("Justin Fields",       27),
    "PHI": ("Jalen Hurts",         28),
    "PIT": ("Russell Wilson",      38),
    "SF":  ("Brock Purdy",         27),
    "SEA": ("Geno Smith",          36),
    "TB":  ("Baker Mayfield",      31),
    "TEN": ("Cam Ward",            24),  # 2025 draft pick
    "WAS": ("Jayden Daniels",      26),
}

BUDGET_MIN = 2_800   # floor: even bad offenses throw ~2,800 yds to skill players
BUDGET_MAX = 4_800   # ceiling: historically no team has exceeded this


def _qb_age_factor(age):
    """Interpolate QB age factor; clamp to defined boundaries."""
    if age in QB_AGE_FACTORS:
        return QB_AGE_FACTORS[age]
    if age < min(QB_AGE_FACTORS):
        return QB_AGE_FACTORS[min(QB_AGE_FACTORS)]
    return QB_AGE_FACTORS[max(QB_AGE_FACTORS)]


def apply_team_budget(proj_df, seasonal):
    """
    Adjust receiving volume (yards, targets, receptions) for every skill-
    position player so each team's projected receiving total matches a
    QB-age-adjusted passing budget.

    Steps
    -----
    1. Compute 2025 team passing yards from seasonal QB stats.
    2. Apply QB age-factor ratio (factor_age26 / factor_age25) to derive budget.
    3. Clamp budget to [BUDGET_MIN, BUDGET_MAX].
    4. Compute current projected team receiving yards from proj_df.
    5. Scale receiving_yards, targets, receptions by (budget / current).
       Cap scaling factor to ±30 % to prevent over-correction.
    6. Keep TD rates and all other efficiency metrics unchanged.
    7. Recalculate all four fpts columns for scaled players (vectorized).
    8. Print before/after table + top 3 most-affected players per team.
    """
    print(f"\n{'─' * 60}")
    print("  Team Passing Budget — QB Age Adjustment")
    print(f"{'─' * 60}")

    # ── Step 1: 2025 team passing yards ──────────────────────────────────────
    qb_2025 = seasonal[
        (seasonal["season"] == BASE_SEASON) & (seasonal["position"] == "QB")
    ]
    team_pass_yds_2025 = qb_2025.groupby("team")["passing_yards"].sum()

    # ── Steps 2 & 3: Budget per team ─────────────────────────────────────────
    budgets = {}
    for team, (qb_name, age_26) in TEAM_QB_2026.items():
        age_25   = age_26 - 1
        f26      = _qb_age_factor(age_26)
        f25      = _qb_age_factor(age_25)
        ratio    = f26 / f25 if f25 > 0 else 1.0
        base_yds = float(team_pass_yds_2025.get(team, 3_500))
        budget   = int(np.clip(round(base_yds * ratio), BUDGET_MIN, BUDGET_MAX))
        budgets[team] = dict(
            qb_name=qb_name, age_26=age_26, age_25=age_25,
            f26=round(f26, 3), f25=round(f25, 3), ratio=round(ratio, 4),
            pass_yds_25=int(base_yds), budget=budget,
        )

    # ── Steps 4 & 5: Scale volume columns ───────────────────────────────────
    SKILL      = {"WR", "TE", "RB"}
    SCALE_COLS = ["receiving_yards", "targets", "receptions"]

    before_df = proj_df.copy()  # snapshot for reporting

    for team, info in budgets.items():
        budget = info["budget"]
        for scenario in ("bear", "base", "bull"):
            mask = (
                proj_df["team"].eq(team) &
                proj_df["position"].isin(SKILL) &
                proj_df["scenario"].eq(scenario)
            )
            current = proj_df.loc[mask, "receiving_yards"].sum()
            if current <= 0:
                info[f"sf_{scenario}"] = 1.0
                continue
            sf = float(np.clip(budget / current, 0.70, 1.30))
            info[f"sf_{scenario}"] = round(sf, 4)
            for col in SCALE_COLS:
                proj_df.loc[mask, col] = (proj_df.loc[mask, col] * sf).round(0)

    # ── Steps 6 & 7: Recalculate fpts (vectorized over all skill rows) ───────
    sk = proj_df["position"].isin(SKILL)
    for fmt, s in SCORING.items():
        te_bonus = np.where(proj_df.loc[sk, "position"] == "TE", s["te_bonus"], 0.0)
        proj_df.loc[sk, f"fpts_{fmt}"] = (
            proj_df.loc[sk, "passing_yards"]   * s["pass_yd"] +
            proj_df.loc[sk, "passing_tds"]     * s["pass_td"] +
            proj_df.loc[sk, "interceptions"]   * s["int"]     +
            proj_df.loc[sk, "rushing_yards"]   * s["rush_yd"] +
            proj_df.loc[sk, "rushing_tds"]     * s["rush_td"] +
            proj_df.loc[sk, "receptions"]      * (s["rec"] + te_bonus) +
            proj_df.loc[sk, "receiving_yards"] * s["rec_yd"]  +
            proj_df.loc[sk, "receiving_tds"]   * s["rec_td"]
        ).round(1)

    # ── Step 8a: Before/after team totals table ───────────────────────────────
    base_mask = proj_df["scenario"].eq("base") & proj_df["position"].isin(SKILL)

    before_team = (
        before_df[base_mask].groupby("team")["receiving_yards"].sum().rename("before")
    )
    after_team = (
        proj_df[base_mask].groupby("team")["receiving_yards"].sum().rename("after")
    )
    tbl = pd.concat([before_team, after_team], axis=1)
    tbl["budget"]  = tbl.index.map(lambda t: budgets.get(t, {}).get("budget", 0))
    tbl["sf"]      = tbl.index.map(lambda t: budgets.get(t, {}).get("sf_base", 1.0))
    tbl["qb"]      = tbl.index.map(lambda t: budgets.get(t, {}).get("qb_name", "?"))
    tbl["age"]     = tbl.index.map(lambda t: budgets.get(t, {}).get("age_26", 0))
    tbl["delta"]   = (tbl["after"] - tbl["before"]).round(0)

    hdr = f"  {'Team':<5} {'QB':<22} {'Age':>3}  {'SF':>5}  {'Budget':>6}  {'Before':>6}  {'After':>6}  {'Delta':>6}"
    print(f"\n{hdr}")
    print(f"  {'─'*4} {'─'*21} {'─'*3}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for team, row in tbl.sort_values("delta").iterrows():
        sf_str = f"{row['sf']:.3f}" if pd.notna(row.get("sf")) else "  —  "
        print(
            f"  {team:<5} {str(row['qb']):<22} {int(row['age']):>3}  "
            f"{sf_str}  {int(row['budget']):>6}  {int(row['before']):>6}  "
            f"{int(row['after']):>6}  {int(row['delta']):>+6}"
        )

    # ── Step 8b: Top 3 most-affected players per team ────────────────────────
    print(f"\n  Top 3 most-affected players per team (base scenario, |delta fpts_ppr|):")

    before_pts = (
        before_df[before_df["scenario"].eq("base")]
        [["player_id", "fpts_ppr"]]
        .rename(columns={"fpts_ppr": "fpts_before"})
    )
    after_pts = (
        proj_df[proj_df["scenario"].eq("base")]
        [["player_id", "team", "player_name", "position", "fpts_ppr"]]
        .rename(columns={"fpts_ppr": "fpts_after"})
    )
    diff = after_pts.merge(before_pts, on="player_id")
    diff["delta_pts"] = (diff["fpts_after"] - diff["fpts_before"]).round(1)
    diff["abs_delta"]  = diff["delta_pts"].abs()

    for team in sorted(diff["team"].unique()):
        top = (
            diff[(diff["team"] == team) & (diff["abs_delta"] >= 0.5)]
            .sort_values("abs_delta", ascending=False)
            .head(3)
        )
        if top.empty:
            continue
        print(f"\n  {team}  ({budgets.get(team, {}).get('qb_name', '?')}, age {budgets.get(team, {}).get('age_26', '?')}):")
        for _, p in top.iterrows():
            sign = "+" if p["delta_pts"] >= 0 else ""
            print(
                f"    {p['player_name']:<24} {p['position']}  "
                f"{p['fpts_before']:.1f} -> {p['fpts_after']:.1f}  "
                f"({sign}{p['delta_pts']:.1f})"
            )

    print(f"\n  Team budget scaling complete. {len(budgets)} teams processed.")
    return proj_df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Weighted Projection Model  ")
print("=" * 60)

seasonal  = pd.read_csv(DATA / "player_seasonal_stats.csv")
players   = pd.read_csv(DATA / "player_metadata.csv")
variance  = pd.read_csv(DATA / "player_variance.csv")
team_stats = pd.read_csv(DATA / "team_season_stats.csv")

# Calculate player ages for 2026 projection year
players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
players["age_2026"] = ((pd.Timestamp("2026-09-01") - players["birth_date"]).dt.days / 365.25).round(0)

# Merge age into seasonal
seasonal = seasonal.merge(
    players[["player_id", "age_2026"]],
    on="player_id", how="left"
)

PROJ_YEAR = 2026
BASE_SEASON = seasonal["season"].max()
print(f"\n  Projecting: {PROJ_YEAR} season")
print(f"  Most recent data: {BASE_SEASON}")
print(f"  Seasons available: {sorted(seasonal['season'].unique())}")

# ── Positional per-game averages for games-played regression ─────────────────
_base = seasonal[(seasonal["season"] == BASE_SEASON) & (seasonal["games"].fillna(0) > 0)].copy()
POS_AVG_PG = {}
for _p in ("QB", "RB", "WR", "TE"):
    _pdata = _base[_base["position"] == _p]
    POS_AVG_PG[_p] = {}
    for _metric in ["targets", "carries", "attempts"]:
        if _metric in _pdata.columns:
            _vals = _pdata[_metric].fillna(0) / _pdata["games"]
            POS_AVG_PG[_p][_metric] = float(_vals.median())

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED AVERAGE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def weighted_avg(player_id, metric, seasonal_df):
    """
    Compute weighted average of a metric across seasons.
    Recent seasons weighted more heavily per SEASON_WEIGHTS.
    """
    rows = seasonal_df[seasonal_df["player_id"] == player_id].copy()
    if rows.empty or metric not in rows.columns:
        return np.nan

    rows = rows[rows[metric].notna() & (rows["games"] > 0)]
    if rows.empty:
        return np.nan

    # Convert to per-game rates before weighting
    rows["per_game"] = rows[metric] / rows["games"]

    total_weight = 0
    weighted_sum = 0
    for _, row in rows.iterrows():
        w = SEASON_WEIGHTS.get(int(row["season"]), 0.01)
        weighted_sum  += row["per_game"] * w
        total_weight  += w

    return weighted_sum / total_weight if total_weight > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# BUILD WEIGHTED PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building weighted projections...")

# Get most recent team / stats for each player
latest = (
    seasonal.sort_values("season")
    .groupby("player_id")
    .last()
    .reset_index()[["player_id", "team", "position", "age_2026", "games", "carries", "targets", "attempts"]]
)

# ── Eligibility filters ────────────────────────────────────────────────────
# Filter 1: must have played at least 6 games in most recent season
before = len(latest)
latest = latest[latest["games"].fillna(0) >= 6].copy()
print(f"  Filter: games >= 6         → removed {before - len(latest)} players")

# Filter 2: RBs must have real workload (40+ carries OR 20+ targets)
rb_mask      = latest["position"] == "RB"
rb_workload  = (latest["carries"].fillna(0) >= 40) | (latest["targets"].fillna(0) >= 20)
before = len(latest)
latest = latest[~rb_mask | rb_workload].copy()
print(f"  Filter: RB workload        → removed {before - len(latest)} depth RBs")
print(f"  Players remaining: {len(latest)}")

# Get team context from most recent season
latest_team_stats = team_stats[team_stats["season"] == BASE_SEASON]

all_projections = []

for _, player in latest.iterrows():
    pid  = player["player_id"]
    pos  = player["position"]
    team = player["team"]
    age  = player["age_2026"]

    if pos not in ("QB", "RB", "WR", "TE"):
        continue

    # Get player name
    name_row = players[players["player_id"] == pid]
    name = name_row["player_name"].values[0] if not name_row.empty else pid

    # Age multiplier
    age_mult = get_age_multiplier(pos, age)

    # Games-played regression
    games_recent = int(player["games"]) if not np.isnan(player["games"]) else 17
    reg_wt = 0.20 if games_recent < 14 else 0.0
    def regress_pg(val, metric, _pos=pos, _rw=reg_wt):
        if _rw == 0.0 or pd.isna(val):
            return val
        pos_med = POS_AVG_PG.get(_pos, {}).get(metric, val)
        return val * (1 - _rw) + pos_med * _rw

    if games_recent >= 14:
        proj_confidence = "High"
    elif games_recent >= 8:
        proj_confidence = "Medium"
    else:
        proj_confidence = "Low"

    # Team context
    team_row = latest_team_stats[latest_team_stats["team"] == team]
    team_targets  = team_row["avg_targets_game"].values[0]  if not team_row.empty else 35
    team_carries  = team_row["avg_carries_game"].values[0]  if not team_row.empty else 25
    team_pass_att = team_row["avg_pass_attempts_game"].values[0] if not team_row.empty else 35

    # Get variance percentiles for bear/bull
    def get_pct(metric, pct_col):
        row = variance[
            (variance["player_id"] == pid) &
            (variance["metric"] == metric) &
            (variance["season"] == BASE_SEASON)
        ]
        if row.empty:
            return np.nan
        return row[pct_col].values[0]

    for scenario in ("bear", "base", "bull"):
        pct_col = {"bear": "p25", "base": "median", "bull": "p75"}[scenario]

        if pos in ("WR", "TE"):
            # Weighted target share as base, then apply age curve
            w_tgt_share = weighted_avg(pid, "target_share", seasonal)
            w_catch_rate_num = weighted_avg(pid, "receptions", seasonal)
            w_catch_rate_den = weighted_avg(pid, "targets", seasonal)
            w_ypr = weighted_avg(pid, "receiving_yards", seasonal)  # per game
            w_rec = weighted_avg(pid, "receptions", seasonal)        # per game

            catch_rate = (w_catch_rate_num / w_catch_rate_den) if w_catch_rate_den else 0.65
            yds_per_rec = (w_ypr / w_rec) if w_rec else 10.0

            # Bear/base/bull variance on targets
            tgt_p = get_pct("targets", pct_col)
            tgt_per_game = tgt_p if not np.isnan(tgt_p) else (w_tgt_share or 0.15) * team_targets

            # Regress toward positional avg if player missed games
            tgt_per_game = regress_pg(tgt_per_game, "targets")

            # Apply age curve to volume
            tgt_per_game = tgt_per_game * age_mult

            # Efficiency metrics — anchored to weighted history, NOT age-adjusted
            _w_rec_tds = weighted_avg(pid, "receiving_tds", seasonal)
            td_per_tgt = (
                _w_rec_tds / w_catch_rate_den
                if (w_catch_rate_den and not np.isnan(w_catch_rate_den) and not np.isnan(_w_rec_tds))
                else 0.05
            )

            # Small-sample guard: < 60 targets in most recent season → regress 50% toward pos avg
            _POS_AVG_TD_RATE = {"WR": 0.055, "TE": 0.060}
            _recent_tgts = player["targets"] if not pd.isna(player.get("targets", np.nan)) else 0
            if _recent_tgts < 60:
                _pos_avg_td = _POS_AVG_TD_RATE.get(pos, 0.055)
                td_per_tgt = td_per_tgt * 0.5 + _pos_avg_td * 0.5

            # Rate cap: max 9% TD-per-target
            td_per_tgt = min(td_per_tgt, 0.09)

            season_tgt  = tgt_per_game * GAMES
            season_rec  = season_tgt * catch_rate
            season_yds  = season_rec * yds_per_rec
            season_tds  = min(season_tgt * td_per_tgt, 15.0)
            if pos == "WR":
                season_tds = min(season_tds, 12.0)  # WR all-time record cap

            proj = {
                "targets": safe_round(season_tgt),
                "receptions": safe_round(season_rec),
                "receiving_yards": safe_round(season_yds),
                "receiving_tds": safe_round(season_tds, 1),
                "carries": 0, "rushing_yards": 0, "rushing_tds": 0,
                "passing_yards": 0, "passing_tds": 0, "interceptions": 0,
            }

        elif pos == "RB":
            car_p = get_pct("carries", pct_col)
            w_carries = weighted_avg(pid, "carries", seasonal)
            carries_pg = car_p if not np.isnan(car_p) else (w_carries or 8)
            carries_pg = regress_pg(carries_pg, "carries")
            carries_pg = carries_pg * age_mult

            w_rush_yds = weighted_avg(pid, "rushing_yards", seasonal)
            w_rush_tds = weighted_avg(pid, "rushing_tds", seasonal)
            ypc = (w_rush_yds / w_carries) if w_carries else 4.2
            td_per_carry = (w_rush_tds / w_carries) if w_carries else 0.04

            tgt_p = get_pct("targets", pct_col)
            w_tgts = weighted_avg(pid, "targets", seasonal)
            tgt_pg = tgt_p if not np.isnan(tgt_p) else (w_tgts or 2)
            tgt_pg = regress_pg(tgt_pg, "targets")
            tgt_pg = tgt_pg * age_mult

            w_rec = weighted_avg(pid, "receptions", seasonal)
            w_rec_yds = weighted_avg(pid, "receiving_yards", seasonal)
            catch_rate = (w_rec / w_tgts) if w_tgts else 0.75
            ypr = (w_rec_yds / w_rec) if w_rec else 7.0

            proj = {
                "carries": safe_round(carries_pg * GAMES),
                "rushing_yards": safe_round(carries_pg * GAMES * ypc),
                "rushing_tds": safe_round(carries_pg * GAMES * td_per_carry, 1),
                "targets": safe_round(tgt_pg * GAMES),
                "receptions": safe_round(tgt_pg * GAMES * catch_rate),
                "receiving_yards": safe_round(tgt_pg * GAMES * catch_rate * ypr),
                "receiving_tds": safe_round(tgt_pg * GAMES * 0.03, 1),
                "passing_yards": 0, "passing_tds": 0, "interceptions": 0,
            }

        elif pos == "QB":
            w_att  = weighted_avg(pid, "attempts", seasonal)
            w_py   = weighted_avg(pid, "passing_yards", seasonal)
            w_ptd  = weighted_avg(pid, "passing_tds", seasonal)
            w_int  = weighted_avg(pid, "interceptions", seasonal)
            w_comp = weighted_avg(pid, "completions", seasonal)

            ypa    = (w_py  / w_att) if w_att else 7.2
            td_att = (w_ptd / w_att) if w_att else 0.045
            td_att = min(td_att, 0.055)  # cap: ~38 TDs per 700 attempts — regresses to mean
            int_att = (w_int / w_att) if w_att else 0.02

            # QB start filter: proxy for games started via season attempts
            # ~25 att/game × 10 games = 250 attempts threshold
            qb_recent_att = player["attempts"] if not pd.isna(player["attempts"]) else 0
            if qb_recent_att < 250:
                # Backup / spot-starter — override confidence and regress attempts harder
                proj_confidence = "Low"
                qb_att_reg = 0.40
            else:
                qb_att_reg = reg_wt  # standard 20% or 0%

            att_p  = get_pct("attempts", pct_col)
            att_pg = att_p if not np.isnan(att_p) else (w_att or 32)
            # Apply QB-specific regression (may be 0%, 20%, or 40%)
            if qb_att_reg > 0.0 and not pd.isna(att_pg):
                pos_med = POS_AVG_PG.get(pos, {}).get("attempts", att_pg)
                att_pg = att_pg * (1 - qb_att_reg) + pos_med * qb_att_reg
            att_pg = att_pg * age_mult

            w_rush = weighted_avg(pid, "rushing_yards", seasonal)
            w_rtd  = weighted_avg(pid, "rushing_tds", seasonal)

            proj = {
                "attempts": safe_round(att_pg * GAMES),
                "completions": safe_round(att_pg * GAMES * ((w_comp / w_att) if w_att else 0.64)),
                "passing_yards": safe_round(att_pg * GAMES * ypa),
                "passing_tds": safe_round(att_pg * GAMES * td_att, 1),
                "interceptions": safe_round(att_pg * GAMES * int_att, 1),
                "carries": safe_round(weighted_avg(pid, "carries", seasonal) * GAMES or 0),
                "rushing_yards": safe_round((w_rush or 0) * GAMES),
                "rushing_tds": safe_round((w_rtd or 0) * GAMES, 1),
                "targets": 0, "receptions": 0,
                "receiving_yards": 0, "receiving_tds": 0,
            }
        else:
            continue

        # Fantasy points in all formats
        for fmt in SCORING:
            proj[f"fpts_{fmt}"] = calc_fpts(proj, fmt, pos)

        proj.update({
            "player_id":    pid,
            "player_name":  name,
            "position":     pos,
            "team":         team,
            "age":          int(age) if not np.isnan(age) else None,
            "age_multiplier": safe_round(age_mult, 3),
            "scenario":     scenario,
            "season_proj":  PROJ_YEAR,
            "model":        "weighted_v1",
            "projection_confidence": proj_confidence,
            "games_recent": games_recent,
        })
        all_projections.append(proj)

# ─────────────────────────────────────────────────────────────────────────────
# INJURED RETURNER HANDLER
# Players healthy the season before last but who missed most of the most recent
# season.  Re-base their volume on the prior healthy year + 15% rust discount.
# ─────────────────────────────────────────────────────────────────────────────
PRIOR_SEASON    = BASE_SEASON - 1
VOLUME_DISCOUNT = 0.85   # 15% haircut: rust + injury-recurrence risk

_prior_healthy = seasonal[
    (seasonal["season"] == PRIOR_SEASON) &
    (seasonal["games"].fillna(0)              >= 12) &
    (seasonal["fantasy_points_ppr"].fillna(0) >= 150)
][["player_id"]]

_recent_injured = seasonal[
    (seasonal["season"] == BASE_SEASON) &
    (seasonal["games"].fillna(0) < 8)
][["player_id"]]

returner_ids = set(_prior_healthy.merge(_recent_injured, on="player_id")["player_id"])

# ── Print flagged players ────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print(f"  Injured Returner Handler")
print(f"{'─' * 60}")
print(f"  Criteria: {PRIOR_SEASON} (>=12 g, >=150 fpts_ppr)  ->  {BASE_SEASON} (<8 g)")

flagged_info = []
for _pid in sorted(returner_ids):
    _pr = seasonal[(seasonal["player_id"] == _pid) & (seasonal["season"] == PRIOR_SEASON)]
    _ir = seasonal[(seasonal["player_id"] == _pid) & (seasonal["season"] == BASE_SEASON)]
    if _pr.empty:
        continue
    flagged_info.append((
        _pr["player_name"].values[0],
        _pr["position"].values[0],
        int(_pr["games"].values[0]),
        float(_pr["fantasy_points_ppr"].values[0]),
        int(_ir["games"].values[0]) if not _ir.empty else 0,
    ))

if flagged_info:
    print(f"\n  {'Player':<26} {'Pos':<5} {PRIOR_SEASON}              {BASE_SEASON}")
    print(f"  {'-'*26} {'-'*4} {'g':>3}  {'fpts_ppr':>9}   {'g':>3}")
    for _name, _pos, _pg, _pp, _ig in sorted(flagged_info):
        print(f"  {_name:<26} {_pos:<5} {_pg:>3}  {_pp:>9.1f}   {_ig:>3}")
else:
    print("\n  No players matched the injury criteria.")

# ── Rebuild projections for returners ───────────────────────────────────────
rebuilt = []
for proj in all_projections:
    pid = proj["player_id"]
    if pid not in returner_ids:
        proj["returning_from_injury"] = False
        rebuilt.append(proj)
        continue

    pos      = proj["position"]
    scenario = proj["scenario"]
    pct_col  = {"bear": "p25", "base": "median", "bull": "p75"}[scenario]
    age_mult = float(proj["age_multiplier"])

    def prior_pct(metric, _pid=pid, _pc=pct_col):
        """Pull variance percentile from the prior healthy season."""
        row = variance[
            (variance["player_id"] == _pid) &
            (variance["metric"]    == metric) &
            (variance["season"]    == PRIOR_SEASON)
        ]
        return np.nan if row.empty else float(row[_pc].values[0])

    def rescale(existing_total, prior_pg_raw):
        """Ratio to shift existing season total onto prior-year base × discount.
        prior_pg_raw is the raw (pre-age-curve) per-game value from variance.
        existing_total already includes the age curve, so we apply age_mult again
        to the prior figure to keep adjustments consistent.
        """
        if existing_total <= 0:
            return VOLUME_DISCOUNT
        if np.isnan(prior_pg_raw):
            return VOLUME_DISCOUNT
        return (prior_pg_raw * age_mult * VOLUME_DISCOUNT * GAMES) / existing_total

    if pos in ("WR", "TE"):
        r = rescale(proj["targets"], prior_pct("targets"))
        proj["targets"]         = safe_round(proj["targets"]         * r)
        proj["receptions"]      = safe_round(proj["receptions"]      * r)
        proj["receiving_yards"] = safe_round(proj["receiving_yards"] * r)
        proj["receiving_tds"]   = safe_round(proj["receiving_tds"]   * r, 1)

    elif pos == "RB":
        cr = rescale(proj["carries"], prior_pct("carries"))
        proj["carries"]       = safe_round(proj["carries"]       * cr)
        proj["rushing_yards"] = safe_round(proj["rushing_yards"] * cr)
        proj["rushing_tds"]   = safe_round(proj["rushing_tds"]   * cr, 1)

        tr = rescale(proj["targets"], prior_pct("targets"))
        proj["targets"]         = safe_round(proj["targets"]         * tr)
        proj["receptions"]      = safe_round(proj["receptions"]      * tr)
        proj["receiving_yards"] = safe_round(proj["receiving_yards"] * tr)
        proj["receiving_tds"]   = safe_round(proj["receiving_tds"]   * tr, 1)

    elif pos == "QB":
        ar = rescale(proj.get("attempts", 0), prior_pct("attempts"))
        for _k in ["attempts", "completions", "passing_yards"]:
            if _k in proj:
                proj[_k] = safe_round(proj[_k] * ar)
        for _k in ["passing_tds", "interceptions"]:
            if _k in proj:
                proj[_k] = safe_round(proj[_k] * ar, 1)

    # Recalculate fantasy points with adjusted volume
    for fmt_name in SCORING:
        proj[f"fpts_{fmt_name}"] = calc_fpts(proj, fmt_name, pos)

    proj["projection_confidence"] = "Medium"
    proj["returning_from_injury"] = True
    rebuilt.append(proj)

all_projections = rebuilt
n_returners = len({p["player_id"] for p in all_projections if p.get("returning_from_injury")})
print(f"\n  {n_returners} players rebuilt as injured returners")

proj_df = pd.DataFrame(all_projections)
proj_df = apply_team_budget(proj_df, seasonal)
proj_df.to_csv(DATA / "projections_weighted.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
base = proj_df[proj_df["scenario"] == "base"].copy()

print(f"\n  ✓ {len(base)} players projected")
print(f"\n  Top 15 WRs by receiving TDs (base scenario):")
print(
    base[base["position"] == "WR"]
    .sort_values("receiving_tds", ascending=False)
    [["player_name", "team", "age", "targets", "receiving_tds", "fpts_ppr"]]
    .head(15)
    .to_string(index=False)
)

print(f"\n  Top 10 RBs (PPR, base, weighted model):")
print(
    base[base["position"] == "RB"]
    .sort_values("fpts_ppr", ascending=False)
    [["player_name", "team", "age", "age_multiplier", "carries", "rushing_yards", "fpts_ppr"]]
    .head(10)
    .to_string(index=False)
)

print(f"\n  Age curve examples:")
for name, expected_pos in [("Tyreek Hill", "WR"), ("Christian McCaffrey", "RB"), ("Travis Kelce", "TE")]:
    row = base[(base["player_name"] == name) & (base["position"] == expected_pos)]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {name:25} age={r['age']}  mult={r['age_multiplier']}  PPR={r['fpts_ppr']}")

print(f"\n  ✅ Saved to data/projections_weighted.csv")
print("""
Next steps:
  → Run 05_vegas_totals.py   to add game total adjustments
  → Tell Claude Code to update app.py to use projections_weighted.csv
""")
