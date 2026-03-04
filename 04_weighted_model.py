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
    2021: 0.02,
    2020: 0.01,
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
    .reset_index()[["player_id", "team", "position", "age_2026", "games", "carries", "targets"]]
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

            # Apply age curve to volume
            tgt_per_game = tgt_per_game * age_mult

            # Efficiency metrics — anchored to weighted history, NOT age-adjusted
            td_per_tgt = weighted_avg(pid, "receiving_tds", seasonal)
            td_per_tgt = (td_per_tgt / weighted_avg(pid, "targets", seasonal)) if weighted_avg(pid, "targets", seasonal) else 0.05

            season_tgt  = tgt_per_game * GAMES
            season_rec  = season_tgt * catch_rate
            season_yds  = season_rec * yds_per_rec
            season_tds  = season_tgt * (td_per_tgt or 0.05)

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
            carries_pg = carries_pg * age_mult

            w_rush_yds = weighted_avg(pid, "rushing_yards", seasonal)
            w_rush_tds = weighted_avg(pid, "rushing_tds", seasonal)
            ypc = (w_rush_yds / w_carries) if w_carries else 4.2
            td_per_carry = (w_rush_tds / w_carries) if w_carries else 0.04

            tgt_p = get_pct("targets", pct_col)
            w_tgts = weighted_avg(pid, "targets", seasonal)
            tgt_pg = tgt_p if not np.isnan(tgt_p) else (w_tgts or 2)
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
            att_p = get_pct("attempts", pct_col)
            w_att = weighted_avg(pid, "attempts", seasonal)
            att_pg = att_p if not np.isnan(att_p) else (w_att or 32)
            att_pg = att_pg * age_mult

            w_py  = weighted_avg(pid, "passing_yards", seasonal)
            w_ptd = weighted_avg(pid, "passing_tds", seasonal)
            w_int = weighted_avg(pid, "interceptions", seasonal)
            w_comp = weighted_avg(pid, "completions", seasonal)

            ypa    = (w_py  / w_att) if w_att else 7.2
            td_att = (w_ptd / w_att) if w_att else 0.045
            int_att = (w_int / w_att) if w_att else 0.02

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
        })
        all_projections.append(proj)

proj_df = pd.DataFrame(all_projections)
proj_df.to_csv(DATA / "projections_weighted.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
base = proj_df[proj_df["scenario"] == "base"].copy()

print(f"\n  ✓ {len(base)} players projected")
print(f"\n  Top 10 WRs (PPR, base, weighted model):")
print(
    base[base["position"] == "WR"]
    .sort_values("fpts_ppr", ascending=False)
    [["player_name", "team", "age", "age_multiplier", "targets", "receiving_yards", "fpts_ppr"]]
    .head(10)
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
