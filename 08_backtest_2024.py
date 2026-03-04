"""
Model Backtest — 2024 Season
==============================
Trains the weighted projection model on 2020-2023 data only,
projects 2024, then compares to actual 2024 results.

This tells us how accurate our model is before we trust it
to project 2026.

Run:
    py 08_backtest_2024.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math

DATA = Path("data")

# ─────────────────────────────────────────────────────────────────────────────
# SAME CONFIG AS 04_weighted_model.py but capped at 2023
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASON   = 2024
GAMES         = 17
POSITIONS     = ["QB", "RB", "WR", "TE"]

SEASON_WEIGHTS = {
    2023: 0.55,
    2022: 0.25,
    2021: 0.12,
    2020: 0.08,
}

AGE_CURVES = {
    "WR": {22:0.82,23:0.90,24:0.96,25:0.99,26:1.02,27:1.03,28:1.02,29:0.98,
           30:0.93,31:0.87,32:0.80,33:0.72,34:0.63,35:0.55},
    "RB": {22:0.88,23:0.95,24:1.01,25:1.02,26:1.00,27:0.94,28:0.86,29:0.76,
           30:0.65,31:0.54,32:0.44},
    "TE": {22:0.72,23:0.80,24:0.88,25:0.94,26:0.98,27:1.01,28:1.02,29:1.02,
           30:1.00,31:0.97,32:0.92,33:0.86,34:0.79,35:0.71},
    "QB": {22:0.80,23:0.86,24:0.91,25:0.95,26:0.98,27:1.00,28:1.02,29:1.03,
           30:1.03,31:1.02,32:1.01,33:1.00,34:0.97,35:0.93,36:0.88,37:0.82,
           38:0.75,39:0.67,40:0.59},
}

SCORING = {
    "ppr": {"pass_yd":0.04,"pass_td":4,"int":-2,"rush_yd":0.1,"rush_td":6,
            "rec":1.0,"rec_yd":0.1,"rec_td":6,"te_bonus":0},
}

def safe_round(x, n=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return round(float(x), n)

def get_age_multiplier(position, age):
    if pd.isna(age): return 1.0
    age = int(age)
    curve = AGE_CURVES.get(position, {})
    if age in curve: return curve[age]
    if age > max(curve.keys()): return min(curve.values())
    if age < min(curve.keys()): return curve[min(curve.keys())]
    return 1.0

def calc_fpts(proj, position):
    s = SCORING["ppr"]
    pts  = proj.get("passing_yards",0)   * s["pass_yd"]
    pts += proj.get("passing_tds",0)     * s["pass_td"]
    pts += proj.get("interceptions",0)   * s["int"]
    pts += proj.get("rushing_yards",0)   * s["rush_yd"]
    pts += proj.get("rushing_tds",0)     * s["rush_td"]
    te_b = s["te_bonus"] if position == "TE" else 0
    pts += proj.get("receptions",0)      * (s["rec"] + te_b)
    pts += proj.get("receiving_yards",0) * s["rec_yd"]
    pts += proj.get("receiving_tds",0)   * s["rec_td"]
    return round(pts, 1)

def weighted_avg(player_id, metric, df):
    rows = df[df["player_id"] == player_id].copy()
    if rows.empty or metric not in rows.columns: return np.nan
    rows = rows[rows[metric].notna() & (rows["games"] > 0)]
    if rows.empty: return np.nan
    rows["per_game"] = rows[metric] / rows["games"]
    total_w, weighted_sum = 0, 0
    for _, row in rows.iterrows():
        w = SEASON_WEIGHTS.get(int(row["season"]), 0.01)
        weighted_sum += row["per_game"] * w
        total_w += w
    return weighted_sum / total_w if total_w > 0 else np.nan

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Model Backtest — Projecting 2024 from 2020-2023 data")
print("=" * 60)

seasonal  = pd.read_csv(DATA / "player_seasonal_stats.csv")
players   = pd.read_csv(DATA / "player_metadata.csv")
variance  = pd.read_csv(DATA / "player_variance.csv")
team_stats = pd.read_csv(DATA / "team_season_stats.csv")

# Age for 2024 projection
players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
players["age_2024"] = ((pd.Timestamp("2024-09-01") - players["birth_date"]).dt.days / 365.25).round(0)

seasonal = seasonal.merge(players[["player_id","age_2024"]], on="player_id", how="left")

# Training data = 2020-2023 only
train = seasonal[seasonal["season"].isin(TRAIN_SEASONS)].copy()
train = train.rename(columns={"age_2024": "age"})

# Actual 2024 results
actual_2024 = seasonal[seasonal["season"] == TEST_SEASON].copy()

# Minimum games filter
train_qualified = train[train["games"] >= 6]
latest_team = (
    train.sort_values("season")
    .groupby("player_id").last()
    .reset_index()[["player_id","team","position","age"]]
)
latest_team_stats = team_stats[team_stats["season"] == 2023]

print(f"\n  Training seasons: {TRAIN_SEASONS}")
print(f"  Test season: {TEST_SEASON}")
print(f"  Players in training: {train['player_id'].nunique()}")
print(f"  Players in actual 2024: {actual_2024['player_id'].nunique()}")

# ── Positional per-game averages for games-played regression (2023) ───────────
_base23 = train[(train["season"] == 2023) & (train["games"].fillna(0) > 0)].copy()
POS_AVG_PG = {}
for _p in ("QB", "RB", "WR", "TE"):
    _pdata = _base23[_base23["position"] == _p]
    POS_AVG_PG[_p] = {}
    for _metric in ["targets", "carries", "attempts"]:
        if _metric in _pdata.columns:
            _vals = _pdata[_metric].fillna(0) / _pdata["games"]
            POS_AVG_PG[_p][_metric] = float(_vals.median())

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PROJECTIONS (base case only for backtest)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building 2024 projections from 2023 data...")

projections = []

for _, player in latest_team.iterrows():
    pid  = player["player_id"]
    pos  = player["position"]
    team = player["team"]
    age  = player["age"]

    if pos not in POSITIONS:
        continue

    # Minimum activity filter
    recent = train_qualified[
        (train_qualified["player_id"] == pid) &
        (train_qualified["season"] == 2023)
    ]
    if recent.empty:
        continue

    name_row = players[players["player_id"] == pid]
    name = name_row["player_name"].values[0] if not name_row.empty else pid

    age_mult = get_age_multiplier(pos, age)

    # Games-played regression
    games_recent = int(recent["games"].values[0])
    reg_wt = 0.20 if games_recent < 14 else 0.0
    def regress_pg(val, metric, _pos=pos, _rw=reg_wt):
        if _rw == 0.0 or (isinstance(val, float) and math.isnan(val)):
            return val
        pos_med = POS_AVG_PG.get(_pos, {}).get(metric, val)
        return val * (1 - _rw) + pos_med * _rw

    if games_recent >= 14:
        proj_confidence = "High"
    elif games_recent >= 8:
        proj_confidence = "Medium"
    else:
        proj_confidence = "Low"

    team_row = latest_team_stats[latest_team_stats["team"] == team]
    team_targets  = team_row["avg_targets_game"].values[0]  if not team_row.empty else 35
    team_carries  = team_row["avg_carries_game"].values[0]  if not team_row.empty else 25

    def get_pct(metric, pct_col):
        row = variance[
            (variance["player_id"] == pid) &
            (variance["metric"] == metric) &
            (variance["season"] == 2023)
        ]
        if row.empty: return np.nan
        return row[pct_col].values[0]

    pct_col = "median"

    if pos in ("WR", "TE"):
        w_tgt   = weighted_avg(pid, "targets", train)
        w_rec   = weighted_avg(pid, "receptions", train)
        w_yds   = weighted_avg(pid, "receiving_yards", train)
        w_tds   = weighted_avg(pid, "receiving_tds", train)

        catch_rate = (w_rec / w_tgt) if w_tgt else 0.65
        yds_per_rec = (w_yds / w_rec) if w_rec else 10.0
        td_per_tgt = (w_tds / w_tgt) if w_tgt else 0.05
        td_per_tgt = min(td_per_tgt, 0.12)  # cap at 12%

        tgt_p = get_pct("targets", pct_col)
        tgt_pg = tgt_p if not np.isnan(tgt_p) else (w_tgt or 0.15) * team_targets
        tgt_pg = regress_pg(tgt_pg, "targets")
        tgt_pg = tgt_pg * age_mult if not np.isnan(tgt_pg) else 0

        season_tgt = safe_round(tgt_pg * GAMES)
        season_rec = safe_round(season_tgt * (catch_rate if not np.isnan(catch_rate) else 0.65))
        season_yds = safe_round(season_rec * (yds_per_rec if not np.isnan(yds_per_rec) else 10.0))
        season_tds = safe_round(season_tgt * (td_per_tgt if not np.isnan(td_per_tgt) else 0.05), 1)

        proj = {"targets":season_tgt,"receptions":season_rec,
                "receiving_yards":season_yds,"receiving_tds":season_tds,
                "carries":0,"rushing_yards":0,"rushing_tds":0,
                "passing_yards":0,"passing_tds":0,"interceptions":0}

    elif pos == "RB":
        w_car  = weighted_avg(pid, "carries", train)
        w_ryds = weighted_avg(pid, "rushing_yards", train)
        w_rtds = weighted_avg(pid, "rushing_tds", train)
        w_tgt  = weighted_avg(pid, "targets", train)
        w_rec  = weighted_avg(pid, "receptions", train)
        w_reyds = weighted_avg(pid, "receiving_yards", train)

        ypc       = (w_ryds / w_car) if w_car else 4.2
        td_per_car = (w_rtds / w_car) if w_car else 0.04
        catch_rate = (w_rec / w_tgt) if w_tgt else 0.75
        ypr        = (w_reyds / w_rec) if w_rec else 7.0

        car_p = get_pct("carries", pct_col)
        car_pg = car_p if not np.isnan(car_p) else (w_car or 8)
        car_pg = regress_pg(car_pg, "carries")
        car_pg = safe_round(car_pg * age_mult, 1) if not np.isnan(car_pg) else 0

        tgt_p = get_pct("targets", pct_col)
        tgt_pg = tgt_p if not np.isnan(tgt_p) else (w_tgt or 2)
        tgt_pg = regress_pg(tgt_pg, "targets")
        tgt_pg = safe_round(tgt_pg * age_mult, 1) if not np.isnan(tgt_pg) else 0

        proj = {
            "carries":         safe_round(car_pg * GAMES),
            "rushing_yards":   safe_round(car_pg * GAMES * (ypc or 4.2)),
            "rushing_tds":     safe_round(car_pg * GAMES * (td_per_car or 0.04), 1),
            "targets":         safe_round(tgt_pg * GAMES),
            "receptions":      safe_round(tgt_pg * GAMES * (catch_rate or 0.75)),
            "receiving_yards": safe_round(tgt_pg * GAMES * (catch_rate or 0.75) * (ypr or 7.0)),
            "receiving_tds":   safe_round(tgt_pg * GAMES * 0.03, 1),
            "passing_yards":0,"passing_tds":0,"interceptions":0,
        }

    elif pos == "QB":
        w_att  = weighted_avg(pid, "attempts", train)
        w_py   = weighted_avg(pid, "passing_yards", train)
        w_ptd  = weighted_avg(pid, "passing_tds", train)
        w_int  = weighted_avg(pid, "interceptions", train)
        w_comp = weighted_avg(pid, "completions", train)
        w_rush = weighted_avg(pid, "rushing_yards", train)
        w_rtd  = weighted_avg(pid, "rushing_tds", train)

        ypa     = (w_py  / w_att) if w_att else 7.2
        td_att  = (w_ptd / w_att) if w_att else 0.045
        td_att  = min(td_att, 0.055)  # cap: ~38 TDs per 700 attempts — regresses to mean
        int_att = (w_int / w_att) if w_att else 0.02
        comp_rt = (w_comp / w_att) if w_att else 0.64

        att_p  = get_pct("attempts", pct_col)
        att_pg = att_p if not np.isnan(att_p) else (w_att or 32)
        att_pg = regress_pg(att_pg, "attempts")
        att_pg = safe_round(att_pg * age_mult, 1) if not np.isnan(att_pg) else 0

        proj = {
            "attempts":      safe_round(att_pg * GAMES),
            "completions":   safe_round(att_pg * GAMES * (comp_rt or 0.64)),
            "passing_yards": safe_round(att_pg * GAMES * (ypa or 7.2)),
            "passing_tds":   safe_round(att_pg * GAMES * (td_att or 0.045), 1),
            "interceptions": safe_round(att_pg * GAMES * (int_att or 0.02), 1),
            "carries":       safe_round((w_rush or 0) / (w_att / att_pg if w_att and att_pg else 1) * GAMES if w_rush else 0),
            "rushing_yards": safe_round((w_rush or 0) * GAMES),
            "rushing_tds":   safe_round((w_rtd or 0) * GAMES, 1),
            "targets":0,"receptions":0,"receiving_yards":0,"receiving_tds":0,
        }
    else:
        continue

    proj["fpts_ppr_projected"] = calc_fpts(proj, pos)
    proj.update({
        "player_id": pid, "player_name": name,
        "position": pos, "team": team,
        "age": int(age) if not np.isnan(age) else None,
        "age_multiplier": round(age_mult, 3),
        "projection_confidence": proj_confidence,
        "games_recent": games_recent,
    })
    projections.append(proj)

proj_df = pd.DataFrame(projections)
print(f"  ✓ {len(proj_df)} players projected for 2024")

# ─────────────────────────────────────────────────────────────────────────────
# MERGE WITH ACTUAL 2024 RESULTS
# ─────────────────────────────────────────────────────────────────────────────
actual = actual_2024[["player_id","player_name","position","games",
                       "fantasy_points_ppr"]].copy()
actual = actual.rename(columns={"fantasy_points_ppr": "fpts_ppr_actual"})
actual = actual[actual["games"] >= 6]  # min 6 games to evaluate

merged = proj_df.merge(actual, on=["player_id","player_name","position"], how="inner")
merged["error"]     = merged["fpts_ppr_projected"] - merged["fpts_ppr_actual"]
merged["abs_error"] = merged["error"].abs()
merged["pct_error"] = (merged["error"] / merged["fpts_ppr_actual"] * 100).round(1)

merged.to_csv(DATA / "backtest_2024.csv", index=False)
print(f"  ✓ {len(merged)} players matched for evaluation")

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BACKTEST RESULTS — 2024 Season")
print("=" * 60)

for pos in POSITIONS:
    pos_df = merged[merged["position"] == pos]
    if pos_df.empty: continue
    mae  = pos_df["abs_error"].mean()
    rmse = np.sqrt((pos_df["error"]**2).mean())
    bias = pos_df["error"].mean()
    within_20pct = (pos_df["abs_error"] / pos_df["fpts_ppr_actual"] < 0.20).mean() * 100
    print(f"\n  {pos} ({len(pos_df)} players):")
    print(f"    MAE:           {mae:.1f} pts  (avg points off)")
    print(f"    RMSE:          {rmse:.1f} pts")
    print(f"    Bias:          {bias:+.1f} pts  ({'over' if bias>0 else 'under'}-projecting)")
    print(f"    Within 20%:    {within_20pct:.0f}% of players")

print(f"\n  Overall MAE: {merged['abs_error'].mean():.1f} pts")
print(f"  Overall bias: {merged['error'].mean():+.1f} pts")

# ─────────────────────────────────────────────────────────────────────────────
# BIGGEST MISSES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BIGGEST OVER-PROJECTIONS (model was too optimistic)")
print("=" * 60)
over = merged.nlargest(10, "error")[["player_name","position","fpts_ppr_projected","fpts_ppr_actual","error","pct_error"]]
print(over.to_string(index=False))

print("\n" + "=" * 60)
print("  BIGGEST UNDER-PROJECTIONS (model missed breakouts)")
print("=" * 60)
under = merged.nsmallest(10, "error")[["player_name","position","fpts_ppr_projected","fpts_ppr_actual","error","pct_error"]]
print(under.to_string(index=False))

print("\n" + "=" * 60)
print("  MOST ACCURATE PROJECTIONS")
print("=" * 60)
accurate = merged.nsmallest(10, "abs_error")[["player_name","position","fpts_ppr_projected","fpts_ppr_actual","error","pct_error"]]
print(accurate.to_string(index=False))

print(f"\n  ✅ Full results saved to data/backtest_2024.csv")
print("""
Key questions this answers:
  → Which positions does the model project most accurately?
  → Are we systematically over/under projecting any position?
  → Which player types does the model miss (injuries, breakouts)?
  → How does accuracy compare to industry standard (~40-50 MAE)?
""")
