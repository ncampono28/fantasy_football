"""
01_fetch_data.py — Fetch & cache all NFL data using nflreadpy
=============================================================
Produces (in data/):
  player_seasonal_stats.csv
  player_weekly_stats.csv
  player_metadata.csv
  player_variance.csv
  team_season_stats.csv

Run:
    python 01_fetch_data.py
Then:
    python 04_weighted_model.py
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

DATA = Path("data")
DATA.mkdir(exist_ok=True)

SEASONS   = list(range(2020, 2026))   # 2020–2025
POSITIONS = {"QB", "RB", "WR", "TE"}

print("=" * 60)
print("  01_fetch_data — nflreadpy")
print("=" * 60)
print(f"  Seasons: {SEASONS}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEASONAL PLAYER STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading seasonal player stats...")
raw_seasonal = nfl.load_player_stats(seasons=SEASONS, summary_level="reg").to_pandas()

seasonal = raw_seasonal[raw_seasonal["position"].isin(POSITIONS)].copy()
seasonal = seasonal.drop(columns=["player_name"], errors="ignore")
seasonal = seasonal.rename(columns={
    "player_display_name":   "player_name",
    "recent_team":           "team",
    "passing_interceptions": "interceptions",
})

SEASONAL_COLS = [
    "player_id", "player_name", "position", "team", "season", "games",
    "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "fantasy_points", "fantasy_points_ppr",
]
seasonal = seasonal[SEASONAL_COLS].sort_values(["season", "player_name"]).reset_index(drop=True)
seasonal.to_csv(DATA / "player_seasonal_stats.csv", index=False)
print(f"  OK player_seasonal_stats.csv — {len(seasonal)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. WEEKLY PLAYER STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Loading weekly player stats...")
raw_weekly = nfl.load_player_stats(seasons=SEASONS, summary_level="week").to_pandas()

weekly = raw_weekly[
    raw_weekly["position"].isin(POSITIONS) &
    (raw_weekly["season_type"] == "REG")
].copy()
weekly = weekly.drop(columns=["player_name"], errors="ignore")
weekly = weekly.rename(columns={
    "player_display_name":   "player_name",
    "passing_interceptions": "interceptions",
})

WEEKLY_COLS = [
    "player_id", "player_name", "position", "team", "season", "week",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "fantasy_points", "fantasy_points_ppr",
]
weekly = weekly[WEEKLY_COLS].sort_values(["season", "week", "player_name"]).reset_index(drop=True)
weekly.to_csv(DATA / "player_weekly_stats.csv", index=False)
print(f"  OK player_weekly_stats.csv — {len(weekly)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLAYER METADATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Loading player metadata...")
raw_players = nfl.load_players().to_pandas()

raw_players = raw_players.rename(columns={
    "gsis_id":      "player_id",
    "display_name": "player_name",
    "college_name": "college",
    "rookie_season": "rookie_year",
    "draft_year":   "entry_year",
})

# Restrict to players in our stats; get their most-recent team from seasonal
known_ids = set(seasonal["player_id"].dropna().unique())
meta = raw_players[raw_players["player_id"].isin(known_ids)].copy()

latest_team = (
    seasonal.sort_values("season")
    .groupby("player_id")[["player_name", "position", "team"]]
    .last()
    .reset_index()
)

meta = meta.merge(latest_team[["player_id", "team"]], on="player_id", how="left")

META_COLS = [
    "player_id", "player_name", "position", "team", "birth_date",
    "height", "weight", "college", "rookie_year", "entry_year",
    "draft_round", "draft_pick",
]
meta = meta[META_COLS].reset_index(drop=True)
meta.to_csv(DATA / "player_metadata.csv", index=False)
print(f"  OK player_metadata.csv — {len(meta)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLAYER VARIANCE (per-game percentiles, from weekly)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Computing player variance...")

VARIANCE_METRICS = {
    "QB": ["attempts", "passing_yards", "passing_tds"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets"],
    "WR": ["targets", "receiving_yards", "receiving_tds", "target_share"],
    "TE": ["targets", "receiving_yards", "receiving_tds", "target_share"],
}

rows = []
for pos, metrics in VARIANCE_METRICS.items():
    pos_df = weekly[weekly["position"] == pos]
    for (pid, season), g in pos_df.groupby(["player_id", "season"]):
        if len(g) < 4:
            continue
        name = g["player_name"].iloc[0]
        for metric in metrics:
            vals = g[metric].fillna(0)
            rows.append({
                "player_id":   pid,
                "player_name": name,
                "position":    pos,
                "season":      season,
                "metric":      metric,
                "mean":        round(float(vals.mean()), 4),
                "std":         round(float(vals.std(ddof=1)), 4),
                "p25":         round(float(vals.quantile(0.25)), 4),
                "median":      round(float(vals.median()), 4),
                "p75":         round(float(vals.quantile(0.75)), 4),
            })

variance_df = pd.DataFrame(rows)
variance_df.to_csv(DATA / "player_variance.csv", index=False)
print(f"  OK player_variance.csv — {len(variance_df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TEAM SEASON STATS (derived from weekly)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Computing team season stats...")

# Sum per team/season/week, then average across weeks
team_weekly = (
    weekly.groupby(["team", "season", "week"])
    .agg(
        pass_attempts=("attempts", "sum"),
        carries=("carries", "sum"),
        targets=("targets", "sum"),
    )
    .reset_index()
)
team_season = (
    team_weekly.groupby(["team", "season"])
    .agg(
        games_played=("week", "count"),
        avg_pass_attempts_game=("pass_attempts", "mean"),
        avg_carries_game=("carries", "mean"),
        avg_targets_game=("targets", "mean"),
    )
    .reset_index()
)
for col in ["avg_pass_attempts_game", "avg_carries_game", "avg_targets_game"]:
    team_season[col] = team_season[col].round(1)

team_season.to_csv(DATA / "team_season_stats.csv", index=False)
print(f"  OK team_season_stats.csv — {len(team_season)} rows")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DONE All data files saved to data/")
print("=" * 60)
print("\nNext: python 04_weighted_model.py")
