# -*- coding: utf-8 -*-
"""
Re-pulls player_weekly_stats, player_seasonal_stats, player_variance,
and team_season_stats for seasons 2020-2025 using the same SSL-bypass
download_parquet approach as ff.ipynb.
"""
import ssl, urllib.request, io, pandas as pd, numpy as np

POSITIONS = ["QB", "RB", "WR", "TE"]
SEASONS   = list(range(2020, 2026))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

def download_parquet(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r:
        data = r.read()
    return pd.read_parquet(io.BytesIO(data))

# -- 1. players (for full display-name join) -----------------------------------
print("Downloading players...")
players_raw = download_parquet(
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
)
players = players_raw[players_raw["position"].isin(POSITIONS)][[
    "gsis_id", "display_name", "position", "latest_team",
    "birth_date", "height", "weight", "college_name",
    "rookie_season", "draft_year", "draft_round", "draft_pick"
]].rename(columns={
    "gsis_id":        "player_id",
    "display_name":   "player_name",
    "latest_team":    "team",
    "college_name":   "college",
    "rookie_season":  "rookie_year",
    "draft_year":     "entry_year",
})
players.to_csv("data/player_metadata.csv", index=False)
print(f"  OK  {len(players):,} players saved to data/player_metadata.csv")

full_names = players[["player_id", "player_name"]].rename(columns={"player_name": "full_name"})

# -- 2. weekly stats (REG season, 2020-2025) -----------------------------------
print("Downloading player_stats parquet (all seasons, may take ~30s)...")
seasonal_raw = download_parquet(
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"
)
print(f"  raw rows: {len(seasonal_raw):,}  latest seasons: {sorted(seasonal_raw['season'].unique())[-4:]}")

skill = seasonal_raw[
    seasonal_raw["position"].isin(POSITIONS) &
    seasonal_raw["season"].isin(SEASONS)
].copy()

keep_cols = [c for c in [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "fantasy_points", "fantasy_points_ppr"
] if c in skill.columns]

weekly = skill[keep_cols].rename(columns={"recent_team": "team"})

# Attach full display names
weekly = weekly.merge(full_names, on="player_id", how="left")
weekly["player_name"] = weekly["full_name"].fillna(weekly["player_name"])
weekly = weekly.drop(columns=["full_name"])

weekly.to_csv("data/player_weekly_stats.csv", index=False)
print(f"  OK  {len(weekly):,} weekly rows  |  seasons: {sorted(weekly['season'].unique())}")

# -- 3. seasonal aggregates ----------------------------------------------------
print("Building seasonal aggregates...")
seasonal_stats = (
    weekly
    .groupby(["player_id", "player_name", "position", "team", "season"])
    .agg(
        games              = ("week",               "nunique"),
        targets            = ("targets",            "sum"),
        receptions         = ("receptions",         "sum"),
        receiving_yards    = ("receiving_yards",    "sum"),
        receiving_tds      = ("receiving_tds",      "sum"),
        target_share       = ("target_share",       "mean"),
        carries            = ("carries",            "sum"),
        rushing_yards      = ("rushing_yards",      "sum"),
        rushing_tds        = ("rushing_tds",        "sum"),
        attempts           = ("attempts",           "sum"),
        completions        = ("completions",        "sum"),
        passing_yards      = ("passing_yards",      "sum"),
        passing_tds        = ("passing_tds",        "sum"),
        interceptions      = ("interceptions",      "sum"),
        fantasy_points     = ("fantasy_points",     "sum"),
        fantasy_points_ppr = ("fantasy_points_ppr", "sum"),
    )
    .reset_index()
)
seasonal_stats.to_csv("data/player_seasonal_stats.csv", index=False)
print(f"  OK  {len(seasonal_stats):,} seasonal rows saved to data/player_seasonal_stats.csv")

# -- 4. team season context ----------------------------------------------------
print("Building team season stats...")
team_season = (
    weekly
    .groupby(["team", "season", "week"])
    .agg(
        team_pass_attempts = ("attempts", "sum"),
        team_carries       = ("carries",  "sum"),
        team_targets       = ("targets",  "sum"),
    )
    .reset_index()
    .groupby(["team", "season"])
    .agg(
        games_played           = ("week",               "nunique"),
        avg_pass_attempts_game = ("team_pass_attempts", "mean"),
        avg_carries_game       = ("team_carries",       "mean"),
        avg_targets_game       = ("team_targets",       "mean"),
    )
    .reset_index()
)
team_season.to_csv("data/team_season_stats.csv", index=False)
print(f"  OK  {len(team_season)} team-season rows saved to data/team_season_stats.csv")

# -- 5. per-player weekly variance ---------------------------------------------
print("Building variance stats...")
variance_metrics = {
    "WR": ["targets", "receiving_yards", "receiving_tds", "target_share"],
    "TE": ["targets", "receiving_yards", "receiving_tds", "target_share"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receiving_yards"],
    "QB": ["attempts", "passing_yards", "passing_tds"],
}

all_var = []
for pos, metrics in variance_metrics.items():
    pw = weekly[weekly["position"] == pos]
    for metric in metrics:
        if metric not in pw.columns:
            continue
        t = (
            pw.groupby(["player_id", "player_name", "season"])[metric]
            .agg(
                mean   = "mean",
                std    = "std",
                p25    = lambda x: x.quantile(0.25),
                median = "median",
                p75    = lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
        t["metric"]   = metric
        t["position"] = pos
        all_var.append(t)

variance_df = pd.concat(all_var, ignore_index=True)
variance_df.to_csv("data/player_variance.csv", index=False)
print(f"  OK  {len(variance_df):,} variance rows saved to data/player_variance.csv")

# -- sanity check --------------------------------------------------------------
print("\n=== Season coverage (unique players per season) ===")
print(seasonal_stats.groupby("season")["player_id"].nunique().to_string())

print("\n=== Sample 2025 rows ===")
s25 = seasonal_stats[seasonal_stats["season"] == 2025]
if s25.empty:
    print("  WARNING: no 2025 data found - check if nflverse has published it yet")
else:
    print(s25[["player_name","position","team","games","fantasy_points_ppr"]]
          .sort_values("fantasy_points_ppr", ascending=False)
          .head(10)
          .to_string())

print("\nDone.")
