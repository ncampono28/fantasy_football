"""
Draft VORP Analysis
====================
Reads preseason ADP CSVs (fantasy_draft_history/) and player_variance.csv,
calculates per-game VORP for every drafted player across 2021-2025, and
outputs two CSVs consumed by the Draft Intelligence page in app.py:

  data/draft_vorp_full.csv     — one row per player per season, with ADP + actuals
  data/draft_vorp_summary.csv  — aggregated VORP by position x round x season

Run:
    py 07_draft_analysis.py

Requires fantasy_draft_history/ folder at project root with:
    ff_draft_2021_halfppr.csv  (columns: Name, Pos, Overall, ...)
    ff_draft_2022_halfppr.csv
    ff_draft_2023_halfppr.csv
    ff_draft_2024_halfppr.csv
    ff_draft_2025_halfppr.csv  (columns: Name, Position, Overall, ...)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")
DRAFT_DIR = Path("fantasy_draft_history")

SEASONS = [2021, 2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]

# Replacement level thresholds (12-team league)
REPLACEMENT = {"QB": 13, "RB": 25, "WR": 25, "TE": 13}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ADP DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Draft VORP Analysis")
print("=" * 60)

adp_frames = []
for season in SEASONS:
    path = DRAFT_DIR / f"ff_draft_{season}_halfppr.csv"
    if not path.exists():
        print(f"  ⚠️  Missing: {path}")
        continue
    df = pd.read_csv(path)
    # 2025 has different column names
    if season <= 2024:
        df = df.rename(columns={"Name": "player_name", "Pos": "position", "Overall": "adp"})
    else:
        df = df.rename(columns={"Name": "player_name", "Position": "position", "Overall": "adp"})
    df["season"] = season
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["adp_round"] = df["adp"].apply(lambda x: int(np.ceil(x / 12)) if pd.notna(x) else None)
    adp_frames.append(df[["player_name", "position", "adp", "adp_round", "season"]])

adp_all = pd.concat(adp_frames).dropna(subset=["adp"])
adp_all = adp_all[adp_all["position"].isin(POSITIONS)]
print(f"\n  ✓ ADP loaded: {len(adp_all)} rows across {adp_all['season'].nunique()} seasons")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD VARIANCE / ACTUALS
# ─────────────────────────────────────────────────────────────────────────────
variance = pd.read_csv(DATA / "player_variance.csv")
print(f"  ✓ Variance loaded: {len(variance)} rows")


def calc_fpts_pg(row):
    """Half PPR points per game from median per-game stats."""
    pos = str(row.get("position", ""))
    pts  = row.get("passing_yards", 0) * 0.04 + row.get("passing_tds", 0) * 4
    pts += row.get("rushing_yards", 0) * 0.1  + row.get("rushing_tds", 0) * 6
    pts += row.get("receiving_yards", 0) * 0.1 + row.get("receiving_tds", 0) * 6
    catch = {"WR": 0.72, "TE": 0.72, "RB": 0.82, "QB": 0}
    pts += row.get("targets", 0) * catch.get(pos, 0.72) * 0.5
    return round(pts, 2)


def est_games(row):
    pos = str(row.get("position", ""))
    if pos == "QB":  return 15 if row.get("attempts", 0) > 25 else 10
    if pos in ("WR", "TE"): return 15 if row.get("targets", 0) > 5 else 10
    if pos == "RB":  return 15 if row.get("carries", 0) > 8 else 10
    return 12


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE VORP PER SEASON
# ─────────────────────────────────────────────────────────────────────────────
all_results = []

for season in SEASONS:
    s = variance[variance["season"] == season].copy()
    if s.empty:
        print(f"  ⚠️  No variance data for {season}")
        continue

    wide = s.pivot_table(
        index=["player_id", "player_name", "position"],
        columns="metric", values="median"
    ).reset_index()
    wide.columns.name = None
    wide = wide.fillna(0)

    wide["fpts_pg"]     = wide.apply(calc_fpts_pg, axis=1)
    wide["est_games"]   = wide.apply(est_games, axis=1)
    wide["fpts_season"] = (wide["fpts_pg"] * wide["est_games"]).round(1)

    # Replacement levels
    rep_levels = {}
    for pos, threshold in REPLACEMENT.items():
        pos_df = wide[wide["position"] == pos].sort_values("fpts_pg", ascending=False)
        rep_levels[pos] = round(pos_df.iloc[threshold - 1]["fpts_pg"], 2) if len(pos_df) >= threshold else 0

    wide["vorp_pg"]      = wide.apply(lambda r: round(r["fpts_pg"] - rep_levels.get(str(r["position"]), 0), 2), axis=1)
    wide["vorp_season"]  = (wide["vorp_pg"] * wide["est_games"]).round(1)
    wide["season"]       = season

    for pos, val in rep_levels.items():
        wide.loc[wide["position"] == pos, "replacement_pg"] = val

    all_results.append(
        wide[["player_name", "position", "season", "fpts_pg", "fpts_season",
              "vorp_pg", "vorp_season", "replacement_pg", "est_games"]]
    )

all_players = pd.concat(all_results)

# ─────────────────────────────────────────────────────────────────────────────
# MERGE ADP + ACTUALS
# ─────────────────────────────────────────────────────────────────────────────
merged = all_players.merge(
    adp_all[["player_name", "season", "adp", "adp_round"]],
    on=["player_name", "season"], how="inner"
)
merged = merged[merged["position"].isin(POSITIONS)]

# Outcome labels for the dashboard
def label_outcome(vorp):
    if vorp >= 5:   return "Elite"
    if vorp >= 3:   return "Hit"
    if vorp >= 1:   return "Contributor"
    if vorp >= 0:   return "Neutral"
    if vorp >= -2:  return "Bust"
    return "Disaster"

merged["outcome"] = merged["vorp_pg"].apply(label_outcome)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
merged.to_csv(DATA / "draft_vorp_full.csv", index=False)
print(f"\n  ✓ draft_vorp_full.csv — {len(merged)} rows")

# Summary: aggregated by season x position x round
summary = (
    merged.groupby(["season", "position", "adp_round"])
    .agg(
        avg_vorp_pg    = ("vorp_pg",  "mean"),
        median_vorp_pg = ("vorp_pg",  "median"),
        avg_fpts_pg    = ("fpts_pg",  "mean"),
        hit_rate       = ("vorp_pg",  lambda x: (x > 1.0).mean() * 100),
        bust_rate      = ("vorp_pg",  lambda x: (x < 0).mean() * 100),
        elite_rate     = ("vorp_pg",  lambda x: (x > 3.0).mean() * 100),
        n              = ("vorp_pg",  "count"),
    )
    .reset_index()
)
summary = summary[summary["n"] >= 2]
summary["avg_vorp_pg"]    = summary["avg_vorp_pg"].round(2)
summary["median_vorp_pg"] = summary["median_vorp_pg"].round(2)
summary["avg_fpts_pg"]    = summary["avg_fpts_pg"].round(2)
summary["hit_rate"]       = summary["hit_rate"].round(1)
summary["bust_rate"]      = summary["bust_rate"].round(1)
summary["elite_rate"]     = summary["elite_rate"].round(1)

summary.to_csv(DATA / "draft_vorp_summary.csv", index=False)
print(f"  ✓ draft_vorp_summary.csv — {len(summary)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SANITY CHECK — 2025 Top VORP/gm per position")
print("=" * 60)
for pos in POSITIONS:
    top = merged[(merged["season"] == 2025) & (merged["position"] == pos)].nlargest(3, "vorp_pg")
    print(f"\n  {pos}:")
    for _, r in top.iterrows():
        print(f"    {r['player_name']:<25} ADP={r['adp']:.1f} R{r['adp_round']}  VORP={r['vorp_pg']:+.2f}")

print("\n  ✅ Done — run app.py to see the updated Draft Intelligence page")
