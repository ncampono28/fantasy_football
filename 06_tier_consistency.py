"""
Positional Tier Consistency Tracker
=====================================
For every player, every week, calculates their positional rank
and counts how many weeks they finished as:
  - RB1 / RB2 / RB3 / RB4+ (based on league size)
  - WR1 / WR2 / WR3 / WR4+
  - QB1 / QB2 / QB3+
  - TE1 / TE2 / TE3+

Supports all 4 scoring formats and all league sizes.
Outputs one row per player per season with weekly tier counts.

Run:
    py 06_tier_consistency.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")
SEASONS = [2021, 2022, 2023, 2024]
POSITIONS = ["QB", "RB", "WR", "TE"]

# ─────────────────────────────────────────────────────────────────────────────
# SCORING FORMATS
# ─────────────────────────────────────────────────────────────────────────────
SCORING = {
    "standard":   {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 0,   "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "half_ppr":   {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "ppr":        {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0},
    "te_premium": {"pass_yd": 0.04, "pass_td": 4, "int": -2, "rush_yd": 0.1, "rush_td": 6, "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "te_bonus": 0.5},
}

# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE SIZE THRESHOLDS
# How many starters per position determines the tier cutoffs
# ─────────────────────────────────────────────────────────────────────────────
# Standard roster: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE)
# Tier 1 = starter on every team, Tier 2 = starter on most, etc.
LEAGUE_SIZES = [10, 12, 14, 16]

def get_tier_thresholds(position, league_size):
    """
    Returns tier cutoffs for a position based on league size.
    Tier 1 = top N players (weekly starters on all teams)
    Tier 2 = next N players
    etc.
    """
    if position == "QB":
        # 1 QB per team, superflex adds ~half
        t1 = league_size
        t2 = league_size * 2
        t3 = league_size * 3
    elif position in ("RB", "WR"):
        # 2 starters + ~0.5 flex per team
        t1 = league_size
        t2 = league_size * 2
        t3 = league_size * 3
        t4 = league_size * 4
    elif position == "TE":
        # 1 TE per team
        t1 = league_size
        t2 = league_size * 2
        t3 = league_size * 3
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4 if position in ("RB", "WR") else t3}


def calc_fpts(row, fmt, position):
    """Calculate fantasy points for a single game row."""
    s = SCORING[fmt]
    pts  = row.get("passing_yards",  0) * s["pass_yd"]  if pd.notna(row.get("passing_yards"))  else 0
    pts += row.get("passing_tds",    0) * s["pass_td"]  if pd.notna(row.get("passing_tds"))    else 0
    pts += row.get("interceptions",  0) * s["int"]      if pd.notna(row.get("interceptions"))  else 0
    pts += row.get("rushing_yards",  0) * s["rush_yd"]  if pd.notna(row.get("rushing_yards"))  else 0
    pts += row.get("rushing_tds",    0) * s["rush_td"]  if pd.notna(row.get("rushing_tds"))    else 0
    te_b = s["te_bonus"] if position == "TE" else 0
    pts += row.get("receptions",     0) * (s["rec"] + te_b) if pd.notna(row.get("receptions")) else 0
    pts += row.get("receiving_yards",0) * s["rec_yd"]   if pd.notna(row.get("receiving_yards")) else 0
    pts += row.get("receiving_tds",  0) * s["rec_td"]   if pd.notna(row.get("receiving_tds"))  else 0
    return round(pts, 2)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Positional Tier Consistency Tracker  ")
print("=" * 60)

weekly = pd.read_csv(DATA / "player_weekly_stats.csv")
weekly = weekly[weekly["season"].isin(SEASONS)].copy()
weekly = weekly[weekly["week"] <= 17].copy()

print(f"\n  Loaded {len(weekly):,} weekly rows")
print(f"  Seasons: {sorted(weekly['season'].unique())}")

# Fill NaN stats with 0 for calculation
stat_cols = ["passing_yards", "passing_tds", "interceptions",
             "rushing_yards", "rushing_tds", "receptions",
             "receiving_yards", "receiving_tds"]
for col in stat_cols:
    if col in weekly.columns:
        weekly[col] = weekly[col].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE FANTASY POINTS FOR ALL FORMATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Calculating fantasy points for all formats...")

for fmt in SCORING:
    weekly[f"fpts_{fmt}"] = weekly.apply(
        lambda row: calc_fpts(row, fmt, row["position"]), axis=1
    )

print("  ✓ Fantasy points calculated")


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY POSITIONAL RANKINGS
# For each week, rank all players at each position by fantasy points
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building weekly positional rankings...")

for fmt in SCORING:
    col = f"fpts_{fmt}"
    rank_col = f"pos_rank_{fmt}"
    weekly[rank_col] = (
        weekly
        .groupby(["season", "week", "position"])[col]
        .rank(ascending=False, method="min")
        .astype(int)
    )

print("  ✓ Weekly rankings built")


# ─────────────────────────────────────────────────────────────────────────────
# TIER CONSISTENCY AGGREGATION
# Count how many weeks each player hit each tier
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Aggregating tier consistency...")

all_results = []

for season in SEASONS:
    season_df = weekly[weekly["season"] == season].copy()

    for fmt in SCORING:
        rank_col = f"pos_rank_{fmt}"
        fpts_col = f"fpts_{fmt}"

        for league_size in LEAGUE_SIZES:
            for pos in POSITIONS:
                pos_df = season_df[season_df["position"] == pos].copy()
                if pos_df.empty:
                    continue

                thresh = get_tier_thresholds(pos, league_size)

                # Group by player
                player_groups = pos_df.groupby(["player_id", "player_name"])

                for (pid, pname), pg in player_groups:
                    games = len(pg)
                    if games < 1:
                        continue

                    ranks = pg[rank_col].values
                    fpts  = pg[fpts_col].values

                    # Count tier weeks
                    t1_weeks = int((ranks <= thresh["t1"]).sum())
                    t2_weeks = int(((ranks > thresh["t1"]) & (ranks <= thresh["t2"])).sum())
                    t3_weeks = int(((ranks > thresh["t2"]) & (ranks <= thresh["t3"])).sum())
                    t4_weeks = int((ranks > thresh["t3"]).sum())

                    # BOOM METRIC 1 — Tier 1 weeks (WR1/RB1/QB1/TE1)
                    tier1_boom_weeks = t1_weeks

                    # BOOM METRIC 2 — Top 25% of position that week
                    top25_thresh = max(3, league_size // 4)
                    top25_boom_weeks = int((ranks <= top25_thresh).sum())

                    # BOOM METRIC 3 — Scored 20+ fantasy points
                    pts20_boom_weeks = int((fpts >= 20).sum())

                    # Bust weeks = scored fewer than 5 fantasy points
                    bust_weeks = int((fpts < 5).sum())

                    all_results.append({
                        "player_id":        pid,
                        "player_name":      pname,
                        "position":         pos,
                        "season":           season,
                        "league_size":      league_size,
                        "scoring_format":   fmt,
                        "games":            games,
                        "avg_fpts":         round(float(np.mean(fpts)), 1),
                        "total_fpts":       round(float(np.sum(fpts)), 1),
                        "median_fpts":      round(float(np.median(fpts)), 1),
                        "std_fpts":         round(float(np.std(fpts)), 1),
                        f"{pos}1_weeks":    t1_weeks,
                        f"{pos}2_weeks":    t2_weeks,
                        f"{pos}3_weeks":    t3_weeks,
                        f"{pos}4_weeks":    t4_weeks,
                        f"{pos}1_pct":      round(t1_weeks / games * 100, 1),
                        f"{pos}2_pct":      round(t2_weeks / games * 100, 1),
                        # Three boom definitions
                        "boom_tier1_weeks": tier1_boom_weeks,
                        "boom_top25_weeks": top25_boom_weeks,
                        "boom_20pts_weeks": pts20_boom_weeks,
                        "boom_tier1_pct":   round(tier1_boom_weeks / games * 100, 1),
                        "boom_top25_pct":   round(top25_boom_weeks / games * 100, 1),
                        "boom_20pts_pct":   round(pts20_boom_weeks / games * 100, 1),
                        "bust_weeks":       bust_weeks,
                        "bust_pct":         round(bust_weeks / games * 100, 1),
                    })

tiers_df = pd.DataFrame(all_results)
tiers_df.to_csv(DATA / "tier_consistency.csv", index=False)
print(f"  ✓ {len(tiers_df):,} rows saved to data/tier_consistency.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SANITY CHECKS — 12-team PPR 2024")
print("=" * 60)

check = tiers_df[
    (tiers_df["league_size"] == 12) &
    (tiers_df["scoring_format"] == "ppr") &
    (tiers_df["season"] == 2024)
].copy()

for pos in POSITIONS:
    pos_check = check[check["position"] == pos].sort_values("total_fpts", ascending=False)
    t1_col = f"{pos}1_weeks"
    t2_col = f"{pos}2_weeks"
    if t1_col not in pos_check.columns:
        continue

    print(f"\n  Top 10 {pos}s — 2024 PPR 12-team:")
    print(f"  {'Player':<25} {'G':>3} {'TotFP':>7} {'AvgFP':>6} {pos}1W {pos}2W  Tier1%  Top25%  20pts%  Bust%")
    print(f"  {'-'*90}")
    for _, r in pos_check.head(10).iterrows():
        print(f"  {r['player_name']:<25} {r['games']:>3} {r['total_fpts']:>7} "
              f"{r['avg_fpts']:>6} {r[t1_col]:>5} {r[t2_col]:>5} "
              f"{r['boom_tier1_pct']:>6}% {r['boom_top25_pct']:>6}% "
              f"{r['boom_20pts_pct']:>6}% {r['bust_pct']:>5}%")


print("\n" + "=" * 60)
print("  ✅ Tier consistency data complete!")
print("=" * 60)
print(f"""
Files created:
  📄 data/tier_consistency.csv  —  {len(tiers_df):,} rows
     Dimensions: {len(SEASONS)} seasons × {len(SCORING)} formats × {len(LEAGUE_SIZES)} league sizes

Next: Tell Claude Code to add a 'Tier History' tab to app.py that:
  - Lets users pick a player, scoring format, and league size
  - Shows a bar chart of RB1/RB2/RB3/RB4 weeks per season
  - Shows boom% and bust% alongside the tier counts
  - Lets users compare two players side by side
""")
