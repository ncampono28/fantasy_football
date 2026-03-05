"""
Historical Draft Analysis
=========================
Reads local half-PPR draft CSV files from data/ (2020-2025).
Merges with actual seasonal results and classifies each player as
Steal / Hit / Reach / Bust.  Also computes VORP (points above the
positional replacement level in a 12-team league).

Column normalisation:
  2020-2024:  Name / Pos / Overall / Std.
  2025:       Name / Position / Overall / Std. Dev

Output:
  data/draft_analysis.csv
  data/draft_analysis_summary.csv

Run:
    py 07_draft_analysis.py
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA  = Path("data")
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Value-score thresholds (pick-based, 12 picks/round)
STEAL_MIN = 36    # outperformed draft cost by 3+ rounds
BUST_MAX  = -36   # underperformed by 3+ rounds

# Positional replacement rank for VORP (12-team, standard roster)
#   QB: 1 starter  → QB13
#   RB: 2 starters + flex share → RB25
#   WR: 3 starters (incl. flex) → WR37
#   TE: 1 starter  → TE13
REPLACEMENT_RANK = {"QB": 13, "RB": 25, "WR": 37, "TE": 13}
# Use 4-player window around each threshold to smooth replacement fpts
REP_WINDOW = 4

print("=" * 60)
print("  Historical Draft Analysis  2020-2025")
print("=" * 60)


# ─── Name normalisation ────────────────────────────────────────────────────────
_SUFFIX_RE = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", re.IGNORECASE)

def _norm(name: str) -> str:
    name = _SUFFIX_RE.sub("", str(name))
    return re.sub(r"\s+", " ", name).strip().lower()


# ─── Load one season's draft CSV ──────────────────────────────────────────────
def load_draft_csv(year: int) -> pd.DataFrame:
    """Read a local half-PPR draft CSV and return a normalised DataFrame."""
    path = DATA / f"ff_draft_{year}_halfppr.csv"
    if not path.exists():
        print(f"  ! {year}: file not found — {path}")
        return pd.DataFrame()

    # utf-8-sig strips the BOM present in 2020-2024 files
    df = pd.read_csv(path, encoding="utf-8-sig")

    if year <= 2024:
        # Columns: #, Name, Pos, Team, Overall, Std., High, Low
        df = df.rename(columns={
            "Name":    "player_name_raw",
            "Pos":     "position",
            "Overall": "adp",
            "Std.":    "adp_stdev",
        })
    else:
        # 2025 columns: ADP, Overall, Name, Position, Team, Times Drafted,
        #               Std. Dev, High, Low, Bye
        df = df.rename(columns={
            "Name":      "player_name_raw",
            "Position":  "position",
            "Overall":   "adp",
            "Std. Dev":  "adp_stdev",
        })

    df = df[["player_name_raw", "position", "adp", "adp_stdev"]].copy()
    df["adp"]      = pd.to_numeric(df["adp"],      errors="coerce")
    df["adp_stdev"]= pd.to_numeric(df["adp_stdev"],errors="coerce").fillna(0)
    df = df.dropna(subset=["adp"])
    df["position"]  = df["position"].str.upper().str.strip()
    df["season"]    = year
    df["adp_rank"]  = df["adp"].round().astype(int)
    df["name_key"]  = df["player_name_raw"].apply(_norm)

    print(f"  + {year}: {len(df):4d} entries from {path.name}")
    return df


# ─── Load seasonal stats ──────────────────────────────────────────────────────
print("\n  Loading player_seasonal_stats.csv ...")
stats = pd.read_csv(DATA / "player_seasonal_stats.csv")
stats = stats[stats["season"].isin(YEARS)].copy()
stats["name_key"] = stats["player_name"].apply(_norm)

# Overall PPR rank within each season
stats["actual_rank"] = (
    stats.groupby("season")["fantasy_points_ppr"]
    .rank(ascending=False, method="min")
    .astype(int)
)
print(f"  + {len(stats)} player-season rows for {YEARS}")


# ─── Load all draft CSVs ─────────────────────────────────────────────────────
print("\n  Loading local draft CSVs ...")
adp_frames = [load_draft_csv(yr) for yr in YEARS]
adp_frames = [f for f in adp_frames if not f.empty]

if not adp_frames:
    print("\n  ERROR: No draft CSV files found in data/")
    sys.exit(1)

adp_all = pd.concat(adp_frames, ignore_index=True)
print(f"\n  Total ADP rows: {len(adp_all)}")


# ─── Merge ADP with seasonal results ─────────────────────────────────────────
print("\n  Merging ADP with seasonal results ...")
stats_slim = stats[[
    "player_id", "player_name", "position", "team", "season",
    "fantasy_points_ppr", "actual_rank", "name_key",
]].copy()

merged = adp_all.merge(
    stats_slim,
    on=["name_key", "season"],
    how="inner",
    suffixes=("_adp", "_stats"),
)

merged["position"] = merged["position_stats"].fillna(merged["position_adp"])
merged.drop(columns=["position_adp", "position_stats"], inplace=True)
print(f"  + Matched {len(merged)} player-season rows")


# ─── Value score & outcome classification ─────────────────────────────────────
merged["value_score"] = merged["adp_rank"] - merged["actual_rank"]

def classify(v: int) -> str:
    if v >= STEAL_MIN:  return "Steal"
    if v >= 0:          return "Hit"
    if v > BUST_MAX:    return "Reach"
    return "Bust"

merged["outcome"]       = merged["value_score"].apply(classify)
merged["round_drafted"] = np.ceil(merged["adp"] / 12).astype(int)


# ─── VORP — points above positional replacement ───────────────────────────────
print("\n  Computing VORP ...")

# Positional rank within each (season, position)
merged["pos_rank"] = (
    merged.groupby(["season", "position"])["fantasy_points_ppr"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Replacement-level fpts: avg of REP_WINDOW players at the threshold
rep_lookup: dict = {}
for (yr, pos), grp in merged.groupby(["season", "position"]):
    rep_rank = REPLACEMENT_RANK.get(pos)
    if rep_rank is None:
        continue
    rep_pool = grp[grp["pos_rank"].between(rep_rank, rep_rank + REP_WINDOW - 1)]
    rep_lookup[(yr, pos)] = (
        rep_pool["fantasy_points_ppr"].mean() if not rep_pool.empty
        else grp["fantasy_points_ppr"].quantile(0.15)
    )

merged["rep_fpts"] = merged.apply(
    lambda r: rep_lookup.get((r["season"], r["position"]), np.nan), axis=1
)
merged["vorp"] = (merged["fantasy_points_ppr"] - merged["rep_fpts"]).round(1)
merged.drop(columns=["rep_fpts", "pos_rank"], inplace=True)


# ─── Final column order & save ────────────────────────────────────────────────
final = merged[[
    "player_id", "player_name", "position", "team", "season",
    "adp", "adp_stdev", "adp_rank", "round_drafted",
    "fantasy_points_ppr", "actual_rank", "value_score", "vorp", "outcome",
]].sort_values(["season", "adp_rank"]).reset_index(drop=True)

out_path = DATA / "draft_analysis.csv"
final.to_csv(out_path, index=False)
print(f"\n  Saved {len(final)} rows to {out_path}")


# ─── Position × Round summary ─────────────────────────────────────────────────
pos_round = final.groupby(["position", "round_drafted"]).agg(
    avg_value_score = ("value_score",        "mean"),
    avg_vorp        = ("vorp",               "mean"),
    sample_size     = ("value_score",        "count"),
    _steal          = ("outcome", lambda s: (s == "Steal").sum()),
    _hit            = ("outcome", lambda s: (s == "Hit").sum()),
    _reach          = ("outcome", lambda s: (s == "Reach").sum()),
    _bust           = ("outcome", lambda s: (s == "Bust").sum()),
).reset_index()

pos_round["avg_value_score"] = pos_round["avg_value_score"].round(1)
pos_round["avg_vorp"]        = pos_round["avg_vorp"].round(1)
pos_round["steal_pct"] = (pos_round["_steal"] / pos_round["sample_size"] * 100).round(1)
pos_round["hit_pct"]   = (pos_round["_hit"]   / pos_round["sample_size"] * 100).round(1)
pos_round["reach_pct"] = (pos_round["_reach"] / pos_round["sample_size"] * 100).round(1)
pos_round["bust_pct"]  = (pos_round["_bust"]  / pos_round["sample_size"] * 100).round(1)
pos_round.drop(columns=["_steal", "_hit", "_reach", "_bust"], inplace=True)

summary_cols = [
    "position", "round_drafted", "avg_value_score", "avg_vorp",
    "hit_pct", "bust_pct", "steal_pct", "reach_pct", "sample_size",
]
pos_round = pos_round[summary_cols].sort_values(["position", "round_drafted"])

summary_path = DATA / "draft_analysis_summary.csv"
pos_round.to_csv(summary_path, index=False)
print(f"\n  Saved position/round summary to {summary_path}")


# ─── Print summaries ──────────────────────────────────────────────────────────
POSITIONS_ORDER = ["QB", "RB", "WR", "TE"]
print("\n  Position x Round breakdown (avg VORP | avg value score | sample):")
print(f"  {'Pos':<4} {'Rd':<4} {'AvgVORP':>8} {'AvgVal':>7} {'Hit%':>6} {'Steal%':>7} {'Bust%':>6} {'n':>5}")
print("  " + "-" * 57)
for _, row in pos_round[pos_round["position"].isin(POSITIONS_ORDER)].iterrows():
    print(
        f"  {row['position']:<4} {int(row['round_drafted']):<4}"
        f" {row['avg_vorp']:>8.1f}"
        f" {row['avg_value_score']:>7.1f}"
        f" {row['hit_pct']:>6.1f}"
        f" {row['steal_pct']:>7.1f}"
        f" {row['bust_pct']:>6.1f}"
        f" {int(row['sample_size']):>5}"
    )

print("\n  Outcome breakdown by year:")
summary_yr = (
    final.groupby(["season", "outcome"])
    .size()
    .unstack(fill_value=0)
)
for col in ["Steal", "Hit", "Reach", "Bust"]:
    if col not in summary_yr.columns:
        summary_yr[col] = 0
print(summary_yr[["Steal", "Hit", "Reach", "Bust"]].to_string())

print("\n  Top 10 Steals (all years):")
print(
    final[final["outcome"] == "Steal"]
    .sort_values("value_score", ascending=False)
    .head(10)[["season", "player_name", "position", "adp_rank", "actual_rank", "value_score", "vorp"]]
    .to_string(index=False)
)

print("\n  Top 10 Busts (all years):")
print(
    final[final["outcome"] == "Bust"]
    .sort_values("value_score")
    .head(10)[["season", "player_name", "position", "adp_rank", "actual_rank", "value_score", "vorp"]]
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("  Draft analysis complete!")
print("=" * 60)
