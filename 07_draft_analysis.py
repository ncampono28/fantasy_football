"""
Historical Draft Analysis
=========================
Pulls September ADP from Fantasy Football Calculator (PPR, 12-team)
for 2021-2024, merges with actual seasonal results, and classifies
each player as Steal / Hit / Reach / Bust.

Output: data/draft_analysis.csv, data/draft_analysis_summary.csv

Run:
    py 07_draft_analysis.py
"""

import sys
import re
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA  = Path("data")
YEARS = [2021, 2022, 2023, 2024]
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"

# ─── Value score thresholds (pick-based, 12 picks/round) ──────────────────────
# value_score = adp_rank - actual_rank  (positive = outperformed draft cost)
STEAL_MIN  =  36   # outperformed by 3+ rounds → Steal
BUST_MAX   = -36   # underperformed by 3+ rounds → Bust
# 0 <= score < 36  → Hit
# -36 < score < 0  → Reach

print("=" * 60)
print("  Historical Draft Analysis — FFC PPR ADP 2021-2024")
print("=" * 60)


# ─── Name normalisation helpers ────────────────────────────────────────────────
_SUFFIX_RE = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", re.IGNORECASE)

def _norm(name: str) -> str:
    """Lowercase, strip suffixes, collapse whitespace."""
    name = _SUFFIX_RE.sub("", str(name))
    return re.sub(r"\s+", " ", name).strip().lower()


# ─── Pull ADP from Fantasy Football Calculator ─────────────────────────────────
def pull_ffc_adp(year: int) -> pd.DataFrame:
    url = FFC_URL.format(year=year)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; fantasy-football-research/1.0)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ! Year {year}: request failed — {e}")
        return pd.DataFrame()

    players = data.get("players")
    if not players:
        print(f"  ! Year {year}: no 'players' key in response")
        return pd.DataFrame()

    rows = []
    for p in players:
        adp_val = p.get("adp")
        if adp_val is None:
            continue
        try:
            adp_float = float(adp_val)
        except (TypeError, ValueError):
            continue
        rows.append({
            "player_name_raw": str(p.get("name", "")).strip(),
            "position":        str(p.get("position", "")).upper(),
            "adp":             adp_float,
            "adp_stdev":       float(p.get("stdev") or 0),
            "season":          year,
        })

    if not rows:
        print(f"  ! Year {year}: parsed 0 players")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # adp_rank = integer pick number (rounded)
    df["adp_rank"] = df["adp"].round().astype(int)
    df["name_key"] = df["player_name_raw"].apply(_norm)
    print(f"  + {year}: {len(df):4d} ADP entries pulled")
    return df


# ─── Load seasonal stats ───────────────────────────────────────────────────────
print("\n  Loading player_seasonal_stats.csv ...")
stats = pd.read_csv(DATA / "player_seasonal_stats.csv")
stats = stats[stats["season"].isin(YEARS)].copy()
stats["name_key"] = stats["player_name"].apply(_norm)

# Compute PPR actual rank within each season (all positions combined)
stats["actual_rank"] = (
    stats.groupby("season")["fantasy_points_ppr"]
    .rank(ascending=False, method="min")
    .astype(int)
)

print(f"  + {len(stats)} player-season rows for {YEARS}")


# ─── Pull all years ────────────────────────────────────────────────────────────
print("\n  Pulling ADP from Fantasy Football Calculator ...")
adp_frames = []
for yr in YEARS:
    df = pull_ffc_adp(yr)
    if not df.empty:
        adp_frames.append(df)
    time.sleep(1)   # polite delay

if not adp_frames:
    print("\n  ERROR: No ADP data retrieved. Check network / API availability.")
    sys.exit(1)

adp_all = pd.concat(adp_frames, ignore_index=True)
print(f"\n  Total ADP rows: {len(adp_all)}")


# ─── Merge on (name_key, season) ──────────────────────────────────────────────
print("\n  Merging ADP with seasonal results ...")

# Keep only the columns needed from stats
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

# Resolve position column (prefer stats)
merged["position"] = merged["position_stats"].fillna(merged["position_adp"])
merged.drop(columns=["position_adp", "position_stats"], inplace=True)

print(f"  + Matched {len(merged)} player-season rows")


# ─── Value score & outcome classification ─────────────────────────────────────
merged["value_score"] = merged["adp_rank"] - merged["actual_rank"]

def classify(v: int) -> str:
    if v >= STEAL_MIN:
        return "Steal"
    if v >= 0:
        return "Hit"
    if v > BUST_MAX:
        return "Reach"
    return "Bust"

merged["outcome"] = merged["value_score"].apply(classify)

# round_drafted = ceil(adp / 12) for a 12-team league
merged["round_drafted"] = np.ceil(merged["adp"] / 12).astype(int)


# ─── Final column order ────────────────────────────────────────────────────────
final = merged[[
    "player_id",
    "player_name",
    "position",
    "team",
    "season",
    "adp",
    "adp_stdev",
    "adp_rank",
    "round_drafted",
    "fantasy_points_ppr",
    "actual_rank",
    "value_score",
    "outcome",
]].sort_values(["season", "adp_rank"]).reset_index(drop=True)


# ─── Save ──────────────────────────────────────────────────────────────────────
out_path = DATA / "draft_analysis.csv"
final.to_csv(out_path, index=False)
print(f"\n  Saved {len(final)} rows to {out_path}")


# ─── Summary ──────────────────────────────────────────────────────────────────
# ─── Position × Round summary ─────────────────────────────────────────────────
pos_round = final.groupby(["position", "round_drafted"]).agg(
    avg_value_score=("value_score", "mean"),
    sample_size=("value_score", "count"),
    _steal=("outcome", lambda s: (s == "Steal").sum()),
    _hit=("outcome",   lambda s: (s == "Hit").sum()),
    _reach=("outcome", lambda s: (s == "Reach").sum()),
    _bust=("outcome",  lambda s: (s == "Bust").sum()),
).reset_index()

pos_round["avg_value_score"] = pos_round["avg_value_score"].round(1)
pos_round["steal_pct"] = (pos_round["_steal"] / pos_round["sample_size"] * 100).round(1)
pos_round["hit_pct"]   = (pos_round["_hit"]   / pos_round["sample_size"] * 100).round(1)
pos_round["reach_pct"] = (pos_round["_reach"] / pos_round["sample_size"] * 100).round(1)
pos_round["bust_pct"]  = (pos_round["_bust"]  / pos_round["sample_size"] * 100).round(1)
pos_round.drop(columns=["_steal", "_hit", "_reach", "_bust"], inplace=True)

summary_cols = ["position", "round_drafted", "avg_value_score",
                "hit_pct", "bust_pct", "steal_pct", "reach_pct", "sample_size"]
pos_round = pos_round[summary_cols].sort_values(["position", "round_drafted"])

summary_path = DATA / "draft_analysis_summary.csv"
pos_round.to_csv(summary_path, index=False)
print(f"\n  Saved position/round summary to {summary_path}")

# Print the breakdown table
POSITIONS_ORDER = ["QB", "RB", "WR", "TE"]
print("\n  Position × Round breakdown (avg value score | sample):")
print(f"  {'Pos':<4} {'Rd':<4} {'AvgVal':>7} {'Hit%':>6} {'Steal%':>7} {'Bust%':>6} {'n':>5}")
print("  " + "-" * 48)
for _, row in pos_round[pos_round["position"].isin(POSITIONS_ORDER)].iterrows():
    print(
        f"  {row['position']:<4} {int(row['round_drafted']):<4}"
        f" {row['avg_value_score']:>7.1f}"
        f" {row['hit_pct']:>6.1f}"
        f" {row['steal_pct']:>7.1f}"
        f" {row['bust_pct']:>6.1f}"
        f" {int(row['sample_size']):>5}"
    )

print("\n  Outcome breakdown by year:")
summary = (
    final.groupby(["season", "outcome"])
    .size()
    .unstack(fill_value=0)
)
# Ensure all outcome columns present
for col in ["Steal", "Hit", "Reach", "Bust"]:
    if col not in summary.columns:
        summary[col] = 0
print(summary[["Steal", "Hit", "Reach", "Bust"]].to_string())

print("\n  Top 10 Steals (all years):")
steals = (
    final[final["outcome"] == "Steal"]
    .sort_values("value_score", ascending=False)
    .head(10)[["season", "player_name", "position", "adp_rank", "actual_rank", "value_score"]]
)
print(steals.to_string(index=False))

print("\n  Top 10 Busts (all years):")
busts = (
    final[final["outcome"] == "Bust"]
    .sort_values("value_score")
    .head(10)[["season", "player_name", "position", "adp_rank", "actual_rank", "value_score"]]
)
print(busts.to_string(index=False))

print("\n" + "=" * 60)
print("  Draft analysis complete!")
print("=" * 60)
print("""
Output:
  data/draft_analysis.csv
    player_id, player_name, position, team, season,
    adp, adp_stdev, adp_rank,
    fantasy_points_ppr, actual_rank,
    value_score   (adp_rank - actual_rank, positive = outperformed),
    round_drafted (ceil(adp/12)),
    outcome       (Steal >= 36 | Hit >= 0 | Reach > -36 | Bust <= -36)

  data/draft_analysis_summary.csv
    position, round_drafted, avg_value_score,
    hit_pct, bust_pct, steal_pct, reach_pct, sample_size
""")
