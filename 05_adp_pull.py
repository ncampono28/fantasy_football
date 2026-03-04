"""
Sleeper ADP Puller
=====================
Pulls current ADP from the Sleeper API and appends
a dated snapshot to data/adp_history.csv.

Run manually or via refresh_data.py:
    py 05_adp_pull.py

Each run adds one row per player with today's date, building
a historical trend over time.
"""

import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import time

# Force UTF-8 output so emoji/special chars work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA = Path("data")
ADP_FILE = DATA / "adp_history.csv"
POSITIONS = ["QB", "RB", "WR", "TE"]
TODAY = date.today().isoformat()

print("=" * 60)
print("  Sleeper ADP Pull  ")
print("=" * 60)
print(f"  Snapshot date: {TODAY}")


# ─────────────────────────────────────────────────────────────────────────────
# PULL FROM SLEEPER
# Public endpoint — no auth required
# ─────────────────────────────────────────────────────────────────────────────
def pull_sleeper_adp():
    """
    Pull player rankings from the Sleeper /v1/players/nfl endpoint.
    Uses the 'search_rank' field as ADP proxy (lower = higher value).
    Filters to active QB/RB/WR/TE players with a current team.
    Returns a DataFrame sorted by adp ascending.
    """
    url = "https://api.sleeper.app/v1/players/nfl"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()   # dict: {player_id: {player_obj}}

        rows = []
        for player_id, p in data.items():
            pos = p.get("position") or ""
            if pos not in POSITIONS:
                continue
            if not p.get("active"):
                continue
            if not p.get("team"):
                continue
            rank = p.get("search_rank")
            if rank is None:
                continue
            try:
                rank_float = float(rank)
            except (TypeError, ValueError):
                continue
            rows.append({
                "player_name": p.get("full_name") or (
                    f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                ),
                "first_name":  p.get("first_name", ""),
                "last_name":   p.get("last_name", ""),
                "position":    pos,
                "team":        p.get("team") or "",
                "adp":         rank_float,
            })

        if not rows:
            print("  ⚠ Sleeper returned data but no ranked active players.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  ⚠ Sleeper pull failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PROJECTIONS to compute model-implied ADP
# ─────────────────────────────────────────────────────────────────────────────
def get_model_ranks():
    """
    Load weighted projections and compute positional + overall ranks
    in PPR scoring. This becomes our 'model implied ADP'.
    """
    proj_file = DATA / "projections_weighted.csv"
    if not proj_file.exists():
        proj_file = DATA / "projections.csv"
    if not proj_file.exists():
        return pd.DataFrame()

    proj = pd.read_csv(proj_file)
    base = proj[proj["scenario"] == "base"].copy()

    # Overall PPR rank
    base = base.sort_values("fpts_ppr", ascending=False).reset_index(drop=True)
    base["model_overall_rank"] = base.index + 1

    # Positional rank
    base["model_pos_rank"] = (
        base.groupby("position")["fpts_ppr"]
        .rank(ascending=False)
        .astype(int)
    )
    base["model_pos_rank_str"] = base["position"] + base["model_pos_rank"].astype(str)

    return base[["player_name", "position", "team",
                 "fpts_ppr", "model_overall_rank", "model_pos_rank_str"]]


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT + APPEND
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Pulling Sleeper ADP...")
adp_df = pull_sleeper_adp()

if adp_df.empty:
    print("\n  ⚠ Could not pull live ADP — Sleeper API may have changed.")
    print("  Generating sample data for development purposes...")

    # Sample data so the rest of the pipeline works while we fix the API
    sample_players = [
        ("Ja'Marr Chase",      "WR", "CIN",  1.8),
        ("Justin Jefferson",   "WR", "MIN",  3.1),
        ("CeeDee Lamb",        "WR", "DAL",  4.2),
        ("Christian McCaffrey","RB", "SF",   5.1),
        ("Tyreek Hill",        "WR", "MIA",  6.3),
        ("Travis Kelce",       "TE", "KC",   7.4),
        ("Saquon Barkley",     "RB", "PHI",  8.2),
        ("Stefon Diggs",       "WR", "NE",   9.1),
        ("Lamar Jackson",      "QB", "BAL", 10.3),
        ("Davante Adams",      "WR", "LV",  11.2),
        ("Justin Herbert",     "QB", "LAC", 45.1),
        ("Patrick Mahomes",    "QB", "KC",  48.3),
        ("Tony Pollard",       "RB", "TEN", 52.1),
        ("Sam LaPorta",        "TE", "DET", 61.2),
        ("Tee Higgins",        "WR", "CIN", 22.4),
        ("Amon-Ra St. Brown",  "WR", "DET", 18.7),
        ("Bijan Robinson",     "RB", "ATL", 14.2),
        ("Puka Nacua",         "WR", "LAR", 35.6),
        ("Josh Allen",         "QB", "BUF", 12.1),
        ("Breece Hall",        "RB", "NYJ", 16.8),
    ]
    adp_df = pd.DataFrame(sample_players,
                          columns=["player_name", "position", "team", "adp"])
    adp_df["draft_count"] = 0
    print(f"  ✓ Sample data: {len(adp_df)} players")
else:
    print(f"  ✓ Live ADP pulled: {len(adp_df)} players")

# Add snapshot date
adp_df["snapshot_date"] = TODAY

# ─────────────────────────────────────────────────────────────────────────────
# MERGE WITH MODEL RANKS
# ─────────────────────────────────────────────────────────────────────────────
model_ranks = get_model_ranks()
if not model_ranks.empty:
    adp_df = adp_df.merge(
        model_ranks[["player_name", "fpts_ppr", "model_overall_rank", "model_pos_rank_str"]],
        on="player_name", how="left"
    )
    adp_df["rank_diff"] = adp_df["model_overall_rank"] - adp_df["adp"]
    print(f"  ✓ Merged with model projections")

# ─────────────────────────────────────────────────────────────────────────────
# APPEND TO HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if ADP_FILE.exists():
    existing = pd.read_csv(ADP_FILE)
    # Don't double-add if we already ran today
    if TODAY in existing["snapshot_date"].values:
        print(f"\n  ℹ Already have a snapshot for {TODAY} — overwriting today's data.")
        existing = existing[existing["snapshot_date"] != TODAY]
    adp_history = pd.concat([existing, adp_df], ignore_index=True)
else:
    adp_history = adp_df

adp_history.to_csv(ADP_FILE, index=False)
print(f"  ✓ Saved {len(adp_df)} players to adp_history.csv")
print(f"  ✓ Total snapshots in history: {adp_history['snapshot_date'].nunique()} dates")


# ─────────────────────────────────────────────────────────────────────────────
# ADP TREND CALCULATION
# Compares latest snapshot to previous snapshot
# ─────────────────────────────────────────────────────────────────────────────
def calculate_adp_trends(history_df):
    """
    Compare each player's ADP this week vs last week.
    Returns trend direction and magnitude.
    """
    dates = sorted(history_df["snapshot_date"].unique())

    if len(dates) < 2:
        print("\n  ℹ Only one snapshot so far — trends available after next weekly pull.")
        return pd.DataFrame()

    latest_date = dates[-1]
    prev_date   = dates[-2]

    latest = history_df[history_df["snapshot_date"] == latest_date][["player_name", "position", "adp"]]
    prev   = history_df[history_df["snapshot_date"] == prev_date][["player_name", "adp"]].rename(columns={"adp": "adp_prev"})

    trends = latest.merge(prev, on="player_name", how="left")
    trends["adp_change"]  = trends["adp_prev"] - trends["adp"]  # positive = rising (lower ADP)
    trends["adp_change"]  = trends["adp_change"].round(1)

    def trend_arrow(change):
        if pd.isna(change):   return "🆕"   # new player
        if change > 3:        return "🔺🔺"  # big riser
        if change > 1:        return "🔺"    # riser
        if change < -3:       return "🔻🔻"  # big faller
        if change < -1:       return "🔻"    # faller
        return "➡️"                           # stable

    trends["trend"] = trends["adp_change"].apply(trend_arrow)
    return trends.sort_values("adp")

trends = calculate_adp_trends(adp_history)
if not trends.empty:
    print(f"\n  Top ADP risers this week:")
    print(trends[trends["adp_change"] > 0].head(5)[["player_name", "position", "adp", "adp_change", "trend"]].to_string(index=False))
    print(f"\n  Top ADP fallers this week:")
    print(trends[trends["adp_change"] < 0].head(5)[["player_name", "position", "adp", "adp_change", "trend"]].to_string(index=False))

    trends.to_csv(DATA / "adp_trends.csv", index=False)
    print(f"\n  ✓ Trends saved to data/adp_trends.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MARKET INEFFICIENCIES
# Players where model rank vs ADP diverges most
# ─────────────────────────────────────────────────────────────────────────────
if "model_overall_rank" in adp_df.columns:
    adp_df["value_score"] = adp_df["adp"] - adp_df["model_overall_rank"]

    print(f"\n  🟢 Top values (model likes more than market):")
    print(
        adp_df[adp_df["value_score"] > 0]
        .sort_values("value_score", ascending=False)
        .head(8)[["player_name", "position", "team", "adp", "model_overall_rank", "value_score"]]
        .to_string(index=False)
    )

    print(f"\n  🔴 Overvalued (market likes more than model):")
    print(
        adp_df[adp_df["value_score"] < 0]
        .sort_values("value_score")
        .head(8)[["player_name", "position", "team", "adp", "model_overall_rank", "value_score"]]
        .to_string(index=False)
    )

    adp_df.to_csv(DATA / "adp_current.csv", index=False)
    print(f"\n  ✓ Current ADP + model comparison saved to data/adp_current.csv")


print("\n" + "=" * 60)
print("  ✅ ADP pull complete!")
print("=" * 60)
print("""
Files created:
  📄 data/adp_history.csv   — all snapshots with dates
  📄 data/adp_current.csv   — latest ADP + model comparison
  📄 data/adp_trends.csv    — rising/falling trends (2+ snapshots)

Next: Tell Claude Code to add an ADP tab to app.py that shows:
  - Current ADP rankings with trend arrows
  - Model rank vs market rank comparison
  - Market inefficiency highlights
""")
