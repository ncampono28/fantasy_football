"""
Underdog Best Ball ADP Loader
==============================
Reads ADP from CSV files in the ud_adp/ folder.
Filename format: Underdog_Draft_Table_YYYY-MM-DD.csv

Columns expected:
  Rank, Player, Position, Position Rank,
  'ADP on <prev date>', 'ADP on <curr date>', ADP Change

Outputs:
  data/adp_current.csv  — latest ADP + model comparison + value scores
  data/adp_history.csv  — all snapshots (one row per player per date)
  data/adp_trends.csv   — per-player trend arrows

Run:
    py 05_adp_pull.py
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA       = Path("data")
UD_ADP_DIR = Path("ud_adp")
ADP_FILE   = DATA / "adp_history.csv"
POSITIONS  = ["QB", "RB", "WR", "TE"]
TODAY      = date.today().isoformat()

print("=" * 60)
print("  Underdog Best Ball ADP Loader")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD UNDERDOG CSV
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ud_file(csv_path):
    """
    Parse a single Underdog CSV file.
    Returns (DataFrame, snapshot_date_str) or (empty DataFrame, None).
    """
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path.name)
    snapshot_date = date_match.group(1) if date_match else TODAY

    df = pd.read_csv(csv_path)

    # Rename fixed columns
    df = df.rename(columns={
        "Player":        "player_name",
        "Position":      "position",
        "Position Rank": "pos_rank",
        "ADP Change":    "_adp_change_raw",
    })

    # Detect the two "ADP on ..." date columns — use order: first = prev, second = current
    adp_date_cols = [c for c in df.columns if str(c).strip().startswith("ADP on")]
    if len(adp_date_cols) >= 2:
        df = df.rename(columns={adp_date_cols[0]: "adp_prev", adp_date_cols[1]: "adp"})
    elif len(adp_date_cols) == 1:
        df = df.rename(columns={adp_date_cols[0]: "adp"})
        df["adp_prev"] = np.nan
    else:
        print(f"  ERROR: Could not find ADP date columns in {csv_path.name}")
        return pd.DataFrame(), None

    # Sign convention: positive adp_change = player rising (ADP got lower)
    if "_adp_change_raw" in df.columns:
        df["adp_change"] = (-df["_adp_change_raw"]).round(1)
    else:
        df["adp_change"] = (df["adp_prev"] - df["adp"]).round(1)

    df["snapshot_date"] = snapshot_date

    # Keep skill positions only
    df = df[df["position"].isin(POSITIONS)].copy()

    keep = ["player_name", "position", "pos_rank", "adp", "adp_prev",
            "adp_change", "snapshot_date"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    return df, snapshot_date


def load_underdog_csv():
    """
    Reads all CSVs from ud_adp/, sorted by date extracted from filename.
    Returns (latest_DataFrame, latest_snapshot_date_str, all_snapshots_DataFrame).
    """
    csv_files = sorted(UD_ADP_DIR.glob("*.csv"))
    if not csv_files:
        print("  No CSV files found in ud_adp/")
        return pd.DataFrame(), None, pd.DataFrame()

    # Load existing history to know which dates are already present
    existing_dates = set()
    if ADP_FILE.exists():
        existing_dates = set(pd.read_csv(ADP_FILE)["snapshot_date"].unique())

    all_frames = []
    latest_df, latest_date = pd.DataFrame(), None

    for f in csv_files:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not date_match:
            print(f"  Skipping (no date in filename): {f.name}")
            continue
        snap_date = date_match.group(1)
        df, _ = _parse_ud_file(f)
        if df.empty:
            continue
        print(f"  Read {f.name}  →  {len(df)} players  (snapshot: {snap_date})")
        all_frames.append(df)
        latest_df, latest_date = df, snap_date

    all_snapshots = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return latest_df, latest_date, all_snapshots


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RANKS
# ─────────────────────────────────────────────────────────────────────────────
def get_model_ranks():
    proj_file = DATA / "projections_weighted.csv"
    if not proj_file.exists():
        proj_file = DATA / "projections.csv"
    if not proj_file.exists():
        return pd.DataFrame()

    proj = pd.read_csv(proj_file)
    base = proj[proj["scenario"] == "base"].copy()

    base = base.sort_values("fpts_ppr", ascending=False).reset_index(drop=True)
    base["model_overall_rank"] = base.index + 1
    base["model_pos_rank"] = (
        base.groupby("position")["fpts_ppr"]
        .rank(ascending=False)
        .astype(int)
    )
    base["model_pos_rank_str"] = base["position"] + base["model_pos_rank"].astype(str)

    return base[["player_name", "position", "team",
                 "fpts_ppr", "model_overall_rank", "model_pos_rank_str"]]


# ─────────────────────────────────────────────────────────────────────────────
# TREND ARROWS
# ─────────────────────────────────────────────────────────────────────────────
def trend_arrow(change):
    if pd.isna(change): return "NEW"
    if change > 3:      return "++ "
    if change > 1:      return "+  "
    if change < -3:     return "--"
    if change < -1:     return "- "
    return "~"


def build_trends(adp_df):
    """
    Build trend DataFrame from the adp_change column already in the UD file.
    Falls back to history comparison if adp_change is absent.
    """
    if "adp_change" in adp_df.columns and adp_df["adp_change"].notna().any():
        trends = adp_df[["player_name", "position", "adp",
                          "adp_prev", "adp_change"]].copy()
        trends["trend"] = trends["adp_change"].apply(trend_arrow)
        return trends.sort_values("adp")

    # Fallback: compare two history snapshots
    if not ADP_FILE.exists():
        return pd.DataFrame()
    history = pd.read_csv(ADP_FILE)
    dates = sorted(history["snapshot_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    latest = history[history["snapshot_date"] == dates[-1]][["player_name", "position", "adp"]]
    prev   = history[history["snapshot_date"] == dates[-2]][["player_name", "adp"]].rename(
        columns={"adp": "adp_prev"}
    )
    trends = latest.merge(prev, on="player_name", how="left")
    trends["adp_change"] = (trends["adp_prev"] - trends["adp"]).round(1)
    trends["trend"] = trends["adp_change"].apply(trend_arrow)
    return trends.sort_values("adp")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
adp_df, snapshot_date, all_snapshots = load_underdog_csv()

if adp_df.empty:
    print("\n  No Underdog data found — exiting.")
    raise SystemExit(1)

# ── Merge with model ranks ──────────────────────────────────────────────────
model_ranks = get_model_ranks()
if not model_ranks.empty:
    adp_df = adp_df.merge(
        model_ranks[["player_name", "team", "fpts_ppr",
                     "model_overall_rank", "model_pos_rank_str"]],
        on="player_name", how="left",
    )
    print(f"  Merged with model projections ({model_ranks['player_name'].nunique()} projected players)")
else:
    print("  No model projections found — run 04_weighted_model.py first")

# ── Value score  ────────────────────────────────────────────────────────────
if "model_overall_rank" in adp_df.columns:
    adp_df["value_score"] = (adp_df["adp"] - adp_df["model_overall_rank"]).round(1)

# ── Rebuild history from all snapshots in ud_adp/ ───────────────────────────
# Load any pre-existing history rows whose dates are NOT covered by ud_adp/ files
new_dates = set(all_snapshots["snapshot_date"].unique()) if not all_snapshots.empty else set()
if ADP_FILE.exists():
    existing = pd.read_csv(ADP_FILE)
    prior = existing[~existing["snapshot_date"].isin(new_dates)]
    adp_history = pd.concat([prior, all_snapshots], ignore_index=True)
else:
    adp_history = all_snapshots.copy()

adp_history = adp_history.sort_values("snapshot_date").reset_index(drop=True)
adp_history.to_csv(ADP_FILE, index=False)
print(f"  Saved to adp_history.csv  ({adp_history['snapshot_date'].nunique()} snapshot(s))")

# ── Trends ──────────────────────────────────────────────────────────────────
trends = build_trends(adp_df)
if not trends.empty:
    trends.to_csv(DATA / "adp_trends.csv", index=False)
    risers  = trends[trends["adp_change"] > 1].head(5)
    fallers = trends[trends["adp_change"] < -1].head(5)
    if not risers.empty:
        print(f"\n  Top risers (ADP improving):")
        print(risers[["player_name", "position", "adp", "adp_change"]].to_string(index=False))
    if not fallers.empty:
        print(f"\n  Top fallers (ADP worsening):")
        print(fallers[["player_name", "position", "adp", "adp_change"]].to_string(index=False))

# ── Market inefficiencies ───────────────────────────────────────────────────
if "value_score" in adp_df.columns:
    print(f"\n  Best values (model likes more than market):")
    vals = (
        adp_df[adp_df["value_score"].notna() & (adp_df["adp"] <= 300) & (adp_df["value_score"] > 0)]
        .sort_values("value_score", ascending=False)
        .head(8)[["player_name", "position", "adp", "model_overall_rank", "value_score"]]
    )
    print(vals.to_string(index=False))

    print(f"\n  Overvalued (market likes more than model):")
    over = (
        adp_df[adp_df["value_score"].notna() & (adp_df["adp"] <= 300) & (adp_df["value_score"] < 0)]
        .sort_values("value_score")
        .head(8)[["player_name", "position", "adp", "model_overall_rank", "value_score"]]
    )
    print(over.to_string(index=False))

# ── Save adp_current.csv ────────────────────────────────────────────────────
adp_df.to_csv(DATA / "adp_current.csv", index=False)
print(f"\n  Saved adp_current.csv  ({len(adp_df)} players)")

print("\n" + "=" * 60)
print("  Done!")
print("=" * 60)
print("""
Files written:
  data/adp_current.csv   — latest UD Best Ball ADP + model ranks
  data/adp_history.csv   — snapshot history
  data/adp_trends.csv    — per-player trend arrows
""")
