import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="FF Projection Model",
    page_icon="🏈",
    layout="wide",
)

# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    seasonal   = pd.read_csv("data/player_seasonal_stats.csv")
    variance   = pd.read_csv("data/player_variance.csv")
    team_stats = pd.read_csv("data/team_season_stats.csv")
    return seasonal, variance, team_stats


@st.cache_data
def load_tier_data():
    from pathlib import Path
    f = Path("data/tier_consistency.csv")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_data
def load_adp_data():
    from pathlib import Path
    cur_file   = Path("data/adp_current.csv")
    hist_file  = Path("data/adp_history.csv")
    trend_file = Path("data/adp_trends.csv")

    if not cur_file.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cur = pd.read_csv(cur_file)
    cur = cur[cur["adp"] < 9_999_000].copy()

    hist = pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame()
    if not hist.empty:
        hist = hist[hist["adp"] < 9_999_000].copy()

    trends = pd.read_csv(trend_file) if trend_file.exists() else pd.DataFrame()
    return cur, hist, trends


# ── Scoring ──────────────────────────────────────────────────────────────────

SCORING_FORMATS = ["PPR", "Half PPR", "Standard", "TE Premium"]

TIER_SCORING_MAP = {
    "PPR":        "ppr",
    "Half PPR":   "half_ppr",
    "Standard":   "standard",
    "TE Premium": "te_premium",
}


def calc_fp(rec_yds, rec_tds, rec, rush_yds, rush_tds,
            pass_yds, pass_tds, ints, position, fmt):
    pts = (
        rec_yds / 10.0  + rec_tds  * 6.0 +
        rush_yds / 10.0 + rush_tds * 6.0 +
        pass_yds / 25.0 + pass_tds * 4.0 - ints * 2.0
    )
    if fmt == "PPR":
        pts += rec
    elif fmt == "Half PPR":
        pts += rec * 0.5
    elif fmt == "TE Premium":
        pts += rec * (1.5 if position == "TE" else 1.0)
    # Standard: no reception bonus
    return max(round(float(pts), 1), 0.0)


# ── Projection Engine ─────────────────────────────────────────────────────────

def _f(val):
    """Safe float conversion with NaN fallback to 0."""
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


def compute_projection(player_id, seasonal, variance, team_stats,
                        games_proj, scoring,
                        ts_slider=None, cpg_slider=None):
    """
    Returns a dict with bear / base / bull scenario stats and FP,
    plus metadata about the player, team context, and slider defaults.

    ts_slider  : float or None — target share override (0.0–1.0 decimal)
    cpg_slider : float or None — carries per game override
    """
    seas = seasonal[seasonal["player_id"] == player_id].sort_values("season")
    if seas.empty:
        return None

    row  = seas.iloc[-1]
    g    = max(_f(row.get("games", 1)), 1.0)
    pos  = row["position"]
    team = row["team"]
    szn  = int(row["season"])

    # Historical per-game stats
    def hpg(col):
        return _f(row.get(col, 0)) / g

    h_tgt   = hpg("targets")
    h_rec   = hpg("receptions")
    h_ryds  = hpg("receiving_yards")
    h_rtds  = hpg("receiving_tds")
    h_car   = hpg("carries")
    h_rushy = hpg("rushing_yards")
    h_rusht = hpg("rushing_tds")
    h_passy = hpg("passing_yards")
    h_passt = hpg("passing_tds")
    h_ints  = hpg("interceptions")
    h_ts    = _f(row.get("target_share", 0))

    # Efficiency rates (kept constant across scenarios)
    catch_r   = h_rec  / h_tgt  if h_tgt  > 0 else 0.0
    yds_p_tgt = h_ryds / h_tgt  if h_tgt  > 0 else 0.0
    td_p_tgt  = h_rtds / h_tgt  if h_tgt  > 0 else 0.0
    yds_p_car = h_rushy / h_car if h_car  > 0 else 0.0
    td_p_car  = h_rusht / h_car if h_car  > 0 else 0.0

    # Team context
    tr = team_stats[(team_stats["team"] == team) & (team_stats["season"] == szn)]
    if tr.empty:
        tr = team_stats[team_stats["team"] == team].sort_values("season").tail(1)
    team_tgt_pg = _f(tr.iloc[0]["avg_targets_game"]) if not tr.empty else 35.0
    team_car_pg = _f(tr.iloc[0]["avg_carries_game"]) if not tr.empty else 27.0

    # Variance lookup helpers
    var = variance[(variance["player_id"] == player_id) & (variance["season"] == szn)]

    def gv(metric, pct, fallback):
        vr = var[var["metric"] == metric]
        if vr.empty:
            return fallback
        val = vr.iloc[0][pct]
        return _f(val) if pd.notna(val) else fallback

    # Median (base) values from variance, or fall back to historical per-game
    tgt_med   = gv("targets",        "median", h_tgt)
    ryds_med  = gv("receiving_yards","median", h_ryds)
    rtds_med  = gv("receiving_tds",  "median", h_rtds)
    car_med   = gv("carries",        "median", h_car)
    rushy_med = gv("rushing_yards",  "median", h_rushy)
    rusht_med = gv("rushing_tds",    "median", h_rusht)
    passy_med = gv("passing_yards",  "median", h_passy)
    passt_med = gv("passing_tds",    "median", h_passt)
    ts_med    = gv("target_share",   "median", h_ts)

    # P25 / P75 ratios relative to median (for bear / bull scaling)
    def ratio(metric, pct, med, default):
        raw = gv(metric, pct, med * default)
        return raw / med if med > 0 else default

    tgt_bear_r   = ratio("targets",        "p25", tgt_med,   0.65)
    tgt_bull_r   = ratio("targets",        "p75", tgt_med,   1.35)
    car_bear_r   = ratio("carries",        "p25", car_med,   0.65)
    car_bull_r   = ratio("carries",        "p75", car_med,   1.35)
    passy_bear_r = ratio("passing_yards",  "p25", passy_med, 0.65)
    passy_bull_r = ratio("passing_yards",  "p75", passy_med, 1.35)
    passt_bear_r = ratio("passing_tds",    "p25", passt_med, 0.55)
    passt_bull_r = ratio("passing_tds",    "p75", passt_med, 1.45)

    # ── Apply slider overrides to base ──────────────────────────────────────
    # Sliders shift the BASE; bear/bull scale proportionally from the new base.
    base_tgt  = tgt_med
    base_car  = car_med
    base_ts   = ts_med

    if ts_slider is not None:
        new_tgt  = ts_slider * team_tgt_pg
        # Scale receiving yards/TDs proportionally with the target change
        if tgt_med > 0:
            tgt_scale = new_tgt / tgt_med
            ryds_med  = ryds_med  * tgt_scale
            rtds_med  = rtds_med  * tgt_scale
        base_tgt = new_tgt
        base_ts  = ts_slider

    if cpg_slider is not None:
        if car_med > 0:
            car_scale = cpg_slider / car_med
            rushy_med = rushy_med * car_scale
            rusht_med = rusht_med * car_scale
        else:
            rushy_med = cpg_slider * yds_p_car
            rusht_med = cpg_slider * td_p_car
        base_car = cpg_slider

    # ── Build per-game stats for each scenario ───────────────────────────────
    scenarios = {
        "bear": {
            "tgt":   base_tgt  * tgt_bear_r,
            "ryds":  ryds_med  * tgt_bear_r,
            "rtds":  rtds_med  * tgt_bear_r,
            "car":   base_car  * car_bear_r,
            "rushy": rushy_med * car_bear_r,
            "rusht": rusht_med * car_bear_r,
            "passy": passy_med * passy_bear_r,
            "passt": passt_med * passt_bear_r,
            "ints":  h_ints * 1.15,   # slightly more picks in bear
        },
        "base": {
            "tgt":   base_tgt,
            "ryds":  ryds_med,
            "rtds":  rtds_med,
            "car":   base_car,
            "rushy": rushy_med,
            "rusht": rusht_med,
            "passy": passy_med,
            "passt": passt_med,
            "ints":  h_ints,
        },
        "bull": {
            "tgt":   base_tgt  * tgt_bull_r,
            "ryds":  ryds_med  * tgt_bull_r,
            "rtds":  rtds_med  * tgt_bull_r,
            "car":   base_car  * car_bull_r,
            "rushy": rushy_med * car_bull_r,
            "rusht": rusht_med * car_bull_r,
            "passy": passy_med * passy_bull_r,
            "passt": passt_med * passt_bull_r,
            "ints":  h_ints * 0.85,   # fewer picks in bull
        },
    }

    # ── Compute season totals and FP ─────────────────────────────────────────
    g2 = float(games_proj)

    built = {}
    for name, s in scenarios.items():
        rec_season   = s["tgt"] * catch_r * g2
        fp = calc_fp(
            s["ryds"] * g2, s["rtds"] * g2, rec_season,
            s["rushy"] * g2, s["rusht"] * g2,
            s["passy"] * g2, s["passt"] * g2,
            s["ints"]  * g2,
            pos, scoring,
        )
        built[name] = dict(
            fp           = fp,
            targets      = round(s["tgt"]   * g2, 1),
            receptions   = round(rec_season,       1),
            rec_yds      = round(s["ryds"]  * g2, 1),
            rec_tds      = round(s["rtds"]  * g2, 1),
            carries      = round(s["car"]   * g2, 1),
            rush_yds     = round(s["rushy"] * g2, 1),
            rush_tds     = round(s["rusht"] * g2, 1),
            pass_yds     = round(s["passy"] * g2, 1),
            pass_tds     = round(s["passt"] * g2, 1),
            ints         = round(s["ints"]  * g2, 1),
            target_share = round(base_ts * 100,    1),
        )

    return dict(
        pos=pos, team=team, season=szn, games_played=int(g),
        games_proj=games_proj,
        bear=built["bear"], base=built["base"], bull=built["bull"],
        h_ts=h_ts, h_cpg=h_car,
        cur_ts=base_ts, cur_cpg=base_car,
        team_tgt_pg=team_tgt_pg, team_car_pg=team_car_pg,
    )


# ── ADP Tab ───────────────────────────────────────────────────────────────────

def _adp_tab(adp_cur, adp_hist, adp_trends):
    if adp_cur.empty:
        st.info("No ADP data yet — run `py 05_adp_pull.py` to generate it.")
        return

    snapshot_date = adp_cur["snapshot_date"].iloc[0] if "snapshot_date" in adp_cur.columns else "unknown"

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        pos_sel = st.multiselect(
            "Position", ["QB", "RB", "WR", "TE"],
            default=["QB", "RB", "WR", "TE"], key="adp_pos",
        )
    with ctrl2:
        sort_by = st.selectbox(
            "Sort by",
            ["ADP", "Model Rank", "Value Score ↑", "Value Score ↓"],
            key="adp_sort",
        )
    with ctrl3:
        search = st.text_input("Search player", key="adp_search", placeholder="e.g. Mahomes")

    # ── Build display dataframe ────────────────────────────────────────────────
    df = adp_cur.copy()
    if pos_sel:
        df = df[df["position"].isin(pos_sel)]
    if search:
        df = df[df["player_name"].str.contains(search, case=False, na=False)]

    # Merge trend arrows if available
    if not adp_trends.empty and "trend" in adp_trends.columns:
        df = df.merge(
            adp_trends[["player_name", "adp_change", "trend"]],
            on="player_name", how="left",
        )
        df["trend"] = df["trend"].fillna("")
    else:
        df["trend"] = ""

    df["Player"] = df.apply(
        lambda r: f"{r['player_name']} {r['trend']}".strip(), axis=1
    )

    sort_map = {
        "ADP":             ("adp", True),
        "Model Rank":      ("model_overall_rank", True),
        "Value Score ↑":   ("value_score", False),
        "Value Score ↓":   ("value_score", True),
    }
    scol, sasc = sort_map.get(sort_by, ("adp", True))
    if scol in df.columns:
        df = df.sort_values(scol, ascending=sasc, na_position="last")

    # Columns to display
    col_map = {
        "Player":             "Player",
        "position":           "Pos",
        "team":               "Team",
        "adp":                "ADP",
        "model_overall_rank": "Model Rank",
        "model_pos_rank_str": "Pos Rank",
        "fpts_ppr":           "Proj PPR",
        "value_score":        "Value Score",
    }
    avail = [c for c in col_map if c in df.columns or c == "Player"]
    table = df[[c for c in avail if c in df.columns]].rename(columns=col_map)

    def _vs_color(val):
        if pd.isna(val):
            return ""
        if val > 20:
            return "background-color:#b7f5b0;color:#0f4d0a"
        if val > 5:
            return "background-color:#d9f7d6;color:#1e6b18"
        if val < -20:
            return "background-color:#f5b0b0;color:#6b0f0f"
        if val < -5:
            return "background-color:#f7d6d6;color:#8b1c1c"
        return ""

    fmt = {}
    if "ADP"         in table.columns: fmt["ADP"]         = lambda v: f"{int(v)}" if pd.notna(v) else "—"
    if "Model Rank"  in table.columns: fmt["Model Rank"]  = lambda v: f"{int(v)}" if pd.notna(v) else "—"
    if "Proj PPR"    in table.columns: fmt["Proj PPR"]    = lambda v: f"{v:.1f}"  if pd.notna(v) else "—"
    if "Value Score" in table.columns: fmt["Value Score"] = lambda v: f"{v:+.0f}" if pd.notna(v) else "—"

    styler = table.style.format(fmt, na_rep="—")
    if "Value Score" in table.columns:
        styler = styler.applymap(_vs_color, subset=["Value Score"])

    st.subheader(f"Current Rankings — {len(df)} players shown")
    st.dataframe(styler, use_container_width=True, hide_index=True, height=440)
    st.caption(
        f"ADP = Sleeper `search_rank` (lower is better)  ·  "
        f"Value Score = ADP − Model Rank (positive = undervalued by market)  ·  "
        f"Snapshot: {snapshot_date}"
    )

    # ── Market Inefficiencies ──────────────────────────────────────────────────
    if "value_score" in df.columns and "model_overall_rank" in df.columns:
        st.divider()
        st.subheader("Market Inefficiencies")
        ranked = df[df["model_overall_rank"].notna() & df["value_score"].notna()].copy()

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**🟢 Best Values** — model ranks higher than market")
            vals = (
                ranked[ranked["value_score"] > 0]
                .sort_values("value_score", ascending=False)
                .head(8)[["Player", "position", "adp", "model_overall_rank", "value_score"]]
            )
            vals.columns = ["Player", "Pos", "ADP", "Model", "Value"]
            st.dataframe(
                vals.style
                .applymap(lambda _: "background-color:#d9f7d6", subset=["Value"])
                .format({"ADP": "{:.0f}", "Model": "{:.0f}", "Value": "+{:.0f}"}),
                use_container_width=True, hide_index=True,
            )

        with mc2:
            st.markdown("**🔴 Overvalued** — market ranks higher than model")
            over = (
                ranked[ranked["value_score"] < 0]
                .sort_values("value_score")
                .head(8)[["Player", "position", "adp", "model_overall_rank", "value_score"]]
            )
            over.columns = ["Player", "Pos", "ADP", "Model", "Value"]
            st.dataframe(
                over.style
                .applymap(lambda _: "background-color:#f7d6d6", subset=["Value"])
                .format({"ADP": "{:.0f}", "Model": "{:.0f}", "Value": "{:.0f}"}),
                use_container_width=True, hide_index=True,
            )

    # ── ADP Trend Sparkline ────────────────────────────────────────────────────
    st.divider()
    st.subheader("ADP Trend")
    if adp_hist.empty or adp_hist["snapshot_date"].nunique() < 2:
        st.info(
            "ADP trend chart appears once you have 2+ weekly snapshots. "
            "Run `py 05_adp_pull.py` again next week to start tracking trends."
        )
    else:
        player_opts = sorted(adp_hist["player_name"].dropna().unique().tolist())
        pick = st.selectbox("Select player", player_opts, key="adp_spark")
        pdata = adp_hist[adp_hist["player_name"] == pick].sort_values("snapshot_date")
        if len(pdata) >= 2:
            spark = (
                alt.Chart(pdata)
                .mark_line(point=True, strokeWidth=2, color="#2980b9")
                .encode(
                    x=alt.X("snapshot_date:O", title="Date",
                            axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("adp:Q", title="ADP rank (lower = better)",
                            scale=alt.Scale(reverse=True)),
                    tooltip=[
                        alt.Tooltip("snapshot_date:O", title="Date"),
                        alt.Tooltip("adp:Q", title="ADP rank", format=".0f"),
                    ],
                )
                .properties(height=220, title=f"{pick} — ADP over time")
            )
            st.altair_chart(spark, use_container_width=True)
        else:
            st.info("Not enough data points for this player yet.")


# ── Projection Tab ─────────────────────────────────────────────────────────────

def _projection_tab(selected, idx, seasonal, variance, team_stats, proj_games, scoring):
    # ── Resolve player ────────────────────────────────────────────────────────
    prow = idx[idx["player_name"] == selected]
    if prow.empty:
        st.error("Player not found in data.")
        return
    pid = prow.iloc[0]["player_id"]

    init = compute_projection(pid, seasonal, variance, team_stats, proj_games, scoring)
    if init is None:
        st.error("No data available for this player.")
        return

    pos  = init["pos"]
    team = init["team"]
    szn  = init["season"]

    # ── Header ───────────────────────────────────────────────────────────────
    POS_COLOR = {"QB": "#e74c3c", "RB": "#27ae60", "WR": "#2980b9", "TE": "#d68910"}
    color = POS_COLOR.get(pos, "#7f8c8d")

    col_hdr, _ = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            f"## {selected} "
            f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:5px;font-size:0.65em;vertical-align:middle">{pos}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"**{team}**  ·  Base season: **{szn}**  ·  Played **{init['games_played']}** games  ·  {scoring}")
    st.divider()

    # ── Adjustment Sliders ────────────────────────────────────────────────────
    show_ts  = pos in ("WR", "TE") or (pos == "RB" and init["h_ts"] > 0.02)
    show_cpg = pos in ("RB", "QB")

    ts_slider  = None
    cpg_slider = None

    if show_ts or show_cpg:
        st.subheader("Adjust Usage")
        ncols = sum([show_ts, show_cpg])
        slider_cols = st.columns(ncols + 1)

        col_i = 0
        if show_ts:
            default_ts = round(init["h_ts"] * 100, 1)
            max_ts     = float(min(max(default_ts + 15.0, 30.0), 50.0))
            with slider_cols[col_i]:
                ts_pct = st.slider(
                    "Target Share (%)", min_value=0.0, max_value=max_ts,
                    value=default_ts, step=0.5,
                    help="% of team targets going to this player each game",
                )
            ts_slider = ts_pct / 100.0
            col_i += 1

        if show_cpg:
            default_cpg = round(init["h_cpg"], 1)
            max_cpg     = float(min(max(default_cpg + 10.0, 20.0), 35.0))
            with slider_cols[col_i]:
                cpg_val = st.slider(
                    "Carries per Game", min_value=0.0, max_value=max_cpg,
                    value=default_cpg, step=0.5,
                    help="Rushing attempts per game",
                )
            cpg_slider = cpg_val
            col_i += 1

        with slider_cols[col_i]:
            if show_ts and ts_slider is not None:
                implied_tgt = round(ts_slider * init["team_tgt_pg"], 1)
                st.info(
                    f"**{ts_pct:.1f}%** of team's "
                    f"{init['team_tgt_pg']:.1f} targets/game\n\n"
                    f"≈ **{implied_tgt} targets/game**"
                )
            if show_cpg and cpg_slider is not None:
                st.info(
                    f"**{cpg_val:.1f}** carries/game\n\n"
                    f"vs team avg **{init['team_car_pg']:.1f}** total carries/game"
                )

    result = compute_projection(
        pid, seasonal, variance, team_stats,
        proj_games, scoring, ts_slider, cpg_slider,
    )

    # ── Bear / Base / Bull ────────────────────────────────────────────────────
    st.subheader(f"Season Projections — {proj_games} games · {scoring}")

    c1, c2, c3 = st.columns(3)
    bear_fp = result["bear"]["fp"]
    base_fp = result["base"]["fp"]
    bull_fp = result["bull"]["fp"]

    c1.metric(
        "🐻  Bear", f"{bear_fp:.1f} pts",
        delta=f"{bear_fp - init['bear']['fp']:+.1f}" if (ts_slider or cpg_slider) else None,
    )
    c2.metric(
        "📊  Base", f"{base_fp:.1f} pts",
        delta=f"{base_fp - init['base']['fp']:+.1f}" if (ts_slider or cpg_slider) else None,
    )
    c3.metric(
        "🐂  Bull", f"{bull_fp:.1f} pts",
        delta=f"{bull_fp - init['bull']['fp']:+.1f}" if (ts_slider or cpg_slider) else None,
    )

    chart_data = pd.DataFrame({
        "Scenario":       ["Bear", "Base", "Bull"],
        "Fantasy Points": [bear_fp, base_fp, bull_fp],
        "Color":          ["#e74c3c", "#2980b9", "#27ae60"],
    })
    bar = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=60)
        .encode(
            x=alt.X("Scenario", sort=["Bear", "Base", "Bull"],
                    axis=alt.Axis(labelFontSize=14, labelFontWeight="bold")),
            y=alt.Y("Fantasy Points",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(title="Fantasy Points")),
            color=alt.Color("Color", scale=None, legend=None),
            tooltip=[alt.Tooltip("Scenario"), alt.Tooltip("Fantasy Points", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(bar, use_container_width=True)

    # ── Projected Stat Line ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Projected Stat Line")

    bear_s = result["bear"]
    base_s = result["base"]
    bull_s = result["bull"]

    def stat_row(label, key, fmt="{:.1f}"):
        b, m, u = bear_s[key], base_s[key], bull_s[key]
        return {"Stat": label, "🐻 Bear": fmt.format(b), "📊 Base": fmt.format(m), "🐂 Bull": fmt.format(u)}

    def stat_row_str(label, key):
        b, m, u = bear_s[key], base_s[key], bull_s[key]
        return {"Stat": label, "🐻 Bear": f"{b}%", "📊 Base": f"{m}%", "🐂 Bull": f"{u}%"}

    if pos == "QB":
        rows = [
            stat_row("Pass Yards", "pass_yds", "{:.0f}"),
            stat_row("Pass TDs",   "pass_tds", "{:.1f}"),
            stat_row("INTs",       "ints",     "{:.1f}"),
            stat_row("Rush Yards", "rush_yds", "{:.0f}"),
            stat_row("Rush TDs",   "rush_tds", "{:.1f}"),
        ]
    elif pos == "RB":
        rows = [
            stat_row("Carries",    "carries",    "{:.1f}"),
            stat_row("Rush Yards", "rush_yds",   "{:.0f}"),
            stat_row("Rush TDs",   "rush_tds",   "{:.1f}"),
            stat_row("Targets",    "targets",    "{:.1f}"),
            stat_row("Receptions", "receptions", "{:.1f}"),
            stat_row("Rec Yards",  "rec_yds",    "{:.0f}"),
            stat_row("Rec TDs",    "rec_tds",    "{:.1f}"),
        ]
        if init["h_ts"] > 0.02:
            rows.append(stat_row_str("Target Share", "target_share"))
    else:  # WR / TE
        rows = [
            stat_row("Targets",    "targets",    "{:.1f}"),
            stat_row("Receptions", "receptions", "{:.1f}"),
            stat_row("Rec Yards",  "rec_yds",    "{:.0f}"),
            stat_row("Rec TDs",    "rec_tds",    "{:.1f}"),
            stat_row_str("Target Share", "target_share"),
        ]
        if base_s["rush_yds"] > 5:
            rows += [
                stat_row("Rush Yards", "rush_yds", "{:.0f}"),
                stat_row("Rush TDs",   "rush_tds", "{:.1f}"),
            ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Per-Game Averages"):
        pg_rows = []
        for r in rows:
            label = r["Stat"]
            if label == "Target Share":
                pg_rows.append(r)
                continue
            def _pg(val_str, label=label):
                try:
                    return f"{float(val_str) / proj_games:.2f}"
                except (ValueError, TypeError):
                    return val_str
            pg_rows.append({
                "Stat":    label,
                "🐻 Bear": _pg(r["🐻 Bear"]),
                "📊 Base": _pg(r["📊 Base"]),
                "🐂 Bull": _pg(r["🐂 Bull"]),
            })
        st.dataframe(pd.DataFrame(pg_rows), use_container_width=True, hide_index=True)

    with st.expander("Historical Context & Team"):
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Team Targets/Game",  f"{init['team_tgt_pg']:.1f}")
        h2.metric("Team Carries/Game",  f"{init['team_car_pg']:.1f}")
        if init["h_ts"] > 0:
            h3.metric("Historical Target Share", f"{init['h_ts']*100:.1f}%")
        if init["h_cpg"] > 0:
            h4.metric("Historical Carries/Game", f"{init['h_cpg']:.1f}")

        player_seasons = seasonal[
            (seasonal["player_id"] == pid) & (seasonal["season"] >= 2020)
        ].copy()
        if len(player_seasons) > 1:
            st.caption("Season-by-season fantasy points (PPR)")
            spark = (
                alt.Chart(player_seasons)
                .mark_line(point=True)
                .encode(
                    x=alt.X("season:O", title="Season"),
                    y=alt.Y("fantasy_points_ppr:Q", title="FP (PPR)"),
                    tooltip=["season", "team",
                             alt.Tooltip("fantasy_points_ppr:Q", format=".1f", title="FP PPR")],
                )
                .properties(height=160)
            )
            st.altair_chart(spark, use_container_width=True)


# ── Tier History Tab ──────────────────────────────────────────────────────────

def _render_tier_player(pname, pos_data, pos, fmt_label, league_size, view_mode, seasons_to_show):
    """Render stacked bar chart + boom metric cards for one player."""
    player_data = pos_data[
        (pos_data["player_name"] == pname) &
        (pos_data["season"].isin(seasons_to_show))
    ].sort_values("season")

    if player_data.empty:
        st.warning(f"No data for **{pname}** in the selected seasons.")
        return

    t1_col = f"{pos}1_weeks"
    t2_col = f"{pos}2_weeks"
    t3_col = f"{pos}3_weeks"
    t4_col = f"{pos}4_weeks"

    # ── Build chart data ─────────────────────────────────────────────────────
    chart_rows = []
    for _, r in player_data.iterrows():
        szn = str(int(r["season"]))
        chart_rows += [
            {"Season": szn, "Tier": f"{pos}1",  "Weeks": int(r.get(t1_col, 0)), "ord": 1},
            {"Season": szn, "Tier": f"{pos}2",  "Weeks": int(r.get(t2_col, 0)), "ord": 2},
            {"Season": szn, "Tier": f"{pos}3",  "Weeks": int(r.get(t3_col, 0)), "ord": 3},
            {"Season": szn, "Tier": f"{pos}4+", "Weeks": int(r.get(t4_col, 0)), "ord": 4},
        ]
    chart_df = pd.DataFrame(chart_rows)

    # ── Aggregate boom metrics ────────────────────────────────────────────────
    total_games  = int(player_data["games"].sum())
    t1_total     = int(player_data[t1_col].sum()) if t1_col in player_data.columns else 0
    top25_total  = int(player_data["boom_top25_weeks"].sum())
    pts20_total  = int(player_data["boom_20pts_weeks"].sum())

    tier1_pct = round(t1_total   / total_games * 100, 1) if total_games else 0.0
    top25_pct = round(top25_total / total_games * 100, 1) if total_games else 0.0
    pts20_pct = round(pts20_total / total_games * 100, 1) if total_games else 0.0

    # ── Boom metric cards ────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        f"{pos}1%", f"{tier1_pct:.1f}%",
        help=f"Weeks finishing as a {pos}1 ÷ games played",
    )
    mc2.metric(
        "Top 25%", f"{top25_pct:.1f}%",
        help="Weeks finishing in the top 25% of position ÷ games played",
    )
    mc3.metric(
        "20 pts%", f"{pts20_pct:.1f}%",
        help="Weeks scoring 20+ fantasy points ÷ games played",
    )

    # ── Stacked bar chart ────────────────────────────────────────────────────
    tier_domain = [f"{pos}1", f"{pos}2", f"{pos}3", f"{pos}4+"]
    tier_range  = ["#27ae60", "#2980b9", "#f39c12", "#c0392b"]

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Season:O", title="Season"),
            y=alt.Y("Weeks:Q", title="Weeks", stack="zero"),
            color=alt.Color(
                "Tier:N",
                scale=alt.Scale(domain=tier_domain, range=tier_range),
                legend=alt.Legend(title="Tier", orient="right"),
            ),
            order=alt.Order("ord:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("Season:O"),
                alt.Tooltip("Tier:N"),
                alt.Tooltip("Weeks:Q"),
            ],
        )
        .properties(
            height=280,
            title=f"{pname} — {fmt_label} · {league_size}-team",
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Caption
    if view_mode == "Multi-Year":
        season_str = ", ".join(str(int(s)) for s in sorted(seasons_to_show))
        st.caption(f"Seasons: {season_str}  ·  Total games: {total_games}")
    else:
        r = player_data.iloc[0]
        avg  = float(r["avg_fpts"])
        bust = float(r["bust_pct"])
        st.caption(
            f"Season: {int(r['season'])}  ·  Games: {total_games}  ·  "
            f"Avg: {avg:.1f} pts  ·  Bust (<5 pts): {bust:.1f}%"
        )


def _tier_tab(tiers_df):
    if tiers_df.empty:
        st.info("No tier data — run `py 06_tier_consistency.py` to generate it.")
        return

    # ── Row 1: position / scoring / league size / view mode ──────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        pos_sel = st.selectbox("Position", ["QB", "RB", "WR", "TE"], index=2, key="tier_pos")
    with c2:
        fmt_sel = st.selectbox("Scoring Format", list(TIER_SCORING_MAP.keys()), key="tier_fmt")
    with c3:
        league_sel = st.selectbox("League Size", [10, 12, 14, 16], index=1, key="tier_league")
    with c4:
        view_mode = st.radio(
            "View Mode", ["Single Season", "Multi-Year"],
            horizontal=True, key="tier_view",
        )

    fmt_key  = TIER_SCORING_MAP[fmt_sel]
    pos_data = tiers_df[
        (tiers_df["position"] == pos_sel) &
        (tiers_df["scoring_format"] == fmt_key) &
        (tiers_df["league_size"] == league_sel)
    ]
    seasons_available = sorted(pos_data["season"].unique().tolist())

    # ── Row 2: player / season or year-range / compare toggle ────────────────
    r2c1, r2c2, r2c3 = st.columns([3, 2, 2])

    with r2c2:
        if view_mode == "Single Season":
            default_idx = len(seasons_available) - 1  # 2024 by default
            season_sel      = st.selectbox(
                "Season", seasons_available, index=default_idx, key="tier_season",
            )
            seasons_to_show = [season_sel]
        else:
            year_range = st.radio(
                "Year Range",
                ["Last 2 Years", "Last 3 Years", "All 4 Years"],
                horizontal=True, key="tier_years",
            )
            n_years         = {"Last 2 Years": 2, "Last 3 Years": 3, "All 4 Years": 4}[year_range]
            seasons_to_show = sorted(seasons_available)[-n_years:]

    # Build player list from the selected seasons
    players_in_range = pos_data[pos_data["season"].isin(seasons_to_show)]["player_name"].unique()
    player_opts      = sorted(players_in_range.tolist())

    if not player_opts:
        st.warning("No players found for this combination.")
        return

    with r2c1:
        player_sel = st.selectbox("Player", player_opts, key="tier_player")

    with r2c3:
        compare_on  = st.toggle("Compare Players", key="tier_compare")
        player2_sel = None
        if compare_on:
            other_opts = [p for p in player_opts if p != player_sel]
            if other_opts:
                player2_sel = st.selectbox("Compare with", other_opts, key="tier_player2")

    st.divider()

    # ── Render ────────────────────────────────────────────────────────────────
    if compare_on and player2_sel:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### {player_sel}")
            _render_tier_player(
                player_sel, pos_data, pos_sel, fmt_sel, league_sel, view_mode, seasons_to_show,
            )
        with col2:
            st.markdown(f"#### {player2_sel}")
            _render_tier_player(
                player2_sel, pos_data, pos_sel, fmt_sel, league_sel, view_mode, seasons_to_show,
            )
    else:
        _render_tier_player(
            player_sel, pos_data, pos_sel, fmt_sel, league_sel, view_mode, seasons_to_show,
        )


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    seasonal, variance, team_stats = load_data()
    adp_cur, adp_hist, adp_trends  = load_adp_data()
    tiers_df                       = load_tier_data()

    SKILL = {"QB", "RB", "WR", "TE"}
    latest_season = int(seasonal["season"].max())
    idx = (
        seasonal[
            seasonal["position"].isin(SKILL) &
            (seasonal["season"] == latest_season)
        ]
        [["player_id", "player_name", "position", "team", "season"]]
        .drop_duplicates("player_id")
        .sort_values("player_name")
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏈 FF Projections")
        st.divider()

        pos_filter = st.multiselect(
            "Position", ["QB", "RB", "WR", "TE"],
            default=["QB", "RB", "WR", "TE"],
        )

        all_teams = ["All Teams"] + sorted(idx["team"].dropna().unique().tolist())
        team_filter = st.selectbox("Team", all_teams)

        filtered_idx = idx[idx["position"].isin(pos_filter)] if pos_filter else idx
        if team_filter != "All Teams":
            filtered_idx = filtered_idx[filtered_idx["team"] == team_filter]
        name_opts = filtered_idx["player_name"].tolist()

        selected = st.selectbox("Player", name_opts if name_opts else ["(no players)"])

        st.divider()
        scoring = st.radio("Scoring Format", SCORING_FORMATS)

        st.divider()
        proj_games = st.slider("Projected Games", 1, 17, 17)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_proj, tab_adp, tab_tier = st.tabs(
        ["📊 Projections", "📈 ADP & Market", "🏆 Tier History"]
    )

    with tab_proj:
        _projection_tab(selected, idx, seasonal, variance, team_stats, proj_games, scoring)

    with tab_adp:
        _adp_tab(adp_cur, adp_hist, adp_trends)

    with tab_tier:
        _tier_tab(tiers_df)


main()
