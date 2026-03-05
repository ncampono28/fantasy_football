import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="FF Projection Model",
    page_icon="🏈",
    layout="wide",
)

# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=0)
def load_data():
    seasonal   = pd.read_csv("data/player_seasonal_stats.csv")
    team_stats = pd.read_csv("data/team_season_stats.csv")
    return seasonal, team_stats


@st.cache_data(ttl=0)
def load_projections():
    from pathlib import Path
    f = Path("data/projections_weighted.csv")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_data(ttl=0)
def load_tier_data():
    from pathlib import Path
    f = Path("data/tier_consistency.csv")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_data(ttl=0)
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

SCORING_COL = {
    "PPR":        "fpts_ppr",
    "Half PPR":   "fpts_half_ppr",
    "Standard":   "fpts_standard",
    "TE Premium": "fpts_te_premium",
}

TIER_SCORING_MAP = {
    "PPR":        "ppr",
    "Half PPR":   "half_ppr",
    "Standard":   "standard",
    "TE Premium": "te_premium",
}


def _f(val):
    """Safe float conversion with NaN fallback to 0."""
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


def _switch_to_projections_tab():
    """Inject JS to auto-click the Projections tab (works via window.parent DOM)."""
    components.html(
        """<script>
        setTimeout(function() {
            var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            for (var i = 0; i < tabs.length; i++) {
                if (tabs[i].innerText.indexOf("Projections") !== -1) {
                    tabs[i].click();
                    break;
                }
            }
        }, 300);
        </script>""",
        height=0,
        width=0,
    )


# ── ADP Tab ───────────────────────────────────────────────────────────────────

def _adp_tab(adp_cur, adp_hist, adp_trends, tiers_df):
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
            ["Best Ball ADP (Underdog)", "Model Rank", "Value Score ↑", "Value Score ↓"],
            key="adp_sort",
        )
    with ctrl3:
        search = st.text_input("Search player", key="adp_search", placeholder="e.g. Mahomes")

    # ── Build display dataframe ────────────────────────────────────────────────
    # Compute positional ranks on the FULL dataset before any filtering so
    # ranks are always relative to the whole position pool.
    full_df = adp_cur.copy()

    full_df["adp_pos_rank"] = (
        full_df.groupby("position")["adp"]
        .rank(ascending=True, method="min")
        .astype(int)
    )

    if "model_pos_rank_str" in full_df.columns:
        full_df["model_pos_rank_num"] = (
            full_df["model_pos_rank_str"].str.extract(r"(\d+)")[0].astype(float)
        )
    else:
        full_df["model_pos_rank_num"] = np.nan

    # Positional value score: positive = model likes them more than market does
    full_df["pos_value_score"] = (
        full_df["adp_pos_rank"] - full_df["model_pos_rank_num"]
    ).round(0)

    # Pos Rank display: "WR8 · ADP WR15"
    def _pos_rank_display(row):
        pos  = row["position"]
        mstr = row.get("model_pos_rank_str", "")
        apr  = row.get("adp_pos_rank")
        if not mstr or pd.isna(mstr) or pd.isna(apr):
            return mstr if (mstr and not pd.isna(mstr)) else "—"
        return f"{mstr} · ADP {pos}{int(apr)}"

    full_df["pos_rank_display"] = full_df.apply(_pos_rank_display, axis=1)

    # Filter for display
    df = full_df.copy()
    if pos_sel:
        df = df[df["position"].isin(pos_sel)]
    if search:
        df = df[df["player_name"].str.contains(search, case=False, na=False)]

    df["Player"] = df["player_name"]

    # ── Merge tier consistency columns (12-team PPR, most recent season) ──────
    if not tiers_df.empty:
        tier_ppr12 = (
            tiers_df[
                (tiers_df["scoring_format"] == "ppr") &
                (tiers_df["league_size"] == 12)
            ]
            .sort_values("season")
            .groupby("player_name", as_index=False)
            .last()
        )
        df = df.merge(
            tier_ppr12[["player_name", "boom_tier1_pct", "boom_top25_pct", "boom_20pts_pct"]],
            on="player_name", how="left",
        )
        df.rename(columns={
            "boom_tier1_pct": "Tier1%",
            "boom_top25_pct": "Top25%",
            "boom_20pts_pct": "20pts%",
        }, inplace=True)

    sort_map = {
        "Best Ball ADP (Underdog)": ("adp", True),
        "Model Rank":               ("model_overall_rank", True),
        "Value Score ↑":            ("pos_value_score", False),
        "Value Score ↓":            ("pos_value_score", True),
    }
    scol, sasc = sort_map.get(sort_by, ("adp", True))
    if scol in df.columns:
        df = df.sort_values(scol, ascending=sasc, na_position="last")

    # Columns to display
    col_map = {
        "Player":            "Player",
        "position":          "Pos",
        "team":              "Team",
        "adp":               "Best Ball ADP (Underdog)",
        "model_overall_rank":"Model Rank",
        "pos_rank_display":  "Pos Rank",
        "fpts_ppr":          "Proj PPR",
        "pos_value_score":   "Value Score",
        "Tier1%":            "Tier1%",
        "Top25%":            "Top25%",
        "20pts%":            "20pts%",
    }
    avail = [c for c in col_map if c in df.columns or c == "Player"]
    table = df[[c for c in avail if c in df.columns]].rename(columns=col_map)

    def _vs_color(val):
        if pd.isna(val):
            return ""
        a = min(abs(val) / 30, 1.0)
        opacity = round(0.06 + a * 0.14, 3)
        if val > 0:
            return f"background-color:rgba(0,180,80,{opacity})"
        if val < 0:
            return f"background-color:rgba(200,50,50,{opacity})"
        return ""

    def _tier_pct_color(val):
        if pd.isna(val):
            return ""
        if val >= 60:
            return "background-color:rgba(0,180,80,0.15)"
        if val >= 40:
            return "background-color:rgba(0,180,80,0.08)"
        if val < 10:
            return "background-color:rgba(200,50,50,0.10)"
        return ""

    # Reset index so row indices are 0-based for reliable click detection
    table = table.reset_index(drop=True)

    # Normalize Python None in object columns so they render as "—" not "None"
    for _col in table.select_dtypes(include="object").columns:
        table[_col] = table[_col].where(table[_col].notna(), other=None)
        table[_col] = table[_col].replace({None: np.nan, "None": np.nan, "nan": np.nan})

    fmt = {}
    if "Best Ball ADP (Underdog)" in table.columns: fmt["Best Ball ADP (Underdog)"] = lambda v: f"{v:.1f}" if pd.notna(v) else "—"
    if "Model Rank"  in table.columns: fmt["Model Rank"]  = lambda v: f"{int(v)}" if pd.notna(v) else "—"
    if "Pos Rank"    in table.columns: fmt["Pos Rank"]    = lambda v: str(v) if pd.notna(v) else "—"
    if "Proj PPR"    in table.columns: fmt["Proj PPR"]    = lambda v: f"{v:.1f}"  if pd.notna(v) else "—"
    if "Value Score" in table.columns: fmt["Value Score"] = lambda v: f"{v:+.0f}" if pd.notna(v) else "—"
    if "Tier1%"      in table.columns: fmt["Tier1%"]      = lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
    if "Top25%"      in table.columns: fmt["Top25%"]      = lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
    if "20pts%"      in table.columns: fmt["20pts%"]      = lambda v: f"{v:.1f}%" if pd.notna(v) else "—"

    styler = table.style.format(fmt, na_rep="—")
    if "Value Score" in table.columns:
        styler = styler.applymap(_vs_color, subset=["Value Score"])
    for tier_col in ["Tier1%", "Top25%", "20pts%"]:
        if tier_col in table.columns:
            styler = styler.applymap(_tier_pct_color, subset=[tier_col])

    st.subheader(f"Current Rankings — {len(df)} players shown")
    st.caption("Click any player row to jump to their projection →", help="Selects the player in the sidebar and switches to the Projections tab.")
    event = st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=440,
        on_select="rerun",
        selection_mode="single-row",
    )
    # Handle row click → navigate to Projections tab
    if event.selection.rows:
        row_idx = event.selection.rows[0]
        clicked_player = table.iloc[row_idx]["Player"]
        st.session_state["adp_navigate_player"] = clicked_player
        st.rerun()

    st.caption(
        f"Best Ball ADP = Underdog March 4 draft position (lower is better)  ·  "
        f"Value Score = positional ADP rank − model positional rank (positive = undervalued at position)  ·  "
        f"Tier metrics = 12-team PPR, most recent season  ·  "
        f"Snapshot: {snapshot_date}"
    )

    # ── Market Inefficiencies ──────────────────────────────────────────────────
    if "pos_value_score" in full_df.columns and "model_pos_rank_num" in full_df.columns:
        st.divider()
        st.subheader("Market Inefficiencies")

        ranked = full_df[
            full_df["model_pos_rank_num"].notna() &
            full_df["pos_value_score"].notna() &
            (full_df["adp"] <= 300)
        ].copy()
        ranked["Player"] = ranked["player_name"]

        def _pos_label(row):
            pos  = row["position"]
            mnum = row.get("model_pos_rank_num")
            apr  = row.get("adp_pos_rank")
            pvs  = row.get("pos_value_score")
            if pd.isna(mnum) or pd.isna(apr) or pd.isna(pvs):
                return "—"
            sign = f"+{int(pvs)}" if pvs > 0 else str(int(pvs))
            return f"Model: {pos}{int(mnum)}, ADP: {pos}{int(apr)} ({sign})"

        ranked["Positional Gap"] = ranked.apply(_pos_label, axis=1)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**🟢 Best Values** — model ranks higher than market")
            vals = (
                ranked[ranked["pos_value_score"] > 0]
                .sort_values("pos_value_score", ascending=False)
                .head(8)[["Player", "position", "adp", "Positional Gap", "pos_value_score"]]
            )
            vals.columns = ["Player", "Pos", "UD ADP", "Positional Gap", "Value"]
            st.dataframe(
                vals.style
                .applymap(lambda v: _vs_color(v), subset=["Value"])
                .format({"UD ADP": "{:.1f}", "Value": "+{:.0f}"}),
                use_container_width=True, hide_index=True,
            )

        with mc2:
            st.markdown("**🔴 Overvalued** — market ranks higher than model")
            over = (
                ranked[ranked["pos_value_score"] < 0]
                .sort_values("pos_value_score")
                .head(8)[["Player", "position", "adp", "Positional Gap", "pos_value_score"]]
            )
            over.columns = ["Player", "Pos", "UD ADP", "Positional Gap", "Value"]
            st.dataframe(
                over.style
                .applymap(lambda v: _vs_color(v), subset=["Value"])
                .format({"UD ADP": "{:.1f}", "Value": "{:.0f}"}),
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

def _projection_tier_section(pid, pos, tiers_df):
    """Historical tier consistency section appended to the projection card."""
    if tiers_df.empty:
        st.info("Run `py 06_tier_consistency.py` to generate tier data.")
        return

    st.divider()
    st.subheader("Historical Tier Consistency")

    # Controls
    tc1, tc2, tc3 = st.columns([2, 2, 3])
    with tc1:
        tier_fmt = st.selectbox(
            "Scoring", list(TIER_SCORING_MAP.keys()),
            index=0, key="proj_tier_fmt",
        )
    with tc2:
        tier_league = st.selectbox(
            "League Size", [10, 12, 14, 16],
            index=1, key="proj_tier_league",
        )
    with tc3:
        stat_range = st.radio(
            "Stat cards show",
            ["Most Recent", "Last 2 Yrs", "Last 3 Yrs"],
            horizontal=True, key="proj_tier_range",
        )

    fmt_key      = TIER_SCORING_MAP[tier_fmt]
    player_tiers = (
        tiers_df[
            (tiers_df["player_id"] == pid) &
            (tiers_df["scoring_format"] == fmt_key) &
            (tiers_df["league_size"] == tier_league)
        ]
        .sort_values("season")
        .copy()
    )

    if player_tiers.empty:
        st.info("No tier history available for this player.")
        return

    t1_col = f"{pos}1_weeks"
    t2_col = f"{pos}2_weeks"
    t3_col = f"{pos}3_weeks"
    t4_col = f"{pos}4_weeks"

    # ── Stacked bar chart — all available seasons ────────────────────────────
    chart_rows = []
    for _, r in player_tiers.iterrows():
        szn = str(int(r["season"]))
        chart_rows += [
            {"Season": szn, "Tier": f"{pos}1",  "Weeks": int(r.get(t1_col, 0)), "ord": 1},
            {"Season": szn, "Tier": f"{pos}2",  "Weeks": int(r.get(t2_col, 0)), "ord": 2},
            {"Season": szn, "Tier": f"{pos}3",  "Weeks": int(r.get(t3_col, 0)), "ord": 3},
            {"Season": szn, "Tier": f"{pos}4+", "Weeks": int(r.get(t4_col, 0)), "ord": 4},
        ]
    chart_df = pd.DataFrame(chart_rows)

    tier_domain = [f"{pos}1", f"{pos}2", f"{pos}3", f"{pos}4+"]
    tier_colors = ["#27ae60", "#2980b9", "#f39c12", "#c0392b"]

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Season:O", title="Season"),
            y=alt.Y("Weeks:Q", title="Weeks", stack="zero"),
            color=alt.Color(
                "Tier:N",
                scale=alt.Scale(domain=tier_domain, range=tier_colors),
                legend=alt.Legend(title="Tier", orient="right"),
            ),
            order=alt.Order("ord:Q", sort="ascending"),
            tooltip=["Season:O", "Tier:N", "Weeks:Q"],
        )
        .properties(
            height=220,
            title=f"Tier weeks per season — {tier_fmt} · {tier_league}-team",
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # ── Boom stat cards — controlled by the toggle ───────────────────────────
    n_years   = {"Most Recent": 1, "Last 2 Yrs": 2, "Last 3 Yrs": 3}[stat_range]
    stat_data = player_tiers.tail(n_years)

    total_games  = int(stat_data["games"].sum())
    t1_total     = int(stat_data[t1_col].sum()) if t1_col in stat_data.columns else 0
    top25_total  = int(stat_data["boom_top25_weeks"].sum())
    pts20_total  = int(stat_data["boom_20pts_weeks"].sum())

    tier1_pct = round(t1_total    / total_games * 100, 1) if total_games else 0.0
    top25_pct = round(top25_total / total_games * 100, 1) if total_games else 0.0
    pts20_pct = round(pts20_total / total_games * 100, 1) if total_games else 0.0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric(f"{pos}1%",  f"{tier1_pct:.1f}%", help=f"{pos}1 weeks ÷ games played")
    sc2.metric("Top 25%",   f"{top25_pct:.1f}%", help="Weeks finishing top 25% of position ÷ games played")
    sc3.metric("20 pts%",   f"{pts20_pct:.1f}%", help="Weeks scoring 20+ fantasy points ÷ games played")

    seasons_shown = ", ".join(str(int(s)) for s in stat_data["season"].tolist())
    st.caption(f"Stat cards: {seasons_shown}  ·  {total_games} games")


def _projection_tab(selected, proj_df, seasonal, team_stats, scoring, tiers_df):
    fp_col = SCORING_COL.get(scoring, "fpts_ppr")

    # ── Look up all three scenarios from projections_weighted.csv ─────────────
    player_projs = proj_df[proj_df["player_name"] == selected]
    if player_projs.empty:
        st.error(f"No projection data found for '{selected}'. Re-run `py 04_weighted_model.py`.")
        return

    base_row = player_projs[player_projs["scenario"] == "base"]
    bear_row = player_projs[player_projs["scenario"] == "bear"]
    bull_row = player_projs[player_projs["scenario"] == "bull"]

    if base_row.empty:
        st.error("Missing base scenario in projections_weighted.csv.")
        return

    base_s = base_row.iloc[0]
    bear_s = bear_row.iloc[0] if not bear_row.empty else base_s
    bull_s = bull_row.iloc[0] if not bull_row.empty else base_s

    pid        = base_s["player_id"]
    pos        = base_s["position"]
    team       = base_s["team"]
    age        = base_s["age"]
    confidence = base_s.get("projection_confidence", "")
    returning  = bool(base_s.get("returning_from_injury", False))

    base_fp = _f(base_s[fp_col])
    bear_fp = _f(bear_s[fp_col])
    bull_fp = _f(bull_s[fp_col])

    # ── Header ────────────────────────────────────────────────────────────────
    POS_COLOR = {"QB": "#e74c3c", "RB": "#27ae60", "WR": "#2980b9", "TE": "#d68910"}
    color = POS_COLOR.get(pos, "#7f8c8d")

    col_hdr, _ = st.columns([3, 1])
    with col_hdr:
        badges = (
            f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:5px;font-size:0.65em;vertical-align:middle">{pos}</span>'
        )
        if returning:
            badges += (
                ' <span style="background:#8e44ad;color:#fff;padding:3px 8px;'
                'border-radius:5px;font-size:0.65em;vertical-align:middle">↩ Returning</span>'
            )
        st.markdown(f"## {selected} {badges}", unsafe_allow_html=True)
        age_str = f"{int(age)}" if pd.notna(age) else "—"
        st.caption(f"**{team}**  ·  Age: **{age_str}**  ·  {scoring}  ·  Confidence: **{confidence}**")
    st.divider()

    # ── Bear / Base / Bull FP cards ───────────────────────────────────────────
    st.subheader(f"Season Projections — 17 games · {scoring}")

    c1, c2, c3 = st.columns(3)
    c1.metric("🐻  Bear", f"{bear_fp:.1f} pts")
    c2.metric("📊  Base", f"{base_fp:.1f} pts")
    c3.metric("🐂  Bull", f"{bull_fp:.1f} pts")

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

    def sv(row, col):
        """Safe float from a CSV row."""
        try:
            v = float(row.get(col, 0) or 0)
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    def stat_row(label, col, fmt="{:.1f}"):
        b, m, u = sv(bear_s, col), sv(base_s, col), sv(bull_s, col)
        return {"Stat": label, "🐻 Bear": fmt.format(b), "📊 Base": fmt.format(m), "🐂 Bull": fmt.format(u)}

    GAMES = 17
    if pos == "QB":
        rows = [
            stat_row("Pass Yards",  "passing_yards",  "{:.0f}"),
            stat_row("Pass TDs",    "passing_tds",    "{:.1f}"),
            stat_row("INTs",        "interceptions",  "{:.1f}"),
            stat_row("Rush Yards",  "rushing_yards",  "{:.0f}"),
            stat_row("Rush TDs",    "rushing_tds",    "{:.1f}"),
        ]
    elif pos == "RB":
        rows = [
            stat_row("Carries",    "carries",         "{:.1f}"),
            stat_row("Rush Yards", "rushing_yards",   "{:.0f}"),
            stat_row("Rush TDs",   "rushing_tds",     "{:.1f}"),
            stat_row("Targets",    "targets",         "{:.1f}"),
            stat_row("Receptions", "receptions",      "{:.1f}"),
            stat_row("Rec Yards",  "receiving_yards", "{:.0f}"),
            stat_row("Rec TDs",    "receiving_tds",   "{:.1f}"),
        ]
    else:  # WR / TE
        rows = [
            stat_row("Targets",    "targets",         "{:.1f}"),
            stat_row("Receptions", "receptions",      "{:.1f}"),
            stat_row("Rec Yards",  "receiving_yards", "{:.0f}"),
            stat_row("Rec TDs",    "receiving_tds",   "{:.1f}"),
        ]
        if sv(base_s, "rushing_yards") > 5:
            rows += [
                stat_row("Rush Yards", "rushing_yards", "{:.0f}"),
                stat_row("Rush TDs",   "rushing_tds",   "{:.1f}"),
            ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Per-Game Averages"):
        pg_rows = []
        for r in rows:
            try:
                pg_rows.append({
                    "Stat":    r["Stat"],
                    "🐻 Bear": f"{float(r['🐻 Bear']) / GAMES:.2f}",
                    "📊 Base": f"{float(r['📊 Base']) / GAMES:.2f}",
                    "🐂 Bull": f"{float(r['🐂 Bull']) / GAMES:.2f}",
                })
            except (ValueError, TypeError):
                pg_rows.append(r)
        st.dataframe(pd.DataFrame(pg_rows), use_container_width=True, hide_index=True)

    with st.expander("Historical Context & Team"):
        latest_season = int(seasonal["season"].max())
        tr = team_stats[(team_stats["team"] == team) & (team_stats["season"] == latest_season)]
        if tr.empty:
            tr = team_stats[team_stats["team"] == team].sort_values("season").tail(1)

        seas_row = seasonal[
            (seasonal["player_id"] == pid) & (seasonal["season"] == latest_season)
        ]

        h1, h2, h3, h4 = st.columns(4)
        if not tr.empty:
            h1.metric("Team Targets/Game", f"{_f(tr.iloc[0]['avg_targets_game']):.1f}")
            h2.metric("Team Carries/Game",  f"{_f(tr.iloc[0]['avg_carries_game']):.1f}")
        if not seas_row.empty:
            sr  = seas_row.iloc[0]
            g   = max(_f(sr.get("games", 1)), 1.0)
            h_ts  = _f(sr.get("target_share", 0))
            h_cpg = _f(sr.get("carries", 0)) / g
            if h_ts > 0:
                h3.metric("Historical Target Share", f"{h_ts * 100:.1f}%")
            if h_cpg > 0:
                h4.metric("Historical Carries/Game", f"{h_cpg:.1f}")

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

    # ── Historical Tier Consistency ───────────────────────────────────────────
    _projection_tier_section(pid, pos, tiers_df)


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    seasonal, team_stats           = load_data()
    proj_df                        = load_projections()
    adp_cur, adp_hist, adp_trends  = load_adp_data()
    tiers_df                       = load_tier_data()

    # ── Player index from projections_weighted.csv (High/Medium confidence) ───
    if not proj_df.empty:
        idx = (
            proj_df[
                (proj_df["scenario"] == "base") &
                proj_df["projection_confidence"].isin(["High", "Medium"])
            ]
            [["player_id", "player_name", "position", "team"]]
            .drop_duplicates("player_id")
            .sort_values("player_name")
        )
    else:
        # Fallback if projections_weighted.csv not yet generated
        SKILL = {"QB", "RB", "WR", "TE"}
        latest_season = int(seasonal["season"].max())
        idx = (
            seasonal[
                seasonal["position"].isin(SKILL) &
                (seasonal["season"] == latest_season)
            ]
            [["player_id", "player_name", "position", "team"]]
            .drop_duplicates("player_id")
            .sort_values("player_name")
        )

    # Remove entries where player_name looks like a raw player_id (e.g. "00-0037091")
    idx = idx[~idx["player_name"].str.match(r"^\d{2}-\d{7}", na=False)]

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

        # Handle navigation request from ADP table row click
        if "adp_navigate_player" in st.session_state:
            target = st.session_state.pop("adp_navigate_player")
            if target in name_opts:
                st.session_state["proj_player"] = target
                st.session_state["_switch_tab"] = True

        selected = st.selectbox(
            "Player", name_opts if name_opts else ["(no players)"],
            key="proj_player",
        )

        st.divider()
        scoring = st.radio("Scoring Format", SCORING_FORMATS)

    # ── Inject tab-switch JS if navigating from ADP table ────────────────────
    if st.session_state.pop("_switch_tab", False):
        _switch_to_projections_tab()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_adp, tab_proj = st.tabs(["📈 ADP & Market", "📊 Projections"])

    with tab_adp:
        _adp_tab(adp_cur, adp_hist, adp_trends, tiers_df)

    with tab_proj:
        _projection_tab(selected, proj_df, seasonal, team_stats, scoring, tiers_df)


main()
