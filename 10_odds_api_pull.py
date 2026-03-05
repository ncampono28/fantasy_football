"""
10_odds_api_pull.py — The Odds API Integration
================================================
Pulls NFL season-long data from The Odds API:

  1. Team win totals (outrights/futures) — available now
  2. Super Bowl / AFC / NFC championship odds — team strength signals
  3. Weekly game totals — implied team scoring per game (in-season)
  4. Season player props — passing/rushing/receiving yards & TDs (available June-Aug)

How win totals improve the model:
  - A team with 11.5 win total has a strong offense → boost all skill players
  - A team with 5.5 win total has a weak offense → discount all skill players
  - Compare win total to our team_season_stats baseline to find offseason shifts
  
Run:
    py 10_odds_api_pull.py

Outputs:
    data/vegas_win_totals.csv       — team win totals + implied points
    data/vegas_team_context.csv     — team multipliers for projection model
    data/vegas_player_props.csv     — season player props (when available)
    data/vegas_game_totals.csv      — weekly game totals (in-season)

API key: store in environment variable ODDS_API_KEY or paste below
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, date

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — paste your API key here or set as environment variable
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.the-odds-api.com/v4"
DATA = Path("data")

# NFL team name → nflverse abbreviation mapping
TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

SCHEDULE_FILE = DATA / "odds_api_schedule.json"

SCHEDULE_DEFAULTS = {
    "win_totals":    "2026-05-15",
    "player_props":  "2026-06-01",
    "game_totals":   "2026-08-01",
}

SCHEDULE_LABELS = {
    "win_totals":   "Win Totals (team O/U lines post)",
    "player_props": "Player Season Props (passing/rushing/receiving lines post)",
    "game_totals":  "Game Totals (weekly game lines go live)",
}


def _ensure_schedule():
    """Write schedule JSON if it doesn't exist yet."""
    if not SCHEDULE_FILE.exists():
        with open(SCHEDULE_FILE, "w") as f:
            json.dump(SCHEDULE_DEFAULTS, f, indent=2)
        print(f"  Created {SCHEDULE_FILE}")
    with open(SCHEDULE_FILE) as f:
        return json.load(f)


def print_schedule_countdown():
    """Print days-to-go for each market milestone."""
    schedule = _ensure_schedule()
    today = date.today()
    print("\n  --- Odds API Market Schedule ---")
    for key, target_str in schedule.items():
        target = date.fromisoformat(target_str)
        delta = (target - today).days
        label = SCHEDULE_LABELS.get(key, key)
        if delta < 0:
            status = f"PAST (was {target_str})"
        elif delta == 0:
            status = "TODAY"
        else:
            status = f"{delta} days  (target: {target_str})"
        print(f"  {label}")
        print(f"    -> {status}")
    print()


def scan_all_markets():
    """
    Call GET /v4/sports?all=true and print every sport that is
    active=True or has_outrights=True.  Then for any sport whose key
    contains 'nfl' or 'football', attempt to pull outrights and print
    whatever the API returns.
    """
    print("\n--- scan_all_markets() ---")
    r = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY, "all": "true"})
    check_quota(r)
    r.raise_for_status()
    all_sports = r.json()

    relevant = [
        s for s in all_sports
        if s.get("active") or s.get("has_outrights")
    ]
    print(f"  Total sports with active=True or has_outrights=True: {len(relevant)}")
    for s in relevant:
        flag = []
        if s.get("active"):
            flag.append("active")
        if s.get("has_outrights"):
            flag.append("has_outrights")
        print(f"  [{', '.join(flag)}]  {s['key']:55s}  {s.get('title','')}")

    football_sports = [
        s for s in all_sports
        if "nfl" in s["key"].lower() or "football" in s["key"].lower()
    ]
    print(f"\n  Football/NFL sport keys found ({len(football_sports)}):")
    for s in football_sports:
        print(f"    {s['key']}  active={s.get('active')}  has_outrights={s.get('has_outrights')}")

    print("\n  Attempting outrights pull for each football/NFL sport key...")
    for s in football_sports:
        sport_key = s["key"]
        try:
            r2 = requests.get(
                f"{BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey":      API_KEY,
                    "regions":     "us",
                    "markets":     "outrights",
                    "oddsFormat":  "american",
                    "bookmakers":  "draftkings,fanduel,betmgm",
                }
            )
            check_quota(r2)
            if r2.status_code == 200:
                events = r2.json()
                print(f"  {sport_key}: {len(events)} events returned")
                for ev in events[:2]:
                    home = ev.get("home_team", "?")
                    away = ev.get("away_team", "?")
                    bms  = ev.get("bookmakers", [])
                    print(f"    Event: {home} vs {away}  ({len(bms)} bookmakers)")
                    for bm in bms[:1]:
                        for mkt in bm.get("markets", []):
                            outcomes = mkt.get("outcomes", [])
                            print(f"      Market '{mkt['key']}': {len(outcomes)} outcomes")
                            for o in outcomes[:5]:
                                print(f"        {o.get('name')}: {o.get('price')}  point={o.get('point','N/A')}")
            elif r2.status_code == 404:
                print(f"  {sport_key}: 404 — market not available yet")
            elif r2.status_code == 422:
                print(f"  {sport_key}: 422 — invalid market for this sport")
            else:
                print(f"  {sport_key}: HTTP {r2.status_code} — {r2.text[:120]}")
        except Exception as e:
            print(f"  {sport_key}: ERROR — {e}")


def check_quota(response):
    """Print remaining API quota after each call"""
    remaining = response.headers.get("x-requests-remaining", "?")
    used = response.headers.get("x-requests-used", "?")
    print(f"    [API quota: {used} used, {remaining} remaining]")

def get_sports():
    """Verify NFL is available and check what's active"""
    print("\n1. Checking available sports/markets...")
    r = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
    r.raise_for_status()
    sports = r.json()
    nfl = [s for s in sports if "nfl" in s["key"].lower() or "americanfootball" in s["key"].lower()]
    for s in nfl:
        print(f"   {s['key']}: {s['title']} — active={s['active']}, has_outrights={s.get('has_outrights', False)}")
    return nfl

def get_win_totals():
    """
    Pull NFL team win totals (futures/outrights)
    These are the single most useful season-long signal for our model.
    A team's win total encodes Vegas's view of their full offensive environment.
    """
    print("\n2. Pulling NFL win totals...")
    
    # Win totals live under americanfootball_nfl_super_bowl or nfl futures
    # Try both the main NFL futures and the win totals specific endpoint
    endpoints_to_try = [
        "americanfootball_nfl_super_bowl",
        "americanfootball_nfl_championship_winner", 
        "americanfootball_nfl",
    ]
    
    win_totals = []
    
    for sport_key in endpoints_to_try:
        try:
            r = requests.get(
                f"{BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": API_KEY,
                    "regions": "us",
                    "markets": "outrights",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm"
                }
            )
            check_quota(r)
            if r.status_code == 200:
                data = r.json()
                print(f"   Found {len(data)} events for {sport_key}")
                for event in data[:3]:
                    print(f"   Sample event: {event.get('sport_title')} — {event.get('home_team','?')} vs {event.get('away_team','?')}")
                    if event.get('bookmakers'):
                        for bm in event['bookmakers'][:1]:
                            for market in bm.get('markets', []):
                                print(f"   Market: {market['key']} — {len(market.get('outcomes',[]))} outcomes")
                                for outcome in market.get('outcomes', [])[:5]:
                                    print(f"     {outcome.get('name')}: {outcome.get('price')} (point: {outcome.get('point', 'N/A')})")
                break
        except Exception as e:
            print(f"   {sport_key}: {e}")
    
    # Pull win totals specifically — these are under the NFL regular season futures
    print("\n   Trying NFL win totals directly...")
    try:
        r = requests.get(
            f"{BASE_URL}/sports/americanfootball_nfl/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "outrights",
                "oddsFormat": "american",
            }
        )
        check_quota(r)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Events returned: {len(data)}")
            if data:
                # Show structure
                event = data[0]
                print(f"   First event: {json.dumps(event, indent=2)[:500]}")
    except Exception as e:
        print(f"   Error: {e}")

    return win_totals

def get_current_nfl_odds():
    """
    Pull current NFL game odds — useful for game totals and implied team scoring
    This is the weekly data we'll use in-season
    """
    print("\n3. Pulling current NFL game odds (totals)...")
    try:
        r = requests.get(
            f"{BASE_URL}/sports/americanfootball_nfl/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "totals",
                "oddsFormat": "american",
                "bookmakers": "draftkings,fanduel"
            }
        )
        check_quota(r)
        print(f"   Status: {r.status_code}")
        
        if r.status_code == 200:
            games = r.json()
            print(f"   Games with totals: {len(games)}")
            
            if games:
                records = []
                for game in games:
                    home = game['home_team']
                    away = game['away_team']
                    commence = game['commence_time']
                    
                    for bm in game.get('bookmakers', []):
                        for market in bm.get('markets', []):
                            if market['key'] == 'totals':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == 'Over':
                                        records.append({
                                            'home_team': home,
                                            'away_team': away,
                                            'home_abbr': TEAM_MAP.get(home, home),
                                            'away_abbr': TEAM_MAP.get(away, away),
                                            'game_total': outcome['point'],
                                            'implied_home': round(outcome['point'] / 2, 1),
                                            'implied_away': round(outcome['point'] / 2, 1),
                                            'bookmaker': bm['key'],
                                            'commence_time': commence,
                                            'pulled_at': datetime.now().isoformat()
                                        })
                                        break
                
                if records:
                    df = pd.DataFrame(records)
                    df.to_csv(DATA / "vegas_game_totals.csv", index=False)
                    print(f"   Saved {len(df)} game total records")
                    print(df[['home_team','away_team','game_total','commence_time']].to_string(index=False))
                else:
                    print("   No games currently listed (off-season — totals post ~August)")
        
        elif r.status_code == 422:
            print("   NFL not currently in season — game totals not available until August 2026")
            
    except Exception as e:
        print(f"   Error: {e}")

def get_player_season_props():
    """
    Pull NFL player season props — passing/rushing/receiving yards & TDs
    These are the crown jewel for our projection model.
    Available typically June-August before the season.
    
    Market keys for season props:
        player_pass_yds, player_pass_tds, player_rush_yds, 
        player_reception_yds, player_reception_tds, player_receptions
    """
    print("\n4. Checking player season props availability...")
    
    season_prop_markets = [
        "player_pass_yds",
        "player_pass_tds", 
        "player_rush_yds",
        "player_rush_tds",
        "player_reception_yds",
        "player_reception_tds",
        "player_receptions",
    ]
    
    # First get list of events
    try:
        r = requests.get(
            f"{BASE_URL}/sports/americanfootball_nfl/events",
            params={"apiKey": API_KEY}
        )
        # Events endpoint doesn't count against quota
        print(f"   Events status: {r.status_code}")
        
        if r.status_code == 200:
            events = r.json()
            print(f"   NFL events available: {len(events)}")
            
            if events:
                # Try getting props for first event
                event_id = events[0]['id']
                print(f"   Trying props for event: {events[0].get('home_team')} vs {events[0].get('away_team')}")
                
                r2 = requests.get(
                    f"{BASE_URL}/sports/americanfootball_nfl/events/{event_id}/odds",
                    params={
                        "apiKey": API_KEY,
                        "regions": "us",
                        "markets": ",".join(season_prop_markets[:3]),
                        "oddsFormat": "american",
                        "bookmakers": "draftkings"
                    }
                )
                check_quota(r2)
                print(f"   Props status: {r2.status_code}")
                if r2.status_code == 200:
                    prop_data = r2.json()
                    print(f"   Sample: {json.dumps(prop_data, indent=2)[:800]}")
            else:
                print("   No events listed — NFL season props not yet available")
                print("   These will post June-August 2026 before the season")
                
    except Exception as e:
        print(f"   Error: {e}")

def get_nfl_futures():
    """
    Pull Super Bowl, AFC, NFC futures — team strength signals
    """
    print("\n5. Pulling NFL championship futures...")
    
    futures_sports = [
        "americanfootball_nfl_super_bowl",
        "americanfootball_nfl_championship_winner",
    ]
    
    all_futures = []
    
    for sport in futures_sports:
        try:
            r = requests.get(
                f"{BASE_URL}/sports/{sport}/odds",
                params={
                    "apiKey": API_KEY,
                    "regions": "us",
                    "markets": "outrights",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm"
                }
            )
            check_quota(r)
            
            if r.status_code == 200:
                data = r.json()
                print(f"   {sport}: {len(data)} events")
                
                for event in data:
                    for bm in event.get('bookmakers', []):
                        if bm['key'] == 'draftkings':
                            for market in bm.get('markets', []):
                                for outcome in market.get('outcomes', []):
                                    all_futures.append({
                                        'market': sport.replace('americanfootball_nfl_',''),
                                        'team': outcome['name'],
                                        'team_abbr': TEAM_MAP.get(outcome['name'], outcome['name']),
                                        'odds': outcome['price'],
                                        'bookmaker': bm['key'],
                                        'pulled_at': datetime.now().isoformat()
                                    })
            elif r.status_code == 404:
                print(f"   {sport}: Not available yet")
                
        except Exception as e:
            print(f"   {sport}: {e}")
    
    if all_futures:
        df = pd.DataFrame(all_futures)
        
        # Convert American odds to implied probability
        def american_to_prob(odds):
            if odds > 0:
                return round(100 / (odds + 100), 4)
            else:
                return round(abs(odds) / (abs(odds) + 100), 4)
        
        df['implied_prob'] = df['odds'].apply(american_to_prob)
        df.to_csv(DATA / "vegas_futures.csv", index=False)
        print(f"\n   Saved {len(df)} futures records")
        
        # Show Super Bowl odds
        sb = df[df['market'].str.contains('super_bowl', case=False)].sort_values('implied_prob', ascending=False)
        if not sb.empty:
            print("\n   Super Bowl odds (top 16):")
            print(sb[['team','team_abbr','odds','implied_prob']].head(16).to_string(index=False))
    else:
        print("   No futures data returned — may not be posted yet for 2026 season")

def build_team_context():
    """
    Build team_context.csv combining:
    - Win totals (if available)
    - Super Bowl implied probability (team strength proxy)
    - Historical team scoring from team_season_stats.csv
    
    This feeds into 04_weighted_model.py as a multiplier
    """
    print("\n6. Building team context file...")
    
    try:
        team_stats = pd.read_csv(DATA / "team_season_stats.csv")
        latest = team_stats.sort_values('season').groupby('team').last().reset_index()
        print(f"   Loaded {len(latest)} teams from historical stats")
    except:
        print("   Warning: team_season_stats.csv not found — run 01_fetch_data.py first")
        return
    
    # Load futures if we pulled them
    team_context = latest[['team','avg_targets_game','avg_carries_game']].copy()
    
    futures_path = DATA / "vegas_futures.csv"
    if futures_path.exists():
        futures = pd.read_csv(futures_path)
        sb = futures[futures['market'].str.contains('super_bowl', case=False)]
        sb = sb.groupby('team_abbr')['implied_prob'].mean().reset_index()
        sb.columns = ['team', 'sb_prob']
        team_context = team_context.merge(sb, on='team', how='left')
        print(f"   Merged Super Bowl probabilities for {sb['team'].nunique()} teams")
    
    # Derive team strength multiplier
    # League average SB prob = 1/32 = 0.03125
    # Teams above average get a modest boost, below get a discount
    if 'sb_prob' in team_context.columns:
        avg_prob = 1/32
        team_context['team_strength_multiplier'] = (
            team_context['sb_prob'].fillna(avg_prob) / avg_prob
        ).apply(lambda x: min(max(round(x ** 0.25, 3), 0.88), 1.12))
        # Cap multiplier at ±12% to prevent overcorrection
    else:
        team_context['team_strength_multiplier'] = 1.0
    
    team_context['data_as_of'] = datetime.now().strftime('%Y-%m-%d')
    team_context.to_csv(DATA / "vegas_team_context.csv", index=False)
    print(f"   Saved team context for {len(team_context)} teams")
    
    if 'team_strength_multiplier' in team_context.columns:
        print("\n   Team strength multipliers (top 10 and bottom 10):")
        sorted_ctx = team_context.sort_values('team_strength_multiplier', ascending=False)
        print(sorted_ctx[['team','team_strength_multiplier']].head(10).to_string(index=False))
        print("   ...")
        print(sorted_ctx[['team','team_strength_multiplier']].tail(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  The Odds API — NFL Season Data Pull")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Always print countdown — runs even without a valid API key
    print_schedule_countdown()

    if API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: Set your API key!")
        print("   Option 1: Set environment variable: set ODDS_API_KEY=your_key")
        print("   Option 2: Edit API_KEY at top of this file")
        exit(1)

    # Check quota first
    print("\n0. Checking API status...")
    r = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY, "all": "true"})
    check_quota(r)

    try:
        # 0b. Scan every available sport/market
        scan_all_markets()

        # 1. Check what NFL markets are available
        get_sports()

        # 2. Win totals / futures (most valuable for our model)
        get_nfl_futures()

        # 3. Win totals specifically
        get_win_totals()

        # 4. Current game totals (in-season weekly signal)
        get_current_nfl_odds()

        # 5. Player season props
        get_player_season_props()

        # 6. Build combined team context file
        build_team_context()
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if e.response.status_code == 401:
            print("   Invalid API key — check your key at the-odds-api.com")
        elif e.response.status_code == 429:
            print("   Rate limit hit — wait and try again")
    
    print("\n" + "=" * 60)
    print("  What's available NOW vs LATER:")
    print("  ✅ Super Bowl / championship futures — available now")
    print("  ✅ Team win totals — check above, may need nfl_super_bowl key")
    print("  ⏳ Game totals (weekly) — available August 2026 when season posts")
    print("  ⏳ Player season props — available June-August 2026")
    print("  ")
    print("  Next step: Run py 04_weighted_model.py after win totals")
    print("  are integrated into vegas_team_context.csv")
    print("=" * 60)
