import time
import urllib.parse
import requests
from collections import deque
import csv
import os

import concurrent.futures

PROXY_URL = "http://127.0.0.1:7860/api"

def fetch_player_data(tag, session):
    safe_tag = urllib.parse.quote(tag)
    try:
        player_response = session.get(f"{PROXY_URL}/player/{safe_tag}", timeout=10)
        peak_rank = "Unknown"
        if player_response.status_code == 200:
            player_data = player_response.json()
            peak_rank = player_data.get('highestAllTimeRankedElo') or player_data.get('highestAllTimeRankedRank') or player_data.get('highestTrophies', 'Unknown')

        battle_response = session.get(f"{PROXY_URL}/battlelog/{safe_tag}", timeout=10)
        
        if battle_response.status_code == 200:
            return tag, peak_rank, battle_response.json(), None
        else:
            return tag, peak_rank, None, f"Status: {battle_response.status_code}"
    except Exception as e:
        return tag, "Unknown", None, str(e)

def load_seen_matches(filename):
    seen = set()
    if os.path.isfile(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # Use DictReader if header exists, but fallback to checking first column if no header
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    try:
                        time_idx = headers.index('time')
                        for row in reader:
                            if len(row) > time_idx:
                                seen.add(row[time_idx])
                    except ValueError:
                        # Fallback if 'time' is not in header
                        pass
        except Exception as e:
            print(f"Error loading seen matches: {e}")
    return seen

def run_snowball_scraper(seed_tags):
    """Crawls through battlelogs, discovering new players as it goes."""
    
    player_queue = deque(seed_tags)
    seen_players = set(seed_tags)
    seen_matches = load_seen_matches(CSV_FILENAME)
    print(f"Loaded {len(seen_matches)} already seen matches from {CSV_FILENAME}.")
    matches_saved = 0
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    print("Starting continuous Snowball Scrape. Press Ctrl+C to stop.", flush=True)
    
    MAX_WORKERS = 10
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = set()
            
            while player_queue or futures:
                # Fill the executor with tasks
                while len(futures) < MAX_WORKERS and player_queue:
                    current_tag = player_queue.popleft()
                    futures.add(executor.submit(fetch_player_data, current_tag, session))
                    time.sleep(0.05)
                
                if not futures:
                    break
                    
                # Wait for at least one task to complete
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    current_tag, peak_rank, data, error = future.result()
                    
                    if error:
                        print(f"  [!] Failed on {current_tag}: {error}", flush=True)
                        continue
                    if not data:
                        continue
                        
                    # Filter: Only keep players/matches with peak rank >= 6000
                    try:
                        numeric_rank = int(peak_rank)
                    except (ValueError, TypeError):
                        numeric_rank = 0
                    if numeric_rank < 6000:
                        continue
                        
                    items = data.get('items', [])
                    player_batch = []
                    
                    for log in items:
                        battle = log.get('battle', {})
                        event = log.get('event', {})
                        
                        # --- STRICT MAP & FORMAT FILTERING ---
                        map_name = event.get('map')
                        if not map_name or str(map_name).strip().lower() in ['none', 'null', '']:
                            continue 
    
                        if battle.get('type') != 'ranked':
                            continue 
                        if 'trophyChange' in battle:
                            continue 
                        
                        # Ensure mode is a valid 3v3 ranked mode (exclude showdown, air hockey, etc.)
                        valid_ranked_modes = ['brawlBall', 'gemGrab', 'bounty', 'heist', 'hotZone', 'knockout']
                        if event.get('mode') not in valid_ranked_modes:
                            continue
                            
                        teams = battle.get('teams', [])
                        
                        # Must be exactly 2 teams
                        if not isinstance(teams, list) or len(teams) != 2:
                            continue 
                        # Must be exactly 3v3 format
                        if len(teams[0]) != 3 or len(teams[1]) != 3:
                            continue 
                            
                        match_time = log.get('battleTime')
                        if match_time in seen_matches:
                            continue
                        
                        # Snowball logic: add players to the queue
                        for team in teams:
                            for player in team:
                                new_tag = player.get('tag')
                                if new_tag and new_tag not in seen_players:
                                    seen_players.add(new_tag)
                                    player_queue.append(new_tag)
                        
                        # Safe extraction of brawler names
                        try:
                            ally_team = [p['brawler']['name'] for p in teams[0]]
                            enemy_team = [p['brawler']['name'] for p in teams[1]]
                        except KeyError:
                            continue # If the JSON is mangled and 'brawler' or 'name' is missing, ignore it
                        
                        # Ensure no duplicate brawlers (real draft format)
                        if set(ally_team) & set(enemy_team):
                            continue
                        if len(set(ally_team)) < 3 or len(set(enemy_team)) < 3:
                            continue

    
                        row = {
                            'time': match_time,
                            'peak_rank': peak_rank,
                            'map': map_name,
                            'mode': event.get('mode', 'Unknown Mode'),
                            'ally_1': ally_team[0], 'ally_2': ally_team[1], 'ally_3': ally_team[2],
                            'enemy_1': enemy_team[0], 'enemy_2': enemy_team[1], 'enemy_3': enemy_team[2],
                            'result': battle.get('result', 'draw')
                        }
                        
                        player_batch.append(row)
                        seen_matches.add(match_time)
                        matches_saved += 1
                        
                    if player_batch:
                        save_batch_to_csv(player_batch, CSV_FILENAME)
                            
                    print(f"Queue Size: {len(player_queue)} | Total Saved: {matches_saved} | Just Checked: {current_tag} (Found: {len(player_batch)})", flush=True)
            
    except KeyboardInterrupt:
        print(f"\n[!] Graceful exit: Total matches saved in this session: {matches_saved}.")

def save_batch_to_csv(batch, filename):
    file_exists = os.path.isfile(filename)
    keys = batch[0].keys() if batch else []
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(batch)

def get_top_100_tags():
    print("Fetching Top 100 Global Players to seed the scraper...", flush=True)
    try:
        response = requests.get(f"{PROXY_URL}/rankings", timeout=10)
        if response.status_code == 200:
            data = response.json()
            tags = [player['tag'] for player in data.get('items', [])]
            print(f"[OK] Successfully loaded {len(tags)} elite seed tags.", flush=True)
            return tags
        else:
            print(f"[!] Failed to fetch rankings (Status: {response.status_code})", flush=True)
    except Exception as e:
        print(f"[!] Error fetching rankings: {e}", flush=True)
    
    return []

if __name__ == "__main__":
    CSV_FILENAME = "matches.csv"
    
    STARTING_PLAYERS = get_top_100_tags()
    
    if not STARTING_PLAYERS:
        print("Falling back to a default high-ELO tag.", flush=True)
        STARTING_PLAYERS = ["#2PPQVUQ8J"] 
        
    print("Initializing Scraper...")
    run_snowball_scraper(STARTING_PLAYERS)