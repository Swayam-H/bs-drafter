from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import urllib.parse
import os
import functools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CLOUD DATA SYNC ---
import hf_sync
if hf_sync.is_hf_configured():
    print("[HF Sync] Environment detected, downloading latest cloud data...")
    hf_sync.download_from_hf('matches.csv', 'matches.csv')
    hf_sync.download_from_hf('draft_model.pkl', 'draft_model.pkl')
else:
    print("[HF Sync] No Hugging Face token found, using local data.")

# Import the ML logic and the global Brawlers and Map lists from your predictor script
from smart_predictor import get_smart_recommendation, ALL_BRAWLERS, MAP_LIST

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
# Enable CORS for frontend
CORS(app, origins=['http://127.0.0.1:5000', 'http://localhost:5000'])

# Serve the frontend at the root URL
@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, 'index.html')

# --- CONFIGURATION ---
BRAWL_STARS_API_KEY = os.environ.get("BRAWL_STARS_API_KEY")
if not BRAWL_STARS_API_KEY:
    print("WARNING: BRAWL_STARS_API_KEY is not set! Scraping and player stats will not work.")
BASE_URL = "https://api.brawlstars.com/v1"

HEADERS = {
    "Authorization": f"Bearer {BRAWL_STARS_API_KEY}",
    "Accept": "application/json"
}

# --- 1. PLAYER STATS ROUTE ---
@app.route('/api/player/<player_tag>', methods=['GET'])
def get_player_stats(player_tag):
    """Proxy endpoint to fetch a player's profile."""
    safe_tag = urllib.parse.quote(player_tag)
    target_url = f"{BASE_URL}/players/{safe_tag}"
    
    try:
        response = requests.get(target_url, headers=HEADERS)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 2. BATTLELOG SCRAPER ROUTE ---
@app.route('/api/battlelog/<player_tag>', methods=['GET'])
def get_player_battlelog(player_tag):
    """Proxy endpoint to fetch a player's battlelog."""
    # Ensure the hashtag is properly URL-encoded if it isn't already
    if not player_tag.startswith('%23') and not player_tag.startswith('#'):
        player_tag = '%23' + player_tag
    elif player_tag.startswith('#'):
        player_tag = player_tag.replace('#', '%23')
        
    safe_tag = urllib.parse.quote(player_tag)
    
    # Fix double-encoding if it happens (urllib.parse.quote turns %23 into %2523)
    if safe_tag.startswith('%2523'):
        safe_tag = safe_tag.replace('%2523', '%23')
        
    target_url = f"{BASE_URL}/players/{safe_tag}/battlelog"
    
    try:
        response = requests.get(target_url, headers=HEADERS)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 2.5 BRAWLERS ROUTE ---
@app.route('/api/brawlers', methods=['GET'])
def get_brawlers():
    """Returns the list of all unique brawlers in the meta."""
    try:
        return jsonify(ALL_BRAWLERS), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 2.7 MAPS ROUTE ---
@app.route('/api/maps', methods=['GET'])
def get_maps():
    """Returns the clean list of all maps from the training meta."""
    try:
        OFFICIAL_MAPS = {
            "Sneaky Fields", "Super Beach", "Dry Season", "Hard Rock Mine", 
            "Double Swoosh", "Shooting Star", "Backyard Bowl", "New Horizons", 
            "Goldarm Gulch", "Bridge Too Far", "Nullscapes", "Triple Dribble", 
            "Dueling Beetles", "Belle's Rock", "Sunny Soccer", "Pinhole Punt", 
            "Layer Cake", "Below Zero", "Out in the Open", "Pinball Dreams", 
            "Center Stage", "Hot Potato", "Flaring Phoenix", "Undermine", 
            "Snake Prairie", "Parallel Plays", "Deep End", "Rustic Arcade", 
            "Hideout", "Safe Zone", "Ring of Fire", "Split", "Open Zone", 
            "Minecart Madness", "Kaboom Canyon", "Canal Grande", "Flowing Springs",
            "Gem Fort", "Last Stop", "GG Mortuary", "Pit Stop", "Quarter Pounder", 
            "Slippery Steps"
        }
        # Return all official maps, even if they aren't in the training data
        # If a map isn't in MAP_LIST, the ML model will make a generic prediction (all map features = 0)
        return jsonify(sorted(list(OFFICIAL_MAPS))), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@functools.lru_cache(maxsize=128)
def cached_recommendation(target_map, current_allies, current_enemies, available_to_draft, first_pick_team, peak_rank, player_brawlers_power_11, my_pick_index):
    return get_smart_recommendation(
        target_map, 
        list(current_allies), 
        list(current_enemies), 
        list(available_to_draft),
        first_pick_team,
        peak_rank=peak_rank,
        player_brawlers_power_11=list(player_brawlers_power_11) if player_brawlers_power_11 else None,
        my_pick_index=my_pick_index
    )

# --- 3. ML PREDICTOR ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        target_map = data.get('map')
        current_allies = data.get('allies', [])
        current_enemies = data.get('enemies', [])
        banned_brawlers = data.get('banned', [])
        first_pick_team = data.get('firstPickTeam', 'ally')
        peak_rank = data.get('peakRank')
        
        # User account sync parameters
        player_brawlers_power_11 = data.get('playerBrawlersPower11')
        my_pick_index = data.get('myPickIndex')
        
        if peak_rank is not None:
            try:
                peak_rank = int(peak_rank)
            except (ValueError, TypeError):
                peak_rank = None
        
        if not target_map:
            return jsonify({'status': 'error', 'message': 'Map is required'}), 400
            
        print(f"--- INCOMING WEB DRAFT ---")
        print(f"MAP: '{target_map}'")
        print(f"FIRST PICK: '{first_pick_team.upper()}'")
        print(f"RANK: {peak_rank}")
        print(f"ALLIES: {current_allies}")  
        print(f"ENEMIES: {current_enemies}")
        print(f"BANNED: {banned_brawlers}")
        
        # Input validation
        invalid_brawlers = [b for b in current_allies + current_enemies + banned_brawlers if b not in ALL_BRAWLERS]
        if invalid_brawlers:
            return jsonify({'status': 'error', 'message': f'Invalid brawlers: {invalid_brawlers}'}), 400

        picked_and_banned = set(current_allies + current_enemies + banned_brawlers)
        
        # Determine draft turn sequence to check if it's the player's turn to pick
        if first_pick_team == 'ally':
            draft_sequence = [
                {'team': 'ally', 'index': 0},
                {'team': 'enemy', 'index': 0},
                {'team': 'enemy', 'index': 1},
                {'team': 'ally', 'index': 1},
                {'team': 'ally', 'index': 2},
                {'team': 'enemy', 'index': 2}
            ]
        else:
            draft_sequence = [
                {'team': 'enemy', 'index': 0},
                {'team': 'ally', 'index': 0},
                {'team': 'ally', 'index': 1},
                {'team': 'enemy', 'index': 1},
                {'team': 'enemy', 'index': 2},
                {'team': 'ally', 'index': 2}
            ]
            
        current_turn_index = len(current_allies) + len(current_enemies)
        is_my_turn = False
        if my_pick_index is not None and current_turn_index < len(draft_sequence):
            turn = draft_sequence[current_turn_index]
            if turn['team'] == 'ally' and turn['index'] == my_pick_index:
                is_my_turn = True
                
        available_to_draft = [b for b in ALL_BRAWLERS if b not in picked_and_banned]
        
        recommendations = cached_recommendation(
            target_map, 
            tuple(current_allies), 
            tuple(current_enemies), 
            tuple(available_to_draft),
            first_pick_team,
            peak_rank=peak_rank,
            player_brawlers_power_11=tuple(player_brawlers_power_11) if player_brawlers_power_11 else None,
            my_pick_index=my_pick_index
        )
        
        return jsonify({
            'status': 'success',
            'map': target_map,
            'recommendations': recommendations
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- 4. GLOBAL RANKINGS ROUTE ---
@app.route('/api/rankings', methods=['GET'])
def get_top_players():
    """Fetches the top 100 global players to seed the ML scraper."""
    target_url = f"{BASE_URL}/rankings/global/players?limit=100"
    
    try:
        # Changed 'session.get' to 'requests.get' and added your HEADERS
        response = requests.get(target_url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    from apscheduler.schedulers.background import BackgroundScheduler
    import threading
    import subprocess
    import scraper

    # 1. Start the snowball scraper in a background thread
    print("[Server] Starting background scraper thread...")
    seed_tags = scraper.get_top_100_tags()
    if not seed_tags:
        seed_tags = ["#2PPQVUQ8J"]
    scraper_thread = threading.Thread(target=scraper.run_snowball_scraper, args=(seed_tags,), daemon=True)
    scraper_thread.start()

    # 2. Schedule cloud syncs and model training
    scheduler = BackgroundScheduler()
    
    def sync_matches():
        if hf_sync.is_hf_configured():
            hf_sync.upload_to_hf('matches.csv', 'matches.csv')
            
    def run_training_pipeline():
        print("[Scheduler] Starting nightly model training...")
        subprocess.run(["uv", "run", "train_model.py"])
        if hf_sync.is_hf_configured():
            hf_sync.upload_to_hf('draft_model.pkl', 'draft_model.pkl')
            
    scheduler.add_job(sync_matches, 'interval', hours=1)
    # Run model training at 2 AM UTC every day
    scheduler.add_job(run_training_pipeline, 'cron', hour=2, minute=0)
    scheduler.start()

    # Run the server on port provided by the environment (Render) or 7860 (Hugging Face default)
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)