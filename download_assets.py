import os
import pickle
import requests

# Load model data to get the brawler list
try:
    with open('draft_model.pkl', 'rb') as f:
        data = pickle.load(f)
        brawler_list = data['brawler_list']
except FileNotFoundError:
    print("Error: draft_model.pkl not found. Run train_model.py first!")
    exit(1)

# Fetch all brawlers data from BrawlAPI
print("Fetching brawlers metadata from BrawlAPI...")
try:
    api_response = requests.get('https://api.brawlapi.com/v1/brawlers', timeout=15)
    if api_response.status_code == 200:
        brawlers_data = api_response.json().get('list', [])
    else:
        print(f"Error: BrawlAPI returned status code {api_response.status_code}")
        exit(1)
except Exception as e:
    print(f"Error connecting to BrawlAPI: {e}")
    exit(1)

# Create a dictionary mapping upper-case name to BrawlAPI object
brawler_map = {b['name'].upper(): b for b in brawlers_data}

# Naming formatting helper matching script.js
def get_formatted_name(name):
    formatted = name.strip().lower()
    if formatted == "8-bit":
        return "8-Bit"
    if formatted in ["mr. p", "mr p"]:
        return "Mr-P"
    if formatted == "r-t":
        return "R-T"
    
    # Capitalize each word and join with hyphens
    words = formatted.split(' ')
    capitalized = [word.capitalize() for word in words]
    return "-".join(capitalized)

# Create images folder
images_dir = os.path.join("frontend", "images")
os.makedirs(images_dir, exist_ok=True)

print(f"Downloading brawler icons to {images_dir}...")

downloaded_count = 0
for name in brawler_list:
    name_upper = name.upper()
    fmt_name = get_formatted_name(name)
    target_path = os.path.join(images_dir, f"{fmt_name}.png")
    
    # Check if we have the brawler in the BrawlAPI list
    if name_upper in brawler_map:
        brawler_info = brawler_map[name_upper]
        # Use borderless (imageUrl2) if available, otherwise borders (imageUrl)
        url = brawler_info.get('imageUrl2') or brawler_info.get('imageUrl')
    else:
        # Fallback to name-based URL if name is not in mapping
        url = f"https://cdn.brawlify.com/brawlers/{fmt_name}.png"
        print(f"Brawler '{name}' not found in BrawlAPI mapping. Falling back to name-based URL.")

    if not url:
        print(f"Warning: No image URL found for {name}")
        continue

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(target_path, 'wb') as img_f:
                img_f.write(response.content)
            print(f"Downloaded: {fmt_name}.png")
            downloaded_count += 1
        else:
            print(f"Failed: Status {response.status_code} for {name} (URL: {url})")
    except Exception as e:
        print(f"Error downloading {name} from {url}: {e}")

print(f"Asset download completed! Successfully downloaded {downloaded_count} of {len(brawler_list)} brawler icons.")

# Create a dictionary of metadata to save for the frontend
import json
frontend_metadata = {}
for name in brawler_list:
    name_upper = name.upper()
    if name_upper in brawler_map:
        binfo = brawler_map[name_upper]
        frontend_metadata[name_upper] = {
            "class": binfo.get('class', {}).get('name', 'Unknown'),
            "rarity": binfo.get('rarity', {}).get('name', 'Unknown')
        }
    else:
        frontend_metadata[name_upper] = {
            "class": "Unknown",
            "rarity": "Unknown"
        }

metadata_path = os.path.join("frontend", "brawler_metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as mf:
    json.dump(frontend_metadata, mf, indent=2)
print(f"Saved brawler metadata to {metadata_path}")

# --- Compute per-brawler win rates per map ---
import pandas as pd

csv_path = 'global_ranked_drafts.csv'
if os.path.exists(csv_path):
    print("Computing map-specific brawler win rates...")
    df = pd.read_csv(csv_path)
    df = df[df['result'] != 'draw'].drop_duplicates(subset=['time', 'ally_1', 'ally_2', 'ally_3']).reset_index(drop=True)

    # Filter rare maps
    map_counts = df['map'].value_counts()
    valid_maps = map_counts[map_counts >= 15].index
    df = df[df['map'].isin(valid_maps)].reset_index(drop=True)

    map_winrates = {}

    for map_name in df['map'].unique():
        map_df = df[df['map'] == map_name]
        brawler_stats = {}

        for brawler in brawler_list:
            # Count appearances as ally (victory = win for this brawler)
            ally_mask = (map_df['ally_1'] == brawler) | (map_df['ally_2'] == brawler) | (map_df['ally_3'] == brawler)
            ally_appearances = map_df[ally_mask]
            ally_wins = (ally_appearances['result'] == 'victory').sum()
            ally_total = len(ally_appearances)

            # Count appearances as enemy (victory = loss for this brawler)
            enemy_mask = (map_df['enemy_1'] == brawler) | (map_df['enemy_2'] == brawler) | (map_df['enemy_3'] == brawler)
            enemy_appearances = map_df[enemy_mask]
            enemy_wins = (enemy_appearances['result'] == 'defeat').sum()  # defeat for us = victory for enemy brawler
            enemy_total = len(enemy_appearances)

            total_games = ally_total + enemy_total
            total_wins = ally_wins + enemy_wins

            if total_games >= 5:  # Minimum sample size
                brawler_stats[brawler] = round(total_wins / total_games * 100, 1)

        map_winrates[map_name] = brawler_stats

    winrates_path = os.path.join("frontend", "map_winrates.json")
    with open(winrates_path, 'w', encoding='utf-8') as wf:
        json.dump(map_winrates, wf, indent=2)
    print(f"Saved map win rates to {winrates_path} ({len(map_winrates)} maps)")
else:
    print(f"Skipping win rate computation: {csv_path} not found.")

