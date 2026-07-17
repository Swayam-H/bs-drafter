import pandas as pd
import numpy as np
import pickle
import warnings
from smart_predictor import run_greedy

# Suppress XGBoost feature name warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load the trained model and features
print("Loading model...")
try:
    with open('draft_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        brawler_list = data['brawler_list']
        map_list = data['map_list']
except FileNotFoundError:
    print("Error: draft_model.pkl not found. Run train_model.py first!")
    exit()

def get_best_pick(target_map, allies, enemies):
    """
    Simulates picking every available brawler and returns the top 5 
    based on predicted win probability using the robust Stacking Ensemble features.
    """
    # Identify available brawlers (not currently picked)
    unavailable = set(allies + enemies)
    available_brawlers = [b for b in brawler_list if b not in unavailable]
    
    # Delegate to the run_greedy algorithm which handles full feature engineering
    recs = run_greedy(target_map, allies, enemies, available_brawlers)
    
    return [{'brawler': r['our_pick'], 'win_prob': r['win_prob']} for r in recs]

# --- TEST THE PREDICTOR ---
if __name__ == "__main__":
    print("\n--- DRAFT SIMULATION ---")
    
    # You can change these values to test different scenarios!
    test_map = "Hard Rock Mine"
    
    # Brawlers must match exactly how they are spelled in the game/API (ALL CAPS usually)
    current_allies = ["SHELLY"] 
    current_enemies = ["PIPER", "TICK"]
    
    print(f"Map: {test_map}")
    print(f"Allies: {current_allies}")
    print(f"Enemies: {current_enemies}")
    print("\nCalculating best picks...")
    
    recommendations = get_best_pick(test_map, current_allies, current_enemies)
    
    print("\nTop 5 Recommended Picks:")
    for i, rec in enumerate(recommendations, 1):
        # Format the probability as a percentage (e.g., 65.4%)
        prob_pct = rec['win_prob'] * 100
        print(f"{i}. {rec['brawler'].ljust(15)} - {prob_pct:.1f}% Win Probability")