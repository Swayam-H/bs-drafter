import pickle
import itertools
import numpy as np
import pandas as pd
from smart_predictor import run_minimax

# 1. Load the dictionary from the pickle file
print("Loading model and feature lists...")
with open('draft_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

# Extract the components
model = saved_data['model']
ALL_BRAWLERS = saved_data['brawler_list']
MAP_LIST = saved_data['map_list']

def batch_minimax_recommendation(map_name, current_allies, current_enemies, available_brawlers):
    """
    Vectorized Minimax using robust Stacking Ensemble feature engineering from smart_predictor.
    """
    recs = run_minimax(map_name, current_allies, current_enemies, available_brawlers)
    
    results = []
    for r in recs:
        results.append({
            'our_pick': r['our_pick'],
            'expected_win_prob': r['win_prob'],
            'expected_enemy_counter': r['enemy_counter']
        })
        
    return results

if __name__ == "__main__":
    # --- Example Usage ---
    # Make sure this map matches a map from your dataset exactly
    target_map = "Hard Rock Mine" 
    
    my_team = ['BO']
    enemy_team = ['COLT']
    
    # Assuming Piper and Tara were banned
    banned_brawlers = ['PIPER', 'TARA']
    
    # Filter available brawlers
    picked_and_banned = set(my_team + enemy_team + banned_brawlers) 
    available_to_draft = [b for b in ALL_BRAWLERS if b not in picked_and_banned]
    
    print("\nRunning Minimax simulation...")
    top_5 = batch_minimax_recommendation(
        target_map, 
        my_team, 
        enemy_team, 
        available_to_draft
    )
    
    print(f"\n--- Top 5 Safest Picks for {target_map} ---")
    for i, rec in enumerate(top_5, 1):
        print(f"{i}. Pick: {rec['our_pick'].ljust(12)} "
              f"| Worst-case Win Prob: {rec['expected_win_prob']:.2%} "
              f"| Enemy Counter: {rec['expected_enemy_counter']}")