import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import time
from brawler_roles import BRAWLER_ROLES, ALL_ROLES, get_role, get_tier, get_mechanics, is_open_map, is_closed_map, get_range_class, StackedModel
import lightgbm as lgb
import catboost as cb

print("--- BRAWL STARS ML TRAINING (v11 - Stacking Ensemble & Meta-Learning) ---")
HIGH_ELO_ONLY = True  # Set to True to train exclusively on high-level drafts (>6000 Peak Rank)

# 1. Load and Clean the Data
print("Loading data from matches.csv...")

rows = []
with open('matches.csv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 10:
            t, map_name, mode, a1, a2, a3, e1, e2, e3, res = parts
            peak_rank = None
        elif len(parts) == 11:
            t, pr, map_name, mode, a1, a2, a3, e1, e2, e3, res = parts
            try:
                peak_rank = int(pr)
            except (ValueError, TypeError):
                peak_rank = None
        else:
            continue
        rows.append([t, peak_rank, map_name, mode, a1, a2, a3, e1, e2, e3, res])

df = pd.DataFrame(rows, columns=['time', 'peak_rank', 'map', 'mode', 'ally_1', 'ally_2', 'ally_3', 'enemy_1', 'enemy_2', 'enemy_3', 'result'])

# Drop draws and duplicates
df = df[df['result'] != 'draw'].drop_duplicates(subset=['time', 'ally_1', 'ally_2', 'ally_3']).reset_index(drop=True)

# Filter out non-draft matches (where duplicate brawlers exist on either team or across teams)
def is_valid_draft(row):
    allies = {row.ally_1, row.ally_2, row.ally_3}
    enemies = {row.enemy_1, row.enemy_2, row.enemy_3}
    if len(allies) < 3 or len(enemies) < 3:
        return False
    if allies & enemies:
        return False
    return True

valid_draft_mask = df.apply(is_valid_draft, axis=1)
df = df[valid_draft_mask].reset_index(drop=True)

# Filter out custom/rare maps (with less than 15 occurrences)
map_counts = df['map'].value_counts()
valid_maps = map_counts[map_counts >= 15].index
df = df[df['map'].isin(valid_maps)].reset_index(drop=True)


# Fill missing peak_rank with the median of the dataset
median_rank = df['peak_rank'].median()
df['peak_rank'] = df['peak_rank'].fillna(median_rank).astype(float)

# Implement Continuous Exponential ELO Weighting
print("Applying continuous ELO weighting to all matches...")
# Clip peak_rank to [1000, 10000] to prevent extreme outliers from dominating
# Weight formula: (peak_rank / 5000)^3
df['rank_weight'] = np.power(np.clip(df['peak_rank'], 1000, 10000) / 5000.0, 3)


df['target'] = df['result'].apply(lambda x: 1 if x == 'victory' else 0)
y = df['target'].values

print(f"Dataset ready: {len(df)} valid matches after filtering.")
print(f"Median peak rank: {median_rank}")

# --- Strategy 2: Parse timestamps for temporal weighting ---
print("Computing temporal weights (30-day half-life)...")
df['time_parsed'] = pd.to_datetime(df['time'], format='%Y%m%dT%H%M%S.%fZ', errors='coerce')
latest_time = df['time_parsed'].max()
df['days_ago'] = (latest_time - df['time_parsed']).dt.total_seconds() / 86400.0
df['days_ago'] = df['days_ago'].fillna(df['days_ago'].median())
HALF_LIFE_DAYS = 30
df['time_weight'] = np.power(0.5, df['days_ago'] / HALF_LIFE_DAYS)
print(f"  Time weight range: {df['time_weight'].min():.4f} - {df['time_weight'].max():.4f}")

# 2. Extract Unique Brawlers
all_brawlers = pd.concat([
    df['ally_1'], df['ally_2'], df['ally_3'], 
    df['enemy_1'], df['enemy_2'], df['enemy_3']
]).unique()

brawler_list = sorted(list(all_brawlers))
print(f"Found {len(brawler_list)} unique brawlers in the meta.")

brawler_to_idx = {b: i for i, b in enumerate(brawler_list)}

role_feature_names = []
for role in ALL_ROLES:
    role_feature_names.extend([f"ally_{role}_count", f"enemy_{role}_count", f"{role}_advantage"])

role_to_idx = {}
for i, role in enumerate(ALL_ROLES):
    role_to_idx[role] = {
        'ally': i * 3,
        'enemy': i * 3 + 1,
        'advantage': i * 3 + 2,
    }

# Encode all maps
X_maps_temp = pd.get_dummies(df['map'], prefix='map')
map_list = list(X_maps_temp.columns)

map_to_mode = df[['map', 'mode']].drop_duplicates().set_index('map')['mode'].to_dict()

# --- Strategy 4: Compute brawler pick rates before split ---
print("Computing brawler pick rates...")
all_picks = pd.concat([df['ally_1'], df['ally_2'], df['ally_3'], df['enemy_1'], df['enemy_2'], df['enemy_3']])
brawler_pick_rates = (all_picks.value_counts() / len(df)).to_dict()

# 3. Train/Val/Test Split (70/15/15) - Stratified (BEFORE augmentation to prevent leakage)
df_train_orig, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['target'])

# --- Strategy 1: Mirror augmentation ONLY on training set ---
print("Applying mirror symmetry data augmentation (training set only)...")
df_train_flipped = df_train_orig.copy()
df_train_flipped[['ally_1', 'ally_2', 'ally_3', 'enemy_1', 'enemy_2', 'enemy_3']] = \
    df_train_orig[['enemy_1', 'enemy_2', 'enemy_3', 'ally_1', 'ally_2', 'ally_3']].values
df_train_flipped['target'] = 1 - df_train_flipped['target']
df_train = pd.concat([df_train_orig, df_train_flipped], ignore_index=True)
print(f"  Training set: {len(df_train_orig)} -> {len(df_train)} (augmented)")
print(f"  Val set: {len(df_val)} | Test set: {len(df_test)} (clean, no augmentation)")

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# 4. Advanced ML Features (Map compatibility, Synergy, Counters)
print("Calculating advanced features (Map compatibility, Synergy, Counters)...")
start_time = time.time()

# Helper to calculate stats on a specific subset of data
def compute_stats_for_subset(df_sub):
    weights = df_sub['rank_weight'].values
    targets = df_sub['target'].values
    
    map_brawler = {}
    mode_brawler = {}
    synergy = {}
    trio = {}
    counter = {}
    brawler_base = {}
    
    for i, row in enumerate(df_sub.itertuples()):
        w = weights[i]
        t = targets[i]
        m = row.map
        g_mode = row.mode
        allies = [row.ally_1, row.ally_2, row.ally_3]
        enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
        
        # Map-Brawler
        for ally in allies:
            brawler_base[ally] = brawler_base.get(ally, 0.0) + w * t
            brawler_base[ally + '_total'] = brawler_base.get(ally + '_total', 0.0) + w
            key = (m, ally)
            map_brawler[key] = map_brawler.get(key, 0.0) + w * t
            map_brawler[key + ('total',)] = map_brawler.get(key + ('total',), 0.0) + w
        for enemy in enemies:
            brawler_base[enemy] = brawler_base.get(enemy, 0.0) + w * (1.0 - t)
            brawler_base[enemy + '_total'] = brawler_base.get(enemy + '_total', 0.0) + w
            key = (m, enemy)
            map_brawler[key] = map_brawler.get(key, 0.0) + w * (1.0 - t)
            map_brawler[key + ('total',)] = map_brawler.get(key + ('total',), 0.0) + w
            
        # Mode-Brawler
        for ally in allies:
            key = (g_mode, ally)
            mode_brawler[key] = mode_brawler.get(key, 0.0) + w * t
            mode_brawler[key + ('total',)] = mode_brawler.get(key + ('total',), 0.0) + w
        for enemy in enemies:
            key = (g_mode, enemy)
            mode_brawler[key] = mode_brawler.get(key, 0.0) + w * (1.0 - t)
            mode_brawler[key + ('total',)] = mode_brawler.get(key + ('total',), 0.0) + w
            
        # Synergy
        for a, b in [(allies[0], allies[1]), (allies[0], allies[2]), (allies[1], allies[2])]:
            pair = tuple(sorted([a, b]))
            synergy[pair] = synergy.get(pair, 0.0) + w * t
            synergy[pair + ('total',)] = synergy.get(pair + ('total',), 0.0) + w
        for a, b in [(enemies[0], enemies[1]), (enemies[0], enemies[2]), (enemies[1], enemies[2])]:
            pair = tuple(sorted([a, b]))
            synergy[pair] = synergy.get(pair, 0.0) + w * (1.0 - t)
            synergy[pair + ('total',)] = synergy.get(pair + ('total',), 0.0) + w
            
        # Trio Synergy
        a_trio = tuple(sorted(allies))
        trio[a_trio] = trio.get(a_trio, 0.0) + w * t
        trio[a_trio + ('total',)] = trio.get(a_trio + ('total',), 0.0) + w
        
        e_trio = tuple(sorted(enemies))
        trio[e_trio] = trio.get(e_trio, 0.0) + w * (1.0 - t)
        trio[e_trio + ('total',)] = trio.get(e_trio + ('total',), 0.0) + w
            
        # Counters
        for ally in allies:
            for enemy in enemies:
                key1 = (ally, enemy)
                counter[key1] = counter.get(key1, 0.0) + w * t
                counter[key1 + ('total',)] = counter.get(key1 + ('total',), 0.0) + w
                
                key2 = (enemy, ally)
                counter[key2] = counter.get(key2, 0.0) + w * (1.0 - t)
                counter[key2 + ('total',)] = counter.get(key2 + ('total',), 0.0) + w
                
    map_brawler_stats = {}
    for key in list(map_brawler.keys()):
        if isinstance(key, tuple) and len(key) == 2:
            win_val = map_brawler[key]
            tot_val = map_brawler[key + ('total',)]
            map_brawler_stats[key] = (win_val + 10.0 * 0.5) / (tot_val + 10.0)
            
    mode_brawler_stats = {}
    for key in list(mode_brawler.keys()):
        if isinstance(key, tuple) and len(key) == 2:
            win_val = mode_brawler[key]
            tot_val = mode_brawler[key + ('total',)]
            mode_brawler_stats[key] = (win_val + 10.0 * 0.5) / (tot_val + 10.0)
            
    synergy_stats = {}
    for key in list(synergy.keys()):
        if isinstance(key, tuple) and len(key) == 2:
            win_val = synergy[key]
            tot_val = synergy[key + ('total',)]
            synergy_stats[key] = (win_val + 15.0 * 0.5) / (tot_val + 15.0)
            
    trio_stats = {}
    for key in list(trio.keys()):
        if isinstance(key, tuple) and len(key) == 3:
            win_val = trio[key]
            tot_val = trio[key + ('total',)]
            # Stronger bayesian prior for trios
            trio_stats[key] = (win_val + 30.0 * 0.5) / (tot_val + 30.0)
            
    counter_stats = {}
    for key in list(counter.keys()):
        if isinstance(key, tuple) and len(key) == 2:
            win_val = counter[key]
            tot_val = counter[key + ('total',)]
            counter_stats[key] = (win_val + 15.0 * 0.5) / (tot_val + 15.0)
            
    brawler_stats = {}
    for key in list(brawler_base.keys()):
        if isinstance(key, str) and not key.endswith('_total'):
            win_val = brawler_base[key]
            tot_val = brawler_base[key + '_total']
            brawler_stats[key] = (win_val + 50.0 * 0.5) / (tot_val + 50.0)
            
    return brawler_stats, map_brawler_stats, mode_brawler_stats, synergy_stats, trio_stats, counter_stats

# Main wrapper to calculate lookup stats with ELO splits
def calculate_lookup_stats(df_subset):
    c_base, c_map, c_mode, c_syn, c_trio, c_cnt = compute_stats_for_subset(df_subset)
    
    df_low = df_subset[df_subset['peak_rank'] < 5000]
    if len(df_low) > 0:
        l_base, l_map, l_mode, l_syn, l_trio, l_cnt = compute_stats_for_subset(df_low)
    else:
        l_base, l_map, l_mode, l_syn, l_trio, l_cnt = {}, {}, {}, {}, {}, {}
        
    df_high = df_subset[df_subset['peak_rank'] >= 5000]
    if len(df_high) > 0:
        h_base, h_map, h_mode, h_syn, h_trio, h_cnt = compute_stats_for_subset(df_high)
    else:
        h_base, h_map, h_mode, h_syn, h_trio, h_cnt = {}, {}, {}, {}, {}, {}
        
    return {
        'combined': {'base': c_base, 'map': c_map, 'mode': c_mode, 'synergy': c_syn, 'trio': c_trio, 'counter': c_cnt},
        'low': {'base': l_base, 'map': l_map, 'mode': l_mode, 'synergy': l_syn, 'trio': l_trio, 'counter': l_cnt},
        'high': {'base': h_base, 'map': h_map, 'mode': h_mode, 'synergy': h_syn, 'trio': h_trio, 'counter': h_cnt}
    }

def get_split_stat(stats_package, split_key, stat_type, key, default=0.5):
    val = stats_package.get(split_key, {}).get(stat_type, {}).get(key)
    if val is not None:
        return val
    return stats_package.get('combined', {}).get(stat_type, {}).get(key, default)

def compute_advanced_features(df_to_update, stats_package):
    ally_map_scores = []
    enemy_map_scores = []
    ally_mode_scores = []
    enemy_mode_scores = []
    ally_synergy_scores = []
    enemy_synergy_scores = []
    ally_counter_scores = []
    enemy_counter_scores = []
    ally_max_counter_scores = []
    enemy_max_counter_scores = []
    ally_min_map_scores = []
    enemy_min_map_scores = []
    ally_base_winrates = []
    enemy_base_winrates = []
    ally_trio_scores = []
    enemy_trio_scores = []
    ally_role_diversities = []
    enemy_role_diversities = []
    # --- Strategy 4: New feature lists ---
    ally_counter_stds = []
    enemy_counter_stds = []
    ally_counter_mins = []
    enemy_counter_mins = []
    ally_pick_rates = []
    enemy_pick_rates = []
    comp_double_tank = []
    comp_no_support = []
    comp_triple_aggro = []
    comp_enemy_double_tank = []
    comp_enemy_no_support = []
    comp_enemy_triple_aggro = []
    
    for row in df_to_update.itertuples():
        m = row.map
        g_mode = row.mode
        pr = row.peak_rank
        split_key = 'low' if pr < 5000 else 'high'
        
        allies = [row.ally_1, row.ally_2, row.ally_3]
        enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
        
        # Map score
        a_map_scores = [get_split_stat(stats_package, split_key, 'map', (m, b), 0.5) for b in allies]
        e_map_scores = [get_split_stat(stats_package, split_key, 'map', (m, b), 0.5) for b in enemies]
        ally_map_scores.append(sum(a_map_scores))
        enemy_map_scores.append(sum(e_map_scores))
        ally_min_map_scores.append(min(a_map_scores))
        enemy_min_map_scores.append(min(e_map_scores))
        
        # Mode score
        a_mode = sum(get_split_stat(stats_package, split_key, 'mode', (g_mode, b), 0.5) for b in allies)
        e_mode = sum(get_split_stat(stats_package, split_key, 'mode', (g_mode, b), 0.5) for b in enemies)
        ally_mode_scores.append(a_mode)
        enemy_mode_scores.append(e_mode)
        
        # Synergy score
        a_syn = sum(get_split_stat(stats_package, split_key, 'synergy', tuple(sorted([allies[i], allies[j]])), 0.5) for i, j in [(0, 1), (0, 2), (1, 2)])
        e_syn = sum(get_split_stat(stats_package, split_key, 'synergy', tuple(sorted([enemies[i], enemies[j]])), 0.5) for i, j in [(0, 1), (0, 2), (1, 2)])
        ally_synergy_scores.append(a_syn)
        enemy_synergy_scores.append(e_syn)
        
        # Counter score
        a_cnt_scores = [get_split_stat(stats_package, split_key, 'counter', (ally, enemy), 0.5) for ally in allies for enemy in enemies]
        e_cnt_scores = [get_split_stat(stats_package, split_key, 'counter', (enemy, ally), 0.5) for enemy in enemies for ally in allies]
        ally_counter_scores.append(sum(a_cnt_scores))
        enemy_counter_scores.append(sum(e_cnt_scores))
        ally_max_counter_scores.append(max(a_cnt_scores) if a_cnt_scores else 0.5)
        enemy_max_counter_scores.append(max(e_cnt_scores) if e_cnt_scores else 0.5)
        
        # Trio score
        a_trio_val = get_split_stat(stats_package, split_key, 'trio', tuple(sorted(allies)), 0.5)
        e_trio_val = get_split_stat(stats_package, split_key, 'trio', tuple(sorted(enemies)), 0.5)
        ally_trio_scores.append(a_trio_val)
        enemy_trio_scores.append(e_trio_val)
        
        # Base winrate
        ally_base_winrates.append(sum([get_split_stat(stats_package, split_key, 'base', b, 0.5) for b in allies]))
        enemy_base_winrates.append(sum([get_split_stat(stats_package, split_key, 'base', b, 0.5) for b in enemies]))
        
        # Role diversity
        ally_roles = set([get_role(b)[0] for b in allies])
        enemy_roles = set([get_role(b)[0] for b in enemies])
        ally_role_diversities.append(len(ally_roles))
        enemy_role_diversities.append(len(enemy_roles))
        
        # --- Strategy 4: New features ---
        # Counter std/min
        ally_counter_stds.append(np.std(a_cnt_scores) if len(a_cnt_scores) > 1 else 0.0)
        enemy_counter_stds.append(np.std(e_cnt_scores) if len(e_cnt_scores) > 1 else 0.0)
        ally_counter_mins.append(min(a_cnt_scores) if a_cnt_scores else 0.5)
        enemy_counter_mins.append(min(e_cnt_scores) if e_cnt_scores else 0.5)
        
        # Pick rate features
        ally_pick_rates.append(np.mean([brawler_pick_rates.get(b, 0.0) for b in allies]))
        enemy_pick_rates.append(np.mean([brawler_pick_rates.get(b, 0.0) for b in enemies]))
        
        # Composition pattern features
        a_role_list = [get_role(b)[0] for b in allies]
        e_role_list = [get_role(b)[0] for b in enemies]
        a_sec_roles = [get_role(b)[1] for b in allies]
        e_sec_roles = [get_role(b)[1] for b in enemies]
        
        a_tank_count = sum(1 for r in a_role_list if r == 'tank') + sum(1 for r in a_sec_roles if r == 'tank')
        a_support_count = sum(1 for r in a_role_list if r == 'support') + sum(1 for r in a_sec_roles if r == 'support')
        a_aggro_count = sum(1 for r in a_role_list if r == 'aggro') + sum(1 for r in a_sec_roles if r == 'aggro')
        
        e_tank_count = sum(1 for r in e_role_list if r == 'tank') + sum(1 for r in e_sec_roles if r == 'tank')
        e_support_count = sum(1 for r in e_role_list if r == 'support') + sum(1 for r in e_sec_roles if r == 'support')
        e_aggro_count = sum(1 for r in e_role_list if r == 'aggro') + sum(1 for r in e_sec_roles if r == 'aggro')
        
        comp_double_tank.append(1 if a_tank_count >= 2 else 0)
        comp_no_support.append(1 if a_support_count == 0 else 0)
        comp_triple_aggro.append(1 if a_aggro_count >= 3 else 0)
        comp_enemy_double_tank.append(1 if e_tank_count >= 2 else 0)
        comp_enemy_no_support.append(1 if e_support_count == 0 else 0)
        comp_enemy_triple_aggro.append(1 if e_aggro_count >= 3 else 0)
        
    df_to_update['ally_map_score'] = ally_map_scores
    df_to_update['enemy_map_score'] = enemy_map_scores
    df_to_update['map_score_advantage'] = np.array(ally_map_scores) - np.array(enemy_map_scores)
    
    df_to_update['ally_mode_score'] = ally_mode_scores
    df_to_update['enemy_mode_score'] = enemy_mode_scores
    df_to_update['mode_score_advantage'] = np.array(ally_mode_scores) - np.array(enemy_mode_scores)
    
    df_to_update['ally_synergy_score'] = ally_synergy_scores
    df_to_update['enemy_synergy_score'] = enemy_synergy_scores
    df_to_update['synergy_advantage'] = np.array(ally_synergy_scores) - np.array(enemy_synergy_scores)
    
    df_to_update['ally_counter_score'] = ally_counter_scores
    df_to_update['enemy_counter_score'] = enemy_counter_scores
    df_to_update['counter_advantage'] = np.array(ally_counter_scores) - np.array(enemy_counter_scores)
    
    df_to_update['ally_trio_score'] = ally_trio_scores
    df_to_update['enemy_trio_score'] = enemy_trio_scores
    df_to_update['trio_advantage'] = np.array(ally_trio_scores) - np.array(enemy_trio_scores)
    
    df_to_update['ally_role_diversity'] = ally_role_diversities
    df_to_update['enemy_role_diversity'] = enemy_role_diversities
    df_to_update['role_diversity_advantage'] = np.array(ally_role_diversities) - np.array(enemy_role_diversities)
    
    df_to_update['ally_min_map_score'] = ally_min_map_scores
    df_to_update['enemy_min_map_score'] = enemy_min_map_scores
    df_to_update['min_map_score_advantage'] = np.array(ally_min_map_scores) - np.array(enemy_min_map_scores)
    
    df_to_update['ally_max_counter_score'] = ally_max_counter_scores
    df_to_update['enemy_max_counter_score'] = enemy_max_counter_scores
    df_to_update['max_counter_advantage'] = np.array(ally_max_counter_scores) - np.array(enemy_max_counter_scores)
    
    df_to_update['ally_base_winrate'] = ally_base_winrates
    df_to_update['enemy_base_winrate'] = enemy_base_winrates
    df_to_update['base_winrate_advantage'] = np.array(ally_base_winrates) - np.array(enemy_base_winrates)
    
    # --- Strategy 4: Assign new features ---
    df_to_update['ally_counter_std'] = ally_counter_stds
    df_to_update['enemy_counter_std'] = enemy_counter_stds
    df_to_update['counter_std_advantage'] = np.array(ally_counter_stds) - np.array(enemy_counter_stds)
    
    df_to_update['ally_counter_min'] = ally_counter_mins
    df_to_update['enemy_counter_min'] = enemy_counter_mins
    df_to_update['counter_min_advantage'] = np.array(ally_counter_mins) - np.array(enemy_counter_mins)
    
    df_to_update['ally_avg_pick_rate'] = ally_pick_rates
    df_to_update['enemy_avg_pick_rate'] = enemy_pick_rates
    df_to_update['pick_rate_advantage'] = np.array(ally_pick_rates) - np.array(enemy_pick_rates)
    
    df_to_update['comp_double_tank'] = comp_double_tank
    df_to_update['comp_no_support'] = comp_no_support
    df_to_update['comp_triple_aggro'] = comp_triple_aggro
    df_to_update['comp_enemy_double_tank'] = comp_enemy_double_tank
    df_to_update['comp_enemy_no_support'] = comp_enemy_no_support
    df_to_update['comp_enemy_triple_aggro'] = comp_enemy_triple_aggro

# Calculate OOF features on train set
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in [
    'ally_map_score', 'enemy_map_score', 'map_score_advantage',
    'ally_mode_score', 'enemy_mode_score', 'mode_score_advantage',
    'ally_synergy_score', 'enemy_synergy_score', 'synergy_advantage',
    'ally_counter_score', 'enemy_counter_score', 'counter_advantage',
    'ally_trio_score', 'enemy_trio_score', 'trio_advantage',
    'ally_role_diversity', 'enemy_role_diversity', 'role_diversity_advantage',
    'ally_min_map_score', 'enemy_min_map_score', 'min_map_score_advantage',
    'ally_max_counter_score', 'enemy_max_counter_score', 'max_counter_advantage',
    'ally_base_winrate', 'enemy_base_winrate', 'base_winrate_advantage',
    'ally_counter_std', 'enemy_counter_std', 'counter_std_advantage',
    'ally_counter_min', 'enemy_counter_min', 'counter_min_advantage',
    'ally_avg_pick_rate', 'enemy_avg_pick_rate', 'pick_rate_advantage',
    'comp_double_tank', 'comp_no_support', 'comp_triple_aggro',
    'comp_enemy_double_tank', 'comp_enemy_no_support', 'comp_enemy_triple_aggro'
]:
    df_train[col] = 0.0
    df_val[col] = 0.0
    df_test[col] = 0.0

print("Computing Out-Of-Fold features...")
for train_idx, val_idx in skf.split(df_train, df_train['target']):
    df_trn_fold = df_train.iloc[train_idx]
    df_val_fold = df_train.iloc[val_idx].copy()
    
    stats_package = calculate_lookup_stats(df_trn_fold)
    compute_advanced_features(df_val_fold, stats_package)
    
    df_train.iloc[val_idx] = df_val_fold

# Now compute final global lookup stats on whole df_train
global_stats = calculate_lookup_stats(df_train)

# Calculate features for val and test sets using these global stats
compute_advanced_features(df_val, global_stats)
compute_advanced_features(df_test, global_stats)

# 5. Base feature encoding (one-hot maps, roles, rank, draft state, tiers)
def encode_base_features(df_subset, brawler_list, brawler_to_idx, role_feature_names, role_to_idx, map_list):
    draft_features = np.zeros((len(df_subset), len(brawler_list)), dtype=np.int8)
    role_features = np.zeros((len(df_subset), len(role_feature_names)), dtype=np.float32)
    tier_features = np.zeros((len(df_subset), 3), dtype=np.float32)
    
    for i, row in enumerate(df_subset.itertuples()):
        allies = [row.ally_1, row.ally_2, row.ally_3]
        enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
        
        for ally in allies:
            if ally in brawler_to_idx:
                draft_features[i, brawler_to_idx[ally]] = 1
        for enemy in enemies:
            if enemy in brawler_to_idx:
                draft_features[i, brawler_to_idx[enemy]] = -1
        
        for ally in allies:
            primary, secondary = get_role(ally)
            if primary in role_to_idx:
                role_features[i, role_to_idx[primary]['ally']] += 1
            if secondary and secondary in role_to_idx:
                role_features[i, role_to_idx[secondary]['ally']] += 0.5
        
        for enemy in enemies:
            primary, secondary = get_role(enemy)
            if primary in role_to_idx:
                role_features[i, role_to_idx[primary]['enemy']] += 1
            if secondary and secondary in role_to_idx:
                role_features[i, role_to_idx[secondary]['enemy']] += 0.5
        
        for role in ALL_ROLES:
            idx = role_to_idx[role]
            role_features[i, idx['advantage']] = role_features[i, idx['ally']] - role_features[i, idx['enemy']]
            
        # Tier score features
        ally_tier = sum(get_tier(b) for b in allies)
        enemy_tier = sum(get_tier(b) for b in enemies)
        tier_features[i, 0] = ally_tier
        tier_features[i, 1] = enemy_tier
        tier_features[i, 2] = ally_tier - enemy_tier
        
    # Mechanic counts and interaction features
    mechanic_features = np.zeros((len(df_subset), 4), dtype=np.float32)
    range_map_features = np.zeros((len(df_subset), 2), dtype=np.float32)
    for i, row in enumerate(df_subset.itertuples()):
        allies = [row.ally_1, row.ally_2, row.ally_3]
        enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
        m = row.map
        
        def get_team_attrs(team):
            pois = sum(1 for b in team if get_mechanics(b).get('poison'))
            sil = sum(1 for b in team if get_mechanics(b).get('silence'))
            div = sum(1 for b in team if get_mechanics(b).get('diver'))
            spw = sum(1 for b in team if get_mechanics(b).get('spawner'))
            hel = sum(1 for b in team if get_mechanics(b).get('healer'))
            snip = sum(1 for b in team if get_role(b)[0] == 'sniper' or get_role(b)[1] == 'sniper')
            tnk = sum(1 for b in team if get_role(b)[0] == 'tank' or get_role(b)[1] == 'tank')
            return pois, sil, div, spw, hel, snip, tnk
            
        a_pois, a_sil, a_div, a_spw, a_hel, a_snip, a_tnk = get_team_attrs(allies)
        e_pois, e_sil, e_div, e_spw, e_hel, e_snip, e_tnk = get_team_attrs(enemies)
        
        mechanic_features[i, 0] = a_div * e_sil
        mechanic_features[i, 1] = a_tnk * e_pois
        mechanic_features[i, 2] = a_spw * e_snip
        mechanic_features[i, 3] = a_pois * e_hel - e_pois * a_hel
        
        # Range/Openness features
        is_open = 1.0 if is_open_map(m) else 0.0
        is_closed = 1.0 if is_closed_map(m) else 0.0
        
        ally_short_count = sum(1 for b in allies if get_range_class(b) == 'short')
        ally_long_count = sum(1 for b in allies if get_range_class(b) == 'long')
        
        range_map_features[i, 0] = ally_short_count * is_open
        range_map_features[i, 1] = ally_long_count * is_closed
            
    X_draft = pd.DataFrame(draft_features, columns=brawler_list)
    X_roles = pd.DataFrame(role_features, columns=role_feature_names)
    X_tiers = pd.DataFrame(tier_features, columns=['ally_tier_score', 'enemy_tier_score', 'tier_advantage'])
    X_mechs = pd.DataFrame(mechanic_features, columns=[
        'diver_vs_silence_penalty', 'tank_vs_poison_penalty', 
        'spawner_vs_sniper_bonus', 'poison_vs_healer_advantage'
    ])
    X_range_maps = pd.DataFrame(range_map_features, columns=[
        'short_range_on_open_map_penalty', 'long_range_on_closed_map_penalty'
    ])
    
    X_maps = pd.get_dummies(df_subset['map'], prefix='map')
    X_maps = X_maps.reindex(columns=map_list, fill_value=False).astype(float)
    
    X_rank = df_subset[['peak_rank']].copy().reset_index(drop=True)
    
    adv_cols = [
        'ally_map_score', 'enemy_map_score', 'map_score_advantage',
        'ally_mode_score', 'enemy_mode_score', 'mode_score_advantage',
        'ally_synergy_score', 'enemy_synergy_score', 'synergy_advantage',
        'ally_counter_score', 'enemy_counter_score', 'counter_advantage',
        'ally_trio_score', 'enemy_trio_score', 'trio_advantage',
        'ally_role_diversity', 'enemy_role_diversity', 'role_diversity_advantage',
        'ally_min_map_score', 'enemy_min_map_score', 'min_map_score_advantage',
        'ally_max_counter_score', 'enemy_max_counter_score', 'max_counter_advantage',
        'ally_base_winrate', 'enemy_base_winrate', 'base_winrate_advantage',
        'ally_counter_std', 'enemy_counter_std', 'counter_std_advantage',
        'ally_counter_min', 'enemy_counter_min', 'counter_min_advantage',
        'ally_avg_pick_rate', 'enemy_avg_pick_rate', 'pick_rate_advantage',
        'comp_double_tank', 'comp_no_support', 'comp_triple_aggro',
        'comp_enemy_double_tank', 'comp_enemy_no_support', 'comp_enemy_triple_aggro'
    ]
    X_adv = df_subset[adv_cols].copy().reset_index(drop=True)
    
    return pd.concat([
        X_maps.reset_index(drop=True), 
        X_draft.reset_index(drop=True), 
        X_roles.reset_index(drop=True), 
        X_rank, 
        X_adv, 
        X_tiers,
        X_mechs,
        X_range_maps
    ], axis=1)

print("Encoding feature matrices...")
X_train = encode_base_features(df_train, brawler_list, brawler_to_idx, role_feature_names, role_to_idx, map_list)
X_val = encode_base_features(df_val, brawler_list, brawler_to_idx, role_feature_names, role_to_idx, map_list)
X_test = encode_base_features(df_test, brawler_list, brawler_to_idx, role_feature_names, role_to_idx, map_list)

y_train = df_train['target'].values
y_val = df_val['target'].values
y_test = df_test['target'].values

print(f"Total features: {X_train.shape[1]}")
# 6. ELO + temporal weighting during model fit
elo_weight_train = df_train['rank_weight'].values
elo_weight_val = df_val['rank_weight'].values
elo_weight_test = df_test['rank_weight'].values

# Multiply by temporal weight (Strategy 2)
time_weight_train = df_train['time_weight'].values
time_weight_val = df_val['time_weight'].values

sample_weights_train = elo_weight_train * time_weight_train
sample_weights_val = elo_weight_val * time_weight_val
sample_weights_test = elo_weight_test  # Test set uses ELO weight only for fair evaluation
print(f"  Sample weight range (train): {sample_weights_train.min():.4f} - {sample_weights_train.max():.4f}")

# 7. Train tuned XGBoost and LightGBM Models using Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Running Optuna hyperparameter optimization for XGBoost...")
def xgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9)
    }
    
    tmp_model = xgb.XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=30,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        **params
    )
    tmp_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[sample_weights_val],
        verbose=False
    )
    return tmp_model.best_score

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=20)
best_xgb_params = xgb_study.best_params
print(f"Best XGBoost Params: {best_xgb_params} (Logloss: {xgb_study.best_value:.5f})")

print("\nRunning Optuna hyperparameter optimization for LightGBM...")
def lgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9)
    }
    
    tmp_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        **params
    )
    tmp_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sample_weights_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    score = tmp_model.evals_result_['valid_0']['binary_logloss'][tmp_model.best_iteration_ - 1]
    return score

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=20)
best_lgb_params = lgb_study.best_params
print(f"Best LightGBM Params: {best_lgb_params} (Logloss: {lgb_study.best_value:.5f})")

print("\nRunning Optuna hyperparameter optimization for CatBoost...")
def cb_objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'learning_rate': 0.05,
        'iterations': 1000,
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 30
    }
    
    tmp_model = cb.CatBoostClassifier(**params)
    tmp_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return tmp_model.get_best_score()['validation']['Logloss']

cb_study = optuna.create_study(direction='minimize')
cb_study.optimize(cb_objective, n_trials=20)
best_cb_params = cb_study.best_params
print(f"Best CatBoost Params: {best_cb_params} (Logloss: {cb_study.best_value:.5f})")

print("\nRunning Optuna hyperparameter optimization for Random Forest...")
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    tmp_model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    tmp_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    # Evaluate on validation set
    y_pred = tmp_model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import log_loss
    return log_loss(y_val, y_pred, sample_weight=sample_weights_val)

rf_study = optuna.create_study(direction='minimize')
rf_study.optimize(rf_objective, n_trials=10)
best_rf_params = rf_study.best_params
print(f"Best Random Forest Params: {best_rf_params} (Logloss: {rf_study.best_value:.5f})")

# Train final tuned models
print("\nTraining final optimized XGBoost model...")
model_xgb = xgb.XGBClassifier(
    n_estimators=2000,
    early_stopping_rounds=50,
    learning_rate=0.03,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    **best_xgb_params
)
model_xgb.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[sample_weights_val],
    verbose=50
)

print("\nTraining final optimized LightGBM model...")
model_lgb = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    random_state=42,
    verbose=-1,
    **best_lgb_params
)
model_lgb.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[sample_weights_val],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

print("\nTraining final optimized CatBoost model...")
best_cb_params['learning_rate'] = 0.03
best_cb_params['iterations'] = 2000
best_cb_params['eval_metric'] = 'Logloss'
best_cb_params['random_seed'] = 42
best_cb_params['early_stopping_rounds'] = 50
model_cb = cb.CatBoostClassifier(**best_cb_params)
model_cb.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print("\nTraining final optimized Random Forest model...")
model_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)
model_rf.fit(X_train, y_train, sample_weight=sample_weights_train)

# --- Strategy 3: Cross-Validated Meta-Model (OOF Stacking) ---
print("\nTraining Cross-Validated Meta-Model (OOF Stacking)...")
oof_preds = np.zeros((len(X_train), 4))  # 4 base models
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (trn_idx, val_idx) in enumerate(skf_meta.split(X_train, y_train)):
    X_trn_fold = X_train.iloc[trn_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_trn_fold = y_train[trn_idx]
    y_val_fold = y_train[val_idx]
    w_trn_fold = sample_weights_train.iloc[trn_idx] if hasattr(sample_weights_train, 'iloc') else sample_weights_train[trn_idx]
    
    # XGBoost
    tmp_xgb = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30, learning_rate=0.05,
                                  objective='binary:logistic', eval_metric='logloss', random_state=42, **best_xgb_params)
    tmp_xgb.fit(X_trn_fold, y_trn_fold, sample_weight=w_trn_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    oof_preds[val_idx, 0] = tmp_xgb.predict_proba(X_val_fold)[:, 1]
    
    # LightGBM
    tmp_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1, **best_lgb_params)
    tmp_lgb.fit(X_trn_fold, y_trn_fold, sample_weight=w_trn_fold, eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    oof_preds[val_idx, 1] = tmp_lgb.predict_proba(X_val_fold)[:, 1]
    
    # CatBoost
    tmp_cb_params = {**best_cb_params, 'learning_rate': 0.05, 'iterations': 1000, 'eval_metric': 'Logloss',
                     'random_seed': 42, 'early_stopping_rounds': 30, 'verbose': False}
    tmp_cb = cb.CatBoostClassifier(**tmp_cb_params)
    tmp_cb.fit(X_trn_fold, y_trn_fold, sample_weight=w_trn_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    oof_preds[val_idx, 2] = tmp_cb.predict_proba(X_val_fold)[:, 1]
    
    # Random Forest
    tmp_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)
    tmp_rf.fit(X_trn_fold, y_trn_fold, sample_weight=w_trn_fold)
    oof_preds[val_idx, 3] = tmp_rf.predict_proba(X_val_fold)[:, 1]
    
    print(f"  Fold {fold_idx+1}/5 complete")

# Train meta-model on OOF predictions (no leakage!)
meta_model = LogisticRegression(C=0.1, random_state=42)
meta_model.fit(oof_preds, y_train)

print(f"Meta-Model Coefficients (XGB/LGB/CB/RF): {meta_model.coef_[0]}")
print(f"Meta-Model Intercept: {meta_model.intercept_[0]}")

stacked_model = StackedModel(model_xgb, model_lgb, model_cb, model_rf, meta_model)

# 8. Evaluate Stacked Model on Test Set
predictions = stacked_model.predict(X_test)
predict_probs = stacked_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc_roc = roc_auc_score(y_test, predict_probs)

print(f"\n--- ENSEMBLE PERFORMANCE METRICS ---")
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print(f"AUC-ROC Score: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Defeat", "Victory"]))

print(f"\nData Processing Time: {time.time() - start_time:.2f} seconds")

# 9. Save pickle
with open('draft_model.pkl', 'wb') as f:
    pickle.dump({
        'model': stacked_model,
        'brawler_list': brawler_list,
        'map_list': map_list,
        'role_feature_names': role_feature_names,
        'median_rank': median_rank,
        'global_stats': global_stats,
        'map_to_mode': map_to_mode,
        'brawler_pick_rates': brawler_pick_rates
    }, f)

print("Ensemble Model successfully saved to draft_model.pkl")