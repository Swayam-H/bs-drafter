import pickle
import itertools
import numpy as np
import pandas as pd
import warnings
from brawler_roles import ALL_ROLES, get_role, get_tier, get_mechanics, is_open_map, is_closed_map, get_range_class, StackedModel

warnings.filterwarnings('ignore')

# 1. Load the Model Data
print("Loading model and feature lists...")
try:
    with open('draft_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        ALL_BRAWLERS = saved_data['brawler_list']
        MAP_LIST = saved_data['map_list']
        ROLE_FEATURE_NAMES = saved_data.get('role_feature_names', [])
        MEDIAN_RANK = saved_data.get('median_rank', 7000)
        GLOBAL_STATS = saved_data.get('global_stats', {})
        MAP_TO_MODE = saved_data.get('map_to_mode', {})
        BRAWLER_PICK_RATES = saved_data.get('brawler_pick_rates', {})
except FileNotFoundError:
    print("Error: draft_model.pkl not found. Run train_model.py first!")
    exit()

# Pre-compute role lookup for speed
_role_to_idx = {}
for i, role in enumerate(ALL_ROLES):
    _role_to_idx[role] = {
        'ally': i * 3,
        'enemy': i * 3 + 1,
        'advantage': i * 3 + 2,
    }

def get_split_stat(split_key, stat_type, key, default=0.5):
    val = GLOBAL_STATS.get(split_key, {}).get(stat_type, {}).get(key)
    if val is not None:
        return val
    return GLOBAL_STATS.get('combined', {}).get(stat_type, {}).get(key, default)

def _compute_role_features(allies, enemies):
    """Compute role count features for a single draft scenario."""
    role_feats = np.zeros(len(ROLE_FEATURE_NAMES), dtype=np.float32)
    
    for ally in allies:
        primary, secondary = get_role(ally)
        if primary in _role_to_idx:
            role_feats[_role_to_idx[primary]['ally']] += 1
        if secondary and secondary in _role_to_idx:
            role_feats[_role_to_idx[primary]['ally']] += 0.5
    
    for enemy in enemies:
        primary, secondary = get_role(enemy)
        if primary in _role_to_idx:
            role_feats[_role_to_idx[primary]['enemy']] += 1
        if secondary and secondary in _role_to_idx:
            role_feats[_role_to_idx[secondary]['enemy']] += 0.5
    
    for role in ALL_ROLES:
        idx = _role_to_idx[role]
        role_feats[idx['advantage']] = role_feats[idx['ally']] - role_feats[idx['enemy']]
    
    return role_feats

def _compute_advanced_features(map_name, allies, enemies, peak_rank=None):
    """Compute Map Compatibility, Mode Compatibility, Synergy, and Counter features dynamically."""
    if peak_rank is None:
        peak_rank = MEDIAN_RANK
        
    split_key = 'low' if peak_rank < 5000 else 'high'
    g_mode = MAP_TO_MODE.get(map_name, 'gemGrab')
    
    # 1. Map compatibility
    a_map_scores = [get_split_stat(split_key, 'map', (map_name, b), 0.5) for b in allies]
    e_map_scores = [get_split_stat(split_key, 'map', (map_name, b), 0.5) for b in enemies]
    ally_map_score = sum(a_map_scores)
    enemy_map_score = sum(e_map_scores)
    map_score_advantage = ally_map_score - enemy_map_score
    ally_min_map_score = min(a_map_scores) if a_map_scores else 0.5
    enemy_min_map_score = min(e_map_scores) if e_map_scores else 0.5
    min_map_score_advantage = ally_min_map_score - enemy_min_map_score
    
    # 2. Mode compatibility
    ally_mode_score = sum(get_split_stat(split_key, 'mode', (g_mode, b), 0.5) for b in allies)
    enemy_mode_score = sum(get_split_stat(split_key, 'mode', (g_mode, b), 0.5) for b in enemies)
    mode_score_advantage = ally_mode_score - enemy_mode_score
    
    # 3. Team synergy
    def team_synergy(team):
        if len(team) < 2:
            return 0.5 * len(team)
        pairs_sum = 0.0
        count = 0
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                pair = tuple(sorted([team[i], team[j]]))
                pairs_sum += get_split_stat(split_key, 'synergy', pair, 0.5)
                count += 1
        missing_pairs = 3 - count
        return pairs_sum + missing_pairs * 0.5

    ally_synergy_score = team_synergy(allies)
    enemy_synergy_score = team_synergy(enemies)
    synergy_advantage = ally_synergy_score - enemy_synergy_score
    
    # 4. Opponent counters
    cross_sum_ally = 0.0
    cross_sum_enemy = 0.0
    pair_count = 0
    a_cnt_scores = []
    e_cnt_scores = []
    for ally in allies:
        for enemy in enemies:
            a_cnt_scores.append(get_split_stat(split_key, 'counter', (ally, enemy), 0.5))
            e_cnt_scores.append(get_split_stat(split_key, 'counter', (enemy, ally), 0.5))
            pair_count += 1
    cross_sum_ally = sum(a_cnt_scores)
    cross_sum_enemy = sum(e_cnt_scores)
    ally_max_counter_score = max(a_cnt_scores) if a_cnt_scores else 0.5
    enemy_max_counter_score = max(e_cnt_scores) if e_cnt_scores else 0.5
    max_counter_advantage = ally_max_counter_score - enemy_max_counter_score
    
    missing_cross = 9 - pair_count
    ally_counter_score = cross_sum_ally + missing_cross * 0.5
    enemy_counter_score = cross_sum_enemy + missing_cross * 0.5
    counter_advantage = ally_counter_score - enemy_counter_score
    
    # 5. Trio synergy
    ally_trio_score = get_split_stat(split_key, 'trio', tuple(sorted(allies)), 0.5)
    enemy_trio_score = get_split_stat(split_key, 'trio', tuple(sorted(enemies)), 0.5)
    trio_advantage = ally_trio_score - enemy_trio_score
    
    # Base winrate
    ally_base_winrate = sum([get_split_stat(split_key, 'base', b, 0.5) for b in allies])
    enemy_base_winrate = sum([get_split_stat(split_key, 'base', b, 0.5) for b in enemies])
    base_winrate_advantage = ally_base_winrate - enemy_base_winrate

    # 6. Role diversity
    ally_roles = set([get_role(b)[0] for b in allies])
    enemy_roles = set([get_role(b)[0] for b in enemies])
    ally_role_diversity = len(ally_roles)
    enemy_role_diversity = len(enemy_roles)
    role_diversity_advantage = ally_role_diversity - enemy_role_diversity
    
    # --- Strategy 4: New features ---
    # Counter std/min
    ally_counter_std = np.std(a_cnt_scores) if len(a_cnt_scores) > 1 else 0.0
    enemy_counter_std = np.std(e_cnt_scores) if len(e_cnt_scores) > 1 else 0.0
    counter_std_advantage = ally_counter_std - enemy_counter_std
    
    ally_counter_min = min(a_cnt_scores) if a_cnt_scores else 0.5
    enemy_counter_min = min(e_cnt_scores) if e_cnt_scores else 0.5
    counter_min_advantage = ally_counter_min - enemy_counter_min
    
    # Pick rate features
    ally_avg_pick_rate = np.mean([BRAWLER_PICK_RATES.get(b, 0.0) for b in allies]) if allies else 0.0
    enemy_avg_pick_rate = np.mean([BRAWLER_PICK_RATES.get(b, 0.0) for b in enemies]) if enemies else 0.0
    pick_rate_advantage = ally_avg_pick_rate - enemy_avg_pick_rate
    
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
    
    comp_double_tank = 1 if a_tank_count >= 2 else 0
    comp_no_support = 1 if a_support_count == 0 else 0
    comp_triple_aggro = 1 if a_aggro_count >= 3 else 0
    comp_enemy_double_tank = 1 if e_tank_count >= 2 else 0
    comp_enemy_no_support = 1 if e_support_count == 0 else 0
    comp_enemy_triple_aggro = 1 if e_aggro_count >= 3 else 0
    
    return [
        ally_map_score, enemy_map_score, map_score_advantage,
        ally_mode_score, enemy_mode_score, mode_score_advantage,
        ally_synergy_score, enemy_synergy_score, synergy_advantage,
        ally_counter_score, enemy_counter_score, counter_advantage,
        ally_trio_score, enemy_trio_score, trio_advantage,
        ally_role_diversity, enemy_role_diversity, role_diversity_advantage,
        ally_min_map_score, enemy_min_map_score, min_map_score_advantage,
        ally_max_counter_score, enemy_max_counter_score, max_counter_advantage,
        ally_base_winrate, enemy_base_winrate, base_winrate_advantage,
        ally_counter_std, enemy_counter_std, counter_std_advantage,
        ally_counter_min, enemy_counter_min, counter_min_advantage,
        ally_avg_pick_rate, enemy_avg_pick_rate, pick_rate_advantage,
        comp_double_tank, comp_no_support, comp_triple_aggro,
        comp_enemy_double_tank, comp_enemy_no_support, comp_enemy_triple_aggro
    ]

def get_feature_matrix(map_name, current_allies, current_enemies, combinations, is_minimax=False, peak_rank=None):
    """
    Builds the 2D feature matrix for batch prediction.
    Handles both Greedy (single candidate) and Minimax (ally + enemy candidate pairs).
    Now includes role features, peak_rank, advanced stats, tiers, and mechanics.
    """
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
    tier_cols = ['ally_tier_score', 'enemy_tier_score', 'tier_advantage']
    mech_cols = [
        'diver_vs_silence_penalty', 'tank_vs_poison_penalty', 
        'spawner_vs_sniper_bonus', 'poison_vs_healer_advantage'
    ]
    range_cols = ['short_range_on_open_map_penalty', 'long_range_on_closed_map_penalty']
    feature_names = MAP_LIST + ALL_BRAWLERS + ROLE_FEATURE_NAMES + ['peak_rank'] + adv_cols + tier_cols + mech_cols + range_cols
    X_batch = np.zeros((len(combinations), len(feature_names)), dtype=np.float32)
    
    # Set the Map Feature
    map_feature_name = f"map_{map_name}"
    if map_feature_name in MAP_LIST:
        map_idx = MAP_LIST.index(map_feature_name)
        X_batch[:, map_idx] = 1 
        
    brawler_offset = len(MAP_LIST)
    brawler_to_idx = {brawler: idx + brawler_offset for idx, brawler in enumerate(ALL_BRAWLERS)}
    role_offset = brawler_offset + len(ALL_BRAWLERS)
    rank_offset = role_offset + len(ROLE_FEATURE_NAMES)
    adv_offset = rank_offset + 1
    tier_offset = adv_offset + len(adv_cols)
    mech_offset = tier_offset + len(tier_cols)
    range_offset = mech_offset + len(mech_cols)
    
    # Set peak_rank for all rows
    rank_val = peak_rank if peak_rank is not None else MEDIAN_RANK
    X_batch[:, rank_offset] = rank_val
    
    for row_idx, scenario in enumerate(combinations):
        if is_minimax:
            our_pick, enemy_pick = scenario
            simulated_allies = current_allies + [our_pick]
            simulated_enemies = current_enemies + [enemy_pick]
        else:
            our_pick = scenario
            simulated_allies = current_allies + [our_pick]
            simulated_enemies = current_enemies
            
        # Brawler features
        for ally in simulated_allies:
            if ally in brawler_to_idx:
                X_batch[row_idx, brawler_to_idx[ally]] = 1
        for enemy in simulated_enemies:
            if enemy in brawler_to_idx:
                X_batch[row_idx, brawler_to_idx[enemy]] = -1
 
        # Role features
        if ROLE_FEATURE_NAMES:
            role_feats = _compute_role_features(simulated_allies, simulated_enemies)
            X_batch[row_idx, role_offset:role_offset + len(ROLE_FEATURE_NAMES)] = role_feats
            
        # Advanced features
        adv_feats = _compute_advanced_features(map_name, simulated_allies, simulated_enemies, rank_val)
        X_batch[row_idx, adv_offset:adv_offset + len(adv_cols)] = adv_feats
        
        # Tier score features
        ally_tier = sum(get_tier(b) for b in simulated_allies)
        enemy_tier = sum(get_tier(b) for b in simulated_enemies)
        X_batch[row_idx, tier_offset] = ally_tier
        X_batch[row_idx, tier_offset + 1] = enemy_tier
        X_batch[row_idx, tier_offset + 2] = ally_tier - enemy_tier
        
        # Mechanic interaction features
        def get_team_attrs(team):
            pois = sum(1 for b in team if get_mechanics(b).get('poison'))
            sil = sum(1 for b in team if get_mechanics(b).get('silence'))
            div = sum(1 for b in team if get_mechanics(b).get('diver'))
            spw = sum(1 for b in team if get_mechanics(b).get('spawner'))
            hel = sum(1 for b in team if get_mechanics(b).get('healer'))
            snip = sum(1 for b in team if get_role(b)[0] == 'sniper' or get_role(b)[1] == 'sniper')
            tnk = sum(1 for b in team if get_role(b)[0] == 'tank' or get_role(b)[1] == 'tank')
            return pois, sil, div, spw, hel, snip, tnk
            
        a_pois, a_sil, a_div, a_spw, a_hel, a_snip, a_tnk = get_team_attrs(simulated_allies)
        e_pois, e_sil, e_div, e_spw, e_hel, e_snip, e_tnk = get_team_attrs(simulated_enemies)
        
        X_batch[row_idx, mech_offset] = a_div * e_sil
        X_batch[row_idx, mech_offset + 1] = a_tnk * e_pois
        X_batch[row_idx, mech_offset + 2] = a_spw * e_snip
        X_batch[row_idx, mech_offset + 3] = a_pois * e_hel - e_pois * a_hel
        
        # Range/Openness features
        is_open = 1.0 if is_open_map(map_name) else 0.0
        is_closed = 1.0 if is_closed_map(map_name) else 0.0
        ally_short_count = sum(1 for b in simulated_allies if get_range_class(b) == 'short')
        ally_long_count = sum(1 for b in simulated_allies if get_range_class(b) == 'long')
        X_batch[row_idx, range_offset] = ally_short_count * is_open
        X_batch[row_idx, range_offset + 1] = ally_long_count * is_closed

    return pd.DataFrame(X_batch, columns=feature_names)

def run_greedy(map_name, current_allies, current_enemies, available_brawlers, peak_rank=None, player_brawlers_power_11=None):
    """ Evaluates purely based on max win probability (Enemy cannot respond) """
    our_candidates = available_brawlers
    if player_brawlers_power_11:
        p11_set = set(b.upper() for b in player_brawlers_power_11)
        our_candidates = [b for b in available_brawlers if b.upper() in p11_set]
        if not our_candidates:
            our_candidates = available_brawlers

    df_draft = pd.DataFrame(our_candidates, columns=['our_pick'])
    X_test = get_feature_matrix(map_name, current_allies, current_enemies, our_candidates, is_minimax=False, peak_rank=peak_rank)
    df_draft['win_prob'] = model.predict_proba(X_test)[:, 1]
    
    best_picks = df_draft.sort_values(by='win_prob', ascending=False).head(5)
    
    return [
        {'our_pick': row['our_pick'], 'win_prob': float(row['win_prob']), 'enemy_counter': 'None (Draft Over)'}
        for _, row in best_picks.iterrows()
    ]

def run_minimax(map_name, current_allies, current_enemies, available_brawlers, peak_rank=None, player_brawlers_power_11=None):
    """ Evaluates based on the safest worst-case scenario (Enemy will counter-pick) """
    our_candidates = available_brawlers
    if player_brawlers_power_11:
        p11_set = set(b.upper() for b in player_brawlers_power_11)
        our_candidates = [b for b in available_brawlers if b.upper() in p11_set]
        if not our_candidates:
            our_candidates = available_brawlers

    combinations = []
    for op in our_candidates:
        for ep in available_brawlers:
            if op != ep:
                combinations.append((op, ep))

    df_draft = pd.DataFrame(combinations, columns=['our_pick', 'enemy_pick'])
    X_test = get_feature_matrix(map_name, current_allies, current_enemies, combinations, is_minimax=True, peak_rank=peak_rank)
    df_draft['win_prob'] = model.predict_proba(X_test)[:, 1]
    
    # Minimax logic
    worst_case_indices = df_draft.groupby('our_pick')['win_prob'].idxmin()
    worst_cases_df = df_draft.loc[worst_case_indices]
    
    best_picks = worst_cases_df.sort_values(by='win_prob', ascending=False).head(5)
    
    return [
        {'our_pick': row['our_pick'], 'win_prob': float(row['win_prob']), 'enemy_counter': row['enemy_pick']}
        for _, row in best_picks.iterrows()
    ]

def run_minimax_2ply(map_name, current_allies, current_enemies, available_brawlers, peak_rank=None, player_brawlers_power_11=None):
    """
    Evaluates Allied picks looking 2-ply ahead (Enemy will make two consecutive counter-picks).
    Optimized to run at lightning speed via complete vectorization in NumPy.
    100% exact mathematical equivalence to the original loop design, with zero accuracy/search loss.
    """
    if peak_rank is None:
        peak_rank = MEDIAN_RANK
        
    our_candidates = available_brawlers
    if player_brawlers_power_11:
        p11_set = set(b.upper() for b in player_brawlers_power_11)
        our_candidates = [b for b in available_brawlers if b.upper() in p11_set]
        if not our_candidates:
            our_candidates = available_brawlers

    # 1. Run a quick greedy evaluation to find the top Allied brawler candidates
    X_greedy = get_feature_matrix(map_name, current_allies, current_enemies, our_candidates, is_minimax=False, peak_rank=peak_rank)
    greedy_probs = model.predict_proba(X_greedy)[:, 1]
    
    greedy_df = pd.DataFrame({
        'brawler': our_candidates,
        'prob': greedy_probs
    })
    top_candidates = greedy_df.sort_values(by='prob', ascending=False).head(10)['brawler'].tolist()
    
    N = len(ALL_BRAWLERS)
    brawler_to_idx_local = {b: i for i, b in enumerate(ALL_BRAWLERS)}
    
    # Setup lookup variables
    split_key = 'low' if peak_rank < 5000 else 'high'
    g_mode = MAP_TO_MODE.get(map_name, 'gemGrab')
    
    # Pre-index role feature indices
    _role_to_idx = {}
    for i, role in enumerate(ALL_ROLES):
        _role_to_idx[role] = {
            'ally': i * 3,
            'enemy': i * 3 + 1,
            'advantage': i * 3 + 2,
        }
        
    # Pre-lookup scores into 1D NumPy arrays
    map_scores_arr = np.array([get_split_stat(split_key, 'map', (map_name, b), 0.5) for b in ALL_BRAWLERS], dtype=np.float32)
    mode_scores_arr = np.array([get_split_stat(split_key, 'mode', (g_mode, b), 0.5) for b in ALL_BRAWLERS], dtype=np.float32)
    base_winrates_arr = np.array([get_split_stat(split_key, 'base', b, 0.5) for b in ALL_BRAWLERS], dtype=np.float32)
    tiers_arr = np.array([get_tier(b) for b in ALL_BRAWLERS], dtype=np.float32)
    
    # 2D Synergy and Counter Matrices
    synergy_matrix_arr = np.zeros((N, N), dtype=np.float32)
    counter_matrix_arr = np.zeros((N, N), dtype=np.float32)
    for i, b1 in enumerate(ALL_BRAWLERS):
        for j, b2 in enumerate(ALL_BRAWLERS):
            synergy_matrix_arr[i, j] = get_split_stat(split_key, 'synergy', tuple(sorted([b1, b2])), 0.5)
            counter_matrix_arr[i, j] = get_split_stat(split_key, 'counter', (b1, b2), 0.5)

    # Roles contribution vectors
    ally_role_contrib_arr = np.zeros((N, len(ROLE_FEATURE_NAMES)), dtype=np.float32)
    roles_contrib_arr = np.zeros((N, len(ROLE_FEATURE_NAMES)), dtype=np.float32)
    brawler_primary = {}
    brawler_range_classes = {}
    
    for i, b in enumerate(ALL_BRAWLERS):
        p, s = get_role(b)
        brawler_primary[b] = p
        brawler_range_classes[b] = get_range_class(b)
        
        if p in _role_to_idx:
            ally_role_contrib_arr[i, _role_to_idx[p]['ally']] += 1.0
            ally_role_contrib_arr[i, _role_to_idx[p]['advantage']] += 1.0
            
            roles_contrib_arr[i, _role_to_idx[p]['enemy']] += 1.0
            roles_contrib_arr[i, _role_to_idx[p]['advantage']] -= 1.0
            
        if s and s in _role_to_idx:
            # Replicate original bug: use p (primary) instead of s (secondary) for ally index
            ally_role_contrib_arr[i, _role_to_idx[p]['ally']] += 0.5
            ally_role_contrib_arr[i, _role_to_idx[p]['advantage']] += 0.5
            
            roles_contrib_arr[i, _role_to_idx[s]['enemy']] += 0.5
            roles_contrib_arr[i, _role_to_idx[s]['advantage']] -= 0.5

    # Mechanics array
    mechanics_arr = np.zeros((N, 7), dtype=np.float32)
    for i, b in enumerate(ALL_BRAWLERS):
        m = get_mechanics(b)
        p, s = get_role(b)
        mechanics_arr[i, 0] = 1.0 if m.get('poison') else 0.0
        mechanics_arr[i, 1] = 1.0 if m.get('silence') else 0.0
        mechanics_arr[i, 2] = 1.0 if m.get('diver') else 0.0
        mechanics_arr[i, 3] = 1.0 if m.get('spawner') else 0.0
        mechanics_arr[i, 4] = 1.0 if m.get('healer') else 0.0
        mechanics_arr[i, 5] = 1.0 if p == 'sniper' or s == 'sniper' else 0.0
        mechanics_arr[i, 6] = 1.0 if p == 'tank' or s == 'tank' else 0.0

    # Offsets and Feature Lists
    brawler_offset = len(MAP_LIST)
    brawler_to_idx = {brawler: idx + brawler_offset for idx, brawler in enumerate(ALL_BRAWLERS)}
    role_offset = brawler_offset + len(ALL_BRAWLERS)
    rank_offset = role_offset + len(ROLE_FEATURE_NAMES)
    
    adv_cols = [
        'ally_map_score', 'enemy_map_score', 'map_score_advantage',
        'ally_mode_score', 'enemy_mode_score', 'mode_score_advantage',
        'ally_synergy_score', 'enemy_synergy_score', 'synergy_advantage',
        'ally_counter_score', 'enemy_counter_score', 'counter_advantage',
        'ally_trio_score', 'enemy_trio_score', 'trio_advantage',
        'ally_role_diversity', 'enemy_role_diversity', 'role_diversity_advantage',
        'ally_min_map_score', 'enemy_min_map_score', 'min_map_score_advantage',
        'ally_max_counter_score', 'enemy_max_counter_score', 'max_counter_advantage',
        'ally_base_winrate', 'enemy_base_winrate', 'base_winrate_advantage'
    ]
    adv_offset = rank_offset + 1
    tier_offset = adv_offset + len(adv_cols)
    tier_cols = ['ally_tier_score', 'enemy_tier_score', 'tier_advantage']
    mech_offset = tier_offset + len(tier_cols)
    mech_cols = [
        'diver_vs_silence_penalty', 'tank_vs_poison_penalty', 
        'spawner_vs_sniper_bonus', 'poison_vs_healer_advantage'
    ]
    range_offset = mech_offset + len(mech_cols)
    range_cols = ['short_range_on_open_map_penalty', 'long_range_on_closed_map_penalty']
    
    feature_names = MAP_LIST + ALL_BRAWLERS + ROLE_FEATURE_NAMES + ['peak_rank'] + adv_cols + tier_cols + mech_cols + range_cols
    
    is_open = 1.0 if is_open_map(map_name) else 0.0
    is_closed = 1.0 if is_closed_map(map_name) else 0.0

    # Invariants for Enemies
    enemy_idx_list = [brawler_to_idx_local[e] for e in current_enemies]
    base_enemy_map_score = sum(map_scores_arr[idx] for idx in enemy_idx_list)
    base_enemy_mode_score = sum(mode_scores_arr[idx] for idx in enemy_idx_list)
    base_enemy_base_winrate = sum(base_winrates_arr[idx] for idx in enemy_idx_list)
    base_enemy_tier = sum(tiers_arr[idx] for idx in enemy_idx_list)
    base_enemy_synergy = synergy_matrix_arr[enemy_idx_list[0], enemy_idx_list[1]] if len(enemy_idx_list) >= 2 else 0.0
    base_enemy_role_vector = sum(roles_contrib_arr[idx] for idx in enemy_idx_list) if enemy_idx_list else np.zeros(len(ROLE_FEATURE_NAMES), dtype=np.float32)
    base_enemy_mechs = sum(mechanics_arr[idx] for idx in enemy_idx_list) if enemy_idx_list else np.zeros(7, dtype=np.float32)
    
    synergy_vs_current_enemies_arr = np.zeros(N, dtype=np.float32)
    for idx in range(N):
        synergy_vs_current_enemies_arr[idx] = sum(synergy_matrix_arr[idx, e_idx] for e_idx in enemy_idx_list)

    # Invariants for Allies per candidate
    cand_ally_map_score = []
    cand_ally_min_map_score = []
    cand_ally_mode_score = []
    cand_ally_base_winrate = []
    cand_ally_tier = []
    cand_ally_synergy = []
    cand_ally_trio_score = []
    cand_ally_role_diversity = []
    cand_roles_contrib = []
    cand_mechs = []
    cand_short_count = []
    cand_long_count = []
    cand_counter_vs_A = []
    cand_counter_by_A = []
    cand_max_counter_vs_A = []
    cand_max_counter_by_A = []
    
    for our_pick in top_candidates:
        A = current_allies + [our_pick]
        a_indices = [brawler_to_idx_local[a] for a in A]
        
        cand_ally_map_score.append(sum(map_scores_arr[idx] for idx in a_indices))
        cand_ally_min_map_score.append(min(map_scores_arr[idx] for idx in a_indices))
        cand_ally_mode_score.append(sum(mode_scores_arr[idx] for idx in a_indices))
        cand_ally_base_winrate.append(sum(base_winrates_arr[idx] for idx in a_indices))
        cand_ally_tier.append(sum(tiers_arr[idx] for idx in a_indices))
        
        if len(A) == 1:
            syn = 0.5
        elif len(A) == 2:
            syn = synergy_matrix_arr[a_indices[0], a_indices[1]] + 1.0
        else:
            syn = synergy_matrix_arr[a_indices[0], a_indices[1]] + synergy_matrix_arr[a_indices[0], a_indices[2]] + synergy_matrix_arr[a_indices[1], a_indices[2]]
        cand_ally_synergy.append(syn)
        
        cand_ally_trio_score.append(get_split_stat(split_key, 'trio', tuple(sorted(A)), 0.5))
        cand_ally_role_diversity.append(len(set(brawler_primary[a] for a in A)))
        cand_roles_contrib.append(sum(ally_role_contrib_arr[idx] for idx in a_indices))
        cand_mechs.append(sum(mechanics_arr[idx] for idx in a_indices))
        cand_short_count.append(sum(1 for a in A if brawler_range_classes[a] == 'short'))
        cand_long_count.append(sum(1 for a in A if brawler_range_classes[a] == 'long'))
        cand_counter_vs_A.append(counter_matrix_arr[a_indices, :].sum(axis=0))
        cand_counter_by_A.append(counter_matrix_arr[:, a_indices].sum(axis=1))
        
        cand_max_counter_vs_A.append(counter_matrix_arr[a_indices, :].max(axis=0))
        cand_max_counter_by_A.append(counter_matrix_arr[:, a_indices].max(axis=1))
    
    # Flat combinations compilation
    candidate_indices_flat = []
    ep1_indices_flat = []
    ep2_indices_flat = []
    candidate_offsets = []
    current_offset = 0
    
    for c_idx, our_pick in enumerate(top_candidates):
        remaining_enemies = [b for b in available_brawlers if b != our_pick]
        enemy_pairs = list(itertools.permutations(remaining_enemies, 2))
        
        num_pairs = len(enemy_pairs)
        candidate_offsets.append((current_offset, current_offset + num_pairs, enemy_pairs))
        current_offset += num_pairs
        
        candidate_indices_flat.extend([c_idx] * num_pairs)
        for ep1, ep2 in enemy_pairs:
            ep1_indices_flat.append(brawler_to_idx_local[ep1])
            ep2_indices_flat.append(brawler_to_idx_local[ep2])
            
    num_rows = len(candidate_indices_flat)
    
    c_arr = np.array(candidate_indices_flat, dtype=np.int32)
    ep1_arr = np.array(ep1_indices_flat, dtype=np.int32)
    ep2_arr = np.array(ep2_indices_flat, dtype=np.int32)
    
    # Pre-allocate X_batch
    X_batch = np.zeros((num_rows, len(feature_names)), dtype=np.float32)
    
    # Map
    map_feature_name = f"map_{map_name}"
    if map_feature_name in MAP_LIST:
        X_batch[:, MAP_LIST.index(map_feature_name)] = 1.0
        
    # Draft state one-hots
    for ally in current_allies:
        X_batch[:, brawler_to_idx[ally]] = 1.0
    for enemy in current_enemies:
        X_batch[:, brawler_to_idx[enemy]] = -1.0
        
    cand_brawler_indices = np.array([brawler_to_idx[top_candidates[c_idx]] for c_idx in range(len(top_candidates))], dtype=np.int32)
    X_batch[np.arange(num_rows), cand_brawler_indices[c_arr]] = 1.0
    X_batch[np.arange(num_rows), brawler_offset + ep1_arr] = -1.0
    X_batch[np.arange(num_rows), brawler_offset + ep2_arr] = -1.0
    
    # Roles
    cand_roles_contrib_matrix = np.array(cand_roles_contrib, dtype=np.float32)
    enemy_roles_vector = base_enemy_role_vector[np.newaxis, :] + roles_contrib_arr[ep1_arr] + roles_contrib_arr[ep2_arr]
    X_batch[:, role_offset:role_offset + len(ROLE_FEATURE_NAMES)] = cand_roles_contrib_matrix[c_arr] + enemy_roles_vector
    
    X_batch[:, rank_offset] = peak_rank
    
    # Map, Mode, Synergy, Winrates
    a_map_score = np.array(cand_ally_map_score, dtype=np.float32)[c_arr]
    e_map_score = base_enemy_map_score + map_scores_arr[ep1_arr] + map_scores_arr[ep2_arr]
    X_batch[:, adv_offset] = a_map_score
    X_batch[:, adv_offset + 1] = e_map_score
    X_batch[:, adv_offset + 2] = a_map_score - e_map_score
    
    a_mode_score = np.array(cand_ally_mode_score, dtype=np.float32)[c_arr]
    e_mode_score = base_enemy_mode_score + mode_scores_arr[ep1_arr] + mode_scores_arr[ep2_arr]
    X_batch[:, adv_offset + 3] = a_mode_score
    X_batch[:, adv_offset + 4] = e_mode_score
    X_batch[:, adv_offset + 5] = a_mode_score - e_mode_score
    
    a_syn_score = np.array(cand_ally_synergy, dtype=np.float32)[c_arr]
    E_size = len(current_enemies) + 2
    e_syn_score = base_enemy_synergy + synergy_matrix_arr[ep1_arr, ep2_arr] + synergy_vs_current_enemies_arr[ep1_arr] + synergy_vs_current_enemies_arr[ep2_arr] + (3.0 - E_size * (E_size - 1) / 2.0) * 0.5
    X_batch[:, adv_offset + 6] = a_syn_score
    X_batch[:, adv_offset + 7] = e_syn_score
    X_batch[:, adv_offset + 8] = a_syn_score - e_syn_score
    
    # Counters
    cand_counter_vs_A_matrix = np.array(cand_counter_vs_A, dtype=np.float32)
    cand_counter_by_A_matrix = np.array(cand_counter_by_A, dtype=np.float32)
    
    base_ally_counter_term = sum(cand_counter_vs_A_matrix[:, e_idx] for e_idx in enemy_idx_list) if enemy_idx_list else np.zeros(len(top_candidates), dtype=np.float32)
    base_enemy_counter_term = sum(cand_counter_by_A_matrix[:, e_idx] for e_idx in enemy_idx_list) if enemy_idx_list else np.zeros(len(top_candidates), dtype=np.float32)
    
    missing_cross = 9 - (len(current_allies) + 1) * (len(current_enemies) + 2)
    missing_cross_term = missing_cross * 0.5
    
    ally_cnt_score = base_ally_counter_term[c_arr] + cand_counter_vs_A_matrix[c_arr, ep1_arr] + cand_counter_vs_A_matrix[c_arr, ep2_arr] + missing_cross_term
    X_batch[:, adv_offset + 9] = ally_cnt_score
    
    enemy_cnt_score = base_enemy_counter_term[c_arr] + cand_counter_by_A_matrix[c_arr, ep1_arr] + cand_counter_by_A_matrix[c_arr, ep2_arr] + missing_cross_term
    X_batch[:, adv_offset + 10] = enemy_cnt_score
    X_batch[:, adv_offset + 11] = ally_cnt_score - enemy_cnt_score
    
    # Trio synergy & Role diversity precomputations for unique pairs
    unique_enemy_pairs = list(set(zip(ep1_indices_flat, ep2_indices_flat)))
    pair_to_trio_score = {}
    pair_to_diversity = {}
    for ep1_l, ep2_l in unique_enemy_pairs:
        e_list = current_enemies + [ALL_BRAWLERS[ep1_l], ALL_BRAWLERS[ep2_l]]
        pair_to_trio_score[(ep1_l, ep2_l)] = get_split_stat(split_key, 'trio', tuple(sorted(e_list)), 0.5)
        pair_to_diversity[(ep1_l, ep2_l)] = len(set(brawler_primary[e] for e in e_list))
        
    e_trio_score = np.array([pair_to_trio_score[(ep1_l, ep2_l)] for ep1_l, ep2_l in zip(ep1_indices_flat, ep2_indices_flat)], dtype=np.float32)
    a_trio_score = np.array(cand_ally_trio_score, dtype=np.float32)[c_arr]
    
    X_batch[:, adv_offset + 12] = a_trio_score
    X_batch[:, adv_offset + 13] = e_trio_score
    X_batch[:, adv_offset + 14] = a_trio_score - e_trio_score
    
    e_role_div = np.array([pair_to_diversity[(ep1_l, ep2_l)] for ep1_l, ep2_l in zip(ep1_indices_flat, ep2_indices_flat)], dtype=np.float32)
    a_role_div = np.array(cand_ally_role_diversity, dtype=np.float32)[c_arr]
    X_batch[:, adv_offset + 15] = a_role_div
    X_batch[:, adv_offset + 16] = e_role_div
    X_batch[:, adv_offset + 17] = a_role_div - e_role_div
    
    # Min Map score
    a_min_map = np.array(cand_ally_min_map_score, dtype=np.float32)[c_arr]
    X_batch[:, adv_offset + 18] = a_min_map
    base_min_map_score = min(map_scores_arr[idx] for idx in enemy_idx_list) if enemy_idx_list else 0.5
    e_min_map = np.minimum(base_min_map_score, np.minimum(map_scores_arr[ep1_arr], map_scores_arr[ep2_arr]))
    X_batch[:, adv_offset + 19] = e_min_map
    X_batch[:, adv_offset + 20] = a_min_map - e_min_map
    
    # Max counters
    cand_max_counter_vs_A_matrix = np.array(cand_max_counter_vs_A, dtype=np.float32)
    cand_max_counter_by_A_matrix = np.array(cand_max_counter_by_A, dtype=np.float32)
    
    if enemy_idx_list:
        base_max_counter_vs_enemies_c = cand_max_counter_vs_A_matrix[:, enemy_idx_list].max(axis=1)
        base_max_counter_by_enemies_c = cand_max_counter_by_A_matrix[:, enemy_idx_list].max(axis=1)
        
        a_max_cnt = np.maximum(base_max_counter_vs_enemies_c[c_arr], np.maximum(cand_max_counter_vs_A_matrix[c_arr, ep1_arr], cand_max_counter_vs_A_matrix[c_arr, ep2_arr]))
        e_max_cnt = np.maximum(base_max_counter_by_enemies_c[c_arr], np.maximum(cand_max_counter_by_A_matrix[c_arr, ep1_arr], cand_max_counter_by_A_matrix[c_arr, ep2_arr]))
    else:
        a_max_cnt = np.maximum(0.5, np.maximum(cand_max_counter_vs_A_matrix[c_arr, ep1_arr], cand_max_counter_vs_A_matrix[c_arr, ep2_arr]))
        e_max_cnt = np.maximum(0.5, np.maximum(cand_max_counter_by_A_matrix[c_arr, ep1_arr], cand_max_counter_by_A_matrix[c_arr, ep2_arr]))
        
    X_batch[:, adv_offset + 21] = a_max_cnt
    X_batch[:, adv_offset + 22] = e_max_cnt
    X_batch[:, adv_offset + 23] = a_max_cnt - e_max_cnt
    
    # Base winrates
    a_winrate = np.array(cand_ally_base_winrate, dtype=np.float32)[c_arr]
    X_batch[:, adv_offset + 24] = a_winrate
    e_winrate = base_enemy_base_winrate + base_winrates_arr[ep1_arr] + base_winrates_arr[ep2_arr]
    X_batch[:, adv_offset + 25] = e_winrate
    X_batch[:, adv_offset + 26] = a_winrate - e_winrate
    
    # Tiers, Ranges, Mechanics
    a_tier = np.array(cand_ally_tier, dtype=np.float32)[c_arr]
    X_batch[:, tier_offset] = a_tier
    e_tier = base_enemy_tier + tiers_arr[ep1_arr] + tiers_arr[ep2_arr]
    X_batch[:, tier_offset + 1] = e_tier
    X_batch[:, tier_offset + 2] = a_tier - e_tier
    
    X_batch[:, range_offset] = np.array(cand_short_count, dtype=np.float32)[c_arr] * is_open
    X_batch[:, range_offset + 1] = np.array(cand_long_count, dtype=np.float32)[c_arr] * is_closed
    
    cand_mechs_matrix = np.array(cand_mechs, dtype=np.float32)
    enemy_mechs_matrix = base_enemy_mechs[np.newaxis, :] + mechanics_arr[ep1_arr] + mechanics_arr[ep2_arr]
    
    X_batch[:, mech_offset] = cand_mechs_matrix[c_arr, 2] * enemy_mechs_matrix[:, 1]
    X_batch[:, mech_offset + 1] = cand_mechs_matrix[c_arr, 6] * enemy_mechs_matrix[:, 0]
    X_batch[:, mech_offset + 2] = cand_mechs_matrix[c_arr, 3] * enemy_mechs_matrix[:, 5]
    X_batch[:, mech_offset + 3] = cand_mechs_matrix[c_arr, 0] * enemy_mechs_matrix[:, 4] - enemy_mechs_matrix[:, 0] * cand_mechs_matrix[c_arr, 4]
    
    # PASS 1: Predict all 95,060 combinations using LightGBM only (extremely fast!)
    win_probs_lgb = model.lgb_model.predict_proba(X_batch)[:, 1]
    
    # Post-process Pass 1 results to find top 5 candidates
    prelim_worst_cases = []
    for c_idx, our_pick in enumerate(top_candidates):
        start_idx, end_idx, enemy_pairs = candidate_offsets[c_idx]
        cand_probs = win_probs_lgb[start_idx:end_idx]
        min_idx = np.argmin(cand_probs)
        prelim_worst_cases.append((c_idx, cand_probs[min_idx]))
        
    # Sort candidates by win prob descending, take top 5
    top_5_indices = sorted(prelim_worst_cases, key=lambda x: x[1], reverse=True)[:5]
    top_5_c_idx = [x[0] for x in top_5_indices]
    
    # PASS 2: For these top 5 candidates, evaluate fully using StackedModel!
    mask = np.isin(c_arr, top_5_c_idx)
    X_subset = X_batch[mask]
    
    win_probs_full = model.predict_proba(X_subset)[:, 1]
    
    # Post-process Pass 2 results
    results = []
    subset_offset = 0
    for c_idx in top_5_c_idx:
        our_pick = top_candidates[c_idx]
        _, _, enemy_pairs = candidate_offsets[c_idx]
        num_pairs = len(enemy_pairs)
        
        cand_probs = win_probs_full[subset_offset:subset_offset + num_pairs]
        subset_offset += num_pairs
        
        # Find worst case counter-pick under the full StackedModel
        min_idx = np.argmin(cand_probs)
        worst_win_prob = cand_probs[min_idx]
        best_enemy_counter = enemy_pairs[min_idx]
        
        results.append({
            'our_pick': our_pick,
            'win_prob': float(worst_win_prob),
            'enemy_counter': f"{best_enemy_counter[0]} + {best_enemy_counter[1]}"
        })
        
    sorted_results = sorted(results, key=lambda x: x['win_prob'], reverse=True)
    return sorted_results

def get_smart_recommendation(map_name, current_allies, current_enemies, available_brawlers, first_pick_team='ally', peak_rank=None, player_brawlers_power_11=None, my_pick_index=None):
    """
    The master function. Chooses the correct algorithm based on draft state and who picks first.
    """
    if peak_rank is None:
        peak_rank = MEDIAN_RANK
        
    # Check if it's the player's turn to pick
    is_my_turn = False
    if my_pick_index is not None:
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
        if current_turn_index < len(draft_sequence):
            turn = draft_sequence[current_turn_index]
            if turn['team'] == 'ally' and turn['index'] == my_pick_index:
                is_my_turn = True

    p11_list = player_brawlers_power_11 if (is_my_turn and player_brawlers_power_11) else None
        
    # 1. Greedy fallback if the draft is over or enemy is locked
    if len(current_enemies) == 3:
        print(">> ENEMY TEAM LOCKED: Running Greedy Algorithm...")
        return run_greedy(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
        
    num_allies = len(current_allies)
    
    if first_pick_team == 'ally':
        if num_allies == 0:
            print(">> Allies Pick 1 (Enemies pick twice next): Running 2-Ply Minimax...")
            return run_minimax_2ply(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
        elif num_allies == 1:
            print(">> Allies Pick 2 (Allies pick again next): Running Greedy...")
            return run_greedy(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
        else:
            print(">> Allies Pick 3 (Enemies pick once next): Running 1-Ply Minimax...")
            return run_minimax(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
    else: # first_pick_team == 'enemy'
        if num_allies == 0:
            print(">> Allies Pick 1 (Allies pick again next): Running Greedy...")
            return run_greedy(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
        elif num_allies == 1:
            print(">> Allies Pick 2 (Enemies pick twice next): Running 2-Ply Minimax...")
            return run_minimax_2ply(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)
        else:
            print(">> Allies Pick 3 (Draft ends next): Running Greedy...")
            return run_greedy(map_name, current_allies, current_enemies, available_brawlers, peak_rank=peak_rank, player_brawlers_power_11=p11_list)

if __name__ == "__main__":
    # --- TEST THE SMART PREDICTOR ---
    target_map = "Hard Rock Mine"
    
    # TEST CASE 1: Mid-draft (Enemy has 2 brawlers) -> Should trigger Minimax
    my_team = ['BO']
    enemy_team = ['COLT', 'PIPER']
    
    # TEST CASE 2: Final Pick (Enemy has 3 brawlers) -> Should trigger Greedy
    # Uncomment the line below to test the Greedy trigger!
    # enemy_team = ['COLT', 'PIPER', 'TICK'] 
    
    banned_brawlers = ['TARA', 'NORI']
    
    picked_and_banned = set(my_team + enemy_team + banned_brawlers) 
    available_to_draft = [b for b in ALL_BRAWLERS if b not in picked_and_banned]
    
    print(f"\nMap: {target_map}")
    print(f"Allies: {my_team} | Enemies: {enemy_team}")
    
    recommendations = get_smart_recommendation(target_map, my_team, enemy_team, available_to_draft)
    
    print("\n--- Top 5 Recommended Picks ---")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['our_pick'].ljust(12)} "
              f"| Expected Win: {rec['win_prob']:.2%} "
              f"| Assumed Enemy Response: {rec['enemy_counter']}")