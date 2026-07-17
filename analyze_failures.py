import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from brawler_roles import ALL_ROLES, get_role, get_tier, get_mechanics, is_open_map, is_closed_map, get_range_class, StackedModel

print("Loading model and data...")
with open('draft_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
brawler_list = saved['brawler_list']
map_list = saved['map_list']
role_feature_names = saved.get('role_feature_names', [])
median_rank = saved.get('median_rank', 7000)

# Load data (handle mixed column formats)
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

map_counts = df['map'].value_counts()
valid_maps = map_counts[map_counts >= 15].index
df = df[df['map'].isin(valid_maps)].reset_index(drop=True)

df['peak_rank'] = df['peak_rank'].fillna(median_rank).astype(float)

y = df['result'].apply(lambda x: 1 if x == 'victory' else 0).values

# Build features (same pipeline as train_model.py)
brawler_to_idx = {b: i for i, b in enumerate(brawler_list)}

_role_to_idx = {}
for i, role in enumerate(ALL_ROLES):
    _role_to_idx[role] = {'ally': i * 3, 'enemy': i * 3 + 1, 'advantage': i * 3 + 2}

draft_features = np.zeros((len(df), len(brawler_list)), dtype=np.int8)
role_features = np.zeros((len(df), len(role_feature_names)), dtype=np.float32)

for i, row in enumerate(df.itertuples()):
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
        if primary in _role_to_idx:
            role_features[i, _role_to_idx[primary]['ally']] += 1
        if secondary and secondary in _role_to_idx:
            role_features[i, _role_to_idx[secondary]['ally']] += 0.5
    for enemy in enemies:
        primary, secondary = get_role(enemy)
        if primary in _role_to_idx:
            role_features[i, _role_to_idx[primary]['enemy']] += 1
        if secondary and secondary in _role_to_idx:
            role_features[i, _role_to_idx[secondary]['enemy']] += 0.5
    for role in ALL_ROLES:
        idx = _role_to_idx[role]
        role_features[i, idx['advantage']] = role_features[i, idx['ally']] - role_features[i, idx['enemy']]

X_draft = pd.DataFrame(draft_features, columns=brawler_list)
X_roles = pd.DataFrame(role_features, columns=role_feature_names)
X_maps = pd.get_dummies(df['map'], prefix='map')
for m in map_list:
    if m not in X_maps.columns:
        X_maps[m] = 0
X_maps = X_maps[map_list]
X_rank = df[['peak_rank']].copy()

# Advanced features
global_stats = saved.get('global_stats', {})
map_to_mode = saved.get('map_to_mode', {})

def get_split_stat(split_key, stat_type, key, default=0.5):
    val = global_stats.get(split_key, {}).get(stat_type, {}).get(key)
    if val is not None:
        return val
    return global_stats.get('combined', {}).get(stat_type, {}).get(key, default)

ally_map_scores = []
enemy_map_scores = []
ally_mode_scores = []
enemy_mode_scores = []
ally_synergy_scores = []
enemy_synergy_scores = []
ally_counter_scores = []
enemy_counter_scores = []
ally_trio_scores = []
enemy_trio_scores = []
ally_role_diversities = []
enemy_role_diversities = []
ally_min_map_scores = []
enemy_min_map_scores = []
ally_max_counter_scores = []
enemy_max_counter_scores = []
ally_base_winrates = []
enemy_base_winrates = []

for row in df.itertuples():
    m = row.map
    g_mode = row.mode
    pr = row.peak_rank
    split_key = 'low' if pr < 5000 else 'high'
    allies = [row.ally_1, row.ally_2, row.ally_3]
    enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
    
    a_map_scores = [get_split_stat(split_key, 'map', (m, b), 0.5) for b in allies]
    e_map_scores = [get_split_stat(split_key, 'map', (m, b), 0.5) for b in enemies]
    ally_map_scores.append(sum(a_map_scores))
    enemy_map_scores.append(sum(e_map_scores))
    ally_min_map_scores.append(min(a_map_scores) if a_map_scores else 0.5)
    enemy_min_map_scores.append(min(e_map_scores) if e_map_scores else 0.5)
    
    ally_base_winrates.append(sum([get_split_stat(split_key, 'base', b, 0.5) for b in allies]))
    enemy_base_winrates.append(sum([get_split_stat(split_key, 'base', b, 0.5) for b in enemies]))
    
    a_mode = sum(get_split_stat(split_key, 'mode', (g_mode, b), 0.5) for b in allies)
    e_mode = sum(get_split_stat(split_key, 'mode', (g_mode, b), 0.5) for b in enemies)
    ally_mode_scores.append(a_mode)
    enemy_mode_scores.append(e_mode)
    
    a_syn = sum(get_split_stat(split_key, 'synergy', tuple(sorted([allies[i], allies[j]])), 0.5) for i, j in [(0, 1), (0, 2), (1, 2)])
    e_syn = sum(get_split_stat(split_key, 'synergy', tuple(sorted([enemies[i], enemies[j]])), 0.5) for i, j in [(0, 1), (0, 2), (1, 2)])
    ally_synergy_scores.append(a_syn)
    enemy_synergy_scores.append(e_syn)
    
    a_cnt_scores = [get_split_stat(split_key, 'counter', (ally, enemy), 0.5) for ally in allies for enemy in enemies]
    e_cnt_scores = [get_split_stat(split_key, 'counter', (enemy, ally), 0.5) for enemy in enemies for ally in allies]
    ally_counter_scores.append(sum(a_cnt_scores))
    enemy_counter_scores.append(sum(e_cnt_scores))
    ally_max_counter_scores.append(max(a_cnt_scores) if a_cnt_scores else 0.5)
    enemy_max_counter_scores.append(max(e_cnt_scores) if e_cnt_scores else 0.5)
    
    ally_trio_scores.append(get_split_stat(split_key, 'trio', tuple(sorted(allies)), 0.5))
    enemy_trio_scores.append(get_split_stat(split_key, 'trio', tuple(sorted(enemies)), 0.5))
    
    ally_role_diversities.append(len(set(get_role(b)[0] for b in allies)))
    enemy_role_diversities.append(len(set(get_role(b)[0] for b in enemies)))
    
df['ally_map_score'] = ally_map_scores
df['enemy_map_score'] = enemy_map_scores
df['map_score_advantage'] = np.array(ally_map_scores) - np.array(enemy_map_scores)

df['ally_mode_score'] = ally_mode_scores
df['enemy_mode_score'] = enemy_mode_scores
df['mode_score_advantage'] = np.array(ally_mode_scores) - np.array(enemy_mode_scores)

df['ally_synergy_score'] = ally_synergy_scores
df['enemy_synergy_score'] = enemy_synergy_scores
df['synergy_advantage'] = np.array(ally_synergy_scores) - np.array(enemy_synergy_scores)

df['ally_counter_score'] = ally_counter_scores
df['enemy_counter_score'] = enemy_counter_scores
df['counter_advantage'] = np.array(ally_counter_scores) - np.array(enemy_counter_scores)

df['ally_trio_score'] = ally_trio_scores
df['enemy_trio_score'] = enemy_trio_scores
df['trio_advantage'] = np.array(ally_trio_scores) - np.array(enemy_trio_scores)

df['ally_role_diversity'] = ally_role_diversities
df['enemy_role_diversity'] = enemy_role_diversities
df['role_diversity_advantage'] = np.array(ally_role_diversities) - np.array(enemy_role_diversities)

df['ally_min_map_score'] = ally_min_map_scores
df['enemy_min_map_score'] = enemy_min_map_scores
df['min_map_score_advantage'] = np.array(ally_min_map_scores) - np.array(enemy_min_map_scores)

df['ally_max_counter_score'] = ally_max_counter_scores
df['enemy_max_counter_score'] = enemy_max_counter_scores
df['max_counter_advantage'] = np.array(ally_max_counter_scores) - np.array(enemy_max_counter_scores)

df['ally_base_winrate'] = ally_base_winrates
df['enemy_base_winrate'] = enemy_base_winrates
df['base_winrate_advantage'] = np.array(ally_base_winrates) - np.array(enemy_base_winrates)

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
X_adv = df[adv_cols].copy()

# Tier features
tier_features = np.zeros((len(df), 3), dtype=np.float32)
for i, row in enumerate(df.itertuples()):
    allies = [row.ally_1, row.ally_2, row.ally_3]
    enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
    ally_tier = sum(get_tier(b) for b in allies)
    enemy_tier = sum(get_tier(b) for b in enemies)
    tier_features[i, 0] = ally_tier
    tier_features[i, 1] = enemy_tier
    tier_features[i, 2] = ally_tier - enemy_tier
X_tiers = pd.DataFrame(tier_features, columns=['ally_tier_score', 'enemy_tier_score', 'tier_advantage'])

# Mechanic features
mechanic_features = np.zeros((len(df), 4), dtype=np.float32)
for i, row in enumerate(df.itertuples()):
    allies = [row.ally_1, row.ally_2, row.ally_3]
    enemies = [row.enemy_1, row.enemy_2, row.enemy_3]
    
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
range_map_features = np.zeros((len(df), 2), dtype=np.float32)
for i, row in enumerate(df.itertuples()):
    allies = [row.ally_1, row.ally_2, row.ally_3]
    m = row.map
    is_open = 1.0 if is_open_map(m) else 0.0
    is_closed = 1.0 if is_closed_map(m) else 0.0
    ally_short_count = sum(1 for b in allies if get_range_class(b) == 'short')
    ally_long_count = sum(1 for b in allies if get_range_class(b) == 'long')
    range_map_features[i, 0] = ally_short_count * is_open
    range_map_features[i, 1] = ally_long_count * is_closed

X_mechs = pd.DataFrame(mechanic_features, columns=[
    'diver_vs_silence_penalty', 'tank_vs_poison_penalty', 
    'spawner_vs_sniper_bonus', 'poison_vs_healer_advantage'
])
X_range_maps = pd.DataFrame(range_map_features, columns=[
    'short_range_on_open_map_penalty', 'long_range_on_closed_map_penalty'
])

X = pd.concat([X_maps, X_draft, X_roles, X_rank, X_adv, X_tiers, X_mechs, X_range_maps], axis=1)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

print("Making predictions...")
predict_probs = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

test_df = df.iloc[indices_test].copy()
test_df['true_result'] = y_test
test_df['prediction'] = predictions
test_df['win_probability'] = predict_probs

failures = test_df[test_df['true_result'] != test_df['prediction']]
successes = test_df[test_df['true_result'] == test_df['prediction']]

print(f"Total Test Set: {len(test_df)}")
print(f"Total Failures: {len(failures)} ({(len(failures)/len(test_df))*100:.2f}%)")

print("\n--- Map Analysis (Failure Rate by Map) ---")
map_stats = test_df.groupby('map').apply(lambda g: pd.Series({
    'total': len(g),
    'fails': len(g[g['true_result'] != g['prediction']]),
    'fail_rate': len(g[g['true_result'] != g['prediction']]) / len(g)
})).sort_values('fail_rate', ascending=False)
print(map_stats[map_stats['total'] > 20].head(10))

print("\n--- Most Confident Failures (Model was >70% sure but was wrong) ---")
high_conf_fails = failures[(failures['win_probability'] > 0.7) | (failures['win_probability'] < 0.3)]
print(f"Number of highly confident failures: {len(high_conf_fails)}")
for _, row in high_conf_fails.head(5).iterrows():
    pred_str = "Victory" if row['prediction'] == 1 else "Defeat"
    true_str = "Victory" if row['true_result'] == 1 else "Defeat"
    prob = row['win_probability'] if row['prediction'] == 1 else 1 - row['win_probability']
    print(f"Map: {row['map']}, Allies: {row['ally_1']},{row['ally_2']},{row['ally_3']}, Enemies: {row['enemy_1']},{row['enemy_2']},{row['enemy_3']}")
    print(f"Predicted {pred_str} ({prob*100:.1f}%) but True was {true_str}")

print("\n--- Brawler Analysis (Brawlers present in failures vs successes) ---")
def brawler_freq(df_subset):
    brawlers = pd.concat([df_subset['ally_1'], df_subset['ally_2'], df_subset['ally_3'], 
                          df_subset['enemy_1'], df_subset['enemy_2'], df_subset['enemy_3']])
    return brawlers.value_counts() / len(df_subset)

fail_freq = brawler_freq(failures)
succ_freq = brawler_freq(successes)
diff_freq = (fail_freq - succ_freq).dropna().sort_values(ascending=False)

print("Brawlers whose presence most increases failure rate (Higher in Fails):")
print(diff_freq.head(5))

print("\nBrawlers whose presence most increases success rate (Higher in Successes):")
print(diff_freq.tail(5))
