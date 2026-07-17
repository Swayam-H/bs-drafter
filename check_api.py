import requests

tags_to_try = ['%232PPQVUQ8J', '%23LRQG2CVG', '%23Y0JGPLJ']
for tag in tags_to_try:
    resp = requests.get(f'http://127.0.0.1:5000/api/battlelog/{tag}', timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n--- Player: {tag} ---")
        for log in data.get('items', []):
            battle = log.get('battle', {})
            event = log.get('event', {})
            btype = battle.get('type')
            mode = event.get('mode')
            map_name = event.get('map')
            result = battle.get('result')
            has_trophy = 'trophyChange' in battle
            teams = battle.get('teams', [])
            
            # Check for duplicate brawlers
            dupes = ""
            if teams and len(teams) == 2 and len(teams[0]) == 3 and len(teams[1]) == 3:
                t0 = set(p['brawler']['name'] for p in teams[0])
                t1 = set(p['brawler']['name'] for p in teams[1])
                shared = t0 & t1
                if shared:
                    dupes = f" [DUPE: {shared}]"
            
            print(f"  Type: {btype} | Mode: {mode} | Map: {map_name} | Result: {result} | Trophy: {has_trophy}{dupes}")
    else:
        print(f"Player {tag}: Status {resp.status_code}")
