# Brawler Role Mapping for ML Features
# Each brawler has a primary_role and secondary_role (or None).

BRAWLER_ROLES = {
    # --- Aggro ---
    "EDGAR":      {"primary": "aggro",      "secondary": "assassin"},
    "EL PRIMO":   {"primary": "aggro",      "secondary": "tank"},
    "BULL":       {"primary": "aggro",      "secondary": "tank"},
    "ROSA":       {"primary": "aggro",      "secondary": "tank"},
    "BIBI":       {"primary": "aggro",      "secondary": "anti_aggro"},
    "DARRYL":     {"primary": "aggro",      "secondary": "assassin"},
    "BUZZ":       {"primary": "aggro",      "secondary": "assassin"},
    "FANG":       {"primary": "aggro",      "secondary": "assassin"},
    "ASH":        {"primary": "aggro",      "secondary": "tank"},
    "KENJI":      {"primary": "aggro",      "secondary": "assassin"},
    "SURGE":      {"primary": "aggro",      "secondary": "hybrid"},
    "MICO":       {"primary": "aggro",      "secondary": "assassin"},
    "SAM":        {"primary": "aggro",      "secondary": "tank"},
    "DOUG":       {"primary": "support",    "secondary": "tank"},

    # --- Anti-Aggro ---
    "EMZ":        {"primary": "anti_aggro",  "secondary": "control"},
    "GRIFF":      {"primary": "anti_aggro",  "secondary": "anti_tank"},
    "COLETTE":    {"primary": "anti_aggro",  "secondary": "anti_tank"},
    "NORI":       {"primary": "anti_aggro",  "secondary": "control"},
    "DAMIAN":     {"primary": "anti_aggro",  "secondary": "control"},
    "GALE":       {"primary": "anti_aggro",  "secondary": "control"},
    "SQUEAK":     {"primary": "anti_aggro",  "secondary": "control"},
    "LOU":        {"primary": "anti_aggro",  "secondary": "control"},
    "JACKY":      {"primary": "anti_aggro",  "secondary": "tank"},
    "FRANK":      {"primary": "anti_aggro",  "secondary": "tank"},
    "PAM":        {"primary": "anti_aggro",  "secondary": "support"},

    # --- Assassin ---
    "LEON":       {"primary": "assassin",    "secondary": "aggro"},
    "CROW":       {"primary": "assassin",    "secondary": "control"},
    "MORTIS":     {"primary": "assassin",    "secondary": "aggro"},
    "STU":        {"primary": "assassin",    "secondary": "aggro"},
    "MELODIE":    {"primary": "assassin",    "secondary": "aggro"},
    "CORDELIUS":  {"primary": "assassin",    "secondary": "hybrid"},
    "LILY":       {"primary": "assassin",    "secondary": "aggro"},
    "SHADE":      {"primary": "assassin",    "secondary": "aggro"},
    "KAZE":       {"primary": "assassin",    "secondary": "aggro"},
    "KIT":        {"primary": "assassin",    "secondary": "support"},
    "CARL":       {"primary": "assassin",    "secondary": "aggro"},
    "MAX":        {"primary": "support",     "secondary": "aggro"},

    # --- Tank ---
    "NITA":       {"primary": "tank",        "secondary": "support"},
    "BUSTER":     {"primary": "tank",        "secondary": "anti_aggro"},
    "MEG":        {"primary": "tank",        "secondary": "control"},
    "HANK":       {"primary": "tank",        "secondary": "control"},
    "TRUNK":      {"primary": "tank",        "secondary": "aggro"},
    "CLANCY":     {"primary": "tank",        "secondary": "anti_tank"},

    # --- Anti-Tank ---
    "SPIKE":      {"primary": "anti_tank",   "secondary": "control"},
    "AMBER":      {"primary": "anti_tank",   "secondary": "control"},
    "SHELLY":     {"primary": "anti_tank",   "secondary": "aggro"},
    "BEA":        {"primary": "anti_tank",   "secondary": "sniper"},

    # --- Sniper ---
    "PIPER":      {"primary": "sniper",      "secondary": None},
    "BROCK":      {"primary": "sniper",      "secondary": "control"},
    "BELLE":      {"primary": "sniper",      "secondary": "anti_tank"},
    "MANDY":      {"primary": "sniper",      "secondary": None},
    "NANI":       {"primary": "sniper",      "secondary": None},
    "COLT":       {"primary": "sniper",      "secondary": "anti_tank"},
    "8-BIT":      {"primary": "sniper",      "secondary": "support"},
    "PIERCE":     {"primary": "sniper",      "secondary": "control"},
    "BOLT":       {"primary": "sniper",      "secondary": "hybrid"},
    "LOLA":       {"primary": "sniper",      "secondary": "control"},
    "SIRIUS":     {"primary": "sniper",      "secondary": "anti_aggro"},
    "STARR NOVA": {"primary": "sniper",      "secondary": "control"},

    # --- Thrower ---
    "DYNAMIKE":       {"primary": "thrower",  "secondary": None},
    "BARLEY":         {"primary": "thrower",  "secondary": "control"},
    "TICK":           {"primary": "thrower",  "secondary": "control"},
    "GROM":           {"primary": "thrower",  "secondary": "control"},
    "SPROUT":         {"primary": "thrower",  "secondary": "control"},
    "PENNY":          {"primary": "control",  "secondary": "support"},
    "LARRY & LAWRIE": {"primary": "thrower",  "secondary": "tank"},

    # --- Control ---
    "TARA":       {"primary": "control",     "secondary": "assassin"},
    "GENE":       {"primary": "control",     "secondary": "support"},
    "SANDY":      {"primary": "control",     "secondary": "support"},
    "BO":         {"primary": "control",     "secondary": "sniper"},
    "JESSIE":     {"primary": "control",     "secondary": "support"},
    "RICO":       {"primary": "control",     "secondary": "anti_aggro"},
    "MR. P":      {"primary": "control",     "secondary": "support"},
    "OTIS":       {"primary": "control",     "secondary": "anti_aggro"},
    "EVE":        {"primary": "control",     "secondary": "hybrid"},
    "WILLOW":     {"primary": "control",     "secondary": "assassin"},
    "OLLIE":      {"primary": "control",     "secondary": "tank"},
    "MEEPLE":     {"primary": "control",     "secondary": "support"},
    "GLOWY":      {"primary": "control",     "secondary": "support"},
    "FINX":       {"primary": "control",     "secondary": "sniper"},
    "R-T":        {"primary": "sniper",      "secondary": "assassin"},
    "ZIGGY":      {"primary": "control",     "secondary": "support"},
    "JUJU":       {"primary": "control",     "secondary": "support"},

    # --- Support ---
    "POCO":       {"primary": "support",     "secondary": "anti_aggro"},
    "BYRON":      {"primary": "support",     "secondary": "sniper"},
    "GUS":        {"primary": "support",     "secondary": None},
    "RUFFS":      {"primary": "support",     "secondary": "sniper"},
    "GRAY":       {"primary": "support",     "secondary": "control"},
    "BERRY":      {"primary": "support",     "secondary": "thrower"},
    "ALLI":       {"primary": "support",     "secondary": "anti_aggro"},
    "MINA":       {"primary": "support",     "secondary": "anti_aggro"},
    "CHESTER":    {"primary": "control",     "secondary": "anti_tank"},
    "CHUCK":      {"primary": "assassin",    "secondary": "control"},
    "ANGELO":     {"primary": "sniper",      "secondary": "assassin"},
    "BONNIE":     {"primary": "sniper",      "secondary": "assassin"},
    "CHARLIE":    {"primary": "support",     "secondary": "control"},
    "DRACO":      {"primary": "support",     "secondary": "aggro"},
    "LUMI":       {"primary": "support",     "secondary": "control"},
    "MOE":        {"primary": "support",     "secondary": "thrower"},
    "JAE-YONG":   {"primary": "support",     "secondary": "aggro"},
    "GIGI":       {"primary": "support",     "secondary": "control"},
    "NAJIA":      {"primary": "support",     "secondary": "sniper"},
    "JANET":      {"primary": "support",     "secondary": "sniper"},

    "PEARL":      {"primary": "support",     "secondary": "anti_aggro"},
    "MAISIE":     {"primary": "support",     "secondary": "sniper"},
}

# All unique roles for one-hot encoding
ALL_ROLES = ["aggro", "anti_aggro", "assassin", "tank", "anti_tank", "sniper", "thrower", "control", "support", "hybrid"]

def get_role(brawler_name):
    """Returns (primary_role, secondary_role) for a brawler. Defaults to ('hybrid', None) if unknown."""
    info = BRAWLER_ROLES.get(brawler_name, {"primary": "hybrid", "secondary": None})
    return info["primary"], info["secondary"]

# Brawler Tiers (S to F) for Meta Strength
# S: 1.0, A: 0.8, B: 0.6, C: 0.4, D: 0.2, F: 0.0
BRAWLER_TIERS = {
    # --- S Tier ---
    "SURGE": 1.0, "EDGAR": 1.0, "MORTIS": 1.0, "CROW": 1.0, "COLT": 1.0, "COLETTE": 1.0, "BOLT": 1.0, "DAMIAN": 1.0,
    
    # --- A Tier ---
    "STARR NOVA": 0.8, "SIRIUS": 0.8, "BROCK": 0.8, "NORI": 0.8, "OTIS": 0.8, "CHESTER": 0.8, "KIT": 0.8, "MEG": 0.8, 
    "PIERCE": 0.8, "8-BIT": 0.8, "MAX": 0.8, "BIBI": 0.8, "GRIFF": 0.8, "LUMI": 0.8, "MINA": 0.8, "KENJI": 0.8,
    
    # --- B Tier ---
    "SHELLY": 0.6, "BULL": 0.6, "EMZ": 0.6, "SHADE": 0.6, "FRANK": 0.6, "SANDY": 0.6, "BYRON": 0.6, "CLANCY": 0.6, 
    "MANDY": 0.6, "PIPER": 0.6, "TARA": 0.6, "DYNAMIKE": 0.6, "ANGELO": 0.6, "MOE": 0.6, "LARRY & LAWRIE": 0.6, 
    "BERRY": 0.6, "RICO": 0.6, "LILY": 0.6, "BUZZ": 0.6, "FANG": 0.6,
    
    # --- C Tier ---
    "BO": 0.4, "JESSIE": 0.4, "POCO": 0.4, "GALE": 0.4, "LOU": 0.4, "SQUEAK": 0.4, "NITA": 0.4, "BUSTER": 0.4, 
    "HANK": 0.4, "TRUNK": 0.4, "BARLEY": 0.4, "TICK": 0.4, "GROM": 0.4, "SPROUT": 0.4, "PENNY": 0.4, "GENE": 0.4, 
    "EVE": 0.4, "WILLOW": 0.4, "RUFFS": 0.4, "GRAY": 0.4, "CHARLIE": 0.4, "DRACO": 0.4, "PEARL": 0.4, "MAISIE": 0.4, 
    "JANET": 0.4,
    
    # --- D Tier ---
    "DOUG": 0.2, "DARRYL": 0.2, "ASH": 0.2, "MICO": 0.2, "SAM": 0.2, "JACKY": 0.2, "LOLA": 0.2, "GUS": 0.2, 
    "BONNIE": 0.2, "JAE-YONG": 0.2, "GIGI": 0.2, "NAJIA": 0.2, "EL PRIMO": 0.2, "GLOWY": 0.2, "FINX": 0.2, 
    "R-T": 0.2, "ZIGGY": 0.2, "JUJU": 0.2, "ALLI": 0.2,
    
    # --- F Tier ---
    "MR. P": 0.0, "PAM": 0.0, "MEEPLE": 0.0
}

def get_tier(brawler_name):
    """Returns the numerical tier score of a brawler. Defaults to 0.4 (C-tier) if unknown."""
    return BRAWLER_TIERS.get(brawler_name.upper(), 0.4)

BRAWLER_MECHANICS = {
    # HAS_POISON: reduces healing or keeps health regeneration blocked
    "CROW": {"poison": True},
    "BYRON": {"poison": True, "healer": True},
    
    # HAS_SILENCE_STUN: can silence/stun, countering dive/aggro
    "OTIS": {"silence": True},
    "FRANK": {"silence": True},
    "LOU": {"silence": True},
    "BUZZ": {"silence": True, "diver": True},
    "GALE": {"silence": True},
    
    # IS_DIVER_MOBILE: highly mobile, weak to silence/stun
    "MORTIS": {"diver": True},
    "EDGAR": {"diver": True},
    "FANG": {"diver": True},
    "KENJI": {"diver": True},
    "MAX": {"diver": True},
    "MELODIE": {"diver": True},
    "MICO": {"diver": True},
    "NORI": {"diver": True},
    "STU": {"diver": True},
    "LILY": {"diver": True},
    
    # IS_SPAWNER: spawns pets/shields, distracting single-target snipers
    "NITA": {"spawner": True},
    "JESSIE": {"spawner": True},
    "TARA": {"spawner": True},
    "MR. P": {"spawner": True},
    "LOLA": {"spawner": True},
    
    # IS_HEALER: heals allies
    "POCO": {"healer": True},
    "PAM": {"healer": True},
    "KIT": {"healer": True},
    "BERRY": {"healer": True},
    "DOUG": {"healer": True}
}

def get_mechanics(brawler_name):
    """Returns a dict of mechanics (poison, silence, diver, spawner, healer) for a brawler."""
    return BRAWLER_MECHANICS.get(brawler_name.upper(), {})

# Map Openness (open vs closed)
MAP_OPENNESS = {
    # Open Maps (Sniper-friendly)
    "DRY SEASON": "open",
    "SHOOTING STAR": "open",
    "OPEN BUSINESS": "open",
    "HIDEOUT": "open",
    "BACKYARD BOWL": "open",
    "NEW HORIZONS": "open",
    
    # Closed Maps (Wall/Bush heavy, Tank-friendly)
    "SNEAKY FIELDS": "closed",
    "DOUBLE SWOOSH": "closed",
    "PINBALL DREAMS": "closed",
    "HARD ROCK MINE": "closed",
    "SPIRALING OUT": "closed",
    "PARALLEL PLAYS": "closed",
    "CONTROLLER CHAOS": "closed"
}

def is_open_map(map_name):
    """Returns True if the map is wide-open and favors snipers."""
    return MAP_OPENNESS.get(map_name.upper(), "normal") == "open"

def is_closed_map(map_name):
    """Returns True if the map is closed/bushy and favors tanks."""
    return MAP_OPENNESS.get(map_name.upper(), "normal") == "closed"

# Brawler Range Classes
# short: melee, tanks, shotgunners
# long: snipers, throwers, long-range controls
BRAWLER_RANGES = {
    # Melee/Short-range brawlers
    "EDGAR": "short", "EL PRIMO": "short", "BULL": "short", "ROSA": "short", 
    "BIBI": "short", "DARRYL": "short", "BUZZ": "short", "FANG": "short", 
    "ASH": "short", "KENJI": "short", "SURGE": "short", "MICO": "short", 
    "SAM": "short", "DOUG": "short", "JACKY": "short", "FRANK": "short", 
    "PAM": "short", "LEON": "short", "MORTIS": "short", "MELODIE": "short", 
    "CORDELIUS": "short", "LILY": "short", "SHADE": "short", "KAZE": "short", 
    "KIT": "short", "CARL": "short", "NITA": "short", "BUSTER": "short", 
    "HANK": "short", "TRUNK": "short", "CLANCY": "short", "SHELLY": "short", 
    "DRACO": "short", "OLLIE": "short", "GLOWY": "short",
    
    # Long-range / Snipers / Throwers
    "PIPER": "long", "BROCK": "long", "BELLE": "long", "MANDY": "long", 
    "NANI": "long", "COLT": "long", "8-BIT": "long", "PIERCE": "long", 
    "BOLT": "long", "LOLA": "long", "SIRIUS": "long", "STARR NOVA": "long",
    "DYNAMIKE": "long", "BARLEY": "long", "TICK": "long", "GROM": "long", 
    "SPROUT": "long", "LARRY & LAWRIE": "long", "BYRON": "long", "BERRY": "long", 
    "ANGELO": "long", "BONNIE": "long", "MOE": "long", "NAJIA": "long", "MAISIE": "long"
}

def get_range_class(brawler_name):
    """Returns 'short' or 'long'. Defaults to 'medium' if not listed."""
    return BRAWLER_RANGES.get(brawler_name.upper(), "medium")

class StackedModel:
    def __init__(self, xgb_model, lgb_model, cb_model, rf_model, meta_model):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.rf_model = rf_model
        self.meta_model = meta_model
        
    def predict_proba(self, X):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(m.predict_proba, X)
                for m in [self.xgb_model, self.lgb_model, self.cb_model, self.rf_model]
            ]
            probs = [f.result()[:, 1] for f in futures]
            
        P_base = np.column_stack(probs)
        return self.meta_model.predict_proba(P_base)
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)




