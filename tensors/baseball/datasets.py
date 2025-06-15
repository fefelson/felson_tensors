from ..core import CustomDataset

######################################################################
######################################################################

BATTER_NUMERIC = ["EXTRACT(YEAR FROM AGE(g.game_date, batter.birthdate))::integer AS batter_age", 
                    "(EXTRACT(YEAR FROM g.game_date) - batter.rookie_season)::integer AS batter_exp",
                    "batter.height AS batter_height",
                    "batter.weight AS batter_weight"]
BATTER_CATEGORICAL = ["batter.bats = pitcher.throws AS batter_faces_break"]
BATTER_FEATURES = BATTER_NUMERIC+BATTER_CATEGORICAL

PITCHER_NUMERIC = ["EXTRACT(YEAR FROM AGE(g.game_date, pitcher.birthdate))::integer AS pitcher_age", 
                    "(EXTRACT(YEAR FROM g.game_date) - pitcher.rookie_season)::integer AS pitcher_exp",
                    "pitcher.height AS pitcher_height",
                    "pitcher.weight AS pitcher_weight"]
PITCHER_CATEGORICAL = ["pitcher.throws = 'L' AS pitcher_throws_lefty"]
PITCHER_FEATURES = PITCHER_NUMERIC+PITCHER_CATEGORICAL

COUNT_CATEGORICAL = ["balls", "strikes"]
COUNT_NUMERIC = ["pitch_count", "sequence"]
COUNT_FEATURES = COUNT_CATEGORICAL+COUNT_NUMERIC


PITCH_CATEGORICAL = ["pitch_type_id", "pitch_result_id", "pitch_location"]
PITCH_NUMERIC = ["velocity"]
PITCH_FEATURES = PITCH_CATEGORICAL+PITCH_NUMERIC

PITCH_JOINS = [{"name": "games", "inner": True, "join_tables": [{"name": "pitches", "keys": ["game_id",]}] },
                {"name": "pitch_types", "inner": True, "join_tables": [{"name": "pitches", "keys": ["pitch_type_name",]}] },
                {"name": "pitch_result_types", "inner": True, "join_tables": [{"name": "pitches", "keys": ["pitch_result_name",]}] },
                {"name": "players AS batter", "inner": False, "join_tables": [{"name": "pitches", "keys": ["player_id", "batter_id"]}] },
                {"name": "players AS pitcher", "inner": False, "join_tables": [{"name": "pitches", "keys": ["player_id", "pitcher_id"]}] },
]
SWING_JOINS = PITCH_JOINS+[{"name": "swing_results", "inner": True, "join_tables": [{"name": "pitches", "keys": ["pitch_result_name", ]}] },
]


HIT_CATEGORICAL = ["hit_style", "hit_hardness"]
HIT_NUMERIC = ["hit_distance", "hit_angle"]
HIT_FEATURES = HIT_CATEGORICAL+HIT_NUMERIC

CONTACT_JOINS = [{"name": "games", "inner": True, "join_tables": [{"name": "at_bats", "keys": ["game_id",]}] },
                {"name": "contact_types", "inner": True, "join_tables": [{"name": "at_bats", "keys": ["at_bat_type_id",]}] },
                {"name": "at_bat_types", "inner": True, "join_tables": [{"name": "at_bats", "keys": ["at_bat_type_id",]}] },
]


######################################################################
######################################################################


class SwingResultDataset(CustomDataset):

    _main_table = "pitches"
    _main_id = "pitch_id"
    _joins = SWING_JOINS
    _select_features = COUNT_FEATURES+PITCH_FEATURES+BATTER_FEATURES+PITCHER_FEATURES+["swing_result_id"]
    _categorical_features = COUNT_CATEGORICAL+PITCH_CATEGORICAL+BATTER_CATEGORICAL+PITCHER_CATEGORICAL+["swing_result_id"]
    _numeric_features = COUNT_NUMERIC+PITCH_NUMERIC+BATTER_NUMERIC+PITCHER_NUMERIC
    _features = {"pitch":PITCH_FEATURES, "count":COUNT_FEATURES, "batter":["batter_age", "batter_exp"], "pitcher": ["pitcher_throws_lefty",]}
    _label = "swing_result_id"


######################################################################
###                         Batters
######################################################################


class BatterSwingDataset(SwingResultDataset):

    _featured_id = "batter_id"



class IsSwingDataset(CustomDataset):

    _main_table = "pitches"
    _main_id = "pitch_id"
    _joins = PITCH_JOINS
    _select_features = COUNT_FEATURES+PITCH_FEATURES+BATTER_FEATURES+PITCHER_FEATURES+["is_swing"]
    _categorical_features = COUNT_CATEGORICAL+PITCH_CATEGORICAL+BATTER_CATEGORICAL+PITCHER_CATEGORICAL+["is_swing"]
    _numeric_features = COUNT_NUMERIC+PITCH_NUMERIC+BATTER_NUMERIC+PITCHER_NUMERIC
    _featured_id = "batter_id"
    _features = {"pitch":PITCH_FEATURES, "count":COUNT_FEATURES, "batter":["batter_age", "batter_exp"], "pitcher": ["pitcher_throws_lefty",]}
    _label = "is_swing"



######################################################################
######################################################################




######################################################################
###                         Pitchers
######################################################################


class PitchTypeDataset(CustomDataset):

    _main_table = "pitches"
    _main_id = "pitch_id"
    _joins = PITCH_JOINS
    _select_features = COUNT_FEATURES+PITCH_FEATURES+BATTER_FEATURES+PITCHER_FEATURES
    _categorical_features = COUNT_CATEGORICAL+PITCH_CATEGORICAL+BATTER_CATEGORICAL+PITCHER_CATEGORICAL
    _numeric_features = COUNT_NUMERIC+PITCH_NUMERIC+BATTER_NUMERIC+PITCHER_NUMERIC
    _featured_id = "pitcher_id"
    _features = {"count":COUNT_FEATURES, "batter":["batter_age", "batter_exp", "batter_faces_break"], "pitcher": ["pitcher_age", "pitcher_exp", "pitcher_throws_lefty",]}
    _label = "pitch_type_id"


class PitchLocationDataset(PitchTypeDataset):

    _features = {"count":COUNT_FEATURES, "batter":["batter_age", "batter_exp", "batter_faces_break"], "pitcher": ["pitcher_age", "pitcher_exp", "pitcher_throws_lefty",], "pitch":["pitch_type_id",]}
    _label = "pitch_location"


class PitchVelocityDataset(PitchTypeDataset):

    _features = {"count":COUNT_FEATURES, "batter":["batter_age", "batter_exp", "batter_faces_break"], "pitcher": ["pitcher_age", "pitcher_exp", "pitcher_throws_lefty",], "pitch":["pitch_type_id",]}
    _label = "velocity"



class PitcherSwingDataset(SwingResultDataset):

    _featured_id = "pitcher_id"


######################################################################
###                         Stadiums
######################################################################


class IsHitDataset(CustomDataset):

    _main_table = "at_bats"
    _main_id = "at_bat_id"
    _joins = CONTACT_JOINS
    _select_features = HIT_FEATURES+["is_hit"]
    _categorical_features = HIT_CATEGORICAL+["is_hit"]
    _numeric_features = HIT_NUMERIC
    _featured_id = "stadium_id"
    _features = {"hit":HIT_FEATURES}
    _label = "is_hit"



######################################################################
######################################################################


class IsHRDataset(CustomDataset):
    
    _main_table = "at_bats"
    _main_id = "at_bat_id"
    _joins = CONTACT_JOINS
    _select_features = HIT_FEATURES+["ct.contact_type_id = 7 AS is_hr"]
    _categorical_features = HIT_CATEGORICAL+["is_hr"]
    _numeric_features = HIT_NUMERIC
    _featured_id = "stadium_id"
    _features = {"hit":HIT_FEATURES}
    _label = "is_hr"



######################################################################
######################################################################



class NumBasesIfHitDataset(CustomDataset):
    
    _main_table = "at_bats"
    _main_id = "at_bat_id"
    _joins = CONTACT_JOINS
    _select_features = HIT_FEATURES+["num_bases -1 AS num_bases"]
    _conditionStmt = "is_hit = True"
    _categorical_features = HIT_CATEGORICAL+["num_bases"]
    _numeric_features = HIT_NUMERIC
    _featured_id = "stadium_id"
    _features = {"hit":HIT_FEATURES}
    _label = "num_bases"