import os
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import requests
from kloppy import skillcorner
import warnings
warnings.filterwarnings("ignore")

def get_raw(file_list, destination_directory):
    """
    Downloads a list of files to a specified destination directory with progress bars.

    Args:
        file_list (list): A list of dictionaries, where each dictionary
                          contains 'url' (the URL of the file) and 'filename'
                          (the desired local filename).
        destination_directory (str): The path to the directory where files
                                     should be saved.
    """
    if not os.path.exists(destination_directory):
        print(f"Error directory {destination_directory} does not exist")
        return

    for file_info in tqdm(file_list, desc="Overall Download Progress"):
        url = file_info['url']
        filename = file_info['filename']
        local_filepath = os.path.join(destination_directory, filename)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(local_filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Downloading {filename}", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"Successfully downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {filename}: {e}")


run_cols = ['event_id',
 'match_id',
 'frame_start',
 'frame_end',
 'time_start',
 'time_end',
 'minute_start',
 'second_start',
 'duration',
 'period',
 'event_subtype', # Type of run
 'player_id',
 'player_name',
 'player_position',
 'team_id',
 'team_shortname',
 'x_start',
 'y_start',
 'x_end',
 'y_end',
 'game_state',
 'team_score',
 'opponent_team_score',
 'phase_index',
 'team_in_possession_phase_type',
 'team_out_of_possession_phase_type',
 'lead_to_shot', # possession lead to shot
 'lead_to_goal', # possession lead to goal
 'distance_covered',
 'trajectory_angle',
 'trajectory_direction',
 'separation_start',
 'separation_end',
 'separation_gain',
 'last_defensive_line_x_start',
 'last_defensive_line_x_end',
 'delta_to_last_defensive_line_start',
 'delta_to_last_defensive_line_end',
 'delta_to_last_defensive_line_gain',
 'last_defensive_line_height_start',
 'last_defensive_line_height_end',
 'last_defensive_line_height_gain',
 'inside_defensive_shape_start',
 'inside_defensive_shape_end',
 'location_to_player_in_possession_start',
 'location_to_player_in_possession_end',
 'distance_to_player_in_possession_start',
 'distance_to_player_in_possession_end',
 'player_in_possession_x_start',
 'player_in_possession_y_start',
 'player_in_possession_x_end',
 'player_in_possession_y_end',
 'targeted',
 'received', # Run received pass
 'dangerous', # run was dangerous
 'difficult_pass_target', # run was difficult to pass to
 'xthreat',
 'xpass_completion', # probability of succesfully passing to runner
 'passing_option_score',
 'predicted_passing_option',
 'n_simultaneous_runs', # number of runs going on at same time
 'passing_option_at_start',
 'n_opponents_ahead_end',
 'n_opponents_ahead_start',
 'n_opponents_overtaken', # Players passed by run
]
def get_run_events_from_match(match_id):
    event_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
    events_df =  pd.read_csv(event_data_github_url)
    runs = events_df[events_df["event_type"] == "off_ball_run"]
    open_play_runs = runs[runs["team_in_possession_phase_type"].apply(lambda type_: type_ not in {"set_play","chaotic"})][run_cols]
    return open_play_runs

def get_tracking_data_from_match_id(match_id):
    tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

    dataset = skillcorner.load(
        meta_data=meta_data_github_url,
        raw_data=tracking_data_github_url,
        # Optional Parameters
        coordinates="skillcorner",  # or specify a different coordinate system
        #sample_rate=(1 / 2),  # changes the data from 10fps to 5fps
    )

    tracking_df = (
        dataset.transform(
            to_orientation="static_home_away"
        )  # Now, all attacks happen from left to right for home_team
        .to_df(
            engine="pandas"
        )  # Convert to a Polars DataFrame, or use engine="pandas" for a Pandas DataFrame
    )
    
    meta_json = json.loads(requests.get(meta_data_github_url).text)
    player_id_to_team_id = pd.json_normalize(meta_json["players"])[["id","team_id"]]
    home_team = meta_json["home_team"]["id"]
    # for i,player_id in player_id_to_team_id.iterrows():
    #     tracking_df[f"{player_id["id"]}_team_id"] = player_id["team_id"]
    
    return tracking_df, player_id_to_team_id, home_team

def add_run_curve_ratio(tracking_df,run_features,player_id):
    def run_curviness_ratio(x, y):
        try:
            path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            straight_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
            return path_length / straight_distance - 1
        except IndexError:
            return -1
    active_frames = tracking_df[tracking_df["run_active"]]
    x = active_frames.loc[:,f"{player_id}_x"].values
    y = active_frames.loc[:,f"{player_id}_y"].values
    
    curve_ratio = run_curviness_ratio(x,y)
    
    run_features["run_curve_ratio"] = curve_ratio
    
    return run_features

def get_runs_from_match(match_id):
    run_events = get_run_events_from_match(match_id=match_id)
    tracking_df,player_to_team, home_team = get_tracking_data_from_match_id(match_id=match_id)
    run_features_all = []
    run_tracking_all = []
    for i,run_row in run_events.iterrows():
        event_id = run_row["event_id"]
        player_id = run_row["player_id"]
        team_id = run_row["team_id"]
        run_id = run_row["event_id"]
        start_frame = run_row["frame_start"]
        end_frame = run_row["frame_end"]
        mask = (tracking_df["frame_id"]>=start_frame) & (tracking_df["frame_id"]<=end_frame)
        run_tracking = tracking_df[(tracking_df["frame_id"]>=start_frame-20) & (tracking_df["frame_id"]<=end_frame+20)] # get all the frames that occur from 2 seconds before run all the way to 2 seconds after run
        #run_tracking = tracking_df[mask]
        if run_tracking.shape[0] == 0: # Runs that occur while the ball is dead
            print("No tracking data for run")
            continue
        
        if team_id!=home_team: # Flip to make sure the away team's run is represented as attacking left to right
            for col in run_tracking.columns:
                if col.endswith("_x") or col.endswith("_y"):
                    run_tracking[col] = -run_tracking[col]                    
        
        run_tracking["match_id"] = match_id
        run_tracking["run_id"] = run_id
        run_tracking.loc[mask,"run_active"] = True
        run_tracking.loc[~mask,"run_active"] = False
        
        meta_cols = [
            'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
            'ball_x', 'ball_y', 'ball_z', 'ball_speed', 'match_id', 'run_id',
            'run_active']

        player_ids = sorted({
            col.split('_')[0] for col in run_tracking.columns 
            if col not in meta_cols
        })
        for pid in player_ids:
            x_col, y_col = f"{pid}_x", f"{pid}_y"
            s_col, d_col = f"{pid}_s", f"{pid}_d"

            if x_col in run_tracking.columns and y_col in run_tracking.columns:
                # Velocity components
                vx = run_tracking[x_col].diff() / .1
                vy = run_tracking[y_col].diff() / .1

                # Speed (magnitude of velocity vector)
                run_tracking[s_col] = np.sqrt(vx**2 + vy**2)

                # Direction (angle in radians, atan2 handles quadrants)
                run_tracking[d_col] = np.arctan2(vy, vx)
        vx = run_tracking["ball_x"].diff() / .1
        vy = run_tracking["ball_y"].diff() / .1
        vz = run_tracking["ball_z"].diff() / .1

        # Compute 3D speed (magnitude of velocity vector)
        run_tracking["ball_speed"] = np.sqrt(vx**2 + vy**2 + vz**2)
        
        run_features = run_row[run_cols]
        run_features = add_run_curve_ratio(tracking_df=run_tracking,run_features=run_features,player_id=player_id)
        run_features["id"] = event_id
        run_tracking["id"] = event_id
        # run_object = run(run_features,run_tracking,player_to_team)
        # run_object.add_run_curve_ratio()
        #run_objects.append(run_object)
        run_features_all.append(run_features)
        run_tracking_all.append(run_tracking)
        
    run_features_all = pd.concat(run_features_all,axis=1).T
    run_tracking_all = pd.concat(run_tracking_all)
    meta_cols = [
        'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
        'ball_x', 'ball_y', 'ball_z', 'ball_speed', 'match_id', 'run_id',
        'run_active','id']

    player_cols = [c for c in run_tracking_all.columns if c not in meta_cols]

    tracking_long = (
        run_tracking_all
        .melt(
            id_vars=meta_cols,
            value_vars=player_cols,
            var_name='player_feature',
            value_name='value'
        )
    )

    tracking_long[['player', 'feature']] = tracking_long['player_feature'].str.extract(r'(\d+)_(x|y|d|s)')

    tracking_long = (
        tracking_long
        .pivot(index=meta_cols + ['player'], columns='feature', values='value')
        .reset_index()
    )

    tracking_long = tracking_long[meta_cols + ['player', 'x', 'y', 'd', 's']]
    
        
        
        
    return run_features_all, tracking_long, player_to_team

def get_possessions_from_match_id(match_id):
    phases_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_phases_of_play.csv"
    phases_df =  pd.read_csv(phases_data_github_url)

    possessions = []
    curr_poss = set()
    curr_types = set()
    poss_frame_start = phases_df.iloc[0]["frame_start"]
    poss_frame_end = phases_df.iloc[0]["frame_end"]
    for i in range(phases_df.shape[0]-1):
        phase = phases_df.iloc[i]
        next_phase = phases_df.iloc[i+1]
        if phase.team_in_possession_id == next_phase.team_in_possession_id and phase.frame_end==next_phase.frame_start:
            curr_poss.add(phase["index"])
            curr_types.add(phase.team_in_possession_phase_type)
            curr_poss.add(next_phase["index"])
            curr_types.add(next_phase.team_in_possession_phase_type)
        else:
            curr_poss.add(phase["index"])
            curr_types.add(phase.team_in_possession_phase_type)
            poss_frame_end = phase.frame_end
            possessions.append([phase.match_id,phase.period,curr_poss,poss_frame_start,poss_frame_end,phase.team_in_possession_id,curr_types,phase.team_possession_lead_to_shot,phase.team_possession_lead_to_goal,phase.penalty_area_end])
            poss_frame_start = next_phase.frame_start
            curr_poss = set()
            curr_types = set()
    
    phase = phases_df.iloc[-1]
    curr_poss.add(phase["index"])
    poss_frame_end = phase.frame_end
    curr_types.add(phase.team_in_possession_phase_type)
    possessions.append([phase.match_id,phase.period,curr_poss,poss_frame_start,poss_frame_end,phase.team_in_possession_id,curr_types,phase.team_possession_lead_to_shot,phase.team_possession_lead_to_goal,phase.penalty_area_end])

        

    possessions = pd.DataFrame(possessions,columns = ["match_id","period","phases_indexes","frame_start","frame_end","team_id_possession","phase_types","possession_lead_to_shot","possession_lead_to_goal","possession_leads_to_box"])
    possessions = possessions.reset_index(names="possession_index")
    return possessions
            
def collect_all_data():
    matches_info_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json"
    response = requests.get(matches_info_github_url)
    response.raise_for_status()  # raises error if download failed
    matches_info_json = json.loads(response.text)
    matches_info = pd.json_normalize(matches_info_json)
    
    possessions = []
    run_features = []
    tracking_data = []
    player_team = []
    for match_id in tqdm(matches_info["id"]):
        poss = get_possessions_from_match_id(match_id=match_id)
        possessions.append(poss)
        
        run_feat, tracking, player_to_team = get_runs_from_match(match_id=match_id)
        run_features.append(run_feat)
        tracking_data.append(tracking)
        player_team.append(player_to_team)
        
    possessions = pd.concat(possessions)
    run_features = pd.concat(run_features)
    tracking_data = pd.concat(tracking_data)
    player_team = pd.concat(player_team)

    run_features = pd.merge(run_features,possessions.explode(["phases_indexes"])[["match_id","phases_indexes","possession_index"]],left_on=["match_id","phase_index"],right_on=["match_id","phases_indexes"],how="left")
    
    run_features = run_features.dropna()
    
    tracking_data_2 = tracking_data[(~((tracking_data["ball_x"].isna()) & (tracking_data["ball_y"].isna()))) & (~((tracking_data["x"].isna()) & (tracking_data["y"].isna())))].drop(["ball_owning_team_id","ball_state","ball_z"],axis=1)
    tracking_data_2["s"] = tracking_data_2["s"].fillna(0)
    tracking_data_2["d"] = tracking_data_2["d"].fillna(0)
    tracking_data_2["ball_speed"] = tracking_data_2["ball_speed"].fillna(0)
    
    player_team = player_team.drop_duplicates()
    
    tracking_data_2 = add_velocity_acceleration_to_tracking(tracking_data_2)
    
    tracking_data_2["player"] = tracking_data_2["player"].astype(int)
    
    merged = pd.merge(possessions,run_features,on=["match_id","possession_index"],how="outer",suffixes=("_possession","_run"))

    merged["possession_lead_to_shot"] = (merged["possession_lead_to_shot"] | merged["lead_to_shot"])# Need to update wheter runs and possessions lead to shots on values that conflict
    merged["possession_lead_to_goal"] = (merged["possession_lead_to_goal"] | merged["lead_to_goal"])

    
    return possessions, run_features, tracking_data_2, player_team.set_index("id"), merged

def add_velocity_acceleration_to_tracking(tracking_data):
    tracking_data = tracking_data.sort_values(
        ["match_id","run_id","player", "timestamp"]
    )

    tracking_data["dx"] = (
        tracking_data["x"]
        - tracking_data.groupby(["match_id","player", "run_id"])["x"].shift(1)
    )
    tracking_data["dy"] = (
        tracking_data["y"]
        - tracking_data.groupby(["match_id","player", "run_id"])["y"].shift(1)
    )

    tracking_data["ax"] = (
        tracking_data["dx"]
        - tracking_data.groupby(["match_id","player", "run_id"])["dx"].shift(1)
    )
    tracking_data["ay"] = (
        tracking_data["dy"]
        - tracking_data.groupby(["match_id","player", "run_id"])["dy"].shift(1)
    )
    # Fill na first rows
    tracking_data[["dx", "dy","ax","ay"]] = (
        tracking_data.groupby(["match_id", "run_id", "player"])[["dx", "dy","ax","ay"]]
        .bfill()
    )

    #Speed
    FPS = 10

    tracking_data["speed"] = (
        np.sqrt(
            tracking_data["dx"]**2 +
            tracking_data["dy"]**2
        ) * FPS
    )
    tracking_data["speed_direction"] = np.arctan2(
        tracking_data["dy"],
        tracking_data["dx"]
    )
    eps = 1e-6

    vx = tracking_data["dx"]
    vy = tracking_data["dy"]
    speed_frame = np.sqrt(vx**2 + vy**2)

    vhat_x = vx / (speed_frame + eps)
    vhat_y = vy / (speed_frame + eps)

    tracking_data["acceleration"] = (
        (tracking_data["ax"] * vhat_x +
        tracking_data["ay"] * vhat_y)
        * FPS**2
    )
    tracking_data["acc_direction"] = np.arctan2(
        tracking_data["ay"],
        tracking_data["ax"]
    )
    
    # Ball Speed and acceleration
    tracking_data["ball_dx"] = (
        tracking_data["ball_x"]
        - tracking_data.groupby(["match_id", "period_id","run_id"])["ball_x"].shift(1)
    )

    tracking_data["ball_dy"] = (
        tracking_data["ball_y"]
        - tracking_data.groupby(["match_id", "period_id","run_id"])["ball_y"].shift(1)
    )

    tracking_data["ball_ax"] = (
        tracking_data["ball_dx"]
        - tracking_data.groupby(["match_id", "period_id","run_id"])["ball_dx"].shift(1)
    )

    tracking_data["ball_ay"] = (
        tracking_data["ball_dy"]
        - tracking_data.groupby(["match_id", "period_id","run_id"])["ball_dy"].shift(1)
    )
    # Fill na first rows
    tracking_data[["ball_dx", "ball_dy","ball_ax","ball_ay"]] = (
        tracking_data.groupby(["match_id", "run_id", "player"])[["ball_dx", "ball_dy","ball_ax","ball_ay"]]
        .bfill()
    )

    tracking_data["ball_speed"] = (
        np.sqrt(
            tracking_data["ball_dx"]**2 +
            tracking_data["ball_dy"]**2
        ) * FPS
    )

    tracking_data["ball_speed_direction"] = np.arctan2(
        tracking_data["ball_dy"],
        tracking_data["ball_dx"]
    )
    vx = tracking_data["ball_dx"]
    vy = tracking_data["ball_dy"]

    speed_frame = np.sqrt(vx**2 + vy**2)

    vhat_x = vx / (speed_frame + eps)
    vhat_y = vy / (speed_frame + eps)

    tracking_data["ball_acceleration"] = (
        (tracking_data["ball_ax"] * vhat_x +
        tracking_data["ball_ay"] * vhat_y)
        * FPS**2
    )

    tracking_data["ball_acc_direction"] = np.arctan2(
        tracking_data["ball_ay"],
        tracking_data["ball_ax"]
    )
    return tracking_data

def collect_data_from_matches(match_ids):    
    possessions = []
    run_features = []
    tracking_data = []
    player_team = []
    for match_id in tqdm(match_ids):
        poss = get_possessions_from_match_id(match_id=match_id)
        possessions.append(poss)
        
        run_feat, tracking, player_to_team = get_runs_from_match(match_id=match_id)
        run_features.append(run_feat)
        tracking_data.append(tracking)
        player_team.append(player_to_team)
        
    possessions = pd.concat(possessions)
    run_features = pd.concat(run_features)
    tracking_data = pd.concat(tracking_data)
    player_team = pd.concat(player_team)

    run_features = pd.merge(run_features,possessions.explode(["phases_indexes"])[["match_id","phases_indexes","possession_index"]],left_on=["match_id","phase_index"],right_on=["match_id","phases_indexes"],how="left")
    
    run_features = run_features.dropna()
    
    tracking_data_2 = tracking_data[(~((tracking_data["ball_x"].isna()) & (tracking_data["ball_y"].isna()))) & (~((tracking_data["x"].isna()) & (tracking_data["y"].isna())))].drop(["ball_owning_team_id","ball_state","ball_z"],axis=1)
    tracking_data_2["s"] = tracking_data_2["s"].fillna(0)
    tracking_data_2["d"] = tracking_data_2["d"].fillna(0)
    tracking_data_2["ball_speed"] = tracking_data_2["ball_speed"].fillna(0)
    
    player_team = player_team.drop_duplicates()
    
    tracking_data_2 = add_velocity_acceleration_to_tracking(tracking_data_2)
    
    tracking_data_2["player"] = tracking_data_2["player"].astype(int)
    
    merged = pd.merge(possessions,run_features,on=["match_id","possession_index"],how="outer",suffixes=("_possession","_run"))

    merged["possession_lead_to_shot"] = (merged["possession_lead_to_shot"] | merged["lead_to_shot"])# Need to update wheter runs and possessions lead to shots on values that conflict
    merged["possession_lead_to_goal"] = (merged["possession_lead_to_goal"] | merged["lead_to_goal"])

    
    return possessions, run_features, tracking_data_2, player_team.set_index("id"), merged

