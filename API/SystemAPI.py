import os 
import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter



# Load the user segments
user_segments = pd.DataFrame()
PROJECT_ROOT = os.path.abspath(os.getcwd()) 
DATA_PATH = os.path.join(PROJECT_ROOT, "data/")

# Load the uswe segment data
user_segments = {}
with open(os.path.join(DATA_PATH, "user_segments_viz.pkl"), "rb") as f:
    data = pkl.load(f)
    user_segments = pd.DataFrame.from_dict(data, orient="index").reset_index()
    user_segments.rename(columns={'index': 'id'}, inplace=True)

def load_user_segments():
    segms = user_segments.copy()
    segments = []
    for _, row in segms.iterrows():
        segment_obj = {
            "id": row["id"],
            "title": row["title"],
            "unique_users": row["df"].user_pseudo_id.nunique(),
            "trend": (
                row["df"]
                .groupby("event_date")
                .session_id_unique.count()
                .fillna(0)
                .tolist()
            ), 
        }
        segments.append(segment_obj)
    segments 
    return segments

def load_user_segments_detail(id):
    def get_n_weeks(df):
      return np.median(list(Counter(list(map(lambda x: x.weekday(), df.event_date.unique()))).values()))
  
    segms = user_segments.copy()
    id = int(id)
    segment_det = segms[segms["id"] == id]

    if segment_det.empty:
        return {"error": f"No segment found with id {id}"}

    row = segment_det.iloc[0]

    # MEtrics calculation
    engage_time = float(row["df"][(row["df"]["true_engagement"] == True) & (row["df"]["diff"] > 0) & (row["df"]["diff"] < 30*60)]["diff"].mean() / 60)
    n_eng = row["df"]["true_engagement"].sum()
    n_not_eng = row["df"].shape[0] - n_eng

    # Time perspective
    # 1
    target_cluster_temporal = row["df"].groupby("event_on_weekend").id.count()
    n_weeks = get_n_weeks(row["df"])
    week, weekend = target_cluster_temporal / [5*n_weeks, 2*n_weeks]
    #2 
    bins = list(range(0, 25, 1))
    labels = [f'{s:02}:00-{e-1:02}:59' for s, e in zip(bins[:-1], bins[1:])]
    row["df"].loc[:, 'time_bin'] = pd.cut(
    row["df"].event_time.dt.hour, bins, labels=labels, right=False)
    day_consumption = row["df"].groupby("time_bin").event_date.count()
    day_consumption_times = [str(k) for k in day_consumption.index]
    day_consumption_values = [int(v) for v in day_consumption.values]
    #3
    df_day_count = row["df"].groupby(row["df"].event_date.dt.day_name()).count().loc[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
    day_cnt_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_cnt_vals = [int(v) for v in df_day_count['event_date'].values]

    # transition matrix las 
    mm_cat = row["seq_model"]["df"].iloc[:-1,:-1]

    segment_obj = {
      "id": int(row["id"]),                             
      "title": str(row["title"]),
      "desc": str(row["desc"]),
      "unique_users": int(row["df"].user_pseudo_id.nunique()),  
      "regions": row["regions"],                         
      "freq_users": int(row["user_type_cnt"]['frequent']),              
      "not_freq_users": int(row["user_type_cnt"]['nonfrequent']),              
      "regions_desc": str(row["regions_desc"]),
      "engage_time": engage_time,
      "unique_sessions": int(len(row["df"]["session_id_unique"])),
      "engaged": int(n_eng),
      "not_engaged": int(n_not_eng),
      # Time perspective
      "weekday": int(week),
      "weekend": int(weekend),
      "day_consumption": {
        "times": day_consumption_times,
        "values": day_consumption_values
      },
      "day_count":{
        "day": day_cnt_names,
        "value": day_cnt_vals
      },
      # Transition Matrix 
      "mm_cat": mm_cat
    }
    return segment_obj



