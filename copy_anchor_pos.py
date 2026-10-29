import pandas as pd

TRAIN_FILE_PATH = '/home/shenjie/ws/generate_data/45_135_025_045/D0a-cubeOnly_episodes.csv_ordered.csv' 
EVAL_FILE_PATH = '/home/shenjie/ws/generate_data/45_135_025_045_eval/D0a-cubeOnly_episodes.csv_ordered.csv'

df_train = pd.read_csv(TRAIN_FILE_PATH)
df_eval = pd.read_csv(EVAL_FILE_PATH)

# Only need the box rows
df_train_box = df_train[df_train["role"] == "box"]

# fix generation bug: col color
df_train_box["color"] = "blue"
df_eval.loc[df_eval["role"] == "box", "color"] = "blue"

index_cols = ["episode_id", "role"]
update_cols = ["x", "y", "orientation_deg"]

df_train_box = df_train_box.set_index(["episode_id", "role"])
df_eval = df_eval.set_index(["episode_id", "role"])

df_eval.update(df_train_box[update_cols])

df_eval = df_eval.reset_index()

df_eval.to_csv(EVAL_FILE_PATH.split("csv_ordered")[0]+"_update_box.csv", index=False)
