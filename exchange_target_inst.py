import pandas as pd
import copy

INPUT_FILE_PATH = '/home/shenjie/ws/generate_data/45_135_025_045_train_1anchor/D1_episodes_50ep_to_pair2.csv'
OUTPUT_FILE_PATH = '/home/shenjie/ws/generate_data/45_135_025_045_train_1anchor/D1_episodes_50ep_to_pair2_epid.csv'

df = pd.read_csv(INPUT_FILE_PATH) #.tail(60)
df_flipped = copy.deepcopy(df)

for episode_id, group in df.groupby("episode_id"):
    # # 当前 combo 中的 cube
    # cube_mask = (df_flipped["episode_id"] == episode_id) & (df_flipped["role"] == "cube")

    # # 反转 is_target
    # df_flipped.loc[cube_mask, "is_target"] = ~df_flipped.loc[cube_mask, "is_target"].astype(bool)

    # # 找新的 target cube
    # target_cube = df_flipped.loc[cube_mask & (df_flipped["is_target"] == True)].iloc[0]

    # new_color = target_cube["color"]

    # # 修改整个 combo 的 instruction
    episode_mask = df_flipped["episode_id"] == episode_id

    # df_flipped.loc[episode_mask, "instruction"] = (
    #     f"pick up the {new_color} cube and place it in the box"
    # )

    df_flipped.loc[episode_mask, "episode_id"] = int(episode_id) - 50

df_flipped.to_csv(OUTPUT_FILE_PATH, index=False)