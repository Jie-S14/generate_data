import pandas as pd


CSV_FILE_PATH = '/home/shenjie/ws/generate_data/D0a/D0a-cubeOnly_1anchor_1ori.csv'
df = pd.read_csv(CSV_FILE_PATH)

# df.loc[df["role"] == "box", "orientation_deg"] = 90.0
# df.loc[df["role"] == "box", "x"] = 0.08
# df.loc[df["role"] == "box", "y"] = 0.3
df.loc[df["role"] == "box", "color"] = "blue"

df.to_csv('/home/shenjie/ws/generate_data/D0a/D0a-cubeOnly_1anchor_1ori_test.csv', index=False)
