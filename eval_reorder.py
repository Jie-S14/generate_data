import pandas as pd


CSV_FILE_PATH = '/home/shenjie/ws/generate_data/45_135_025_045_eval/D0a-cubeOnly_episodes_same_box.csv'
df = pd.read_csv(CSV_FILE_PATH)

# # 1. 每个 episode 找到 role=box 的 y，作为该 episode 的排序依据
# episode_order = (
#     df[df["role"] == "box"]
#     .sort_values("y")
#     ["episode_id"]
#     .tolist()
# )

# # 2. 按 box.y 对整个 episode 排序
# df["episode_sort"] = pd.Categorical(
#     df["episode_id"],
#     categories=episode_order,
#     ordered=True
# )

# df = df.sort_values(
#     ["episode_sort", "role"],
#     # 如果你想让每个 episode 内 cube 在前、box 在后
#     key=lambda col: col.map({"cube": 0, "box": 1})
#     if col.name == "role" else col
# )

# # 3. 重新编号 episode_id
# episode_id_mapping = {
#     old_id: new_id
#     for new_id, old_id in enumerate(episode_order)
# }

# df["episode_id"] = df["episode_id"].map(episode_id_mapping)

# # 4. 删除临时排序列
# df = df.drop(columns=["episode_sort"])

# 找到每个 episode 对应的 box.y
box_y = (
    df[df["role"] == "box"]
    .set_index("episode_id")["y"]
)

# 按 box.y 从小到大排列 episode_id
episode_order = box_y.sort_values().index.tolist()

# 按新的 episode 顺序重新排列所有行
df = (
    df.set_index("episode_id")
      .loc[episode_order]
      .reset_index()
)

# 重新编号 episode_id: 0, 1, 2, ...
new_id = {
    old_id: i
    for i, old_id in enumerate(episode_order)
}

df["episode_id"] = df["episode_id"].map(new_id)


# 5. 写回 CSV
df.to_csv(CSV_FILE_PATH+"_ordered.csv", index=False)

print("Done!")