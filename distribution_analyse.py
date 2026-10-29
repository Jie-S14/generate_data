import pandas as pd
import numpy as np

df = pd.read_csv("/home/shenjie/ws/generate_data/45_135_025_045_train_1anchor/D1_episodes_paired.csv")#.head(150)

# 只看每个episode里"被选为target的cube"这一行——这才是我们要对比的"目标位置"
target_cubes = df[(df["role"] == "cube") & (df["is_target"] == True)]

def nearest_neighbor_stats(group: pd.DataFrame, label: str):
    pts = group[["x", "y"]].to_numpy()
    n = len(pts)
    if n < 2:
        print(f"{label}: 样本数({n})不足，跳过")
        return
    # 计算每个点到"自己这组里"最近的另一个点的距离
    dists = np.full(n, np.inf)
    for i in range(n):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf  # 排除自己
        dists[i] = d.min()
    print(f"{label}: n={n}")
    print(f"  最近邻距离: mean={dists.mean():.4f}, min={dists.min():.4f}, "
          f"max={dists.max():.4f}, std={dists.std():.4f}")
    print(f"  覆盖范围: x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
          f"(std={pts[:,0].std():.4f}), "
          f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] (std={pts[:,1].std():.4f})")

for color in target_cubes["color"].unique():
    group = target_cubes[target_cubes["color"] == color]
    nearest_neighbor_stats(group, f"color={color}")