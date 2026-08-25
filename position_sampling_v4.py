"""
position_sampling.py (v4)
==========================
D0~D3 四阶段物体位置 + 朝向(orientation)采样。

v4 相比 v3 的改动
-----------------
新增每个物体的随机朝向(orientation)采样：
- cube: uniform(-0.523599, 0.523599) rad (±30°)
- box:  uniform(-0.174533, 0.174533) rad (±10°)

orientation 采样是独立于 position 的简单 1D 均匀采样，不做 blue-noise
覆盖处理——如果后续你想把 orientation 也当作一个需要"覆盖度"追踪的
target 维度（呼应你 compound OOD 里 orientation 是独立瓶颈的发现），
可以参考 target position 的池子思路，对 orientation 值本身做 1D
poisson-disk 式的间距采样，本版本先给最简单可用的版本。

设计延续 v3
-----------
- target_min_dist(大)：只用来生成 target 角色的位置池，保证全数据集
  尺度上的均匀覆盖
- object_min_dist(小)：只用于任意两物体间的碰撞检查
- distractor 角色：自由采样，不追求跨episode覆盖均匀性，只管不碰撞
"""

from __future__ import annotations

import csv
import itertools
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# =====================================================================
# 1. 极坐标面积均匀采样修正
# =====================================================================

def sample_polar_uniform(
    r_min: float, r_max: float, theta_min: float, theta_max: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
    theta = rng.uniform(theta_min, theta_max)
    return r * np.cos(theta), r * np.sin(theta)


def in_polar_region(
    x: float, y: float, r_min: float, r_max: float,
    theta_min: float, theta_max: float,
) -> bool:
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    if theta_min <= theta_max:
        theta_ok = theta_min <= theta <= theta_max
    else:
        theta_ok = theta >= theta_min or theta <= theta_max
    return (r_min <= r <= r_max) and theta_ok


# =====================================================================
# 2. Orientation 采样(独立于 position 的简单均匀采样)
# =====================================================================

def sample_orientation(rad_min: float, rad_max: float, rng: np.random.Generator) -> float:
    """在给定角度范围内做简单均匀采样，返回弧度值。"""
    return float(rng.uniform(rad_min, rad_max))


# =====================================================================
# 3. Poisson-disk (Bridson's algorithm) —— 只用于生成 target 覆盖池
# =====================================================================

def poisson_disk_polar(
    r_min: float, r_max: float, theta_min: float, theta_max: float,
    min_dist: float, k: int = 30, max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    cell_size = min_dist / np.sqrt(2)
    x_min, x_max = -r_max, r_max
    y_min, y_max = -r_max, r_max
    grid_w = int(np.ceil((x_max - x_min) / cell_size)) + 1
    grid_h = int(np.ceil((y_max - y_min) / cell_size)) + 1
    grid = -np.ones((grid_w, grid_h), dtype=int)

    def grid_coords(p):
        return int((p[0] - x_min) / cell_size), int((p[1] - y_min) / cell_size)

    points: List[np.ndarray] = []

    def fits(p) -> bool:
        if not in_polar_region(p[0], p[1], r_min, r_max, theta_min, theta_max):
            return False
        gx, gy = grid_coords(p)
        for i in range(max(gx - 2, 0), min(gx + 3, grid_w)):
            for j in range(max(gy - 2, 0), min(gy + 3, grid_h)):
                idx = grid[i, j]
                if idx != -1:
                    q = points[idx]
                    if np.hypot(p[0] - q[0], p[1] - q[1]) < min_dist:
                        return False
        return True

    x0 = y0 = None
    for _ in range(1000):
        cx, cy = sample_polar_uniform(r_min, r_max, theta_min, theta_max, rng)
        if in_polar_region(cx, cy, r_min, r_max, theta_min, theta_max):
            x0, y0 = cx, cy
            break
    if x0 is None:
        raise RuntimeError("无法在给定极坐标区域内生成初始种子点，检查参数范围")

    points.append(np.array([x0, y0]))
    gx, gy = grid_coords(points[0])
    grid[gx, gy] = 0
    active = [0]

    while active:
        if max_points is not None and len(points) >= max_points:
            break
        pick = rng.integers(len(active))
        idx = active[pick]
        base = points[idx]
        found = False
        for _ in range(k):
            rad = rng.uniform(min_dist, 2 * min_dist)
            ang = rng.uniform(0, 2 * np.pi)
            cand = base + rad * np.array([np.cos(ang), np.sin(ang)])
            if fits(cand):
                points.append(cand)
                gx, gy = grid_coords(cand)
                grid[gx, gy] = len(points) - 1
                active.append(len(points) - 1)
                found = True
                break
        if not found:
            active.pop(pick)

    return np.array(points)


# =====================================================================
# 4. Workspace：碰撞距离与覆盖间距分开配置
# =====================================================================

@dataclass
class Workspace:
    """极坐标工作空间标定参数。"""
    r_min: float = 0.25
    r_max: float = 0.45
    theta_min: float = -0.785398
    theta_max: float = 0.785398
    object_min_dist: float = 0.05   # 任意两物体的碰撞/可抓取安全距离(小)
    target_min_dist: float = 0.05   # target 位置在全数据集内的覆盖间距(大)

    def __post_init__(self):
        if self.target_min_dist < self.object_min_dist:
            warnings.warn(
                "target_min_dist 小于 object_min_dist，通常覆盖间距应该"
                "不小于碰撞距离，请确认这是有意为之。"
            )


# =====================================================================
# 5. target 角色：全局覆盖池
# =====================================================================

def generate_manipulation_positions(
    n_episodes: int, ws: Workspace, rng: np.random.Generator, k: int = 30,
) -> np.ndarray:
    pool = poisson_disk_polar(
        ws.r_min, ws.r_max, ws.theta_min, ws.theta_max,
        ws.target_min_dist, k=k, max_points=None, rng=rng,
    )
    if len(pool) == 0:
        raise RuntimeError("target_min_dist 过大或区域过小，无法生成任何点")
    if len(pool) < n_episodes:
        warnings.warn(
            f"target_min_dist={ws.target_min_dist} 下该 workspace 最多生成 "
            f"{len(pool)} 个点，少于所需 {n_episodes} 个，将循环复用。"
            f"如需更高位置多样性，请调小 target_min_dist 或扩大 workspace。"
        )

    positions = []
    remaining = n_episodes
    while remaining > 0:
        order = rng.permutation(len(pool))
        take = min(remaining, len(pool))
        positions.append(pool[order[:take]])
        remaining -= take
    return np.concatenate(positions, axis=0)


def pair_targets_with_repair(
    pos_a: np.ndarray, pos_b: np.ndarray, min_dist: float,
    rng: np.random.Generator, max_attempts: int = 50,
) -> np.ndarray:
    pos_b = pos_b.copy()
    n = len(pos_a)
    unresolved = 0
    for i in range(n):
        d = np.hypot(*(pos_a[i] - pos_b[i]))
        if d >= min_dist:
            continue
        fixed = False
        for _ in range(max_attempts):
            j = rng.integers(n)
            if j == i:
                continue
            d_i_new = np.hypot(*(pos_a[i] - pos_b[j]))
            d_j_new = np.hypot(*(pos_a[j] - pos_b[i]))
            if d_i_new >= min_dist and d_j_new >= min_dist:
                pos_b[[i, j]] = pos_b[[j, i]]
                fixed = True
                break
        if not fixed:
            unresolved += 1
    if unresolved > 0:
        warnings.warn(
            f"{unresolved} 个 episode 未能通过交换修复 target 间冲突，"
            f"建议增大 workspace 或调整 target_min_dist / object_min_dist 比例。"
        )
    return pos_b


# =====================================================================
# 6. distractor 角色：自由采样，只管碰撞
# =====================================================================

def sample_free_position(
    existing_points: List[np.ndarray], ws: Workspace,
    rng: np.random.Generator, max_tries: int = 300,
) -> np.ndarray:
    for _ in range(max_tries):
        x, y = sample_polar_uniform(ws.r_min, ws.r_max, ws.theta_min, ws.theta_max, rng)
        if all(np.hypot(x - ex[0], y - ex[1]) >= ws.object_min_dist for ex in existing_points):
            return np.array([x, y])
    raise RuntimeError(
        "无法为 distractor 找到满足 object_min_dist 的位置，"
        "检查 object_min_dist 是否相对 workspace 过大"
    )


def round_robin_assignment(items: List[str], cycle_idx: int) -> List[str]:
    n = len(items)
    shift = cycle_idx % n
    return items[shift:] + items[:shift]


def make_object(role: str, color: str, x: float, y: float, rng: np.random.Generator,
                 is_target: Optional[bool] = None) -> Dict[str, Any]:
    """统一构造一个物体字典，自动附上按角色类型采样的朝向。"""
    rad_min, rad_max = ORIENTATION_RANGES[role]
    obj = {
        "role": role, "color": color,
        "x": float(x), "y": float(y),
        "orientation_rad": sample_orientation(rad_min, rad_max, rng),
    }
    if is_target is not None:
        obj["is_target"] = is_target
    return obj


# =====================================================================
# 7. 各阶段 episode 生成器
# =====================================================================

def generate_D0_dataset(n_episodes: int, ws: Workspace, seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cube_pos = generate_manipulation_positions(n_episodes, ws, rng)
    box_pos = generate_manipulation_positions(n_episodes, ws, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, ws.object_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        episodes.append({
            "episode_id": ep, "stage": "D0",
            "objects": [
                make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng),
                make_object("box", "red", box_pos[ep, 0], box_pos[ep, 1], rng),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def generate_D1_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 1,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    total = episodes_per_combo * len(colors)
    cube_target_pos = generate_manipulation_positions(total, ws, rng)
    box_pos = generate_manipulation_positions(total, ws, rng)
    box_pos = pair_targets_with_repair(cube_target_pos, box_pos, ws.object_min_dist, rng)

    episodes = []
    for ep in range(total):
        target_color = colors[ep % len(colors)]
        distractor_color = colors[(ep + 1) % len(colors)]
        distractor_pos = sample_free_position([cube_target_pos[ep], box_pos[ep]], ws, rng)

        episodes.append({
            "episode_id": ep, "stage": "D1",
            "objects": [
                make_object("cube", target_color, *cube_target_pos[ep], rng, is_target=True),
                make_object("cube", distractor_color, *distractor_pos, rng, is_target=False),
                make_object("box", "neutral", *box_pos[ep], rng),
            ],
            "instruction": f"pick up the {target_color} cube and place it in the box",
        })
    return episodes


def generate_D2_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 2,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    total = episodes_per_combo * len(colors)
    cube_pos = generate_manipulation_positions(total, ws, rng)
    box_target_pos = generate_manipulation_positions(total, ws, rng)
    box_target_pos = pair_targets_with_repair(cube_pos, box_target_pos, ws.object_min_dist, rng)

    episodes = []
    for ep in range(total):
        target_color = colors[ep % len(colors)]
        distractor_color = colors[(ep + 1) % len(colors)]
        distractor_pos = sample_free_position([cube_pos[ep], box_target_pos[ep]], ws, rng)

        episodes.append({
            "episode_id": ep, "stage": "D2",
            "objects": [
                make_object("cube", "neutral", *cube_pos[ep], rng),
                make_object("box", target_color, *box_target_pos[ep], rng, is_target=True),
                make_object("box", distractor_color, *distractor_pos, rng, is_target=False),
            ],
            "instruction": f"pick up the cube and place it in the {target_color} box",
        })
    return episodes


def generate_D3_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 3,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(colors, colors))
    total = episodes_per_combo * len(combos)

    cube_target_pos = generate_manipulation_positions(total, ws, rng)
    box_target_pos = generate_manipulation_positions(total, ws, rng)
    box_target_pos = pair_targets_with_repair(cube_target_pos, box_target_pos, ws.object_min_dist, rng)

    episodes = []
    ep = 0
    for _ in range(episodes_per_combo):
        order = rng.permutation(len(combos))
        for combo_idx in order:
            src_color, dst_color = combos[combo_idx]
            src_distractor = colors[1] if src_color == colors[0] else colors[0]
            dst_distractor = colors[1] if dst_color == colors[0] else colors[0]

            placed = [cube_target_pos[ep], box_target_pos[ep]]
            cube_distractor_pos = sample_free_position(placed, ws, rng)
            placed.append(cube_distractor_pos)
            box_distractor_pos = sample_free_position(placed, ws, rng)

            episodes.append({
                "episode_id": ep, "stage": "D3", "combo": f"{src_color}->{dst_color}",
                "objects": [
                    make_object("cube", src_color, *cube_target_pos[ep], rng, is_target=True),
                    make_object("cube", src_distractor, *cube_distractor_pos, rng, is_target=False),
                    make_object("box", dst_color, *box_target_pos[ep], rng, is_target=True),
                    make_object("box", dst_distractor, *box_distractor_pos, rng, is_target=False),
                ],
                "instruction": f"pick up the {src_color} cube and place it in the {dst_color} box",
            })
            ep += 1
    return episodes


# =====================================================================
# 8. 导出 & 验证
# =====================================================================

def export_json(episodes: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)


def export_csv(episodes: List[Dict[str, Any]], path: str) -> None:
    rows = []
    for epd in episodes:
        for obj in epd["objects"]:
            rows.append({
                "episode_id": epd["episode_id"], "stage": epd["stage"],
                "combo": epd.get("combo", ""), "instruction": epd["instruction"], **obj,
            })
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def check_min_distance_violations(episodes: List[Dict[str, Any]], object_min_dist: float) -> int:
    violations = 0
    for epd in episodes:
        objs = epd["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                d = np.hypot(objs[i]["x"] - objs[j]["x"], objs[i]["y"] - objs[j]["y"])
                if d < object_min_dist - 1e-9:
                    violations += 1
                    print(f"  [违规] episode {epd['episode_id']}: "
                          f"{objs[i]['role']}({objs[i]['color']}) 与 "
                          f"{objs[j]['role']}({objs[j]['color']}) 距离 {d:.4f} < {object_min_dist}")
    return violations


def check_target_coverage(episodes: List[Dict[str, Any]], role: str) -> None:
    from scipy.spatial import cKDTree
    pts = []
    for epd in episodes:
        for obj in epd["objects"]:
            if obj["role"] == role and obj.get("is_target", True):
                pts.append([obj["x"], obj["y"]])
    pts = np.array(pts)
    if len(pts) < 3:
        print(f"  [{role}] 点数不足，跳过覆盖度检验")
        return
    n_unique = len(np.unique(pts, axis=0))
    print(f"  [{role}] target 共 {len(pts)} 个位置，其中 {n_unique} 个物理坐标唯一 "
          f"(复用倍数 ~{len(pts)/max(n_unique,1):.1f}x)")

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    if nn.mean() > 0:
        print(f"  [{role}] target 最近邻距离: mean={nn.mean():.4f}, std={nn.std():.4f}, "
              f"std/mean={nn.std()/nn.mean():.3f} (越小越接近理想blue-noise)")
    else:
        print(f"  [{role}] 大量坐标完全重合(最近邻距离=0)，"
              f"说明 target_min_dist 相对 workspace 面积过大导致点位复用严重")


def check_orientation_ranges(episodes: List[Dict[str, Any]]) -> None:
    """验证各角色的朝向是否都落在对应的合法范围内。"""
    by_role: Dict[str, List[float]] = {}
    for epd in episodes:
        for obj in epd["objects"]:
            by_role.setdefault(obj["role"], []).append(obj["orientation_rad"])
    for role, vals in by_role.items():
        vals = np.array(vals)
        rad_min, rad_max = ORIENTATION_RANGES[role]
        out_of_range = np.sum((vals < rad_min) | (vals > rad_max))
        print(f"  [{role}] orientation: n={len(vals)}, range=[{vals.min():.3f}, {vals.max():.3f}], "
              f"越界数={out_of_range}")


# =====================================================================
# 9. 示例用法
# =====================================================================

# 各物体类型的朝向范围 (rad_min, rad_max)
ORIENTATION_RANGES: Dict[str, Tuple[float, float]] = {
    "cube": (-0.523599, 0.523599),   # ±30°
    "box":  (-0.174533, 0.174533),   # ±10°
}

if __name__ == "__main__":
    ws = Workspace(
        r_min=0.25, r_max=0.45, theta_min=-0.785398, theta_max=0.785398,
        object_min_dist=0.15, target_min_dist=0.05,
    )

    master_seed = 0

    d0 = []
    d1 = []
    d2 = []
    d3 = []

    d0 = generate_D0_dataset(n_episodes=100, ws=ws, seed=0 + master_seed)
    # d1 = generate_D1_dataset(episodes_per_combo=100, ws=ws, seed=1 + master_seed)
    # d2 = generate_D2_dataset(episodes_per_combo=100, ws=ws, seed=2 + master_seed)
    # d3 = generate_D3_dataset(episodes_per_combo=100, ws=ws, seed=3 + master_seed)

    for name, data in [("D0", d0), ("D1", d1), ("D2", d2), ("D3", d3)]:
        if len(data) > 0:
            # export_json(data, f"/mnt/user-data/outputs/{name}_episodes.json")
            export_csv(data, f"45_135_025_045/{name}_episodes.csv")
            n_bad = check_min_distance_violations(data, ws.object_min_dist)
            print(f"{name}: {len(data)} episodes, 碰撞违规数 = {n_bad}")
            check_target_coverage(data, "cube")
            check_target_coverage(data, "box")
            check_orientation_ranges(data)
            print()
