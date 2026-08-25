"""
position_sampling.py
=====================
用于 D0~D3 四阶段数据集采集的物体位置采样工具。

设计目标
--------
1. Poisson-disk (blue-noise) 采样：保证小样本量(如 100ep/组合)下位置覆盖
   仍然均匀，避免 uniform random 在小样本下产生的局部聚集/空洞。
2. 极坐标面积均匀修正：Bridson's Poisson-disk 算法的接受性判定直接在
   Cartesian 几何空间做（是否落在环形扇区内 + 是否满足 min_dist），
   因此天然满足"面积均匀"，等价于隐式做了 r ~ sqrt(uniform) 修正 ——
   不会出现原点附近点密集、边缘稀疏的问题。初始种子点额外显式用
   sample_polar_uniform 生成，双重保证起点无偏。
3. Round-robin slot 分配：延续你在 multi-object 数据集设计中已用过的
   思路，让"物理位置(slot)"与"颜色/角色"在多个 episode 间轮换配对，
   打破 颜色-位置 的统计相关性（防止 policy 学到颜色-位置 shortcut）。

四阶段用法
----------
D0: 1 cube + 1 box，无颜色 grounding，只需要 poisson-disk 位置覆盖。
D1: 2 色 cube（target/distractor）+ 1 box，round-robin 颜色-slot 配对，
    指令 50/50 双向覆盖两种颜色。
D2: 1 cube + 2 色 box（target/distractor），同 D1 逻辑，角色对调。
D3: 2 色 cube × 2 色 box，四种 source-destination 颜色组合需配平出现，
    cube 与 box 的 slot 分配各自独立做 round-robin，避免联合相关性。

注意：r_min/r_max/theta_min/theta_max/min_dist 均为占位默认值，
请替换成 PIPER 实际可达工作空间的标定数值。
"""

from __future__ import annotations

import csv
import itertools
import json
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
    """
    在极坐标环形扇区 (annular sector) 内做面积均匀采样。
    r ~ sqrt(uniform(r_min^2, r_max^2)) 修正，避免采样点在原点附近过密。
    返回 Cartesian 坐标 (x, y)。
    """
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
    else:  # 跨越 -pi/pi 边界的情况
        theta_ok = theta >= theta_min or theta <= theta_max
    return (r_min <= r <= r_max) and theta_ok


# =====================================================================
# 2. Poisson-disk (Bridson's algorithm)，接受性判定在极坐标区域内做
# =====================================================================

def poisson_disk_polar(
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    min_dist: float,
    k: int = 30,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    在极坐标环形扇区内生成 Poisson-disk (blue-noise) 点池。

    Parameters
    ----------
    r_min, r_max : 半径范围 (m)
    theta_min, theta_max : 角度范围 (rad)
    min_dist : 任意两点最小间距 (m)，对应你现有的最小物体间距约束
    k : 每个活跃点尝试生成候选新点的次数上限
    max_points : 可选，够用即停，节省计算
    rng : np.random.Generator，用于可复现实验（务必固定 seed）

    Returns
    -------
    (N, 2) ndarray，Cartesian 坐标
    """
    if rng is None:
        rng = np.random.default_rng()

    cell_size = min_dist / np.sqrt(2)
    x_min, x_max = -r_max, r_max
    y_min, y_max = -r_max, r_max
    grid_w = int(np.ceil((x_max - x_min) / cell_size)) + 1
    grid_h = int(np.ceil((y_max - y_min) / cell_size)) + 1
    grid = -np.ones((grid_w, grid_h), dtype=int)

    def grid_coords(p):
        gx = int((p[0] - x_min) / cell_size)
        gy = int((p[1] - y_min) / cell_size)
        return gx, gy

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

    # 初始种子：显式极坐标均匀采样，最多尝试若干次直到落入区域
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
# 3. 单 episode 内的物体 slot 生成（保证互相满足最小间距，slot 顺序稳定）
# =====================================================================

@dataclass
class Workspace:
    """极坐标工作空间标定参数 —— 请替换为 PIPER 实际可达范围的标定值。"""
    r_min: float = 0.15
    r_max: float = 0.45
    theta_min: float = -0.9   # rad
    theta_max: float = 0.9    # rad
    min_dist: float = 0.08    # 物体间最小间距 (m)


def generate_episode_slots(
    n_slots: int,
    ws: Workspace,
    rng: np.random.Generator,
    pool_multiplier: int = 20,
) -> np.ndarray:
    """
    为单个 episode 生成 n_slots 个互不干涉的位置。
    先在 workspace 内生成一个较大的 poisson-disk 候选池，
    再随机挑 n_slots 个点，最后按极角排序得到稳定的 slot 顺序
    （slot 顺序稳定是 round-robin 能对齐"同一物理slot"的前提）。
    """
    pts = poisson_disk_polar(
        ws.r_min, ws.r_max, ws.theta_min, ws.theta_max, ws.min_dist,
        max_points=max(n_slots * pool_multiplier, n_slots + 5), rng=rng,
    )
    if len(pts) < n_slots:
        raise RuntimeError(
            f"Poisson-disk 生成点数不足({len(pts)} < {n_slots})，"
            f"请减小 min_dist 或扩大 workspace 范围"
        )
    chosen_idx = rng.choice(len(pts), size=n_slots, replace=False)
    chosen = pts[chosen_idx]
    thetas = np.arctan2(chosen[:, 1], chosen[:, 0])
    order = np.argsort(thetas)
    return chosen[order]  # shape (n_slots, 2)，slot 0..n_slots-1 顺序按极角固定

def generate_role_pool(n_positions: int, ws: Workspace, rng: np.random.Generator,
                        min_dist: Optional[float] = None) -> np.ndarray:
    """
    为某个"角色"(如 D0 里的所有 cube 位置)一次性生成整个阶段需要的
    n_positions 个位置,整体服从 blue-noise 均匀覆盖。
    这个函数在 episode 循环外只调用一次,而不是每个 episode 调用一次。
    """
    if min_dist is None:
        # 用目标点数反推一个合理的 min_dist,使池子大小接近 n_positions
        area = 0.5 * (ws.r_max**2 - ws.r_min**2) * (ws.theta_max - ws.theta_min)
        min_dist = 0.75 * np.sqrt(area / n_positions)  # 0.75 是常用的2D blue-noise packing经验系数

    pts = poisson_disk_polar(ws.r_min, ws.r_max, ws.theta_min, ws.theta_max,
                              min_dist, max_points=n_positions, rng=rng)
    if len(pts) < n_positions:
        raise RuntimeError(
            f"生成点数不足({len(pts)}<{n_positions}),请减小 min_dist 或减少所需点数"
        )
    rng.shuffle(pts)  # 打乱分配给 episode 的顺序,避免"生长顺序"和"episode_id"产生关联
    return pts

# =====================================================================
# 4. Round-robin 颜色-slot 配对
# =====================================================================

def round_robin_assignment(items: List[str], cycle_idx: int) -> List[str]:
    """
    给定按 slot 顺序排列的 items（如颜色列表），根据 cycle_idx 做循环轮换，
    使同一物理 slot 在不同 episode 间轮流被不同颜色占据。
    对 n=2 的情况（D1/D2），偶数个 episode 后颜色-slot 完全配平(各占50%)。
    """
    n = len(items)
    shift = cycle_idx % n
    return items[shift:] + items[:shift]


# =====================================================================
# 5. 各阶段 episode 生成器
# =====================================================================

def generate_D0_dataset(
    n_episodes: int, ws: Workspace, seed: int = 0,
) -> List[Dict[str, Any]]:
    """D0: 1 cube + 1 box，无颜色 grounding，仅需 poisson-disk 位置覆盖。"""
    rng = np.random.default_rng(seed)
    cube_positions = generate_role_pool(n_episodes, ws, rng)  # 100个位置,一次生成,整体blue-noise
    box_positions = generate_role_pool(n_episodes, ws, rng)  # 同理,独立的一个全局池子

    episodes = []
    for ep in range(n_episodes):
        cx, cy = cube_positions[ep]
        bx, by = box_positions[ep]
        # cube 和 box 是不同角色,只需要检查这一对在本episode内是否够远
        if np.hypot(cx - bx, cy - by) < ws.min_dist:
            # 极少数冲突:和另一个 episode 的 box 位置交换,修复冲突
            for other in range(n_episodes):
                ox, oy = box_positions[other]
                if np.hypot(cx - ox, cy - oy) >= ws.min_dist:
                    box_positions[[ep, other]] = box_positions[[other, ep]]
                    bx, by = box_positions[ep]
                    break
        episodes.append({
            "episode_id": ep,
            "stage": "D0",
            "objects": [
                {"role": "cube", "color": "red", "x": float(slots[0, 0]), "y": float(slots[0, 1])},
                {"role": "box", "color": "red", "x": float(slots[1, 0]), "y": float(slots[1, 1])},
            ],
            "instruction": "pick up the cube and place it in the box.",
        })
    return episodes


def generate_D1_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 1,
) -> List[Dict[str, Any]]:
    """
    D1: source grounding。2 色 cube 同时出现，1 个固定角色的 box。
    指令 50/50 双向覆盖两种颜色，round-robin 决定颜色占据哪个物理 slot。
    """
    rng = np.random.default_rng(seed)
    episodes = []
    total = episodes_per_combo * len(colors)
    for ep in range(total):
        target_color = colors[ep % len(colors)]           # 双向指令覆盖
        distractor_color = colors[(ep + 1) % len(colors)]
        cube_colors_by_slot = round_robin_assignment(
            [target_color, distractor_color], cycle_idx=ep
        )  # 打破 颜色-slot 相关性
        slots = generate_episode_slots(3, ws, rng)  # slot0/1=两个cube, slot2=box
        episodes.append({
            "episode_id": ep,
            "stage": "D1",
            "objects": [
                {"role": "cube", "color": cube_colors_by_slot[0],
                 "is_target": cube_colors_by_slot[0] == target_color,
                 "x": float(slots[0, 0]), "y": float(slots[0, 1])},
                {"role": "cube", "color": cube_colors_by_slot[1],
                 "is_target": cube_colors_by_slot[1] == target_color,
                 "x": float(slots[1, 0]), "y": float(slots[1, 1])},
                {"role": "box", "color": "neutral",
                 "x": float(slots[2, 0]), "y": float(slots[2, 1])},
            ],
            "instruction": f"pick up the {target_color} cube and place it in the box",
        })
    return episodes


def generate_D2_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 2,
) -> List[Dict[str, Any]]:
    """
    D2: destination grounding。1 个 cube，2 色 box 同时出现。
    与 D1 完全对称，角色对调。
    """
    rng = np.random.default_rng(seed)
    episodes = []
    total = episodes_per_combo * len(colors)
    for ep in range(total):
        target_color = colors[ep % len(colors)]
        distractor_color = colors[(ep + 1) % len(colors)]
        box_colors_by_slot = round_robin_assignment(
            [target_color, distractor_color], cycle_idx=ep
        )
        slots = generate_episode_slots(3, ws, rng)  # slot0=cube, slot1/2=两个box
        episodes.append({
            "episode_id": ep,
            "stage": "D2",
            "objects": [
                {"role": "cube", "color": "neutral",
                 "x": float(slots[0, 0]), "y": float(slots[0, 1])},
                {"role": "box", "color": box_colors_by_slot[0],
                 "is_target": box_colors_by_slot[0] == target_color,
                 "x": float(slots[1, 0]), "y": float(slots[1, 1])},
                {"role": "box", "color": box_colors_by_slot[1],
                 "is_target": box_colors_by_slot[1] == target_color,
                 "x": float(slots[2, 0]), "y": float(slots[2, 1])},
            ],
            "instruction": f"pick up the cube and place it in the {target_color} box",
        })
    return episodes


def generate_D3_dataset(
    episodes_per_combo: int, ws: Workspace,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 3,
) -> List[Dict[str, Any]]:
    """
    D3: 联合 grounding。2 色 cube × 2 色 box，四种 source-destination
    颜色组合需配平出现。cube 的 slot round-robin 与 box 的 slot round-robin
    各自独立循环（用不同的 cycle 计数），避免二者相位对齐产生新的相关性。
    """
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(colors, colors))  # 4 种 (source, dest) 组合
    episodes = []
    ep = 0
    for _ in range(episodes_per_combo):
        # 每一轮内部把 4 种组合打乱一次，保证配平且不引入固定顺序
        order = rng.permutation(len(combos))
        for combo_idx in order:
            src_color, dst_color = combos[combo_idx]
            src_distractor = colors[1] if src_color == colors[0] else colors[0]
            dst_distractor = colors[1] if dst_color == colors[0] else colors[0]

            # cube slot 与 box slot 各自独立的 round-robin 计数
            cube_colors_by_slot = round_robin_assignment(
                [src_color, src_distractor], cycle_idx=ep
            )
            box_colors_by_slot = round_robin_assignment(
                [dst_color, dst_distractor], cycle_idx=ep + 1  # 故意错开相位
            )

            slots = generate_episode_slots(4, ws, rng)  # slot0/1=cube, slot2/3=box
            episodes.append({
                "episode_id": ep,
                "stage": "D3",
                "combo": f"{src_color}->{dst_color}",
                "objects": [
                    {"role": "cube", "color": cube_colors_by_slot[0],
                     "is_target": cube_colors_by_slot[0] == src_color,
                     "x": float(slots[0, 0]), "y": float(slots[0, 1])},
                    {"role": "cube", "color": cube_colors_by_slot[1],
                     "is_target": cube_colors_by_slot[1] == src_color,
                     "x": float(slots[1, 0]), "y": float(slots[1, 1])},
                    {"role": "box", "color": box_colors_by_slot[0],
                     "is_target": box_colors_by_slot[0] == dst_color,
                     "x": float(slots[2, 0]), "y": float(slots[2, 1])},
                    {"role": "box", "color": box_colors_by_slot[1],
                     "is_target": box_colors_by_slot[1] == dst_color,
                     "x": float(slots[3, 0]), "y": float(slots[3, 1])},
                ],
                "instruction": f"pick up the {src_color} cube and place it in the {dst_color} box",
            })
            ep += 1
    return episodes


# =====================================================================
# 6. 导出工具
# =====================================================================

def export_json(episodes: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)


def export_csv(episodes: List[Dict[str, Any]], path: str) -> None:
    """展平成一行一个 object 的 csv，便于后续用 pandas 做覆盖度/相关性检验。"""
    rows = []
    for epd in episodes:
        for obj in epd["objects"]:
            row = {
                "episode_id": epd["episode_id"],
                "stage": epd["stage"],
                "combo": epd.get("combo", ""),
                "instruction": epd["instruction"],
                **obj,
            }
            rows.append(row)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# 7. 示例用法
# =====================================================================

if __name__ == "__main__":
    ws = Workspace(
        r_min=0.25, r_max=0.45,
        theta_min=-0.785398, theta_max=0.785398,
        min_dist=0.08,
    )

    d0 = generate_D0_dataset(n_episodes=100, ws=ws, seed=0)
    # d1 = generate_D1_dataset(episodes_per_combo=100, ws=ws, seed=1)   # 2*100=200ep
    # d2 = generate_D2_dataset(episodes_per_combo=100, ws=ws, seed=2)   # 2*100=200ep
    # d3 = generate_D3_dataset(episodes_per_combo=100, ws=ws, seed=3)   # 4*100=400ep

    # export_json(d0, "/mnt/user-data/outputs/D0_episodes.json")
    # export_json(d1, "/mnt/user-data/outputs/D1_episodes.json")
    # export_json(d2, "/mnt/user-data/outputs/D2_episodes.json")
    # export_json(d3, "/mnt/user-data/outputs/D3_episodes.json")

    export_csv(d0, "45_135_025_045/D0_episodes.csv")
    # export_csv(d1, "/mnt/user-data/outputs/D1_episodes.csv")
    # export_csv(d2, "/mnt/user-data/outputs/D2_episodes.csv")
    # export_csv(d3, "/mnt/user-data/outputs/D3_episodes.csv")

    print(f"D0: {len(d0)} episodes")
    # print(f"D1: {len(d1)} episodes")
    # print(f"D2: {len(d2)} episodes")
    # print(f"D3: {len(d3)} episodes")
