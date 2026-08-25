"""
position_sampling.py (v5)
==========================
D0(joint/a/b) ~ D3 阶段物体位置 + 朝向(orientation)采样。

v5 相比 v4 的改动
-----------------
1. 【核心重构】Workspace + 全局 ORIENTATION_RANGES 拆分 -> 统一的 RoleSpec
   每个角色(cube / box)的 position(r, theta) 和 orientation 范围现在打包
   在同一个 RoleSpec 里，因为 v5 起 cube 和 box 的 workspace 不再共用同一套
   (box 的 r/theta 范围明显更窄，且各自 orientation 范围本来就不同)。

2. 【新增】跨物体碰撞距离 CROSS_OBJECT_MIN_DIST 独立于任何一方的 RoleSpec。
   碰撞距离是物理安全边界(夹爪/物体尺寸决定的)，不应该因为你在算 cube 的
   覆盖池还是 box 的覆盖池而变化，所以从 RoleSpec 里抽出来做全局常量。

3. 【新增】D0 阶段拆成三个子数据集，用于诊断 cube 位置/朝向泛化和 box
   位置/朝向泛化是否存在交互效应(interaction effect)：
     - D0-joint      : cube + box 位置/朝向同时变化(即原 D0)
     - D0a-cubeOnly  : cube 位置/朝向自由变化，box 固定在几个 anchor 点位
     - D0b-boxOnly   : box 位置/朝向自由变化，cube 固定在几个 anchor 点位
   三者建议分别独立训练 policy 对比，而不是只在 D0-joint 上做 post-hoc 切片
   eval —— 后者看不出两个维度同时变化时是否存在 interaction 干扰。

4. 【新增】generate_anchors()：从角色自己的 workspace 内部(而非极端边界值)
   用 poisson-disk 采样出 3~5 个 anchor 点，然后 tile + round-robin 打乱分配
   到各 episode，避免"该角色永远固定在同一个点"这种 trivial shortcut。
   固定角色的 orientation 采用该角色 orientation 范围的中点(canonical value)，
   而不是随机挑一个角度，避免引入未声明的额外变量。

5. D1/D2/D3 生成器同步改为传入 cube_spec / box_spec 两个 RoleSpec，而不是单个
   共享 workspace。
"""

from __future__ import annotations

import csv
import itertools
import json
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# =====================================================================
# 1. 极坐标面积均匀采样修正
# =====================================================================

def sample_polar_uniform(
    r_min: float, r_max: float, theta_deg_min: float, theta_deg_max: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
    theta_deg = rng.uniform(theta_deg_min, theta_deg_max)
    theta_rad = np.deg2rad(theta_deg)
    return r * np.cos(theta_rad), r * np.sin(theta_rad)


def in_polar_region(
    x: float, y: float, r_min: float, r_max: float,
    theta_deg_min: float, theta_deg_max: float,
) -> bool:
    r = np.hypot(x, y)
    theta_rad = np.arctan2(y, x)
    theta_deg = np.rad2deg(theta_rad)
    if theta_deg_min <= theta_deg_max:
        theta_ok = theta_deg_min <= theta_deg <= theta_deg_max
    else:
        theta_ok = theta_deg >= theta_deg_min or theta_deg <= theta_deg_max
    return (r_min <= r <= r_max) and theta_ok


# =====================================================================
# 2. Orientation 采样(独立于 position 的简单均匀采样)
# =====================================================================

def sample_orientation(rad_deg_min: float, rad_deg_max: float, rng: np.random.Generator) -> float:
    """在给定角度范围内做简单均匀采样，返回弧度值。"""
    return float(rng.uniform(rad_deg_min, rad_deg_max))


# =====================================================================
# 3. Poisson-disk (Bridson's algorithm) —— 只用于生成覆盖池 / anchor 池
# =====================================================================

def poisson_disk_polar(
    r_min: float, r_max: float, theta_deg_min: float, theta_deg_max: float,
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
        if not in_polar_region(p[0], p[1], r_min, r_max, theta_deg_min, theta_deg_max):
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
        cx, cy = sample_polar_uniform(r_min, r_max, theta_deg_min, theta_deg_max, rng)
        if in_polar_region(cx, cy, r_min, r_max, theta_deg_min, theta_deg_max):
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
# 4. RoleSpec：每个角色自己的 position + orientation 规格
#    (v5 核心改动：不再用单一 Workspace 给 cube 和 box 共用)
# =====================================================================

@dataclass
class RoleSpec:
    """某一角色(cube / box)的极坐标 workspace + orientation 范围。"""
    r_min: float
    r_max: float
    theta_deg_min: float
    theta_deg_max: float
    orientation_deg_min: float
    orientation_deg_max: float
    target_min_dist: float = 0.05   # 该角色 target 位置在全数据集内的覆盖间距

    @property
    def canonical_orientation(self) -> float:
        """该角色 orientation 范围的中点，用于"固定该角色"时的朝向取值。"""
        return (self.orientation_deg_min + self.orientation_deg_max) / 2.0

# 跨物体碰撞检查距离：物理安全边界，与角色无关，不放进 RoleSpec
CROSS_OBJECT_MIN_DIST: float = 0.15

# 各角色的默认规格 —— 按你的最新讨论：box 的 r/theta 范围比 cube 窄很多
CUBE_SPEC = RoleSpec(
    r_min=0.25, r_max=0.45,
    theta_deg_min=-45.0, theta_deg_max=45.0,          # ±45°
    orientation_deg_min=-30.0, orientation_deg_max=30.0,  # ±30°
    target_min_dist=0.05,
)

BOX_SPEC = RoleSpec(
    r_min=0.25, r_max=0.35,
    theta_deg_min=-80.0, theta_deg_max=80.0,   # ±10°
    orientation_deg_min=80.0, orientation_deg_max=100.0,  # ±10°
    target_min_dist=0.05,
)

ROLE_SPECS: Dict[str, RoleSpec] = {"cube": CUBE_SPEC, "box": BOX_SPEC}

# =====================================================================
# 5. target 角色：全局覆盖池 (改为吃 RoleSpec)
# =====================================================================

def generate_manipulation_positions(
    n_episodes: int, spec: RoleSpec, rng: np.random.Generator, k: int = 30,
) -> np.ndarray:
    pool = poisson_disk_polar(
        spec.r_min, spec.r_max, spec.theta_deg_min, spec.theta_deg_max,
        spec.target_min_dist, k=k, max_points=None, rng=rng,
    )
    if len(pool) == 0:
        raise RuntimeError("target_min_dist 过大或区域过小，无法生成任何点")
    if len(pool) < n_episodes:
        warnings.warn(
            f"target_min_dist={spec.target_min_dist} 下该 workspace 最多生成 "
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
    """将 pos_b 中与对应 pos_a 距离过近的点，与 pos_b 内其它行做交换修复。
    min_dist 应传入 CROSS_OBJECT_MIN_DIST（跨物体碰撞距离），而不是任一方
    RoleSpec 自己的 target_min_dist。
    """
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
            f"{unresolved} 个 episode 未能通过交换修复 object 间冲突，"
            f"建议增大 workspace 或调整 target_min_dist / CROSS_OBJECT_MIN_DIST 比例。"
        )
    return pos_b


# =====================================================================
# 6. distractor 角色 / anchor 相关工具
# =====================================================================

def sample_free_position(
    existing_points: List[np.ndarray], spec: RoleSpec, min_dist: float,
    rng: np.random.Generator, max_tries: int = 300,
) -> np.ndarray:
    """在 spec 自己的区域内采样一个点，且与 existing_points 中每个点距离 >= min_dist。
    min_dist 传入 CROSS_OBJECT_MIN_DIST。
    """
    for _ in range(max_tries):
        x, y = sample_polar_uniform(spec.r_min, spec.r_max, spec.theta_deg_min, spec.theta_deg_max, rng)
        if all(np.hypot(x - ex[0], y - ex[1]) >= min_dist for ex in existing_points):
            return np.array([x, y])
    raise RuntimeError(
        "无法为 distractor 找到满足 min_dist 的位置，"
        "检查 min_dist 是否相对 workspace 过大"
    )


def generate_anchors(
    n_anchor: int, spec: RoleSpec, rng: np.random.Generator, k: int = 30,
) -> np.ndarray:
    """从角色自己的 workspace 内部(非边界极值)用 poisson-disk 采样出 n_anchor 个
    anchor 点，用于"固定该角色"的诊断子数据集(D0a / D0b)。
    """
    pool = poisson_disk_polar(
        spec.r_min, spec.r_max, spec.theta_deg_min, spec.theta_deg_max,
        spec.target_min_dist, k=k, rng=rng,
    )
    if len(pool) < n_anchor:
        warnings.warn(
            f"该 workspace 在 target_min_dist={spec.target_min_dist} 下只能生成 "
            f"{len(pool)} 个候选 anchor，少于请求的 {n_anchor} 个，将全部使用。"
        )
        n_anchor = len(pool)
    idx = rng.choice(len(pool), size=n_anchor, replace=False)
    return pool[idx]


def tile_anchors(anchors: np.ndarray, n_episodes: int, rng: np.random.Generator) -> np.ndarray:
    """把少量 anchor 点循环铺满 n_episodes 长度，并打乱顺序(round-robin 式分配)，
    避免"同一个 anchor 连续出现"引入的顺序相关性。
    """
    reps = int(np.ceil(n_episodes / len(anchors)))
    tiled = np.tile(anchors, (reps, 1))[:n_episodes].copy()
    rng.shuffle(tiled)
    return tiled


def round_robin_assignment(items: List[str], cycle_idx: int) -> List[str]:
    n = len(items)
    shift = cycle_idx % n
    return items[shift:] + items[:shift]


def make_object(
    role: str, color: str, x: float, y: float, rng: np.random.Generator,
    role_specs: Dict[str, RoleSpec] = ROLE_SPECS,
    is_target: Optional[bool] = None,
    fixed_orientation: Optional[float] = None,
) -> Dict[str, Any]:
    """统一构造一个物体字典。
    fixed_orientation 不为 None 时直接使用该值(用于 D0a/D0b 里被固定的角色，
    取该角色 orientation 范围的中点，即 role_specs[role].canonical_orientation)；
    否则按该角色自己的 orientation 范围随机采样。
    """
    spec = role_specs[role]
    orientation = (
        fixed_orientation if fixed_orientation is not None
        else sample_orientation(spec.orientation_deg_min, spec.orientation_deg_max, rng)
    )
    obj = {
        "role": role, "color": color,
        "x": float(x), "y": float(y),
        "orientation_deg": orientation,
    }
    if is_target is not None:
        obj["is_target"] = is_target
    return obj


# =====================================================================
# 7. D0 三件套：joint / cubeOnly / boxOnly
# =====================================================================

def generate_D0_joint_dataset(
    n_episodes: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST, seed: int = 0,
) -> List[Dict[str, Any]]:
    """cube + box 位置/朝向同时自由变化(原 v4 的 D0)。"""
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}

    cube_pos = generate_manipulation_positions(n_episodes, cube_spec, rng)
    box_pos = generate_manipulation_positions(n_episodes, box_spec, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        episodes.append({
            "episode_id": ep, "stage": "D0-joint",
            "objects": [
                make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng, role_specs),
                make_object("box", "red", box_pos[ep, 0], box_pos[ep, 1], rng, role_specs),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def generate_D0a_cubeOnly_dataset(
    n_episodes: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    n_anchor: int = 4, seed: int = 10,
) -> List[Dict[str, Any]]:
    """cube 位置/朝向自由变化，box 固定在 n_anchor 个 anchor 点位(位置+朝向都固定)。
    用于隔离测试 cube 的位置/朝向泛化能力，不受 box 变化干扰。
    """
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}

    box_anchors = generate_anchors(n_anchor, box_spec, rng)
    cube_pos = generate_manipulation_positions(n_episodes, cube_spec, rng)
    box_pos = tile_anchors(box_anchors, n_episodes, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        episodes.append({
            "episode_id": ep, "stage": "D0a-cubeOnly",
            "objects": [
                make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng,
                            role_specs, is_target=True),
                make_object("box", "red", box_pos[ep, 0], box_pos[ep, 1], rng,
                            role_specs, is_target=False,)
                            # fixed_orientation=box_spec.canonical_orientation),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def generate_D0b_boxOnly_dataset(
    n_episodes: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    n_anchor: int = 4, seed: int = 20,
) -> List[Dict[str, Any]]:
    """box 位置/朝向自由变化，cube 固定在 n_anchor 个 anchor 点位(位置+朝向都固定)。
    用于隔离测试 box 的位置/朝向泛化能力，不受 cube 变化干扰。
    """
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}

    cube_anchors = generate_anchors(n_anchor, cube_spec, rng)
    box_pos = generate_manipulation_positions(n_episodes, box_spec, rng)
    cube_pos = tile_anchors(cube_anchors, n_episodes, rng)
    cube_pos = pair_targets_with_repair(box_pos, cube_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        episodes.append({
            "episode_id": ep, "stage": "D0b-boxOnly",
            "objects": [
                make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng,
                            role_specs, is_target=False,),
                            # fixed_orientation=cube_spec.canonical_orientation),
                make_object("box", "red", box_pos[ep, 0], box_pos[ep, 1], rng,
                            role_specs, is_target=True),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


# =====================================================================
# 8. D1 / D2 / D3 (改为吃 cube_spec / box_spec 两个 RoleSpec)
# =====================================================================

def generate_D1_dataset(
    episodes_per_combo: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 1,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}
    total = episodes_per_combo * len(colors)
    cube_target_pos = generate_manipulation_positions(total, cube_spec, rng)
    box_pos = generate_manipulation_positions(total, box_spec, rng)
    box_pos = pair_targets_with_repair(cube_target_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(total):
        target_color = colors[ep % len(colors)]
        distractor_color = colors[(ep + 1) % len(colors)]
        distractor_pos = sample_free_position(
            [cube_target_pos[ep], box_pos[ep]], cube_spec, cross_min_dist, rng)

        episodes.append({
            "episode_id": ep, "stage": "D1",
            "objects": [
                make_object("cube", target_color, *cube_target_pos[ep], rng, role_specs, is_target=True),
                make_object("cube", distractor_color, *distractor_pos, rng, role_specs, is_target=False),
                make_object("box", "neutral", *box_pos[ep], rng, role_specs),
            ],
            "instruction": f"pick up the {target_color} cube and place it in the box",
        })
    return episodes


def generate_D2_dataset(
    episodes_per_combo: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 2,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}
    total = episodes_per_combo * len(colors)
    cube_pos = generate_manipulation_positions(total, cube_spec, rng)
    box_target_pos = generate_manipulation_positions(total, box_spec, rng)
    box_target_pos = pair_targets_with_repair(cube_pos, box_target_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(total):
        target_color = colors[ep % len(colors)]
        distractor_color = colors[(ep + 1) % len(colors)]
        distractor_pos = sample_free_position(
            [cube_pos[ep], box_target_pos[ep]], box_spec, cross_min_dist, rng)

        episodes.append({
            "episode_id": ep, "stage": "D2",
            "objects": [
                make_object("cube", "neutral", *cube_pos[ep], rng, role_specs),
                make_object("box", target_color, *box_target_pos[ep], rng, role_specs, is_target=True),
                make_object("box", distractor_color, *distractor_pos, rng, role_specs, is_target=False),
            ],
            "instruction": f"pick up the cube and place it in the {target_color} box",
        })
    return episodes


def generate_D3_dataset(
    episodes_per_combo: int,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    colors: Tuple[str, str] = ("red", "blue"), seed: int = 3,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}
    combos = list(itertools.product(colors, colors))
    total = episodes_per_combo * len(combos)

    cube_target_pos = generate_manipulation_positions(total, cube_spec, rng)
    box_target_pos = generate_manipulation_positions(total, box_spec, rng)
    box_target_pos = pair_targets_with_repair(cube_target_pos, box_target_pos, cross_min_dist, rng)

    episodes = []
    ep = 0
    for _ in range(episodes_per_combo):
        order = rng.permutation(len(combos))
        for combo_idx in order:
            src_color, dst_color = combos[combo_idx]
            src_distractor = colors[1] if src_color == colors[0] else colors[0]
            dst_distractor = colors[1] if dst_color == colors[0] else colors[0]

            cube_distractor_pos = sample_free_position(
                [cube_target_pos[ep], box_target_pos[ep]], cube_spec, cross_min_dist, rng)
            box_distractor_pos = sample_free_position(
                [cube_target_pos[ep], box_target_pos[ep], cube_distractor_pos],
                box_spec, cross_min_dist, rng)

            episodes.append({
                "episode_id": ep, "stage": "D3", "combo": f"{src_color}->{dst_color}",
                "objects": [
                    make_object("cube", src_color, *cube_target_pos[ep], rng, role_specs, is_target=True),
                    make_object("cube", src_distractor, *cube_distractor_pos, rng, role_specs, is_target=False),
                    make_object("box", dst_color, *box_target_pos[ep], rng, role_specs, is_target=True),
                    make_object("box", dst_distractor, *box_distractor_pos, rng, role_specs, is_target=False),
                ],
                "instruction": f"pick up the {src_color} cube and place it in the {dst_color} box",
            })
            ep += 1
    return episodes


# =====================================================================
# 9. 导出 & 验证
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


def check_min_distance_violations(episodes: List[Dict[str, Any]], min_dist: float) -> int:
    violations = 0
    for epd in episodes:
        objs = epd["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                d = np.hypot(objs[i]["x"] - objs[j]["x"], objs[i]["y"] - objs[j]["y"])
                if d < min_dist - 1e-9:
                    violations += 1
                    print(f"  [违规] episode {epd['episode_id']}: "
                          f"{objs[i]['role']}({objs[i]['color']}) 与 "
                          f"{objs[j]['role']}({objs[j]['color']}) 距离 {d:.4f} < {min_dist}")
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


def check_orientation_ranges(
    episodes: List[Dict[str, Any]], role_specs: Dict[str, RoleSpec] = ROLE_SPECS,
) -> None:
    """验证各角色的朝向是否都落在对应的合法范围内。"""
    by_role: Dict[str, List[float]] = {}
    for epd in episodes:
        for obj in epd["objects"]:
            by_role.setdefault(obj["role"], []).append(obj["orientation_deg"])
    for role, vals in by_role.items():
        vals = np.array(vals)
        spec = role_specs[role]
        out_of_range = np.sum((vals < spec.orientation_deg_min) | (vals > spec.orientation_deg_max))
        n_unique = len(np.unique(vals))
        print(f"  [{role}] orientation: n={len(vals)}, range=[{vals.min():.3f}, {vals.max():.3f}], "
              f"越界数={out_of_range}, 唯一值数={n_unique} "
              f"(=1 说明该角色朝向在本数据集中被固定)")


# =====================================================================
# 10. 示例用法
# =====================================================================
if __name__ == "__main__":
    master_seed = 42

    # ---- D0 三件套 ----
    # D0-joint 组合空间是 D0a/D0b 的笛卡尔积，给更高的 episode 预算；
    # D0a / D0b 保持相同 episode 数，避免"数据量差异"成为新的 confound。
    d0_joint = generate_D0_joint_dataset(n_episodes=150, seed=0 + master_seed)
    d0a = generate_D0a_cubeOnly_dataset(n_episodes=100, n_anchor=4, seed=10 + master_seed)
    d0b = generate_D0b_boxOnly_dataset(n_episodes=100, n_anchor=4, seed=20 + master_seed)

    # ---- D1 / D2 / D3 (需要时取消注释) ----
    d1: List[Dict[str, Any]] = []
    d2: List[Dict[str, Any]] = []
    d3: List[Dict[str, Any]] = []
    # d1 = generate_D1_dataset(episodes_per_combo=100, seed=1 + master_seed)
    # d2 = generate_D2_dataset(episodes_per_combo=100, seed=2 + master_seed)
    # d3 = generate_D3_dataset(episodes_per_combo=100, seed=3 + master_seed)

    all_datasets = [
        ("D0-joint", d0_joint), ("D0a-cubeOnly", d0a), ("D0b-boxOnly", d0b),
        ("D1", d1), ("D2", d2), ("D3", d3),
    ]

    for name, data in all_datasets:
        if len(data) > 0:
            export_csv(data, f"45_135_025_045/{name}_episodes.csv")
            n_bad = check_min_distance_violations(data, CROSS_OBJECT_MIN_DIST)
            print(f"{name}: {len(data)} episodes, 碰撞违规数 = {n_bad}")
            check_target_coverage(data, "cube")
            check_target_coverage(data, "box")
            check_orientation_ranges(data)
            print()
