"""
position_sampling_ood.py
========================
Cube position 的"外推 OOD"(训练区域 CUBE_SPEC 之外)采样。

思路
----
1. 在一个更大的 OUTER_CUBE_SPEC 上做 Poisson-disk，得到覆盖整个外扩区域的 blue-noise 池；
2. 过滤掉落在 (膨胀了 margin 的) 训练区域 CUBE_SPEC 内的点；
3. 过滤掉与 box anchor 距离 < CROSS_OBJECT_MIN_DIST 的点
   (否则单 anchor 时 pair_targets_with_repair 无法通过交换修复碰撞)；
4. 剩下的池子按原来的方式打乱、复用，给每个点打上 ood_dist / ood_region 标签，
   方便之后按"离训练区域的距离"分层统计成功率。

orientation 沿用 OUTER_CUBE_SPEC 里继承自 CUBE_SPEC 的范围(±30°)，
保证只有 position 是 OOD 变量。

用法：与 position_sampling_v5.py 放在同一目录，
    from position_sampling_ood import *
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from position_sampling_v5 import (
    CUBE_SPEC, BOX_SPEC, CROSS_OBJECT_MIN_DIST, RoleSpec,
    poisson_disk_polar, in_polar_region, tile_anchors,
    pair_targets_with_repair, make_object, generate_manipulation_positions,
)

# 外扩候选区域：只放宽 r / theta，orientation 与 target_min_dist 继承自 CUBE_SPEC
OUTER_CUBE_SPEC = replace(
    CUBE_SPEC,
    r_min=0.20, r_max=0.50,
    theta_deg_min=-50.0, theta_deg_max=50.0,
    target_min_dist=0.03
)

INNER_CUBE_SPEC = replace(
    CUBE_SPEC,
    r_min=0.3, r_max=0.4,
    theta_deg_min=-35.0, theta_deg_max=35.0,
    target_min_dist=0.03
)

DENSE_CUBE_SPEC = replace(
    CUBE_SPEC,
    r_min=0.26, r_max=0.44,
    theta_deg_min=-44.0, theta_deg_max=44.0,
    target_min_dist=0.03
)


def _expand(spec: RoleSpec, margin_r: float, margin_theta_deg: float) -> RoleSpec:
    """把 spec 的 r / theta 范围向外膨胀 margin，用作 OOD 排除区。"""
    return replace(
        spec,
        r_min=spec.r_min - margin_r, r_max=spec.r_max + margin_r,
        theta_deg_min=spec.theta_deg_min - margin_theta_deg,
        theta_deg_max=spec.theta_deg_max + margin_theta_deg,
    )


def ood_info(x: float, y: float, train_spec: RoleSpec = CUBE_SPEC) -> Tuple[float, str]:
    """返回 (离训练区域的近似距离[m], 越界方向标签)。
    距离 = hypot(径向越界量, 弧长方向越界量)，是环扇形的近似，用于分层统计足够。
    """
    r = float(np.hypot(x, y))
    th = float(np.rad2deg(np.arctan2(y, x)))
    dr = max(train_spec.r_min - r, 0.0, r - train_spec.r_max)
    dth = max(train_spec.theta_deg_min - th, 0.0, th - train_spec.theta_deg_max)
    dist = float(np.hypot(dr, r * np.deg2rad(dth)))
    tags = []
    if r < train_spec.r_min:
        tags.append("r_low")
    elif r > train_spec.r_max:
        tags.append("r_high")
    if th < train_spec.theta_deg_min:
        tags.append("theta_low")
    elif th > train_spec.theta_deg_max:
        tags.append("theta_high")
    return dist, ("+".join(tags) if tags else "in_dist")


def build_ood_pool(
    rng: np.random.Generator,
    outer_spec: RoleSpec = OUTER_CUBE_SPEC,
    train_spec: RoleSpec = CUBE_SPEC,
    margin_r: float = 0.02,
    margin_theta_deg: float = 5.0,
    avoid_points: Optional[np.ndarray] = None,
    avoid_dist: float = CROSS_OBJECT_MIN_DIST,
    camera_x_max: float = 0.55,
    camera_y_abs_max: float = 0.36,
) -> np.ndarray:
    """OOD 候选点池：outer 区域的 Poisson 池，去掉训练区域(+margin)和 box anchor 附近的点。"""
    pool = poisson_disk_polar(
        outer_spec.r_min, outer_spec.r_max,
        outer_spec.theta_deg_min, outer_spec.theta_deg_max,
        outer_spec.target_min_dist, rng=rng,
    )
    excl = _expand(train_spec, margin_r, margin_theta_deg)
    keep = []
    for p in pool:
        if in_polar_region(p[0], p[1], excl.r_min, excl.r_max,
                           excl.theta_deg_min, excl.theta_deg_max):
            continue
        if not in_camera_fov(p[0], p[1], camera_x_max, camera_y_abs_max):   # camera perspective range
            continue
        if avoid_points is not None and any(
                np.hypot(p[0] - a[0], p[1] - a[1]) < avoid_dist for a in avoid_points):
            continue
        keep.append(p)
    if not keep:
        raise RuntimeError("过滤后 OOD 池为空：检查 outer_spec 是否真的大于训练区域")
    return np.array(keep)


def draw_from_pool(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """与 generate_manipulation_positions 相同的取点逻辑：每轮不放回，池不够则循环复用。"""
    if len(pool) < n:
        warnings.warn(
            f"OOD 池只有 {len(pool)} 个点，少于所需 {n} 个，将循环复用；"
            f"可减小 outer_spec.target_min_dist 或扩大 outer_spec。"
        )
    out, remaining = [], n
    while remaining > 0:
        order = rng.permutation(len(pool))
        take = min(remaining, len(pool))
        out.append(pool[order[:take]])
        remaining -= take
    return np.concatenate(out, axis=0)


def sample_orientation_ood(
    outer_min: float, outer_max: float,
    train_min: float, train_max: float,
    margin_deg: float, rng: np.random.Generator,
) -> float:
    """在 outer 角度范围内采样，排除训练区间(+margin)。"""
    excl_min, excl_max = train_min - margin_deg, train_max + margin_deg
    for _ in range(1000):
        val = float(rng.uniform(outer_min, outer_max))
        if val < excl_min or val > excl_max:
            return val
    raise RuntimeError("outer 角度范围可能没有真正大于训练区间+margin")


def generate_OOD_orientation_dataset(
    n_episodes: int,
    box_anchors: Sequence[Sequence[float]],
    cube_spec: RoleSpec = CUBE_SPEC,      # position 用训练时的 spec，保持 in-distribution
    box_spec: RoleSpec = BOX_SPEC,
    outer_orientation: Tuple[float, float] = (-45.0, 45.0),
    margin_deg: float = 5.0,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    seed: int = 300,
    stage: str = "OOD-orientation",
) -> List[Dict[str, Any]]:
    """cube 位置沿用训练分布(CUBE_SPEC)，只有 orientation 落在训练区间之外。"""
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}
    box_anchors = np.asarray(box_anchors, dtype=float).reshape(-1, 2)

    cube_pos = generate_manipulation_positions(n_episodes, cube_spec, rng)
    box_pos = tile_anchors(box_anchors, n_episodes, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        ori = sample_orientation_ood(
            *outer_orientation,
            cube_spec.orientation_deg_min, cube_spec.orientation_deg_max,
            margin_deg, rng,
        )
        cube = make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng,
                            role_specs, is_target=True, fixed_orientation=ori)
        cube["ood_dist_deg"] = min(abs(ori - cube_spec.orientation_deg_min),
                                    abs(ori - cube_spec.orientation_deg_max))
        episodes.append({
            "episode_id": ep, "stage": stage,
            "objects": [
                cube,
                make_object("box", "blue", box_pos[ep, 0], box_pos[ep, 1],
                            rng, role_specs, is_target=False),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def generate_OOD_cube_dataset(
    n_episodes: int,
    box_anchors: Sequence[Sequence[float]],
    outer_spec: RoleSpec = OUTER_CUBE_SPEC,
    train_spec: RoleSpec = CUBE_SPEC,
    box_spec: RoleSpec = BOX_SPEC,
    margin_r: float = 0.01,
    margin_theta_deg: float = 0.5,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    seed: int = 100,
    stage: str = "OOD-cube",
) -> List[Dict[str, Any]]:
    """D0a 风格：单 cube 在训练区域外，box 固定在 box_anchors(应与训练时相同)。"""
    rng = np.random.default_rng(seed)
    role_specs = {"cube": outer_spec, "box": box_spec}
    box_anchors = np.asarray(box_anchors, dtype=float).reshape(-1, 2)

    pool = build_ood_pool(rng, outer_spec, train_spec, margin_r, margin_theta_deg,
                          avoid_points=box_anchors, avoid_dist=cross_min_dist)
    cube_pos = draw_from_pool(pool, n_episodes, rng)
    box_pos = tile_anchors(box_anchors, n_episodes, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        cube = make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1],
                           rng, role_specs, is_target=True)
        cube["ood_dist"], cube["ood_region"] = ood_info(cube["x"], cube["y"], train_spec)
        episodes.append({
            "episode_id": ep, "stage": stage,
            "objects": [
                cube,
                make_object("box", "blue", box_pos[ep, 0], box_pos[ep, 1],
                            rng, role_specs, is_target=False),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def generate_ID_outer_cube_dataset(
    n_episodes: int,
    box_anchors: Sequence[Sequence[float]],
    outer_spec: RoleSpec = DENSE_CUBE_SPEC,
    train_spec: RoleSpec = INNER_CUBE_SPEC,
    box_spec: RoleSpec = BOX_SPEC,
    margin_r: float = 0.01,
    margin_theta_deg: float = 0.5,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    seed: int = 100,
    stage: str = "OOD-cube",
) -> List[Dict[str, Any]]:
    """D0a 风格：单 cube 在训练区域外，box 固定在 box_anchors(应与训练时相同)。"""
    rng = np.random.default_rng(seed)
    role_specs = {"cube": outer_spec, "box": box_spec}
    box_anchors = np.asarray(box_anchors, dtype=float).reshape(-1, 2)

    pool = build_ood_pool(rng, outer_spec, train_spec, margin_r, margin_theta_deg,
                          avoid_points=box_anchors, avoid_dist=cross_min_dist)
    cube_pos = draw_from_pool(pool, n_episodes, rng)
    box_pos = tile_anchors(box_anchors, n_episodes, rng)
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        cube = make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1],
                           rng, role_specs, is_target=True)
        cube["ood_dist"], cube["ood_region"] = ood_info(cube["x"], cube["y"], train_spec)
        episodes.append({
            "episode_id": ep, "stage": stage,
            "objects": [
                cube,
                make_object("box", "blue", box_pos[ep, 0], box_pos[ep, 1],
                            rng, role_specs, is_target=False),
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes


def in_camera_fov(x: float, y: float, x_max: float = 0.55, y_abs_max: float = 0.36) -> bool:
    """摄像头能拍到的笛卡尔矩形范围：x <= x_max 且 |y| <= y_abs_max。"""
    return x <= x_max and abs(y) <= y_abs_max


def check_ood_disjoint(episodes: List[Dict[str, Any]], train_spec: RoleSpec = CUBE_SPEC,
                       role: str = "cube") -> None:
    """断言所有 OOD 物体都不在训练区域内，并打印按方向的分布 / 距离统计。"""
    from collections import Counter
    n_in, regions, dists = 0, Counter(), []
    for epd in episodes:
        for o in epd["objects"]:
            if o["role"] != role:
                continue
            if in_polar_region(o["x"], o["y"], train_spec.r_min, train_spec.r_max,
                               train_spec.theta_deg_min, train_spec.theta_deg_max):
                n_in += 1
            d, tag = ood_info(o["x"], o["y"], train_spec)
            regions[tag] += 1
            dists.append(d)
    assert n_in == 0, f"{n_in} 个 {role} 落在训练区域内，不是 OOD"
    dists = np.array(dists)
    print(f"  [{role}] OOD 检查通过：全部在训练区域外; "
          f"离训练区域距离 min/mean/max = {dists.min():.3f}/{dists.mean():.3f}/{dists.max():.3f} m")
    print(f"  [{role}] 方向分布: {dict(regions)}")


# =====================================================================
# Box 位置 OOD：cube 位置/朝向与训练一致，只有 box 位置变化
# =====================================================================

def best_candidate_rect(
    n: int, x_range: Tuple[float, float], y_range: Tuple[float, float],
    rng: np.random.Generator, k: int = 30,
    avoid_points: Optional[np.ndarray] = None, avoid_dist: float = 0.0,
) -> np.ndarray:
    """矩形区域内恰好生成 n 个互不重复、分布均匀的点(Mitchell's best-candidate, blue-noise)。
    每加一个点：抽 k 个候选，取离已有点最远的那个。不需要调 min_dist，点数精确等于 n。
    avoid_points / avoid_dist：候选点距 avoid_points 中任何点 < avoid_dist 时被丢弃。
    """
    def draw() -> np.ndarray:
        for _ in range(10000):
            c = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            if avoid_points is None or avoid_dist <= 0 or all(
                    np.hypot(*(c - a)) >= avoid_dist for a in avoid_points):
                return c
        raise RuntimeError("矩形内找不到满足 avoid 条件的点，检查 avoid_dist")

    pts = [draw()]
    while len(pts) < n:
        cands = np.stack([draw() for _ in range(k)])
        arr = np.array(pts)
        d = np.hypot(cands[:, None, 0] - arr[None, :, 0], cands[:, None, 1] - arr[None, :, 1]).min(1)
        pts.append(cands[int(np.argmax(d))])
    return np.array(pts)


def generate_OOD_box_dataset(
    n_episodes: int,
    x_range: Tuple[float, float] = (0.03, 0.18),
    y_range: Tuple[float, float] = (0.20, 0.35),
    train_box_anchors: Optional[Sequence[Sequence[float]]] = None,
    exclude_radius: float = 0.0,
    cube_pos: Optional[np.ndarray] = None,
    cube_spec: RoleSpec = CUBE_SPEC, box_spec: RoleSpec = BOX_SPEC,
    cross_min_dist: float = CROSS_OBJECT_MIN_DIST,
    seed: int = 200, stage: str = "OOD-box",
) -> List[Dict[str, Any]]:
    """cube 位置/朝向来自训练分布(CUBE_SPEC)，box 在笛卡尔矩形内取 n_episodes 个互不重复的位置。

    train_box_anchors : 训练时 box 的位置(如 D1 的硬编码 anchor)。给了就会：
                        (a) 在每个 box 上记录 ood_dist = 到最近训练 anchor 的距离；
                        (b) 若 exclude_radius>0，则矩形内距 anchor < exclude_radius 的区域不采样。
    cube_pos          : 可选，直接传入训练用的 cube 位置(n_episodes,2)，做到与训练"完全一样"；
                        None 时按 CUBE_SPEC 重新采样(分布一样，但具体坐标不同)。
    box orientation   : 沿用 box_spec 的训练范围，只有 position 是 OOD 变量。
    """
    rng = np.random.default_rng(seed)
    role_specs = {"cube": cube_spec, "box": box_spec}
    anchors = None if train_box_anchors is None else np.asarray(train_box_anchors, float).reshape(-1, 2)

    box_pos = best_candidate_rect(n_episodes, x_range, y_range, rng,
                                  avoid_points=anchors, avoid_dist=exclude_radius)
    if cube_pos is None:
        cube_pos = generate_manipulation_positions(n_episodes, cube_spec, rng)
    else:
        cube_pos = np.asarray(cube_pos, float)
        assert cube_pos.shape == (n_episodes, 2), "cube_pos 形状应为 (n_episodes, 2)"
    # 只交换 box 顺序来修复 cube-box 碰撞，不产生新点、不改 cube 分布
    box_pos = pair_targets_with_repair(cube_pos, box_pos, cross_min_dist, rng)

    episodes = []
    for ep in range(n_episodes):
        box = make_object("box", "blue", box_pos[ep, 0], box_pos[ep, 1], rng, role_specs, is_target=True)
        if anchors is not None:
            box["ood_dist"] = float(np.hypot(anchors[:, 0] - box["x"], anchors[:, 1] - box["y"]).min())
        episodes.append({
            "episode_id": ep, "stage": stage,
            "objects": [
                make_object("cube", "red", cube_pos[ep, 0], cube_pos[ep, 1], rng, role_specs, is_target=True),
                box,
            ],
            "instruction": "pick up the cube and place it in the box",
        })
    return episodes
