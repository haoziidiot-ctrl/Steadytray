"""locomotion 环境使用的事件函数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, Sdf, UsdGeom, Vt
import omni.usd
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_all_rel_pos = torch.tensor([])  # 跨回合缓存物体相对位置
_all_scales = torch.tensor([])  # 跨回合缓存物体缩放参数（半径与高度）
_all_cuboid_scales = torch.tensor([])  # 跨回合缓存 cuboid 缩放参数 (sx, sy, sz)，prestartup 一次写入后保持不变

# 作用：计算并设置物体相对机器人的位置与姿态。
def _compute_and_set_object_state(
    env: ManagerBasedRLEnv,
    qualified_env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]],
    relative_velocity: dict[str, float],
    z_offset: float | None = None,
) -> torch.Tensor:
    """计算并设置物体相对机器人的位置与姿态。"""
    global _all_scales
    
    base_asset: Articulation | RigidObject = env.scene[base_asset_cfg.name]
    target_object: RigidObject = env.scene[target_asset_cfg.name]
    
    object_base_height = getattr(target_object.cfg.spawn, 'height', 0.1)
    
    if isinstance(base_asset, Articulation):
        if isinstance(base_asset_cfg.body_ids, slice):
            raise ValueError("Body IDs for articulation must be specified as a list or int, not slice.")
        else:
            body_idx = base_asset_cfg.body_ids[0]
        base_pos = base_asset.data.body_link_pos_w[qualified_env_ids, body_idx]
        base_quat = base_asset.data.body_link_quat_w[qualified_env_ids, body_idx]
    else:
        base_pos = base_asset.data.root_pos_w[qualified_env_ids]
        base_quat = base_asset.data.root_quat_w[qualified_env_ids]

    num_resets = len(qualified_env_ids)

    if isinstance(relative_pose["x"], tuple):
        rel_pos_x = torch.rand(num_resets, device=env.device) * (relative_pose["x"][1] - relative_pose["x"][0]) + relative_pose["x"][0]
    else:
        rel_pos_x = torch.full((num_resets,), float(relative_pose["x"]), device=env.device)
        
    if isinstance(relative_pose["y"], tuple):
        rel_pos_y = torch.rand(num_resets, device=env.device) * (relative_pose["y"][1] - relative_pose["y"][0]) + relative_pose["y"][0]
    else:
        rel_pos_y = torch.full((num_resets,), float(relative_pose["y"]), device=env.device)

    if z_offset is not None:
        z_gap = torch.full((num_resets,), float(z_offset), device=env.device)
        
        if _all_scales.numel() > 0 and _all_scales.shape[0] >= env.num_envs:
            height_scales = _all_scales[qualified_env_ids, 1]  # 读取这些环境当前的高度缩放
        else:
            height_scales = torch.ones(num_resets, device=env.device)
        
        scaled_half_height = (object_base_height * height_scales) / 2.0
        rel_pos_z = z_gap + scaled_half_height
    else:
        if isinstance(relative_pose["z"], tuple):
            rel_pos_z = torch.rand(num_resets, device=env.device) * (relative_pose["z"][1] - relative_pose["z"][0]) + relative_pose["z"][0]
        else:
            rel_pos_z = torch.full((num_resets,), float(relative_pose["z"]), device=env.device)
    
    rel_pos_local = torch.stack([rel_pos_x, rel_pos_y, rel_pos_z], dim=1)
    
    rel_pos_world = math_utils.quat_apply(base_quat, rel_pos_local)
    object_pos = base_pos + rel_pos_world

    roll_spec = relative_pose.get("roll", 0.0)
    if isinstance(roll_spec, tuple):
        rel_roll = torch.rand(num_resets, device=env.device) * (roll_spec[1] - roll_spec[0]) + roll_spec[0]
    else:
        rel_roll = torch.full((num_resets,), float(roll_spec), device=env.device)
    
    pitch_spec = relative_pose.get("pitch", 0.0)
    if isinstance(pitch_spec, tuple):
        rel_pitch = torch.rand(num_resets, device=env.device) * (pitch_spec[1] - pitch_spec[0]) + pitch_spec[0]
    else:
        rel_pitch = torch.full((num_resets,), float(pitch_spec), device=env.device)
    
    yaw_spec = relative_pose.get("yaw", 0.0)
    if isinstance(yaw_spec, tuple):
        rel_yaw = torch.rand(num_resets, device=env.device) * (yaw_spec[1] - yaw_spec[0]) + yaw_spec[0]
    else:
        rel_yaw = torch.full((num_resets,), float(yaw_spec), device=env.device)
    
    rel_quat = math_utils.quat_from_euler_xyz(rel_roll, rel_pitch, rel_yaw)
    
    object_quat = math_utils.quat_mul(base_quat, rel_quat)

    rel_lin_vel_local = torch.stack([
        torch.full((num_resets,), float(relative_velocity["x"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["y"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["z"]), device=env.device)
    ], dim=1)
    
    rel_ang_vel_local = torch.stack([
        torch.full((num_resets,), float(relative_velocity["roll"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["pitch"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["yaw"]), device=env.device)
    ], dim=1)
    
    object_lin_vel = math_utils.quat_apply(base_quat, rel_lin_vel_local)
    object_ang_vel = math_utils.quat_apply(base_quat, rel_ang_vel_local)
    
    object_root_state = target_object.data.default_root_state[qualified_env_ids].clone()
    object_root_state[:, :3] = object_pos
    object_root_state[:, 3:7] = object_quat
    object_root_state[:, 7:10] = object_lin_vel
    object_root_state[:, 10:13] = object_ang_vel
    
    target_object.write_root_state_to_sim(object_root_state, qualified_env_ids)

    return rel_pos_local

# 作用：把刚体放置到机器人参考系下的相对位姿。
def set_rigid_object_relative_to_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]] = {
        "x": (-0.05, 0.05),  # 在 -5cm 到 +5cm 内随机扰动 x/y 位置
        "y": (-0.05, 0.05),
        "z": 0.08,           # 物体高于托盘的高度（若提供 z_offset 则表示间隙）
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0, 
    },
    relative_velocity: dict[str, float] = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
    z_offset: float | None = None,
) -> torch.Tensor:
    """把刚体放置到机器人参考系下的相对位姿。"""
    global _all_rel_pos

    if _all_rel_pos.numel() == 0 or _all_rel_pos.shape[0] < env.num_envs:
        _all_rel_pos = torch.zeros((env.num_envs, 3), device=env.device)

    if _all_rel_pos.device != env.device:
        _all_rel_pos = _all_rel_pos.to(env.device)

    base_asset_cfg.resolve(env.scene)
    target_asset_cfg.resolve(env.scene)
    
    rel_pos_local = _compute_and_set_object_state(
        env,
        env_ids,
        base_asset_cfg,
        target_asset_cfg,
        relative_pose,
        relative_velocity,
        z_offset,
    )

    _all_rel_pos[env_ids] = rel_pos_local

    return _all_rel_pos

# 作用：随机缩放圆柱物体并同步更新质量与碰撞属性。
def randomize_cylinder_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    radius_scale_range: tuple[float, float],
    height_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
) -> torch.Tensor:
    """随机缩放圆柱物体并同步更新质量与碰撞属性。"""
    global _all_scales
    
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    if _all_scales.numel() == 0 or _all_scales.shape[0] < env.num_envs:
        _all_scales = torch.ones((env.num_envs, 2), device=env.device)
    
    if _all_scales.device != env.device:
        _all_scales = _all_scales.to(env.device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    radius_samples = math_utils.sample_uniform(
        radius_scale_range[0], radius_scale_range[1], (len(env_ids),), device="cpu"
    )
    height_samples = math_utils.sample_uniform(
        height_scale_range[0], height_scale_range[1], (len(env_ids),), device="cpu"
    )
    
    sampled_scales = torch.stack([radius_samples, height_samples], dim=1).to(env.device)
    _all_scales[env_ids] = sampled_scales
    
    rand_samples = torch.stack([radius_samples, radius_samples, height_samples], dim=1)
    rand_samples = rand_samples.tolist()

    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
    
    return _all_scales

# 作用：随机缩放刚体长方体并写入 USD scale，prestartup 阶段每 env 一次性写入。
def randomize_cuboid_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    x_scale_range: tuple[float, float],
    y_scale_range: tuple[float, float],
    z_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
) -> torch.Tensor:
    """随机缩放 cuboid 物体的三轴尺寸。"""
    global _all_cuboid_scales

    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported."
        )

    if _all_cuboid_scales.numel() == 0 or _all_cuboid_scales.shape[0] < env.num_envs:
        _all_cuboid_scales = torch.ones((env.num_envs, 3), device=env.device)

    if _all_cuboid_scales.device != env.device:
        _all_cuboid_scales = _all_cuboid_scales.to(env.device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    x_samples = math_utils.sample_uniform(x_scale_range[0], x_scale_range[1], (len(env_ids),), device="cpu")
    y_samples = math_utils.sample_uniform(y_scale_range[0], y_scale_range[1], (len(env_ids),), device="cpu")
    z_samples = math_utils.sample_uniform(z_scale_range[0], z_scale_range[1], (len(env_ids),), device="cpu")

    sampled_scales = torch.stack([x_samples, y_samples, z_samples], dim=1).to(env.device)
    _all_cuboid_scales[env_ids] = sampled_scales

    rand_samples = torch.stack([x_samples, y_samples, z_samples], dim=1).tolist()

    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    return _all_cuboid_scales

# 作用：随机扰动刚体质心位置。
def randomize_rigid_body_com_fixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """随机扰动刚体质心位置。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    coms = asset.root_physx_view.get_coms().clone()
    coms[:,:3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)