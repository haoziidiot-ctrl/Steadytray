from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_foot_vel_z_prev: torch.Tensor | None = None  # 缓存每只脚上一时刻的竖直速度

"""
Joint penalties.
"""

# 作用：惩罚机器人关节能耗。
def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚机器人关节能耗。"""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

# 作用：奖励关节位置接近默认姿态。
def joint_deviation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """奖励关节位置接近默认姿态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)
    return torch.exp(-lambda_exp * deviation)

"""
Feet rewards.
"""

# 作用：奖励摆动脚达到目标离地高度。
def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """奖励摆动脚达到目标离地高度。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

# 作用：奖励足部竖直速度变化平滑。
def feet_smooth_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    lambda_exp: float = 1.0,
) -> torch.Tensor:
    """奖励足部竖直速度变化平滑。"""
    global _foot_vel_z_prev

    robot: Articulation = env.scene[asset_cfg.name]

    foot_vel_z = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # 形状: [num_envs, num_feet]

    if _foot_vel_z_prev is None or _foot_vel_z_prev.shape[0] != env.num_envs:
        _foot_vel_z_prev = foot_vel_z.clone()
        return torch.ones(env.num_envs, device=env.device)  # 第一步直接返回最大奖励

    delta_v_z_sq = torch.square(foot_vel_z - _foot_vel_z_prev)

    total_dv = torch.sum(delta_v_z_sq, dim=1)

    _foot_vel_z_prev = foot_vel_z.clone()

    return torch.exp(-lambda_exp * total_dv)

"""
Body-specific velocity and height rewards.
"""

# 作用：奖励目标 body 的竖直线速度接近零。
def body_lin_vel_z_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 2.0
) -> torch.Tensor:
    """奖励目标 body 的竖直线速度接近零。"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))

# 作用：奖励目标 body 的横滚与俯仰角速度接近零。
def body_ang_vel_xy_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """奖励目标 body 的横滚与俯仰角速度接近零。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    return torch.exp(-lambda_exp * ang_vel_xy_norm_sq)

# 作用：奖励目标 body 保持直立。
def body_upright_bonus_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 4.0
) -> torch.Tensor:
    """奖励目标 body 保持直立。"""
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    gravity_vec = asset.data.GRAVITY_VEC_W
    projected_gravity = math_utils.quat_apply_inverse(body_quat, gravity_vec)

    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)

# 作用：奖励目标 body 维持在期望高度附近。
def body_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    lambda_exp: float = 10.0
) -> torch.Tensor:
    """奖励目标 body 维持在期望高度附近。"""
    asset: Articulation = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target_height = target_height

    current_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_error = current_height - adjusted_target_height

    return torch.exp(-lambda_exp * torch.abs(height_error))

# 作用：奖励 body 坐标系下的线速度跟踪表现。
def track_lin_vel_xy_yaw_body_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励 body 坐标系下的线速度跟踪表现。"""

    asset: Articulation = env.scene[asset_cfg.name]

    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], :3]

    vel_yaw = quat_apply_inverse(yaw_quat(body_quat_w), body_lin_vel_w)

    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

# 作用：奖励 body 坐标系下的偏航角速度跟踪表现。
def track_ang_vel_z_body_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励 body 坐标系下的偏航角速度跟踪表现。"""
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :]

    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

"""
Action penalty
"""

# 作用：惩罚连续动作之间的变化率。
def action_rate_l2_clipped(env: ManagerBasedRLEnv, max_penalty: float = 1.0) -> torch.Tensor:
    """惩罚连续动作之间的变化率。"""
    action_rate_penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(action_rate_penalty, max=max_penalty)

"""
Object rewards.
"""

# 作用：奖励物体保持直立。
def object_upright_bonus_exp(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, lambda_exp: float = 4.0) -> torch.Tensor:
    """奖励物体保持直立。"""
    object: RigidObject = env.scene[object_cfg.name]

    projected_gravity = object.data.projected_gravity_b
    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)

# 作用：惩罚物体相对重力方向的横滚/俯仰倾斜。
def object_tilt_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """惩罚物体相对重力方向的横滚/俯仰倾斜。"""
    object: RigidObject = env.scene[object_cfg.name]
    projected_gravity = object.data.projected_gravity_b
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

# 作用：奖励物体横滚与俯仰角速度接近零。
def object_ang_vel_xy_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """奖励物体横滚与俯仰角速度接近零。"""
    object: RigidObject = env.scene[object_cfg.name]
    ang_vel_xy = object.data.body_ang_vel_w[:, 0, :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    return torch.exp(-lambda_exp * ang_vel_xy_norm_sq)

# 作用：奖励物体竖直线速度接近零。
def object_lin_vel_z_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lambda_exp: float = 2.0
) -> torch.Tensor:
    """奖励物体竖直线速度接近零。"""
    object: RigidObject = env.scene[object_cfg.name]
    vel_z = object.data.body_lin_vel_w[:, 0, 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))

# 作用：计算两个实体姿态差的 L1 度量。
def entity_quat_l1(env: ManagerBasedRLEnv, entity1_cfg: SceneEntityCfg, entity2_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算两个实体姿态差的 L1 度量。"""
    entity1 = env.scene[entity1_cfg.name]
    entity2 = env.scene[entity2_cfg.name]

    if isinstance(entity1, RigidObject):
        entity1_quat = entity1.data.body_quat_w[:, 0, :]
    else:  # Articulation 资产
        entity1_quat = entity1.data.body_quat_w[:, entity1_cfg.body_ids[0], :]

    if isinstance(entity2, RigidObject):
        entity2_quat = entity2.data.body_quat_w[:, 0, :]
    else:  # Articulation 资产
        entity2_quat = entity2.data.body_quat_w[:, entity2_cfg.body_ids[0], :]

    quat_dot = torch.sum(entity1_quat * entity2_quat, dim=-1)
    quat_dot = torch.clamp(torch.abs(quat_dot), max=1.0)
    orientation_difference = 2.0 * torch.acos(quat_dot)

    return orientation_difference

# 作用：用指数核奖励两个实体姿态接近。
def entity_quat_exp(
    env: ManagerBasedRLEnv,
    entity1_cfg: SceneEntityCfg,
    entity2_cfg: SceneEntityCfg,
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """用指数核奖励两个实体姿态接近。"""
    orientation_error_l1 = entity_quat_l1(
        env=env,
        entity1_cfg=entity1_cfg,
        entity2_cfg=entity2_cfg,
    )

    return torch.exp(-lambda_exp * orientation_error_l1)

"""
Contact rewards.
"""

# 作用：统计并奖励期望的接触数量。
def desired_contacts_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """统计并奖励期望的接触数量。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    contacts = (
        contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :].norm(dim=-1) > threshold
    )
    total_active_contacts = contacts.sum(dim=(1, 2, 3)).float()

    return total_active_contacts

# 作用：用指数核约束接触力大小。
def contact_force_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    lambda_exp: float = 0.01
) -> torch.Tensor:
    """用指数核约束接触力大小。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]

    force_magnitudes = torch.norm(contact_forces, dim=-1)

    total_force = force_magnitudes.sum(dim=(1, 2))

    return torch.exp(-lambda_exp * total_force)
