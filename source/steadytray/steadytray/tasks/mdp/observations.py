from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import NoiseCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 作用：读取指定刚体在本体系下的投影重力。
def rigid_body_projected_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rigid_object")
) -> torch.Tensor:
    """读取指定刚体在本体系下的投影重力。"""
    rigid_object: RigidObject = env.scene[asset_cfg.name]

    projected_gravity = rigid_object.data.projected_gravity_b

    return projected_gravity

# 作用：读取物体相对参考系的位置。
def object_rel_pos(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object"
) -> torch.Tensor:
    """读取物体相对参考系的位置。"""
    frame_transformer = env.scene[sensor_cfg.name]

    relative_pos = frame_transformer.data.target_pos_source[:, frame_transformer.data.target_frame_names.index(target_frame_name), :]

    return relative_pos

# 作用：读取物体相对参考系的姿态四元数。
def object_rel_quat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object"
) -> torch.Tensor:
    """读取物体相对参考系的姿态四元数。"""
    frame_transformer = env.scene[sensor_cfg.name]

    relative_quat = frame_transformer.data.target_quat_source[:, frame_transformer.data.target_frame_names.index(target_frame_name), :]

    return relative_quat

# 作用：读取带姿态噪声的相对四元数观测。
def object_rel_quat_with_noise(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object",
    noise_std: float = 0.05
) -> torch.Tensor:
    """读取带姿态噪声的相对四元数观测。"""
    relative_quat = object_rel_quat(env, sensor_cfg=sensor_cfg, target_frame_name=target_frame_name)

    random_axis = torch.randn(env.num_envs, 3, device=env.device)
    random_axis = random_axis / (torch.norm(random_axis, dim=-1, keepdim=True) + 1e-8)

    random_angles = torch.randn(env.num_envs, device=env.device) * noise_std

    half_angles = random_angles / 2.0
    noise_quat = torch.zeros(env.num_envs, 4, device=env.device)
    noise_quat[:, 0] = torch.cos(half_angles)  # 四元数的 w 分量
    noise_quat[:, 1:] = torch.sin(half_angles).unsqueeze(-1) * random_axis  # 四元数的 xyz 分量

    noisy_quat = math_utils.quat_mul(noise_quat, relative_quat)

    return noisy_quat

# 作用：计算目标实体相对参考实体的线速度。
def object_rel_lin_vel(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis")
) -> torch.Tensor:
    """计算目标实体相对参考实体的线速度。"""
    target_asset = env.scene[target_asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]

    if isinstance(target_asset, RigidObject):
        target_lin_vel_w = target_asset.data.root_link_lin_vel_w
    elif isinstance(target_asset, Articulation):
        if isinstance(target_asset_cfg.body_ids, slice):
            raise ValueError("target_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = target_asset_cfg.body_ids[0] if isinstance(target_asset_cfg.body_ids, list) else target_asset_cfg.body_ids
        target_lin_vel_w = target_asset.data.body_link_lin_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported target asset type: {type(target_asset)}")

    if isinstance(reference_asset, RigidObject):
        reference_lin_vel_w = reference_asset.data.root_link_lin_vel_w
    elif isinstance(reference_asset, Articulation):
        if isinstance(reference_asset_cfg.body_ids, slice):
            raise ValueError("reference_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = reference_asset_cfg.body_ids[0] if isinstance(reference_asset_cfg.body_ids, list) else reference_asset_cfg.body_ids
        reference_lin_vel_w = reference_asset.data.body_link_lin_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported reference asset type: {type(reference_asset)}")

    rel_lin_vel = target_lin_vel_w - reference_lin_vel_w

    return rel_lin_vel

# 作用：计算目标实体相对参考实体的角速度。
def object_rel_ang_vel(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis")
) -> torch.Tensor:
    """计算目标实体相对参考实体的角速度。"""
    target_asset = env.scene[target_asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]

    if isinstance(target_asset, RigidObject):
        target_ang_vel_w = target_asset.data.root_link_ang_vel_w
    elif isinstance(target_asset, Articulation):
        if isinstance(target_asset_cfg.body_ids, slice):
            raise ValueError("target_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = target_asset_cfg.body_ids[0] if isinstance(target_asset_cfg.body_ids, list) else target_asset_cfg.body_ids
        target_ang_vel_w = target_asset.data.body_link_ang_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported target asset type: {type(target_asset)}")

    if isinstance(reference_asset, RigidObject):
        reference_ang_vel_w = reference_asset.data.root_link_ang_vel_w
    elif isinstance(reference_asset, Articulation):
        if isinstance(reference_asset_cfg.body_ids, slice):
            raise ValueError("reference_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = reference_asset_cfg.body_ids[0] if isinstance(reference_asset_cfg.body_ids, list) else reference_asset_cfg.body_ids
        reference_ang_vel_w = reference_asset.data.body_link_ang_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported reference asset type: {type(reference_asset)}")

    rel_ang_vel = target_ang_vel_w - reference_ang_vel_w

    return rel_ang_vel

# 作用：读取物体顶部相对参考系的位置。
def object_rel_pos_top(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale_event_term_name: str = "random_object_scale"
) -> torch.Tensor:
    """读取物体顶部相对参考系的位置。"""
    frame_transformer = env.scene[sensor_cfg.name]
    target_idx = frame_transformer.data.target_frame_names.index(target_frame_name)

    relative_pos = frame_transformer.data.target_pos_source[:, target_idx, :]

    relative_quat = frame_transformer.data.target_quat_source[:, target_idx, :]

    target_object: RigidObject = env.scene[object_cfg.name]
    object_base_height = getattr(target_object.cfg.spawn, 'height', 0.1)

    try:
        scales = env.event_manager.get_term_return_value(scale_event_term_name)
        if scales is not None and scales.numel() > 0:
            height_scales = scales[:, 1]
        else:
            height_scales = torch.ones(env.num_envs, device=env.device)
    except (ValueError, AttributeError, IndexError):
        height_scales = torch.ones(env.num_envs, device=env.device)

    half_scaled_height = (object_base_height * height_scales) / 2.0

    offset_object_frame = torch.zeros(env.num_envs, 3, device=env.device)
    offset_object_frame[:, 2] = half_scaled_height

    offset_camera_frame = math_utils.quat_apply(relative_quat, offset_object_frame)

    relative_pos_top = relative_pos + offset_camera_frame

    return relative_pos_top

# 作用：读取 tray holder 的接触力观测。
def tray_holder_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tray_contact_sensor"),
) -> torch.Tensor:
    """读取 tray holder 的接触力观测。"""
    force_matrix = env.scene.sensors[sensor_cfg.name].data.force_matrix_w

    holder_forces = force_matrix.squeeze(1)

    force_vectors = holder_forces.view(holder_forces.shape[0], -1)

    return force_vectors

# 作用：组合相机特征与物体状态观测。
class CombinedCameraObjectObservations(ManagerTermBase):
    """组合相机特征与物体状态观测。"""

    # 作用：初始化组合观测所需的配置和缓存。
    def __init__(
        self,
        cfg: SceneEntityCfg,
        env: ManagerBasedRLEnv,
        camera_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_camera_transform"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        target_frame_name: str = "object",
        scale_event_term_name: str = "random_object_scale",
        include_pos: bool = True,
        include_quat: bool = True,
        pos_scale: float = 1.0,
        quat_scale: float = 1.0,
        pos_clip: tuple[float, float] | None = None,
        quat_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        quat_noise_std: float = 0.0,
    ):
        super().__init__(cfg, env)

        camera_sensor_cfg.resolve(env.scene)
        object_asset_cfg.resolve(env.scene)

        self.camera_sensor_cfg = camera_sensor_cfg
        self.object_asset_cfg = object_asset_cfg
        self.target_frame_name = target_frame_name
        self.scale_event_term_name = scale_event_term_name

        self.include_pos = include_pos
        self.include_quat = include_quat

        self.pos_scale = pos_scale
        self.quat_scale = quat_scale

        self.pos_clip = pos_clip
        self.quat_clip = quat_clip

        self.pos_noise = pos_noise
        self.quat_noise_std = quat_noise_std

    # 作用：计算拼接后的相机与物体联合观测。
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        camera_sensor_cfg: SceneEntityCfg | None = None,
        object_asset_cfg: SceneEntityCfg | None = None,
        target_frame_name: str | None = None,
        scale_event_term_name: str | None = None,
        include_pos: bool | None = None,
        include_quat: bool | None = None,
        pos_scale: float | None = None,
        quat_scale: float | None = None,
        pos_clip: tuple[float, float] | None = None,
        quat_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        quat_noise_std: float | None = None,
    ) -> torch.Tensor:
        observations = []

        if self.include_pos:
            pos_rel = object_rel_pos_top(
                env,
                sensor_cfg=self.camera_sensor_cfg,
                target_frame_name=self.target_frame_name,
                object_cfg=self.object_asset_cfg,
                scale_event_term_name=self.scale_event_term_name
            )
            if self.pos_noise is not None:
                pos_rel = self.pos_noise.func(pos_rel, self.pos_noise)
            if self.pos_clip is not None:
                pos_rel = torch.clamp(pos_rel, self.pos_clip[0], self.pos_clip[1])
            pos_rel = pos_rel * self.pos_scale
            observations.append(pos_rel)

        if self.include_quat:
            quat_rel = object_rel_quat_with_noise(
                env,
                sensor_cfg=self.camera_sensor_cfg,
                target_frame_name=self.target_frame_name,
                noise_std=self.quat_noise_std
            )
            if self.quat_clip is not None:
                quat_rel = torch.clamp(quat_rel, self.quat_clip[0], self.quat_clip[1])
            quat_rel = quat_rel * self.quat_scale
            observations.append(quat_rel)

        if observations:
            combined = torch.cat(observations, dim=-1)
        else:
            combined = torch.zeros(env.num_envs, 0, device=env.device)

        return combined

# 作用：输出字典形式的组合物体观测。
class CombinedObjectObservationsDict(ManagerTermBase):
    """输出字典形式的组合物体观测。"""

    # 作用：初始化字典观测组合器。
    def __init__(
        self,
        cfg: SceneEntityCfg,
        env: ManagerBasedRLEnv,
        object_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
        target_frame_name: str = "object",
        include_pos: bool = True,
        include_ang_vel: bool = True,
        include_lin_vel: bool = True,
        include_gravity: bool = True,
        pos_scale: float = 1.0,
        ang_vel_scale: float = 1.0,
        lin_vel_scale: float = 1.0,
        gravity_scale: float = 1.0,
        pos_clip: tuple[float, float] | None = None,
        ang_vel_clip: tuple[float, float] | None = None,
        lin_vel_clip: tuple[float, float] | None = None,
        gravity_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        ang_vel_noise: NoiseCfg | None = None,
        lin_vel_noise: NoiseCfg | None = None,
        gravity_noise: NoiseCfg | None = None,
    ):
        super().__init__(cfg, env)

        object_sensor_cfg.resolve(env.scene)
        object_asset_cfg.resolve(env.scene)
        robot_torso_cfg.resolve(env.scene)

        self.object_sensor_cfg = object_sensor_cfg
        self.object_asset_cfg = object_asset_cfg
        self.robot_torso_cfg = robot_torso_cfg
        self.target_frame_name = target_frame_name

        self.include_pos = include_pos
        self.include_ang_vel = include_ang_vel
        self.include_lin_vel = include_lin_vel
        self.include_gravity = include_gravity

        self.pos_scale = pos_scale
        self.ang_vel_scale = ang_vel_scale
        self.lin_vel_scale = lin_vel_scale
        self.gravity_scale = gravity_scale

        self.pos_clip = pos_clip
        self.ang_vel_clip = ang_vel_clip
        self.lin_vel_clip = lin_vel_clip
        self.gravity_clip = gravity_clip

        self.pos_noise = pos_noise
        self.ang_vel_noise = ang_vel_noise
        self.lin_vel_noise = lin_vel_noise
        self.gravity_noise = gravity_noise

    # 作用：计算并返回字典形式的组合观测。
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_sensor_cfg: SceneEntityCfg | None = None,
        object_asset_cfg: SceneEntityCfg | None = None,
        robot_torso_cfg: SceneEntityCfg | None = None,
        target_frame_name: str | None = None,
        include_pos: bool | None = None,
        include_ang_vel: bool | None = None,
        include_lin_vel: bool | None = None,
        include_gravity: bool | None = None,
        pos_scale: float | None = None,
        ang_vel_scale: float | None = None,
        lin_vel_scale: float | None = None,
        gravity_scale: float | None = None,
        pos_clip: tuple[float, float] | None = None,
        ang_vel_clip: tuple[float, float] | None = None,
        lin_vel_clip: tuple[float, float] | None = None,
        gravity_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        ang_vel_noise: NoiseCfg | None = None,
        lin_vel_noise: NoiseCfg | None = None,
        gravity_noise: NoiseCfg | None = None,
    ) -> torch.Tensor:
        observations = []

        if self.include_pos:
            pos_rel = object_rel_pos(
                env,
                sensor_cfg=self.object_sensor_cfg,
                target_frame_name=self.target_frame_name
            )
            if self.pos_noise is not None:
                pos_rel = self.pos_noise.func(pos_rel, self.pos_noise)
            if self.pos_clip is not None:
                pos_rel = torch.clamp(pos_rel, self.pos_clip[0], self.pos_clip[1])
            pos_rel = pos_rel * self.pos_scale
            observations.append(pos_rel)

        if self.include_ang_vel:
            ang_vel_rel = object_rel_ang_vel(
                env,
                target_asset_cfg=self.object_asset_cfg,
                reference_asset_cfg=self.robot_torso_cfg
            )
            if self.ang_vel_noise is not None:
                ang_vel_rel = self.ang_vel_noise.func(ang_vel_rel, self.ang_vel_noise)
            if self.ang_vel_clip is not None:
                ang_vel_rel = torch.clamp(ang_vel_rel, self.ang_vel_clip[0], self.ang_vel_clip[1])
            ang_vel_rel = ang_vel_rel * self.ang_vel_scale
            observations.append(ang_vel_rel)

        if self.include_lin_vel:
            lin_vel_rel = object_rel_lin_vel(
                env,
                target_asset_cfg=self.object_asset_cfg,
                reference_asset_cfg=self.robot_torso_cfg
            )
            if self.lin_vel_noise is not None:
                lin_vel_rel = self.lin_vel_noise.func(lin_vel_rel, self.lin_vel_noise)
            if self.lin_vel_clip is not None:
                lin_vel_rel = torch.clamp(lin_vel_rel, self.lin_vel_clip[0], self.lin_vel_clip[1])
            lin_vel_rel = lin_vel_rel * self.lin_vel_scale
            observations.append(lin_vel_rel)

        if self.include_gravity:
            proj_grav = rigid_body_projected_gravity(
                env,
                asset_cfg=self.object_asset_cfg
            )
            if self.gravity_noise is not None:
                proj_grav = self.gravity_noise.func(proj_grav, self.gravity_noise)
            if self.gravity_clip is not None:
                proj_grav = torch.clamp(proj_grav, self.gravity_clip[0], self.gravity_clip[1])
            proj_grav = proj_grav * self.gravity_scale
            observations.append(proj_grav)

        if observations:
            combined = torch.cat(observations, dim=-1)
        else:
            combined = torch.zeros(env.num_envs, 0, device=env.device)

        return combined
