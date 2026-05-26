# 作用：定义 X2 stage1（站立托盘）训练环境。机器人按 keyframe `tray_hold` 复位、托盘随机刷新、双手 wrist_roll_link 夹托盘。无走路命令、无物体。

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from steadytray.assets.robots.x2_delay import X2_DELAY_CFG, X2_DELAY_ACTION_SCALE
from steadytray.tasks import mdp

from .locomotion_env_cfg import FLAT_TERRAINS_CFG


# 托盘相对 pelvis 的标称位置（来自 x2_tray_hold.xml keyframe：tray@(0.40225,0,0.874), pelvis@(0,0,0.68)）
TRAY_NOMINAL_POS = [0.40225, 0.0, 0.194]


@configclass
class X2TrayStage1SceneCfg(InteractiveSceneCfg):
    """X2 stage1 场景：FLAT 地形 + X2 机器人 + 托盘 + 接触传感器 + frame transformer。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # 用平面地形，env_spacing 才生效，机器人才能按网格分散
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = X2_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 托盘：和 G1 stage2 同型，质量随机化在 events 里
    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        spawn=sim_utils.CuboidCfg(
            size=(0.214, 0.416, 0.016),  # 与 x2_tray_hold.xml plate 一致
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                compliant_contact_stiffness=8e4, compliant_contact_damping=400
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # 接触传感器：托盘 ↔ 双 wrist_roll_link（含手指 mesh）
    tray_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        track_air_time=True,
        history_length=10,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_wrist_roll_link",
            "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
        ],
    )

    # torso 系的托盘位姿（critic 看 lin/ang_vel 时也用 torso 作 reference）
    robot_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                name="tray",
            ),
        ],
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class X2TrayStage1EventCfg:
    """X2 stage1 事件：随机化机器人 / 托盘物理参数，reset 用 keyframe 默认姿态，托盘随机刷新。"""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 双 wrist_roll_link 摩擦单独抬高（夹托盘需要）
    wrist_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_roll_link"),
            "static_friction_range": (1.5, 2.5),
            "dynamic_friction_range": (1.2, 2.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

    random_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    random_tray_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "static_friction_range": (1.2, 2.0),
            "dynamic_friction_range": (1.0, 1.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 256,
        },
    )

    random_tray_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "mass_distribution_params": (0.3, 0.7),
            "operation": "abs",
        },
    )

    # 托盘 y 方向（宽度）随机化 ±0.1m，对应 y_scale ≈ (0.760, 1.240)。x/z 保持 1.0。
    # nominal y = 0.416 → 实际 y ∈ [0.316, 0.516]
    random_tray_width = EventTerm(
        func=mdp.randomize_cuboid_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "x_scale_range": (1.0, 1.0),
            "y_scale_range": (0.7596, 1.2404),
            "z_scale_range": (1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # reset 复用 X2_DELAY_CFG 的 default joint_pos（即 keyframe `tray_hold`），不做关节随机扰动
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 托盘随机刷新：相对 pelvis，标称 (0.40225, 0, 0.194) + 用户指定的随机扰动范围
    reset_tray_pos = EventTerm(
        func=mdp.set_rigid_object_relative_to_robot,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "target_asset_cfg": SceneEntityCfg("tray"),
            "relative_pose": {
                "x": (TRAY_NOMINAL_POS[0] - 0.05, TRAY_NOMINAL_POS[0] + 0.05),
                "y": (TRAY_NOMINAL_POS[1] - 0.03, TRAY_NOMINAL_POS[1] + 0.03),
                "z": (TRAY_NOMINAL_POS[2] - 0.02, TRAY_NOMINAL_POS[2] + 0.02),
                "roll": (-0.3, 0.3),    # ±0.3 rad = ±17.2°（每边），总跨度 34.4°
                "pitch": (-0.3, 0.3),   # 同上
                "yaw": (-0.1, 0.1),     # ±0.1 rad = ±5.7°（每边），总跨度 11.5°
            },
            "relative_velocity": {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        },
    )

    # 训练中扰动机器人 base，模拟外力
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class X2TrayStage1CommandCfg:
    """站着不动：base_velocity 全 0。保留 cfg 是因为 mdp.generated_commands 仍会被观测/奖励引用。"""

    base_velocity = mdp.UniformLevelVelocityBodyCommandCfg(
        asset_name="robot",
        body_name="torso_link",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        rel_heading_envs=0.0,
        rel_standing_envs=1.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityBodyCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityBodyCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class X2TrayStage1ActionCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=X2_DELAY_ACTION_SCALE,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class X2TrayStage1ObservationsCfg:
    """三组观测：policy（actor proprio）、encoder（proprio + 托盘 4 路 combined + delay）、critic（特权全集）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class EncoderCfg(ObsGroup):
        # proprio
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-25.0, 25.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)

        # 托盘 4 路（pos / ang_vel / lin_vel / gravity，全部 torso 系），共享 noise/delay buffer
        combined_tray_obs = ObsTerm(
            func=mdp.CombinedObjectObservationsDict,
            params={
                "object_sensor_cfg": SceneEntityCfg("robot_transform"),
                "object_asset_cfg": SceneEntityCfg("tray"),
                "robot_torso_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "target_frame_name": "tray",
                "include_pos": True,
                "include_ang_vel": True,
                "include_lin_vel": True,
                "include_gravity": True,
                "pos_noise": Unoise(n_min=-0.03, n_max=0.03),
                "ang_vel_noise": Unoise(n_min=-0.2, n_max=0.2),
                "lin_vel_noise": Unoise(n_min=-0.1, n_max=0.1),
                "gravity_noise": Unoise(n_min=-0.05, n_max=0.05),
                "pos_clip": (-1.0, 1.0),
                "ang_vel_clip": (-50.0, 50.0),
                "lin_vel_clip": (-10.0, 10.0),
                "gravity_clip": None,
                "pos_scale": 1.0,
                "ang_vel_scale": 0.2,
                "lin_vel_scale": 0.5,
                "gravity_scale": 1.0,
            },
            delay_min_lag=0,
            delay_max_lag=3,
            delay_per_env=True,
            delay_hold_prob=0.2,
            delay_update_period=10,
            delay_per_env_phase=True,
        )

        # 托盘宽度（y 方向，0.416 × y_scale），episode 内常量，部署时手动喂实物宽度
        tray_width = ObsTerm(
            func=mdp.cuboid_width_y,
            params={"asset_cfg": SceneEntityCfg("tray"), "nominal_size_y": 0.416},
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 32
            self.enable_corruption = True
            self.flatten_history_dim = False

    encoder: EncoderCfg = EncoderCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # proprio + 真值 base 速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-3.0, 3.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-25.0, 25.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)

        # 托盘特权：torso 系的 pos / ang_vel / lin_vel / 重力投影（无 noise/delay）
        tray_pos_rel = ObsTerm(
            func=mdp.object_rel_pos,
            params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"},
            clip=(-1.0, 1.0),
        )
        tray_ang_vel_rel = ObsTerm(
            func=mdp.object_rel_ang_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("tray"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            scale=0.2,
            clip=(-50.0, 50.0),
        )
        tray_lin_vel_rel = ObsTerm(
            func=mdp.object_rel_lin_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("tray"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            scale=0.5,
            clip=(-10.0, 10.0),
        )
        tray_projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")}
        )

        # 接触力（左右 wrist_roll_link 各 3 维 = 6 维）
        tray_holder_contact_forces = ObsTerm(
            func=mdp.tray_holder_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("tray_contact_sensor")},
            scale=0.1,
            clip=(-50.0, 50.0),
        )

        # 托盘宽度（与 encoder 同源），让 critic 区分宽窄板在相同 proprio/tray pose 下的 V 差异
        tray_width = ObsTerm(
            func=mdp.cuboid_width_y,
            params={"asset_cfg": SceneEntityCfg("tray"), "nominal_size_y": 0.416},
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class X2TrayStage1RewardsCfg:
    """X2 stage1 奖励：托盘水平为主导，contact + force 双约束 + 手臂回 keyframe + 力矩惩罚。"""

    # 站立朝向
    upright_torso = RewTerm(
        func=mdp.body_upright_bonus_exp,
        weight=0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "lambda_exp": 4.0},
    )
    torso_height = RewTerm(
        func=mdp.body_height_exp,
        weight=0.5,
        params={
            "target_height": 0.84,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "lambda_exp": 10.0,
        },
    )
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # 手臂保持 keyframe 姿态（参考 default joint pos）
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            ),
            "lambda_exp": 0.3,
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )

    # 托盘姿态：保持水平（stage1 主导奖励）
    tray_flat_orientation = RewTerm(
        func=mdp.object_upright_bonus_exp,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 4.0},
    )
    tray_lin_vel = RewTerm(
        func=mdp.object_lin_vel_z_exp,
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 2.0},
    )
    tray_ang_vel = RewTerm(
        func=mdp.object_ang_vel_xy_exp,
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 1.0},
    )

    # 接触：左右 wrist_roll_link 持续接触 + 力越小越好
    tray_contact = RewTerm(
        func=mdp.desired_contacts_count,
        weight=0.01,
        params={"sensor_cfg": SceneEntityCfg("tray_contact_sensor"), "threshold": 0.1},
    )
    tray_contact_force = RewTerm(
        func=mdp.contact_force_exp,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("tray_contact_sensor"), "lambda_exp": 0.005},
    )

    # 关节平滑性惩罚
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2_clipped, weight=-0.05, params={"max_penalty": 100.0})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2, weight=-2e-5, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 站立时禁止 torso 等非脚部位接触地面
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class X2TrayStage1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4},
        track_only=False,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        track_only=False,
    )
    tray_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("tray")},
        track_only=True,
        track_only_delay=1.0,
    )


@configclass
class X2TrayStage1CurriculumCfg:
    """X2 stage1 不需要课程学习（FLAT 平地、零命令）。"""

    pass


@configclass
class X2TrayStage1EnvCfg(ManagerBasedRLEnvCfg):
    """X2 stage1 训练入口。"""

    scene: X2TrayStage1SceneCfg = X2TrayStage1SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: X2TrayStage1ObservationsCfg = X2TrayStage1ObservationsCfg()
    actions: X2TrayStage1ActionCfg = X2TrayStage1ActionCfg()
    commands: X2TrayStage1CommandCfg = X2TrayStage1CommandCfg()
    rewards: X2TrayStage1RewardsCfg = X2TrayStage1RewardsCfg()
    terminations: X2TrayStage1TerminationsCfg = X2TrayStage1TerminationsCfg()
    events: X2TrayStage1EventCfg = X2TrayStage1EventCfg()
    curriculum: X2TrayStage1CurriculumCfg = X2TrayStage1CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 2**24

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class X2TrayStage1PlayEnvCfg(X2TrayStage1EnvCfg):
    """X2 stage1 可视化调试用配置（少量 env、立即终止）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "torso_link"
        self.viewer.eye = (3.2, -0.6, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.35)
        self.viewer.resolution = (1920, 1080)

        # play 模式下立即终止，便于看效果
        self.terminations.tray_fallen.track_only_delay = 0.0
