# 作用：定义 Stage 4 蒸馏环境。这里重点决定 student 看什么观测、相机系物体观测怎么构造，以及 play 模式下如何缩小环境。

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .steady_object_env_cfg import SteadyObjectEnvCfg, ObjectObservationsCfg, ObjectSceneCfg, ObjectTerminationsPlayCfg

@configclass
# 作用：为蒸馏阶段额外定义相机参考系到物体的变换，用来构造 student 的相机系物体观测。
class DistillSceneCfg(ObjectSceneCfg):
    """Configuration for the scene in the steady object distillation environment."""

    object_camera_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link",  # Reference frame
        update_period= 1.0 / 30.0,   # Set to camera frame rate
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                name="object",
            ),
        ],
    )


@configclass
# 作用：定义 Stage 4 的观测组；核心是新增 student_encoder 这组有限观测。
class DistillObservationsCfg(ObjectObservationsCfg):
    """Configuration for observations in the steady object distillation environment."""
    
    @configclass
    # 作用：定义 student encoder 的时序输入，包含本体状态和相机系下的物体观测。
    class StudentEncoderCfg(ObsGroup):
        """Configuration for student encoder observations."""

        # Locomotion robot observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-25.0, 25.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)

        # Combined object observations in camera frame with synchronized delays
        camera_object_obs = ObsTerm(
            func=mdp.CombinedCameraObjectObservations,
            params={
                "camera_sensor_cfg": SceneEntityCfg("object_camera_transform"),
                "object_asset_cfg": SceneEntityCfg("object"),
                "target_frame_name": "object",
                "scale_event_term_name": "random_object_scale",
                "include_pos": True,
                "include_quat": True,
                "pos_scale": 1.0,
                "quat_scale": 1.0,
                "pos_clip": (-1.0, 1.0),
                "quat_clip": None,
                "pos_noise": Unoise(n_min=-0.05, n_max=0.05),
                "quat_noise_std": 0.2,
            },
            delay_min_lag=0,        # Minimum delay in steps (0 = no delay)
            delay_max_lag=3,        # Maximum delay in steps (3 steps x 0.02s = 0.06s)
            delay_per_env=True,     # Each environment has independent delay
            delay_hold_prob=0.2,    # 20% chance to keep previous lag
            delay_update_period=10, # Update lag every 10 steps (0.2s)
            delay_per_env_phase=True, # Stagger lag updates across environments
        )

        # 作用：设置 student_encoder 观测的历史长度、噪声开关和是否保留时间维。
        # 作用：设置 student_encoder 观测的历史长度、噪声开关和是否保留时间维。
        def __post_init__(self):
            self.history_length = 32
            self.enable_corruption = True
            self.flatten_history_dim = False

    student_encoder: StudentEncoderCfg = StudentEncoderCfg()


@configclass
# 作用：定义蒸馏阶段使用的速度命令采样配置。
class DistillCommandCfg:
    """Configuration for the velocity command in the steady tray with object environment."""

    base_velocity = mdp.DelayedUniformVelocityCommandCfg(
        asset_name="robot",
        body_name="torso_link",
        delay_time=1.0,  # Zero velocity for first second
        heading_command=True,
        rel_heading_envs=0.5,
        rel_standing_envs=0.02,
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.DelayedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14)
        ),
        limit_ranges=mdp.DelayedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14)
        ),
        debug_vis=True,
        class_type=mdp.DelayedUniformVelocityCommand,
    )


@configclass
# 作用：把蒸馏阶段的 scene 与 observations 组装成正式训练环境。
class SteadyObjectDistillEnvCfg(SteadyObjectEnvCfg):
    """Configuration for the steady object distillation environment."""

    scene: DistillSceneCfg = DistillSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: DistillObservationsCfg = DistillObservationsCfg()


@configclass
# 作用：提供蒸馏阶段的调试/可视化环境配置，便于少量环境下观察行为。
class SteadyObjectDistillPlayEnvCfg(SteadyObjectDistillEnvCfg):

    terminations: ObjectTerminationsPlayCfg = ObjectTerminationsPlayCfg()

    # 作用：缩小 play 模式的环境规模，并限制命令范围，方便可视化调试。
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges