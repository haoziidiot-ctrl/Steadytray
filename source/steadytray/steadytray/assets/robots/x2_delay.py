# 作用：定义 X2 29-DOF 机器人的 ArticulationCfg，含 Delayed PD 执行器、默认 keyframe `tray_hold` 姿态、按力矩等级分组的 stiffness/damping。

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from pathlib import Path

from . import unitree

X2_ASSETS_DIR = (Path(__file__).resolve().parents[1] / "usds").as_posix()

# TODO: 将 deploy/deploy_mujoco/x2t2d5_description_0525/ 下的 URDF 转换为 USD 后，把路径替换为实际产物。
X2_USD_PATH = f"{X2_ASSETS_DIR}/x2_29dof_hand.usd"

MAX_DELAY = 4  # 与 G1 一致

# X2 keyframe `tray_hold`: 双肘 -1.57、双肩 roll ±0.061、其余 0；初始 pelvis z=0.68（来自 MJCF）
X2_DELAY_CFG = unitree.UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=X2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos={
            # 未列出的关节（head/finger 等）默认 0.0
            # shoulder_roll 限位 ±[-0.061, 2.993]，keyframe tray_hold 用边界值 ±0.061，
            # 这里向内收 1e-3 避免 USD 浮点精度判为越界
            "left_shoulder_roll_joint": -0.060,
            "right_shoulder_roll_joint": 0.060,
            ".*_elbow_joint": -1.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 大力关节：髋三轴 + 膝（actuatorfrcrange ±120）
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.0,
            stiffness=150.0,
            damping=5.0,
            armature=0.02906,
            min_delay=0,
            max_delay=MAX_DELAY,
        ),
        # 踝：pitch ±36, roll ±24
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim=37.0,
            stiffness=50.0,
            damping=2.0,
            armature={
                ".*_ankle_pitch_joint": 0.00884,
                ".*_ankle_roll_joint": 0.003435,
            },
            min_delay=0,
            max_delay=MAX_DELAY,
        ),
        # 腰：yaw ±120, pitch/roll ±48
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 120.0,
                "waist_pitch_joint": 48.0,
                "waist_roll_joint": 48.0,
            },
            velocity_limit_sim=20.0,
            stiffness=100.0,
            damping=3.0,
            armature={
                "waist_yaw_joint": 0.02906,
                "waist_pitch_joint": 0.00687,
                "waist_roll_joint": 0.00687,
            },
            min_delay=0,
            max_delay=MAX_DELAY,
        ),
        # 肩 + 肘：shoulder pitch/roll ±36, shoulder_yaw/elbow ±24
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
            },
            velocity_limit_sim=37.0,
            stiffness=80.0,
            damping=2.0,
            armature={
                ".*_shoulder_pitch_joint": 0.00884,
                ".*_shoulder_roll_joint": 0.00884,
                ".*_shoulder_yaw_joint": 0.003435,
                ".*_elbow_joint": 0.003435,
            },
            min_delay=0,
            max_delay=MAX_DELAY,
        ),
        # 腕：yaw ±24, pitch/roll ±4.8（手腕很弱，单独低增益分组）
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wrist_yaw_joint", ".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim=22.0,
            stiffness={
                ".*_wrist_yaw_joint": 30.0,
                ".*_wrist_pitch_joint": 15.0,
                ".*_wrist_roll_joint": 15.0,
            },
            damping={
                ".*_wrist_yaw_joint": 1.0,
                ".*_wrist_pitch_joint": 0.5,
                ".*_wrist_roll_joint": 0.5,
            },
            armature=0.003435,
            min_delay=0,
            max_delay=MAX_DELAY,
        ),
    },
)


X2_JOINT_SDK_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
X2_DELAY_CFG.joint_sdk_names = X2_JOINT_SDK_NAMES


X2_DELAY_ACTION_SCALE = {}
for _a in X2_DELAY_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for _n in _names:
        if _n in _e and _n in _s and _s[_n]:
            X2_DELAY_ACTION_SCALE[_n] = 0.25 * _e[_n] / _s[_n]
