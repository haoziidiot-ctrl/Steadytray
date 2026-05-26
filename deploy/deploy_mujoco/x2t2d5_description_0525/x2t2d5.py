import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DelayedPDActuatorCfg,IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from textop_tracker.assets import ASSET_DIR

REDUCTION_RATIO_PF90 = 21.9
REDUCTION_RATIO_PF72 = 20.0
REDUCTION_RATIO_PF59 = 19.43

MOTOR_ARMATURE_PF90 = 82.26e-6
MOTOR_ARMATURE_PF72 = 25.74e-6
MOTOR_ARMATURE_PF59 = 10.05e-6

ARMATURE_PF90 = MOTOR_ARMATURE_PF90 * REDUCTION_RATIO_PF90**2
ARMATURE_PF72 = MOTOR_ARMATURE_PF72 * REDUCTION_RATIO_PF72**2
ARMATURE_PF59 = MOTOR_ARMATURE_PF59 * REDUCTION_RATIO_PF59**2

JOINT_EFFORT_PF90 = 112.5 # 100~125
JOINT_EFFORT_PF72 = 41 # 32~50
JOINT_EFFORT_PF59 = 26.5 # 18~35
JOINT_EFFORT_WRIST = 3.2

JOINT_VEL_PF90 = 118/60 * 2 * 3.1415926535
JOINT_VEL_PF72 = 139/60 * 2 * 3.1415926535
JOINT_VEL_PF59 = 179/60 * 2 * 3.1415926535
JOINT_VEL_WRIST = 4.15


X2T2D5_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/x2t2d5_description/x2_29dof_hand_simple_collision.urdf",
        # asset_path=f"{ASSET_DIR}/robot_model-t2.5-v1.2.1/x2_ultra_simple_collision.urdf",# 
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": JOINT_EFFORT_PF90,
                ".*_hip_roll_joint": JOINT_EFFORT_PF90,
                ".*_hip_pitch_joint": JOINT_EFFORT_PF90,
                ".*_knee_joint": JOINT_EFFORT_PF90,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint":JOINT_VEL_PF90,
                ".*_hip_roll_joint": JOINT_VEL_PF90,
                ".*_hip_pitch_joint": JOINT_VEL_PF90,
                ".*_knee_joint": JOINT_VEL_PF90,
            },
            stiffness={
                ".*_hip_pitch_joint": 120.,
                ".*_hip_roll_joint": 120.,
                ".*_hip_yaw_joint": 120.,
                ".*_knee_joint": 150.,
            },
            damping={
                ".*_hip_pitch_joint": 5.,
                ".*_hip_roll_joint": 5.,
                ".*_hip_yaw_joint": 5.,
                ".*_knee_joint": 5.,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_PF90,
                ".*_hip_roll_joint": ARMATURE_PF90,
                ".*_hip_yaw_joint": ARMATURE_PF90,
                ".*_knee_joint": ARMATURE_PF90,
            },
            # friction=0.2,         # 添加静态摩擦  
            # dynamic_friction=0.05,  # 添加动态摩擦
            # viscous_friction=0.1,  # 添加粘性摩擦 
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim={
                ".*_ankle_pitch_joint": JOINT_EFFORT_PF72,
                ".*_ankle_roll_joint": JOINT_EFFORT_PF59,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": JOINT_VEL_PF72,
                ".*_ankle_roll_joint": JOINT_VEL_PF59,
            },
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 40,
                ".*_ankle_roll_joint": 30,
            },
            damping={
                ".*_ankle_pitch_joint": 3.,
                ".*_ankle_roll_joint": 2.,
            },
            armature= {
                ".*_ankle_pitch_joint": ARMATURE_PF72,
                ".*_ankle_roll_joint": ARMATURE_PF59,
            },
            # friction=0.2,        # 添加静态摩擦  
            # dynamic_friction=0.05,  # 添加动态摩擦
            # viscous_friction=0.1,  # 添加粘性摩擦 
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=2.0 * JOINT_EFFORT_PF59,
            velocity_limit_sim= JOINT_VEL_PF59,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=80.0,
            damping=5.0,
            armature=2.0 * ARMATURE_PF59,
            # friction=0.2,         # 添加静态摩擦  
            # dynamic_friction=0.05,  # 添加动态摩擦
            # viscous_friction=0.1,  # 添加粘性摩擦 
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=JOINT_EFFORT_PF90,
            velocity_limit_sim=JOINT_VEL_PF90,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=160.0,
            damping=5.0,
            armature=ARMATURE_PF90,
            # friction=0.2,         # 添加静态摩擦  
            # dynamic_friction=0.05,  # 添加动态摩擦
            # viscous_friction=0.1,  # 添加粘性摩擦 
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": JOINT_EFFORT_PF72,
                ".*_shoulder_roll_joint": JOINT_EFFORT_PF72,
                ".*_shoulder_yaw_joint": JOINT_EFFORT_PF59,
                ".*_elbow_joint": JOINT_EFFORT_PF59,
                ".*_wrist_yaw_joint": JOINT_EFFORT_PF59,
                ".*_wrist_pitch_joint": JOINT_EFFORT_WRIST,
                ".*_wrist_roll_joint": JOINT_EFFORT_WRIST,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": JOINT_VEL_PF72,
                ".*_shoulder_roll_joint": JOINT_VEL_PF72,
                ".*_shoulder_yaw_joint": JOINT_VEL_PF59,
                ".*_elbow_joint": JOINT_VEL_PF59,
                ".*_wrist_yaw_joint": JOINT_VEL_PF59,
                ".*_wrist_pitch_joint": JOINT_VEL_WRIST,
                ".*_wrist_roll_joint": JOINT_VEL_WRIST,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40,
                ".*_shoulder_roll_joint": 40,
                ".*_shoulder_yaw_joint": 40,
                ".*_elbow_joint": 40,
                ".*_wrist_yaw_joint": 20,
                ".*_wrist_pitch_joint": 20,
                ".*_wrist_roll_joint": 20,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_joint": 1.0,
                ".*_wrist_yaw_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_PF72,
                ".*_shoulder_roll_joint": ARMATURE_PF72,
                ".*_shoulder_yaw_joint": ARMATURE_PF59,
                ".*_elbow_joint": ARMATURE_PF59,
                ".*_wrist_yaw_joint": ARMATURE_PF59,
                ".*_wrist_pitch_joint": ARMATURE_PF59,
                ".*_wrist_roll_joint": ARMATURE_PF59,
            },
            # friction=0.2,         # 添加静态摩擦  
            # dynamic_friction=0.05,  # 添加动态摩擦
            # viscous_friction=0.1,  # 添加粘性摩擦 
        ),
    },
)


X2T2D5_ACTION_SCALE = {}
for a in X2T2D5_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            X2T2D5_ACTION_SCALE[n] = 0.5
