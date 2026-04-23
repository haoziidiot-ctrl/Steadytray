"""Common observation processing utilities for deployment."""

import numpy as np
from collections import deque
import torch
from typing import Any, Tuple, Optional, Mapping, Union
from scripts.config import Config

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def detect_policy_type(policy: torch.jit.ScriptModule) -> str:
    """
    Detect whether the policy is standard, teacher/adapted, or distillation.

    Args:
        policy: Loaded torch.jit policy network

    Returns:
        "standard", "teacher", or "distillation"
    """
    try:
        # Method 1: Check state_dict for explicit encoder components.
        state_dict = policy.state_dict()

        # Stage 4 distillation exports use the student encoder directly.
        if "student_encoder.embed.weight" in state_dict:
            print("Detected distillation policy: found student_encoder in state_dict")
            return "distillation"

        # Stage 3 teacher/adapted exports use the history encoder.
        has_history_encoder = "history_encoder.embed.weight" in state_dict
        has_film_actor = "actor_body.0.base.weight" in state_dict
        has_residual_actor = (
            "frozen_actor.0.weight" in state_dict
            or "residual_adapter.residual_mlp.0.weight" in state_dict
        )
        if has_history_encoder and (has_film_actor or has_residual_actor):
            print("Detected teacher policy: found history_encoder with adapted actor components")
            return "teacher"

        # Method 2: inspect the exported forward signature.
        try:
            code = policy.code
            if "student_encoder_obs" in code:
                print("Detected distillation policy: found student_encoder_obs in code")
                return "distillation"
            if "encoder_obs" in code and "policy_obs" in code:
                print("Detected teacher policy: found encoder_obs + policy_obs in code")
                return "teacher"
        except Exception:
            pass

        # Method 3: inspect graph inputs.
        try:
            graph = policy.graph
            graph_str = str(graph)

            if "student_encoder_obs" in graph_str:
                print("Detected distillation policy: found student_encoder_obs in graph")
                return "distillation"
            if "encoder_obs" in graph_str and "policy_obs" in graph_str:
                print("Detected teacher policy: found encoder_obs + policy_obs in graph")
                return "teacher"

            # Dual-input exports have 3 graph inputs: self + encoder_obs + policy_obs.
            inputs = list(graph.inputs())
            if len(inputs) > 2:
                print(f"Detected teacher policy: found {len(inputs)} inputs (>2) in graph")
                return "teacher"
        except Exception:
            pass

        print("Detected standard policy: no teacher/distillation components found")
        return "standard"

    except Exception as e:
        # Default to standard if detection fails
        print(f"Policy type detection failed: {e}, defaulting to standard")
        return "standard"

def detect_encoder_obs_size(policy: torch.jit.ScriptModule) -> int:
    """
    Automatically detect the encoder observation size from a dual-input policy.

    Args:
        policy: Loaded torch.jit policy network

    Returns:
        Encoder observation dimension for either teacher/history_encoder or
        distillation/student_encoder inputs.
    """
    try:
        state_dict = policy.state_dict()

        if "student_encoder.embed.weight" in state_dict:
            embed_weight = state_dict["student_encoder.embed.weight"]
            encoder_obs_dim = int(embed_weight.shape[1])
            print(f"Auto-detected distillation encoder observation size from state_dict: {encoder_obs_dim}")
            return encoder_obs_dim

        if "history_encoder.embed.weight" in state_dict:
            embed_weight = state_dict["history_encoder.embed.weight"]
            encoder_obs_dim = int(embed_weight.shape[1])
            print(f"Auto-detected teacher encoder observation size from state_dict: {encoder_obs_dim}")
            return encoder_obs_dim

    except Exception as e:
        print(f"Could not auto-detect from state_dict: {e}")

    policy_type = detect_policy_type(policy)
    if policy_type == "standard":
        raise ValueError("Standard policies do not use encoder observations.")

    num_actions = None
    try:
        state_dict = policy.state_dict()
        if "action_head.weight" in state_dict:
            num_actions = int(state_dict["action_head.weight"].shape[0])
        else:
            frozen_actor_linear_weights = [
                value for key, value in state_dict.items()
                if key.startswith("frozen_actor.") and key.endswith(".weight") and value.ndim == 2
            ]
            if frozen_actor_linear_weights:
                num_actions = int(frozen_actor_linear_weights[-1].shape[0])
    except Exception:
        pass

    if num_actions is None:
        raise ValueError("Unable to auto-detect encoder observation size from the exported policy.")

    base_obs_dim = 9 + 3 * num_actions
    if policy_type == "distillation":
        default_size = base_obs_dim + 7
    else:
        default_size = base_obs_dim + 21

    print(f"Using fallback encoder observation size: {default_size}")
    return default_size

def compute_policy_action(
    policy: torch.jit.ScriptModule,
    frame_stack: deque,
    qj: np.ndarray,
    dqj: np.ndarray,
    quat: np.ndarray,
    omega: np.ndarray,
    cmd: np.ndarray,
    previous_action: np.ndarray,
    config: Config,
    object_obs: Optional[np.ndarray] = None,
    teacher_obs_terms: Optional[Mapping[str, np.ndarray]] = None,
    policy_type: Optional[str] = None,
    encoder_frame_stack: Optional[deque] = None,
    return_debug: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """
    Process sensor observations and compute policy action.

    This function combines observation normalization, frame stacking, policy inference,
    and action transformation. It can be used by both MuJoCo simulation and real robot deployment.
    Supports standard, teacher/adapted, and distillation policies.

    Args:
        policy: Loaded torch.jit policy network
        frame_stack: Deque with maxlen=5 containing observation history for the policy obs
        qj: Raw joint positions (robot order)
        dqj: Raw joint velocities (robot order)
        quat: Root quaternion in wxyz format
        omega: Raw angular velocity (3,)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate]
        previous_action: Previous action in robot order (num_actions,)
        config: Configuration object with necessary parameters
        object_obs: Optional object observations for distillation policies (7,)
                   Format: [pos(3), quat(4)] = position + quaternion
        teacher_obs_terms: Optional teacher observation terms for stage-3 adapted policies.
        policy_type: Optional policy type ("standard", "teacher", or "distillation"). If None, auto-detect.
        encoder_frame_stack: Optional deque with maxlen=32 for encoder observation history.
                           Required for teacher and distillation policies.

    Returns:
        action_robot: Action in robot order (num_actions,)
        target_dof_pos: Target joint positions in robot order (num_actions,)
        policy_debug: Optional debug payload when return_debug=True.
    """

    if policy_type is None:
        policy_type = detect_policy_type(policy)

    default_angles = config.default_angles[config.policy_to_robot]

    # Normalize observations (vectorized operations)
    qj_normalized = (qj - default_angles) * config.dof_pos_scale
    dqj_normalized = dqj * config.dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    omega_normalized = omega * config.ang_vel_scale

    # Process observation with frame stacking (group-major order)
    big_group_major = process_observation_with_frame_stack(
        frame_stack=frame_stack,
        omega=omega_normalized,
        gravity_orientation=gravity_orientation,
        cmd=cmd,
        qj=qj_normalized[config.robot_to_policy],
        dqj=dqj_normalized[config.robot_to_policy],
        action=previous_action[config.robot_to_policy],
        cmd_scale=config.cmd_scale,
        num_obs=config.num_obs,
        num_actions=config.num_actions,
    )

    with torch.no_grad():
        if policy_type in {"teacher", "distillation"}:
            assert encoder_frame_stack is not None, (
                "encoder_frame_stack must be provided for teacher/distillation policies"
            )

            if policy_type == "distillation":
                current_encoder_obs = build_student_encoder_obs(
                    omega=omega_normalized,
                    gravity_orientation=gravity_orientation,
                    cmd=cmd,
                    qj=qj_normalized[config.robot_to_policy],
                    dqj=dqj_normalized[config.robot_to_policy],
                    action=previous_action[config.robot_to_policy],
                    cmd_scale=config.cmd_scale,
                    num_actions=config.num_actions,
                    object_obs=object_obs,
                )
            else:
                assert teacher_obs_terms is not None, (
                    "teacher_obs_terms must be provided for teacher policies"
                )
                current_encoder_obs = build_teacher_encoder_obs(
                    omega=omega_normalized,
                    gravity_orientation=gravity_orientation,
                    cmd=cmd,
                    qj=qj_normalized[config.robot_to_policy],
                    dqj=dqj_normalized[config.robot_to_policy],
                    action=previous_action[config.robot_to_policy],
                    cmd_scale=config.cmd_scale,
                    num_actions=config.num_actions,
                    base_lin_vel=teacher_obs_terms["base_lin_vel"],
                    tray_projected_gravity=teacher_obs_terms["tray_projected_gravity"],
                    tray_pos_rel=teacher_obs_terms["tray_pos_rel"],
                    object_pos_rel=teacher_obs_terms["object_pos_rel"],
                    object_ang_vel_rel=teacher_obs_terms["object_ang_vel_rel"],
                    object_lin_vel_rel=teacher_obs_terms["object_lin_vel_rel"],
                    object_projected_gravity=teacher_obs_terms["object_projected_gravity"],
                )

            # Match IsaacLab history reset behavior: on the first push, fill the whole
            # sequence with the current observation instead of zero-padding.
            if len(encoder_frame_stack) == 0:
                history_len = encoder_frame_stack.maxlen or 1
                for _ in range(history_len):
                    encoder_frame_stack.append(current_encoder_obs.copy())
            else:
                encoder_frame_stack.append(current_encoder_obs)

            # Convert encoder frame stack to tensor with shape [seq_len, batch, obs_dim].
            encoder_obs_array = np.array(encoder_frame_stack, dtype=np.float32)
            encoder_obs_tensor = torch.from_numpy(encoder_obs_array).unsqueeze(1)

            # Convert policy observations to tensor [batch, policy_obs_dim].
            policy_obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)
            action_policy = policy(encoder_obs_tensor, policy_obs_tensor).squeeze(0).numpy()
        else:
            obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)
            action_policy = policy(obs_tensor).squeeze(0).numpy()

    action_policy = np.asarray(action_policy, dtype=np.float32)
    action_robot = action_policy[config.policy_to_robot]
    target_dof_pos = action_robot * config.action_scale[config.policy_to_robot] + default_angles

    if not return_debug:
        return action_robot, target_dof_pos

    q_model = np.asarray(qj, dtype=np.float32).copy()
    dq_model = np.asarray(dqj, dtype=np.float32).copy()
    action_model = np.asarray(action_robot, dtype=np.float32).copy()
    target_model = np.asarray(target_dof_pos, dtype=np.float32).copy()
    default_model = np.asarray(default_angles, dtype=np.float32).copy()
    q_policy = q_model[config.robot_to_policy]
    dq_policy = dq_model[config.robot_to_policy]
    action_policy_debug = action_model[config.robot_to_policy]
    q_delta = q_model - default_model
    target_delta = target_model - default_model

    def max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values))) if values.size else 0.0

    policy_debug = {
        "q_model": q_model,
        "dq_model": dq_model,
        "q_policy": q_policy,
        "dq_policy": dq_policy,
        "action_model": action_model,
        "action_policy": action_policy_debug,
        "target_model": target_model,
        "default_model": default_model,
        "q_delta": q_delta,
        "target_delta": target_delta,
        "q_policy_abs_max": max_abs(q_policy),
        "dq_policy_abs_max": max_abs(dq_policy),
        "action_policy_max": max_abs(action_policy_debug),
        "target_delta_max": max_abs(target_delta),
        "q_delta_max": max_abs(q_delta),
    }
    return action_robot, target_dof_pos, policy_debug

def build_student_encoder_obs(
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_actions: int,
    object_obs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build student encoder observations for a single timestep.
    
    For distillation policies, the student encoder receives observations that match
    the StudentEncoderCfg structure:
    - base_ang_vel (3)
    - projected_gravity (3)
    - velocity_commands (3)
    - joint_pos_rel (29)
    - joint_vel_rel (29)
    - last_action (29)
    - object_pos_cam (3) - object position
    - object_quat_cam (4) - object quaternion
    
    Total: 96 + 7 = 103 dimensions per timestep.
    
    Args:
        omega: Angular velocity (3,) - normalized (corresponds to base_ang_vel)
        gravity_orientation: Gravity vector in body frame (3,) (corresponds to projected_gravity)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate] (corresponds to velocity_commands)
        qj: Joint positions (num_actions,) - normalized, policy order (corresponds to joint_pos_rel)
        dqj: Joint velocities (num_actions,) - normalized, policy order (corresponds to joint_vel_rel)
        action: Previous action (num_actions,) - policy order (corresponds to last_action)
        cmd_scale: Scale factors for commands (3,)
        num_actions: Number of actions (29 for G1)
        object_obs: Object observations (7,)
                   Format: [pos(3), quat(4)] = position + quaternion
                   Must be provided for distillation policies.
        
    Returns:
        student_encoder_obs: Student encoder observations for one timestep
                            Shape: (103,) = 96 base + 7 object
    """
    # Base proprioceptive observations: omega(3) + gravity(3) + cmd(3) + qj(29) + dqj(29) + action(29) = 96
    base_obs_dim = 3 + 3 + 3 + num_actions + num_actions + num_actions
    
    # Determine object observation size
    if object_obs is not None and len(object_obs) > 0:
        object_obs_size = len(object_obs)
        # Ensure we have at least position (3 dims)
        if object_obs_size < 3:
            print(f"Warning: object_obs size {object_obs_size} < 3, padding with zeros")
            object_data = np.zeros(6, dtype=np.float32)  # Default to 6 dims
            object_data[:object_obs_size] = object_obs
            object_obs_size = 6
        else:
            object_data = object_obs.astype(np.float32)
    else:
        raise ValueError("object_obs must be provided for distillation policies")
    
    # Total dimension per timestep: base_obs + object_obs_size
    total_dim = base_obs_dim + object_obs_size
    student_obs = np.zeros(total_dim, dtype=np.float32)
    
    # Fill in observations following StudentEncoderCfg order
    idx = 0
    
    # base_ang_vel (3)
    student_obs[idx:idx+3] = omega
    idx += 3
    
    # projected_gravity (3)
    student_obs[idx:idx+3] = gravity_orientation
    idx += 3
    
    # velocity_commands (3)
    student_obs[idx:idx+3] = cmd * cmd_scale
    idx += 3
    
    # joint_pos_rel (29)
    student_obs[idx:idx+num_actions] = qj
    idx += num_actions
    
    # joint_vel_rel (29)
    student_obs[idx:idx+num_actions] = dqj
    idx += num_actions
    
    # last_action (29)
    student_obs[idx:idx+num_actions] = action
    idx += num_actions
    
    # object_pos_cam (first 3 dims) + object_quat_cam (remaining 4 dims)
    student_obs[idx:idx+3] = object_data[:3]
    idx += 3
    
    student_obs[idx:idx+4] = object_data[3:7]
    idx += 4
    
    return student_obs


def build_teacher_encoder_obs(
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_actions: int,
    base_lin_vel: np.ndarray,
    tray_projected_gravity: np.ndarray,
    tray_pos_rel: np.ndarray,
    object_pos_rel: np.ndarray,
    object_ang_vel_rel: np.ndarray,
    object_lin_vel_rel: np.ndarray,
    object_projected_gravity: np.ndarray,
) -> np.ndarray:
    """
    Build stage-3 teacher encoder observations for a single timestep.

    Order matches `ObjectObservationsCfg.EncoderCfg`:
    - base_ang_vel (3)
    - projected_gravity (3)
    - velocity_commands (3)
    - joint_pos_rel (num_actions)
    - joint_vel_rel (num_actions)
    - last_action (num_actions)
    - base_lin_vel (3)
    - tray_projected_gravity (3)
    - tray_pos_rel (3)
    - object_pos_rel (3)
    - object_ang_vel_rel (3), scale=0.2
    - object_lin_vel_rel (3), scale=0.5
    - object_projected_gravity (3)
    """
    teacher_obs = np.zeros(9 + 3 * num_actions + 21, dtype=np.float32)

    idx = 0
    teacher_obs[idx:idx+3] = omega
    idx += 3

    teacher_obs[idx:idx+3] = gravity_orientation
    idx += 3

    teacher_obs[idx:idx+3] = cmd * cmd_scale
    idx += 3

    teacher_obs[idx:idx+num_actions] = qj
    idx += num_actions

    teacher_obs[idx:idx+num_actions] = dqj
    idx += num_actions

    teacher_obs[idx:idx+num_actions] = action
    idx += num_actions

    teacher_obs[idx:idx+3] = np.clip(np.asarray(base_lin_vel, dtype=np.float32), -3.0, 3.0)
    idx += 3

    teacher_obs[idx:idx+3] = np.asarray(tray_projected_gravity, dtype=np.float32)
    idx += 3

    teacher_obs[idx:idx+3] = np.clip(np.asarray(tray_pos_rel, dtype=np.float32), -1.0, 1.0)
    idx += 3

    teacher_obs[idx:idx+3] = np.clip(np.asarray(object_pos_rel, dtype=np.float32), -1.0, 1.0)
    idx += 3

    teacher_obs[idx:idx+3] = np.clip(np.asarray(object_ang_vel_rel, dtype=np.float32), -50.0, 50.0) * 0.2
    idx += 3

    teacher_obs[idx:idx+3] = np.clip(np.asarray(object_lin_vel_rel, dtype=np.float32), -10.0, 10.0) * 0.5
    idx += 3

    teacher_obs[idx:idx+3] = np.asarray(object_projected_gravity, dtype=np.float32)

    return teacher_obs


def process_observation_with_frame_stack(
    frame_stack: deque,
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_obs: int,
    num_actions: int
) -> np.ndarray:
    """
    Process observations with frame stacking and group-major ordering.
    
    This function builds a single frame observation, adds it to the frame stack,
    and then restructures all frames into group-major order (all omega across frames,
    then all gravity, etc.) as expected by the policy network.
    
    Args:
        frame_stack: Deque with maxlen=5 containing observation history
        omega: Angular velocity (3,)
        gravity_orientation: Gravity vector in body frame (3,)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate]
        qj: Joint positions (num_actions,) - already normalized/scaled
        dqj: Joint velocities (num_actions,) - already normalized/scaled
        action: Previous action (num_actions,)
        cmd_scale: Scale factors for commands (3,)
        num_obs: Observation dimension per frame (typically 96)
        num_actions: Number of actions (typically 29)
        
    Returns:
        big_group_major: Stacked observations in group-major order (5 * num_obs,)
                        Total dimensions: 5 frames × num_obs = 480 for 29-DOF
    """
    # Create temporary observation buffer
    obs = np.zeros(num_obs, dtype=np.float32)
    
    # Build single frame observation (num_obs dimensions: 3+3+3+29+29+29 = 96)
    # Use slicing for faster assignment
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd * cmd_scale
    obs[9 : 9 + num_actions] = qj
    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
    
    # Match IsaacLab history reset behavior: the first real observation fills the
    # entire history buffer instead of mixing with zero-initialized frames.
    if len(frame_stack) == 0:
        history_len = frame_stack.maxlen or 1
        for _ in range(history_len):
            frame_stack.append(obs.copy())
    else:
        frame_stack.append(obs)
    
    # Convert deque to array once (faster than multiple operations)
    stacked_obs = np.array(frame_stack, dtype=np.float32)  # Shape: (5, num_obs)
    
    # Extract features using direct slicing (much faster than reshape chains)
    # Shape is already (5, num_obs), so we can directly slice columns
    obs_omega = stacked_obs[:, :3].ravel()  # ravel() is faster than reshape(-1)
    obs_gravity_orientation = stacked_obs[:, 3:6].ravel()
    obs_cmd = stacked_obs[:, 6:9].ravel()
    obs_pos = stacked_obs[:, 9:9 + num_actions].ravel()
    obs_vel = stacked_obs[:, 9 + num_actions : 9 + 2 * num_actions].ravel()
    obs_action = stacked_obs[:, 9 + 2 * num_actions : 9 + 3 * num_actions].ravel()
    
    # Concatenate all features in group-major order
    # Pre-allocate output array for better performance
    total_size = 3*5 + 3*5 + 3*5 + num_actions*5*3  # 15+15+15+435 = 480
    big_group_major = np.empty(total_size, dtype=np.float32)
    
    # Use direct assignment instead of concatenate (faster)
    idx = 0
    big_group_major[idx:idx+15] = obs_omega
    idx += 15
    big_group_major[idx:idx+15] = obs_gravity_orientation
    idx += 15
    big_group_major[idx:idx+15] = obs_cmd
    idx += 15
    big_group_major[idx:idx+num_actions*5] = obs_pos
    idx += num_actions*5
    big_group_major[idx:idx+num_actions*5] = obs_vel
    idx += num_actions*5
    big_group_major[idx:idx+num_actions*5] = obs_action
    
    return big_group_major
