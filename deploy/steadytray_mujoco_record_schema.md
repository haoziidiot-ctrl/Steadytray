# SteadyTray MuJoCo Record Schema

This repository keeps SteadyTray as a real-robot deployment only. For comparison, export the following JSONL records from the external MuJoCo deployment and pass the file to:

```bash
python3 /home/jerry/RoboMimic_Deploy/tools/compare_steadytray_real_mujoco.py \
  --mujoco /path/to/external_mujoco_steadytray.jsonl
```

## Required Events

Write one JSON object per line. The first useful `obs` and first useful `policy_debug` event are enough.

### `meta`

```json
{
  "event": "meta",
  "run_kind": "mujoco_external",
  "num_joints": 23,
  "config_path": "/path/to/g1_23dof_walk.yaml",
  "model_path": "/path/to/stage4_policy_14800.pt",
  "control_dt": 0.02
}
```

### `obs`

Record exactly the values used to build the policy observation at the first SteadyTray inference frame.

```json
{
  "event": "obs",
  "run_kind": "mujoco_external",
  "cmd_raw": [0.0, 0.0, 0.0],
  "cmd_scaled": [0.0, 0.0, 0.0],
  "quat": [1.0, 0.0, 0.0, 0.0],
  "gravity": [0.0, 0.0, -1.0],
  "omega": [0.0, 0.0, 0.0],
  "object_obs": [0.320389087, -0.01753, 0.044079678, 0.914959668, 0.0, -0.403545296, 0.0],
  "object_source": "mujoco",
  "object_fresh": true
}
```

### `policy_debug`

Record exactly the tensors/vectors around the first SteadyTray policy inference.

```json
{
  "event": "policy_debug",
  "run_kind": "mujoco_external",
  "q_model": [0.0],
  "dq_model": [0.0],
  "q_policy": [0.0],
  "dq_policy": [0.0],
  "action_model": [0.0],
  "action_policy": [0.0],
  "target_model": [0.0],
  "default_model": [0.0],
  "q_delta": [0.0],
  "target_delta": [0.0],
  "q_policy_abs_max": 0.0,
  "dq_policy_abs_max": 0.0,
  "action_policy_max": 0.0,
  "target_delta_max": 0.0,
  "q_delta_max": 0.0
}
```

All joint vectors above must have length `23`. Use the same 23-DoF MuJoCo/model order as SteadyTray deployment:

```text
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll
```

## Minimum Comparison Set

If you do not want to dump everything, the minimum useful fields are:

- `obs.gravity`
- `obs.omega`
- `obs.object_obs`
- `policy_debug.q_model`
- `policy_debug.dq_model`
- `policy_debug.action_model`
- `policy_debug.target_model`
- `policy_debug.default_model`

But the full schema above is preferred because it lets the comparison script identify whether the mismatch is input state, model output, or target mapping.
