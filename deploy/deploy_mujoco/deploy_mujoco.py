import time
import argparse
import array
import errno
import fcntl
import glob
import json
import os
import struct
import sys
import mujoco.viewer
import mujoco
import numpy as np
import torch
from collections import deque
from typing import Optional

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add parent directory to path for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.policy_runner import (
    compute_policy_action,
    detect_policy_type,
    detect_encoder_obs_size,
    get_gravity_orientation,
)
from scripts.config import Config
from scipy.spatial.transform import Rotation

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    InputDevice = None
    ecodes = None
    list_devices = None


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_object_pose(data, model, object_half_height=0.05):
    """
    Get object pose in camera frame from MuJoCo simulation.
    
    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        object_half_height: Half of the object height in meters (default: 0.05m = 5cm for half height)
                           This is added to get the top surface position instead of center
    
    Returns:
        object_obs: Position + quaternion observation array (7,)
    """
    # Get camera pose in world frame
    cam_site_id = model.site("d435_camera_frame").id
    cam_pos_world = data.site_xpos[cam_site_id]
    cam_rot_world = data.site_xmat[cam_site_id].reshape(3, 3)

    # Get object pose in world frame (center of object)
    object_body_id = model.body("object").id
    object_pos_world = data.xpos[object_body_id]
    object_rot_world = data.xmat[object_body_id].reshape(3, 3)
    
    # Add half height to get top surface position
    # The offset is in the object's local frame (z-axis points up in object frame)
    top_surface_offset_local = np.array([0, 0, object_half_height], dtype=np.float32)
    top_surface_offset_world = object_rot_world @ top_surface_offset_local
    object_top_pos_world = object_pos_world + top_surface_offset_world

    # Build transformation matrices (using top surface position)
    object_world_transform = np.eye(4)
    object_world_transform[:3, :3] = object_rot_world
    object_world_transform[:3, 3] = object_top_pos_world  # Use top surface position

    camera_world_transform = np.eye(4)
    camera_world_transform[:3, :3] = cam_rot_world
    camera_world_transform[:3, 3] = cam_pos_world

    # Transform object pose to camera frame
    object_camera_transform = np.linalg.inv(camera_world_transform) @ object_world_transform
    object_camera_pos = object_camera_transform[:3, 3]  # This is now the top surface position
    object_camera_rotation = Rotation.from_matrix(object_camera_transform[:3, :3])
    
    # Convert to wxyz quaternion format
    object_camera_quat_xyzw = object_camera_rotation.as_quat()
    object_camera_quat = np.array([object_camera_quat_xyzw[3], object_camera_quat_xyzw[0], 
                                   object_camera_quat_xyzw[1], object_camera_quat_xyzw[2]], dtype=np.float32)

    # Position + quaternion (3+4)
    object_obs = np.concatenate([object_camera_pos, object_camera_quat], axis=0).astype(np.float32)
    
    return object_obs


def format_object_obs_lines(object_obs):
    """Return display lines for the exact 7D object observation."""
    if object_obs is None:
        return ["object_obs policy input", "not available"]

    return [
        "object_obs policy input",
        "x   {:+.4f}".format(object_obs[0]),
        "y   {:+.4f}".format(object_obs[1]),
        "z   {:+.4f}".format(object_obs[2]),
        "qw  {:+.4f}".format(object_obs[3]),
        "qx  {:+.4f}".format(object_obs[4]),
        "qy  {:+.4f}".format(object_obs[5]),
        "qz  {:+.4f}".format(object_obs[6]),
    ]


def update_object_obs_overlay(viewer, object_obs, data, model):
    """Show object_obs in the MuJoCo viewer, compatible with old and new viewers."""
    lines = format_object_obs_lines(object_obs)

    if hasattr(viewer, "set_texts"):
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            lines[0],
            "\n".join(lines[1:]),
        ))
        return

    user_scn = getattr(viewer, "user_scn", None)
    if user_scn is None:
        return

    user_scn.ngeom = 0
    try:
        object_body_id = model.body("object").id
        label_base_pos = np.array(data.xpos[object_body_id], dtype=np.float32)
        label_base_pos = label_base_pos + np.array([0.0, 0.0, 0.35], dtype=np.float32)
    except Exception:
        label_base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mat = np.eye(3, dtype=np.float32).reshape(-1)
    rgba = np.array([1.0, 0.9, 0.1, 1.0], dtype=np.float32)
    size = np.array([0.02, 0.02, 0.02], dtype=np.float32)

    for idx, line in enumerate(lines):
        if user_scn.ngeom >= len(user_scn.geoms):
            break

        geom = user_scn.geoms[user_scn.ngeom]
        pos = label_base_pos + np.array([0.0, 0.0, -0.035 * idx], dtype=np.float32)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=size,
            pos=pos,
            mat=mat,
            rgba=rgba,
        )
        geom.label = line
        user_scn.ngeom += 1


def resolve_body_name(model: mujoco.MjModel, *candidates: str) -> str:
    """Return the first body name present in the MuJoCo model."""
    for name in candidates:
        try:
            model.body(name)
            return name
        except Exception:
            continue
    raise KeyError(f"None of the candidate bodies exist in the model: {candidates}")


def get_body_pose(data: mujoco.MjData, model: mujoco.MjModel, body_name: str):
    """Get body position, rotation matrix, and quaternion in world frame."""
    body_id = model.body(body_name).id
    pos = np.array(data.xpos[body_id], dtype=np.float32)
    rot = np.array(data.xmat[body_id], dtype=np.float32).reshape(3, 3)
    quat = np.array(data.xquat[body_id], dtype=np.float32)
    return pos, rot, quat


def get_body_velocity(data: mujoco.MjData, model: mujoco.MjModel, body_name: str, local_frame: bool = False):
    """Get body angular and linear velocity from MuJoCo."""
    body_id = model.body(body_name).id
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, int(local_frame))
    ang_vel = velocity[:3].astype(np.float32)
    lin_vel = velocity[3:].astype(np.float32)
    return ang_vel, lin_vel


def get_relative_position(data: mujoco.MjData, model: mujoco.MjModel, reference_body_name: str, target_body_name: str) -> np.ndarray:
    """Get target body position expressed in the reference body frame."""
    ref_pos, ref_rot, _ = get_body_pose(data, model, reference_body_name)
    target_pos, _, _ = get_body_pose(data, model, target_body_name)
    relative_pos = ref_rot.T @ (target_pos - ref_pos)
    return relative_pos.astype(np.float32)


def get_body_projected_gravity(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> np.ndarray:
    """Get projected gravity in the body-local frame."""
    _, _, quat = get_body_pose(data, model, body_name)
    return get_gravity_orientation(quat).astype(np.float32)


def get_teacher_observation_terms(data: mujoco.MjData, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Build the extra per-timestep teacher observations from MuJoCo state."""
    base_body_name = resolve_body_name(model, "pelvis", "base", "torso_link")
    torso_body_name = resolve_body_name(model, "torso_link")
    tray_body_name = resolve_body_name(model, "tray", "plate")
    object_body_name = resolve_body_name(model, "object")

    _, base_lin_vel = get_body_velocity(data, model, base_body_name, local_frame=True)

    torso_ang_vel_world, torso_lin_vel_world = get_body_velocity(data, model, torso_body_name, local_frame=False)
    object_ang_vel_world, object_lin_vel_world = get_body_velocity(data, model, object_body_name, local_frame=False)

    teacher_obs_terms = {
        "base_lin_vel": base_lin_vel,
        "tray_projected_gravity": get_body_projected_gravity(data, model, tray_body_name),
        "tray_pos_rel": get_relative_position(data, model, torso_body_name, tray_body_name),
        "object_pos_rel": get_relative_position(data, model, tray_body_name, object_body_name),
        "object_ang_vel_rel": (object_ang_vel_world - torso_ang_vel_world).astype(np.float32),
        "object_lin_vel_rel": (object_lin_vel_world - torso_lin_vel_world).astype(np.float32),
        "object_projected_gravity": get_body_projected_gravity(data, model, object_body_name),
    }
    return teacher_obs_terms


def _scale_stick_value(value: float, cmd_range) -> float:
    """Map normalized stick value [-1, 1] to an asymmetric command range [min, max]."""
    lower, upper = float(cmd_range[0]), float(cmd_range[1])
    if value >= 0.0:
        return value * upper
    return value * abs(lower)


class JoystickCommandInput:
    """Read joystick commands from Linux joystick backends.

    Priority:
    1. `/dev/input/js*` joystick API, which tends to work well for Xbox controllers.
    2. `evdev` event devices as a fallback.
    """

    LEFT_X = "left_x"
    LEFT_Y = "left_y"
    RIGHT_X = "right_x"

    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80
    JSIOCGAXES = 0x80016A11
    JSIOCGAXMAP = 0x80406A32
    JSIOCGNAME_128 = 0x80806A13

    def __init__(self, config: Config, device_path: Optional[str] = None, deadzone: float = 0.1):
        self.config = config
        self.deadzone = max(0.0, min(float(deadzone), 0.95))
        self.device = None
        self.device_name = ""
        self.backend = None
        self.axis_state = {
            self.LEFT_X: 0.0,
            self.LEFT_Y: 0.0,
            self.RIGHT_X: 0.0,
        }
        self.abs_info = {}
        self.axis_codes = {}
        self.axis_indices = {}
        self.available = False

        selected_path, backend = self._select_backend(device_path)
        if selected_path is None or backend is None:
            print("No joystick/gamepad device found, falling back to zero command.")
            return

        if backend == "js":
            self._open_js_device(selected_path)
        else:
            self._open_evdev_device(selected_path)

    def _select_backend(self, device_path: Optional[str]):
        if device_path:
            if "/js" in device_path:
                return device_path, "js"
            return device_path, "evdev"

        js_device = self._find_js_device()
        if js_device is not None:
            return js_device, "js"

        evdev_device = self._find_evdev_device()
        if evdev_device is not None:
            return evdev_device, "evdev"

        return None, None

    def _score_device_name(self, name: str) -> int:
        score = 0
        lowered = name.lower()
        for keyword in ("xbox", "controller", "gamepad", "joystick", "wireless"):
            if keyword in lowered:
                score += 1
        return score

    def _find_js_device(self) -> Optional[str]:
        candidates = []
        for path in glob.glob("/dev/input/js*"):
            name = self._read_js_device_name(path)
            candidates.append((self._score_device_name(name), path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _read_js_device_name(self, path: str) -> str:
        try:
            with open(path, "rb", buffering=0) as device:
                name_buf = array.array("B", [0] * 128)
                fcntl.ioctl(device, self.JSIOCGNAME_128, name_buf)
                return bytes(name_buf).split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        except OSError:
            return os.path.basename(path)

    def _open_js_device(self, path: str) -> None:
        try:
            self.device = open(path, "rb", buffering=0)
            os.set_blocking(self.device.fileno(), False)
            axis_count_buf = array.array("B", [0])
            fcntl.ioctl(self.device, self.JSIOCGAXES, axis_count_buf)
            axis_count = int(axis_count_buf[0])
            axis_map_buf = array.array("B", [0] * 64)
            fcntl.ioctl(self.device, self.JSIOCGAXMAP, axis_map_buf)
            axis_map = list(axis_map_buf[:axis_count])
            self.axis_indices = self._resolve_js_axis_indices(axis_map)
            self.available = all(name in self.axis_indices for name in (self.LEFT_X, self.LEFT_Y, self.RIGHT_X))
            if not self.available:
                print(f"Joystick device '{path}' is missing required js axes, falling back to zero command.")
                self.close()
                return
            self.backend = "js"
            self.device_name = self._read_js_device_name(path)
            print(f"Using joystick device: {path} ({self.device_name}) via js backend")
        except OSError as exc:
            print(f"Failed to open joystick device '{path}': {exc}. Falling back to zero command.")
            self.close()

    def _resolve_js_axis_indices(self, axis_map):
        def first_index(*axis_codes):
            for idx, axis_code in enumerate(axis_map):
                if axis_code in axis_codes:
                    return idx
            return None

        result = {
            self.LEFT_X: first_index(0x00),               # ABS_X
            self.LEFT_Y: first_index(0x01),               # ABS_Y
            self.RIGHT_X: first_index(0x03),              # ABS_RX (right stick horizontal)
        }
        return {k: v for k, v in result.items() if v is not None}

    def _find_evdev_device(self) -> Optional[str]:
        if InputDevice is None or ecodes is None or list_devices is None:
            return None
        candidates = []
        for path in list_devices():
            try:
                device = InputDevice(path)
                caps = device.capabilities()
                if ecodes.EV_ABS not in caps:
                    continue
                candidates.append((self._score_device_name(device.name), path))
            except OSError:
                continue
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _open_evdev_device(self, path: str) -> None:
        if InputDevice is None or ecodes is None:
            return
        try:
            self.device = InputDevice(path)
            self.device_name = self.device.name
            self.abs_info = dict(self.device.capabilities(absinfo=True).get(ecodes.EV_ABS, []))
            self.axis_codes = self._resolve_evdev_axis_codes()
            self.available = all(name in self.axis_codes for name in (self.LEFT_X, self.LEFT_Y, self.RIGHT_X))
            if not self.available:
                print(f"Joystick device '{self.device_name}' is missing required evdev axes, falling back to zero command.")
                self.close()
                return
            self.backend = "evdev"
            print(f"Using joystick device: {self.device.path} ({self.device_name}) via evdev backend")
        except OSError as exc:
            print(f"Failed to open evdev joystick '{path}': {exc}. Falling back to zero command.")
            self.close()

    def _resolve_evdev_axis_codes(self):
        available_codes = set(self.abs_info.keys())

        def first_supported(*codes):
            for code in codes:
                if code in available_codes:
                    return code
            return None

        result = {
            self.LEFT_X: first_supported(ecodes.ABS_X),
            self.LEFT_Y: first_supported(ecodes.ABS_Y),
            self.RIGHT_X: first_supported(ecodes.ABS_RX),
        }
        return {k: v for k, v in result.items() if v is not None}

    def _apply_deadzone(self, normalized: float) -> float:
        normalized = float(np.clip(normalized, -1.0, 1.0))
        if abs(normalized) < self.deadzone:
            return 0.0
        sign = 1.0 if normalized >= 0.0 else -1.0
        return sign * ((abs(normalized) - self.deadzone) / (1.0 - self.deadzone))

    def _normalize_js_axis(self, raw_value: int) -> float:
        return self._apply_deadzone(raw_value / 32767.0)

    def _normalize_evdev_axis(self, code: int, raw_value: int) -> float:
        info = self.abs_info.get(code)
        if info is None:
            return 0.0
        span = max(info.max - info.min, 1)
        center = 0.5 * (info.max + info.min)
        return self._apply_deadzone((raw_value - center) / (0.5 * span))

    def poll(self) -> np.ndarray:
        if not self.available or self.device is None:
            return get_idle_command(self.config)

        if self.backend == "js":
            return self._poll_js()
        return self._poll_evdev()

    def _poll_js(self) -> np.ndarray:
        while True:
            try:
                event = self.device.read(8)
                if event is None or len(event) < 8:
                    break
                _, value, event_type, number = struct.unpack("IhBB", event)
                event_type &= ~self.JS_EVENT_INIT
                if event_type != self.JS_EVENT_AXIS:
                    continue
                for axis_name, axis_index in self.axis_indices.items():
                    if number == axis_index:
                        self.axis_state[axis_name] = self._normalize_js_axis(value)
                        break
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                print(f"Joystick read failed: {exc}. Reverting to zero command.")
                self.available = False
                return get_idle_command(self.config)

        return self._build_command()

    def _poll_evdev(self) -> np.ndarray:
        try:
            for event in self.device.read():
                if event.type != ecodes.EV_ABS:
                    continue
                for axis_name, axis_code in self.axis_codes.items():
                    if event.code == axis_code:
                        self.axis_state[axis_name] = self._normalize_evdev_axis(axis_code, event.value)
                        break
        except BlockingIOError:
            pass
        except OSError as exc:
            print(f"Joystick read failed: {exc}. Reverting to zero command.")
            self.available = False
            return get_idle_command(self.config)

        return self._build_command()

    def _build_command(self) -> np.ndarray:
        cmd = get_idle_command(self.config)
        cmd[0] = _scale_stick_value(-self.axis_state[self.LEFT_Y], self.config.vel_x_cmd)
        cmd[1] = -_scale_stick_value(self.axis_state[self.LEFT_X], self.config.vel_y_cmd)
        cmd[2] = -_scale_stick_value(self.axis_state[self.RIGHT_X], self.config.yaw_cmd)
        return cmd

    def get_axis_state(self) -> dict[str, float]:
        """Return the latest normalized joystick axis values."""
        return {name: float(value) for name, value in self.axis_state.items()}

    def get_raw_command(self) -> np.ndarray:
        """Return normalized joystick axes in the command sign convention."""
        return np.array(
            [
                -self.axis_state[self.LEFT_Y],
                -self.axis_state[self.LEFT_X],
                -self.axis_state[self.RIGHT_X],
            ],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None


def get_idle_command(config: Config) -> np.ndarray:
    """Zero command within the configured command space."""
    return np.zeros(3, dtype=np.float32)


def get_record_raw_command(joystick: Optional[JoystickCommandInput], current_cmd: np.ndarray) -> np.ndarray:
    """Return the best raw command representation for debug recording."""
    if joystick is not None and joystick.available:
        return joystick.get_raw_command()
    return np.asarray(current_cmd, dtype=np.float32)


class SteadyTrayJsonlRecorder:
    """Write the first MuJoCo SteadyTray inference frame in the comparison schema."""

    def __init__(self, path: str, config_path: str, model_path: str, config: Config):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.obs_written = False
        self.policy_debug_written = False
        parent_dir = os.path.dirname(self.path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8"):
            pass
        self.write_event(
            "meta",
            {
                "run_kind": "mujoco_external",
                "num_joints": int(config.num_actions),
                "config_path": os.path.abspath(os.path.expanduser(config_path)),
                "model_path": os.path.abspath(os.path.expanduser(model_path)),
                "control_dt": float(config.control_dt),
            },
        )

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [SteadyTrayJsonlRecorder._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: SteadyTrayJsonlRecorder._to_jsonable(item) for key, item in value.items()}
        return value

    def write_event(self, event: str, payload: dict) -> None:
        record = {
            "time": time.time(),
            "event": event,
            "run_kind": "mujoco_external",
        }
        record.update(self._to_jsonable(payload))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def write_obs_once(
        self,
        cmd_raw: np.ndarray,
        cmd_scaled: np.ndarray,
        quat: np.ndarray,
        gravity: np.ndarray,
        omega: np.ndarray,
        object_obs: Optional[np.ndarray],
    ) -> bool:
        if self.obs_written:
            return False
        self.obs_written = True
        self.write_event(
            "obs",
            {
                "cmd_raw": np.asarray(cmd_raw, dtype=np.float32),
                "cmd_scaled": np.asarray(cmd_scaled, dtype=np.float32),
                "quat": np.asarray(quat, dtype=np.float32),
                "gravity": np.asarray(gravity, dtype=np.float32),
                "omega": np.asarray(omega, dtype=np.float32),
                "object_obs": (
                    np.asarray(object_obs, dtype=np.float32)
                    if object_obs is not None
                    else np.zeros(0, dtype=np.float32)
                ),
                "object_source": "mujoco" if object_obs is not None else "none",
                "object_fresh": object_obs is not None,
            },
        )
        return True

    def write_policy_debug_once(self, policy_debug: dict) -> bool:
        if self.policy_debug_written:
            return False
        self.policy_debug_written = True
        self.write_event("policy_debug", policy_debug)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='g1 deploy mujoco')
    parser.add_argument('--policy', type=str, default="exported/policy_9999.pt",
                       help='Direct path to policy file')
    parser.add_argument('--config', type=str, default="deploy/configs/g1_29dof_walk.yaml",
                       help='Direct path to config file (overrides default config)')
    parser.add_argument('--encoder_seq_len', type=int, default=32,
                       help='Encoder sequence length (number of history frames for teacher/distillation policies)')
    parser.add_argument('--joystick_device', type=str, default=None,
                       help='Optional joystick device path, e.g. /dev/input/js0 or /dev/input/eventX')
    parser.add_argument('--joystick_deadzone', type=float, default=0.2,
                       help='Deadzone for joystick axes in normalized units')
    parser.add_argument('--disable_joystick', action='store_true',
                       help='Use fixed cmd_init instead of joystick input')
    parser.add_argument('--print_cmd', action='store_true',
                       help='Print live velocity command and joystick axis values')
    parser.add_argument('--print_cmd_hz', type=float, default=10.0,
                       help='Maximum terminal refresh rate for --print_cmd')
    parser.add_argument('--print_object_obs', action='store_true',
                       help='Print live 7D object_obs used by the policy')
    parser.add_argument('--print_object_obs_hz', type=float, default=5.0,
                       help='Maximum terminal refresh rate for --print_object_obs')
    parser.add_argument('--no_object_obs_overlay', action='store_true',
                       help='Disable the MuJoCo viewer overlay for the 7D object_obs policy input')
    parser.add_argument('--record_jsonl', type=str,
                       default=os.path.join("logs", "steadytray_debug", "latest_mujoco_23dof.jsonl"),
                       help='Path for the first-inference SteadyTray JSONL debug record')
    parser.add_argument('--no_record_jsonl', action='store_true',
                       help='Disable SteadyTray JSONL debug recording')

    args = parser.parse_args()

    # Load configuration using shared Config class
    config = Config(args.config)
    
    # Get policy path
    if args.policy is not None:
        policy_path = args.policy
    else:
        raise ValueError("Policy path must be provided via command line argument --policy")

    if os.path.exists(policy_path):
        print(f"Using policy: {policy_path}")
    else:
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    # define context variables
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs = np.zeros(config.num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(config.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config.simulation_dt

    default_angles = config.default_angles[config.policy_to_robot]

    # Align MuJoCo startup with the training reset pose instead of the XML zero pose.
    d.qpos[7:7 + config.num_actions] = default_angles
    d.qvel[6:6 + config.num_actions] = 0.0
    mujoco.mj_forward(m, d)

    target_dof_pos = default_angles.copy()

    frame_stack = deque(maxlen=5)


    # Load policy
    policy = torch.jit.load(policy_path)
    policy_type = detect_policy_type(policy)
    print(f"Policy type: {policy_type}")

    # Auto-detect encoder observation size for dual-input policies
    encoder_obs_dim = None

    if policy_type in {'teacher', 'distillation'}:
        encoder_obs_dim = detect_encoder_obs_size(policy)
        print(f"Encoder observation size: {encoder_obs_dim}")

    if policy_type == "distillation":
        print("Using MuJoCo ground-truth camera/object observations for distillation policy")
    elif policy_type == "teacher":
        print("Using MuJoCo ground-truth tray/object observations for teacher policy")

    recorder = None
    if not args.no_record_jsonl:
        recorder = SteadyTrayJsonlRecorder(args.record_jsonl, args.config, policy_path, config)
        print(f"Recording first SteadyTray inference JSONL to: {recorder.path}")

    joystick = None if args.disable_joystick else JoystickCommandInput(
        config=config,
        device_path=args.joystick_device,
        deadzone=args.joystick_deadzone,
    )
    if args.disable_joystick:
        current_cmd = config.cmd_init.copy()
        print(f"Joystick disabled, using fixed cmd_init: {current_cmd.tolist()}")
    elif joystick is None or not joystick.available:
        current_cmd = get_idle_command(config)
        print(
            "Joystick requested but unavailable. Using zero command [0, 0, 0] "
            "instead of cmd_init so the robot does not walk automatically."
        )
    else:
        current_cmd = get_idle_command(config)

    # Initialize encoder frame stack for teacher/distillation policies
    encoder_frame_stack = deque(maxlen=args.encoder_seq_len)
    print_cmd_interval = 0.0 if args.print_cmd_hz <= 0.0 else 1.0 / args.print_cmd_hz
    print_object_obs_interval = (
        0.0 if args.print_object_obs_hz <= 0.0 else 1.0 / args.print_object_obs_hz
    )
    last_cmd_print_time = 0.0
    last_object_obs_print_time = 0.0
    had_live_cmd_output = False

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set up camera to follow the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # Track the pelvis/base body (usually body id 1, adjust if needed)
        # You can find the correct body by looking at the robot's URDF/XML structure
        viewer.cam.trackbodyid = 1  # Changed from 0 (world) to 1 (pelvis/base)
        viewer.cam.distance = 2.5   # Distance from robot (increased for better view)
        viewer.cam.elevation = -20  # Camera angle (negative looks down)
        viewer.cam.azimuth = 90     # Side view angle
        viewer.cam.lookat[:] = [0, 0, 0.5]  # Look at point offset
        
        while viewer.is_running():
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:7 + config.num_actions], config.kps, np.zeros_like(config.kds), d.qvel[6:6 + config.num_actions], config.kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % config.control_decimation == 0:
                # Apply control signal here.
                if joystick is not None:
                    current_cmd = joystick.poll()
                elif args.disable_joystick:
                    current_cmd = config.cmd_init.copy()
                else:
                    current_cmd = get_idle_command(config)

                if args.print_cmd:
                    now = time.time()
                    if print_cmd_interval == 0.0 or (now - last_cmd_print_time) >= print_cmd_interval:
                        if joystick is not None and joystick.available:
                            axis_state = joystick.get_axis_state()
                            axis_text = (
                                f"axes lx={axis_state[JoystickCommandInput.LEFT_X]:+.3f} "
                                f"ly={axis_state[JoystickCommandInput.LEFT_Y]:+.3f} "
                                f"rx={axis_state[JoystickCommandInput.RIGHT_X]:+.3f}"
                            )
                        elif args.disable_joystick:
                            axis_text = "joystick=disabled"
                        else:
                            axis_text = "joystick=unavailable"

                        status = (
                            f"cmd vx={current_cmd[0]:+.3f} "
                            f"vy={current_cmd[1]:+.3f} "
                            f"yaw={current_cmd[2]:+.3f} | {axis_text}"
                        )
                        print(f"\r{status:<120}", end="", flush=True)
                        last_cmd_print_time = now
                        had_live_cmd_output = True

                start_compute = time.time()
                # Get sensor data (explicitly use float32 for consistency with real robot)
                qj = d.qpos[7:7 + config.num_actions].astype(np.float32)
                dqj = d.qvel[6:6 + config.num_actions].astype(np.float32)
                quat = d.qpos[3:7].astype(np.float32)
                omega = d.qvel[3:6].astype(np.float32)

                # Get extra dual-input observations from MuJoCo simulation.
                object_obs = None
                teacher_obs_terms = None

                if policy_type == "distillation":
                    object_obs = get_object_pose(d, m)
                elif policy_type == "teacher":
                    teacher_obs_terms = get_teacher_observation_terms(d, m)

                if not args.no_object_obs_overlay:
                    update_object_obs_overlay(viewer, object_obs, d, m)

                if object_obs is not None and args.print_object_obs:
                    now = time.time()
                    if (
                        print_object_obs_interval == 0.0
                        or (now - last_object_obs_print_time) >= print_object_obs_interval
                    ):
                        last_object_obs_print_time = now
                        print(
                            "\n[object_obs policy input] "
                            f"xyz=({object_obs[0]:+.4f},{object_obs[1]:+.4f},{object_obs[2]:+.4f}) "
                            f"quat_wxyz=({object_obs[3]:+.4f},{object_obs[4]:+.4f},"
                            f"{object_obs[5]:+.4f},{object_obs[6]:+.4f})",
                            flush=True,
                        )

                if recorder is not None:
                    recorder.write_obs_once(
                        cmd_raw=get_record_raw_command(joystick, current_cmd),
                        cmd_scaled=current_cmd * config.cmd_scale,
                        quat=quat,
                        gravity=get_gravity_orientation(quat).astype(np.float32),
                        omega=omega,
                        object_obs=object_obs,
                    )

                # Compute policy action using shared function
                policy_result = compute_policy_action(
                    policy=policy,
                    frame_stack=frame_stack,
                    qj=qj,
                    dqj=dqj,
                    quat=quat,
                    omega=omega,
                    cmd=current_cmd,
                    previous_action=action,
                    config=config,
                    object_obs=object_obs,
                    teacher_obs_terms=teacher_obs_terms,
                    policy_type=policy_type,
                    encoder_frame_stack=encoder_frame_stack,
                    return_debug=recorder is not None,
                )
                if recorder is not None:
                    action, target_dof_pos, policy_debug = policy_result
                    if recorder.write_policy_debug_once(policy_debug):
                        print(f"\n[SteadyTrayRecord] wrote {recorder.path}")
                else:
                    action, target_dof_pos = policy_result
                compute_time = time.time() - start_compute
                if compute_time > config.control_dt:
                    print(f"Warning: Policy compute time {compute_time:.6f} seconds exceeds control_dt {config.control_dt} seconds")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if joystick is not None:
        joystick.close()
    if had_live_cmd_output:
        print()
