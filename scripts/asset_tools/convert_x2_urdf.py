"""X2 专用 URDF→USD 转换：手掌等动态体用 convex_decomposition 保留手指凹陷。"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert X2 URDF to USD with convex_decomposition colliders.")
parser.add_argument("input", type=str, help="Input URDF path.")
parser.add_argument("output", type=str, help="Output USD path.")
parser.add_argument("--merge-joints", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid URDF: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=False,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        # 关键：手掌等动态 link 用 convex_decomposition 而非 convex_hull，保留手指凹陷
        collider_type="convex_decomposition",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",
        ),
    )

    print("-" * 80)
    print(f"Input URDF: {urdf_path}")
    print(f"Output USD: {dest_path}")
    print(f"collider_type: {cfg.collider_type}")
    print_dict(cfg.to_dict(), nesting=0)
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"Generated USD: {converter.usd_path}")

    carb_settings = carb.settings.get_settings()
    local_gui = carb_settings.get("/app/window/enabled")
    livestream_gui = carb_settings.get("/app/livestream/enabled")
    if local_gui or livestream_gui:
        stage_utils.open_stage(converter.usd_path)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
