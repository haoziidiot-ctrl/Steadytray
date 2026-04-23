# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 训练入口脚本。"""

"""先启动 Isaac Sim 模拟器。"""

# 作用：训练总入口。它根据任务名加载环境和 agent 配置，选择普通 PPO、Stage 3 adapter 训练或 Stage 4 distillation 训练。

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_seconds",
    type=float,
    default=None,
    help="Optional video duration in seconds. Overrides --video_length when provided.",
)
parser.add_argument(
    "--video_resolution",
    type=int,
    nargs=2,
    metavar=("WIDTH", "HEIGHT"),
    default=None,
    help="Optional output resolution for the training viewport camera.",
)
parser.add_argument(
    "--video_eye",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional viewport camera eye position override for training video recording.",
)
parser.add_argument(
    "--video_lookat",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional viewport camera lookat override for training video recording.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查所需的最低 RSL-RL 版本。"""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""下面进入主逻辑。"""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from base.on_policy_runner import OnPolicyRunner  # noqa: E402

try:
    from adapter.on_policy_runner import AdapterOnPolicyRunner
    from adapter.env_wrapper import AdapterRslRlVecEnvWrapper
    ADAPTER_POLICY_AVAILABLE = True
except ImportError as e:
    AdapterOnPolicyRunner = None
    AdapterRslRlVecEnvWrapper = None
    ADAPTER_POLICY_AVAILABLE = False

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import steadytray.tasks  # noqa: F401
from steadytray.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 作用：判断当前 agent 配置是否属于 Stage 3 adapter 强化学习。
def is_adapter_config(agent_cfg):
    """判断当前 agent 配置是否属于 Stage 3 adapter 强化学习。"""
    return hasattr(agent_cfg, 'policy') and hasattr(agent_cfg.policy, 'class_name') and \
           agent_cfg.policy.class_name in ["AdaptedActorCritic", "ResidualActorCritic"]

# 作用：判断当前 agent 配置是否属于 Stage 4 蒸馏训练。
def is_distillation_config(agent_cfg):
    """判断当前 agent 配置是否属于 Stage 4 蒸馏训练。"""
    return hasattr(agent_cfg, 'algorithm') and hasattr(agent_cfg.algorithm, 'class_name') and \
           agent_cfg.algorithm.class_name == "Distillation"

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
# 作用：创建环境与 runner，加载 checkpoint，并启动训练循环。
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """创建环境与 runner，加载 checkpoint，并启动训练循环。"""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    video_length = args_cli.video_length

    if args_cli.video:
        if args_cli.task == "G1-Steady-Object":
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.body_name = "torso_link"
            env_cfg.viewer.eye = (3.2, -0.6, 1.5)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.35)
            env_cfg.viewer.resolution = (1920, 1080)

        if args_cli.video_eye is not None:
            env_cfg.viewer.eye = tuple(args_cli.video_eye)
        if args_cli.video_lookat is not None:
            env_cfg.viewer.lookat = tuple(args_cli.video_lookat)
        if args_cli.video_resolution is not None:
            env_cfg.viewer.resolution = tuple(args_cli.video_resolution)

        env_cfg.sim.render.antialiasing_mode = "TAA"
        env_cfg.sim.render.rendering_mode = "quality"
        env_cfg.sim.render.samples_per_pixel = 2
        env_cfg.sim.render.enable_shadows = True
        env_cfg.sim.render.enable_ambient_occlusion = True
        env_cfg.sim.render.enable_direct_lighting = True
        if args_cli.video_seconds is not None:
            video_length = max(1, round(args_cli.video_seconds / (env_cfg.sim.dt * env_cfg.decimation)))

        print("[INFO] Training video camera overrides:")
        print_dict(
            {
                "tracking": env_cfg.viewer.origin_type,
                "eye": env_cfg.viewer.eye,
                "lookat": env_cfg.viewer.lookat,
                "resolution": env_cfg.viewer.resolution,
                "video_length_steps": video_length,
                "video_length_seconds": video_length * env_cfg.sim.dt * env_cfg.decimation,
                "aa": env_cfg.sim.render.antialiasing_mode,
                "rendering_mode": env_cfg.sim.render.rendering_mode,
                "samples_per_pixel": env_cfg.sim.render.samples_per_pixel,
            },
            nesting=4,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    is_adapter = is_adapter_config(agent_cfg)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif is_adapter and agent_cfg.load_run and agent_cfg.load_run != ".*":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Adapter training will load base policy from: {resume_path}")
    else:
        resume_path = None

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    is_adapter = is_adapter_config(agent_cfg)
    is_distillation = is_distillation_config(agent_cfg)
    
    if is_adapter or is_distillation:
        if not ADAPTER_POLICY_AVAILABLE:
            raise ImportError(
                "Adapter policy runner not available. Make sure adapter.adapted_on_policy_runner module is installed."
            )
        
        if is_distillation:
            print("[INFO] Using distillation configuration (online encoder distillation)")
            print("[INFO] Architecture: Trainable student encoder + Frozen teacher encoder + Shared frozen components")
            print("[INFO] Training: Only student encoder parameters will be updated")
        else:
            adapter_type = agent_cfg.policy.class_name
            print("[INFO] Using adapter-based policy configuration (parameter-efficient fine-tuning)")
            print(f"[INFO] Architecture: Frozen base policy + Trainable {adapter_type} + Encoder")
            print(f"[INFO] Policy class: {adapter_type}")
        
        env = AdapterRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        runner = AdapterOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner_type = "distillation" if is_distillation else "adapter"
    else:
        print("[INFO] Using standard policy configuration")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner_type = "standard"
    
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
