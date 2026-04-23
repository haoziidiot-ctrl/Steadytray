# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL checkpoint 回放脚本。"""

"""先启动 Isaac Sim 模拟器。"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_seconds",
    type=float,
    default=None,
    help="Optional video duration in seconds. Overrides --video_length when provided.",
)
parser.add_argument(
    "--video_compat",
    action="store_true",
    default=False,
    help="Generate a broadly compatible H.264/yuv420p sibling MP4 after recording.",
)
parser.add_argument(
    "--video_resolution",
    type=int,
    nargs=2,
    metavar=("WIDTH", "HEIGHT"),
    default=None,
    help="Optional output resolution for the play viewport camera.",
)
parser.add_argument(
    "--video_eye",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional viewport camera eye position override for video recording.",
)
parser.add_argument(
    "--video_lookat",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional viewport camera lookat override for video recording.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--teacher", action="store_true", default=False, help="Run in teacher mode.")
parser.add_argument(
    "--dump-contract-sample",
    type=str,
    default=None,
    help="Optional .npz path to dump one standard-policy sample with policy_obs and action.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""下面进入主逻辑。"""

import gymnasium as gym
import os
import time
import torch
from mp4_faststart import rewrite_faststart_folder_in_place
from mp4_compat import rewrite_compat_folder

from base.on_policy_runner import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
import steadytray.tasks  # noqa: F401

try:
    from adapter.on_policy_runner import AdapterOnPolicyRunner
    from adapter.env_wrapper import AdapterRslRlVecEnvWrapper
    ADAPTER_POLICY_AVAILABLE = True
except ImportError as e:
    AdapterOnPolicyRunner = None
    AdapterRslRlVecEnvWrapper = None
    ADAPTER_POLICY_AVAILABLE = False

from steadytray.utils.parser_cfg import parse_env_cfg

# 作用：判断当前配置是否属于 adapter 强化学习策略。
def is_adapter_config(agent_cfg):
    """判断当前配置是否属于 adapter 强化学习策略。"""
    return hasattr(agent_cfg, 'policy') and hasattr(agent_cfg.policy, 'class_name') and \
           agent_cfg.policy.class_name in ["AdaptedActorCritic", "ResidualActorCritic"]

# 作用：判断当前配置是否属于 student-teacher 蒸馏策略。
def is_distillation_config(agent_cfg):
    """判断当前配置是否属于 student-teacher 蒸馏策略。"""
    return hasattr(agent_cfg, 'algorithm') and hasattr(agent_cfg.algorithm, 'class_name') and \
           agent_cfg.algorithm.class_name == "Distillation"

def sync_agent_cfg_from_checkpoint(agent_cfg, resume_path: str):
    """Align policy/algorithm class settings with the saved checkpoint config when available."""
    agent_cfg_path = os.path.join(os.path.dirname(resume_path), "params", "agent.yaml")
    if not os.path.isfile(agent_cfg_path):
        return agent_cfg

    try:
        saved_agent_cfg = load_yaml(agent_cfg_path)
    except Exception as exc:
        print(f"[WARN] Failed to read saved agent config '{agent_cfg_path}': {exc}")
        return agent_cfg

    saved_policy_cfg = saved_agent_cfg.get("policy", {}) or {}
    saved_algorithm_cfg = saved_agent_cfg.get("algorithm", {}) or {}

    if hasattr(agent_cfg, "policy"):
        for key in (
            "class_name",
            "adapter_type",
            "encoder_type",
            "ctx_dim",
            "encoder_layers",
            "num_heads",
            "encoder_dropout",
            "use_gate",
            "actor_hidden_dims",
            "critic_hidden_dims",
            "activation",
            "init_noise_std",
            "adapter_hidden",
            "clamp_gamma",
            "residual_hidden_dims",
            "clamp_residual",
        ):
            if key in saved_policy_cfg and hasattr(agent_cfg.policy, key):
                old_value = getattr(agent_cfg.policy, key)
                new_value = saved_policy_cfg[key]
                if old_value != new_value:
                    print(f"[INFO] Restoring policy.{key} from checkpoint config: {old_value} -> {new_value}")
                    setattr(agent_cfg.policy, key, new_value)

    if hasattr(agent_cfg, "algorithm") and "class_name" in saved_algorithm_cfg and hasattr(agent_cfg.algorithm, "class_name"):
        old_value = getattr(agent_cfg.algorithm, "class_name")
        new_value = saved_algorithm_cfg["class_name"]
        if old_value != new_value:
            print(f"[INFO] Restoring algorithm.class_name from checkpoint config: {old_value} -> {new_value}")
            setattr(agent_cfg.algorithm, "class_name", new_value)

    return agent_cfg

# 作用：加载 checkpoint，创建推理环境并执行回放或录视频。
def main():
    """加载 checkpoint，创建推理环境并执行回放或录视频。"""
    effective_disable_fabric = args_cli.disable_fabric
    if args_cli.video and args_cli.disable_fabric:
        print("[WARN] Ignoring --disable_fabric during video recording because it can hide robot meshes in headless videos.")
        effective_disable_fabric = False

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not effective_disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    agent_cfg = sync_agent_cfg_from_checkpoint(agent_cfg, resume_path)

    log_dir = os.path.dirname(resume_path)
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
        env_cfg.sim.render.samples_per_pixel = 4
        env_cfg.sim.render.enable_shadows = True
        env_cfg.sim.render.enable_ambient_occlusion = True
        env_cfg.sim.render.enable_direct_lighting = True
        if args_cli.video_seconds is not None:
            video_length = max(1, round(args_cli.video_seconds / (env_cfg.sim.dt * env_cfg.decimation)))

        print("[INFO] Video camera overrides:")
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


    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 1,
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
            raise ImportError("Adapter policy runner not available. Make sure adapter module is installed.")
        
        if is_distillation:
            print("[INFO] Using distillation configuration (online encoder distillation)")
            print("[INFO] Architecture: Trainable student encoder + Frozen teacher encoder + Shared frozen components")
            print("[INFO] Inference: Using student encoder for evaluation")
        else:
            print("[INFO] Using adapter-based policy configuration (parameter-efficient fine-tuning)")
            adapter_desc = "Residual action adapter" if agent_cfg.policy.class_name == "ResidualActorCritic" else "FiLM adapters"
            encoder_desc = getattr(agent_cfg.policy, "encoder_type", "history").upper()
            print(f"[INFO] Architecture: Frozen base policy + Trainable {adapter_desc} + {encoder_desc} encoder")
        
        env = AdapterRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner = AdapterOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_type = "distillation" if is_distillation else "adapter"
        
        policy_nn = None
    else:
        print("[INFO] Using standard policy configuration")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_type = "standard"

        try:
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            policy_nn = ppo_runner.alg.actor_critic

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    
    dt = env.unwrapped.step_dt

    obs, info = env.get_observations()
    timestep = 0
    dumped_contract_sample = False

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if policy_type == "distillation":
                if isinstance(obs, dict):
                    student_encoder_obs = obs["student_encoder"]
                    policy_obs = obs["policy"]
                    actions = policy.__call__(student_encoder_obs, policy_obs)
                else:
                    raise ValueError("Expected dictionary observations for distillation policy")
            elif policy_type == "adapter":
                if isinstance(obs, dict):
                    encoder_obs = obs["encoder"]
                    policy_obs = obs["policy"]
                    actions = policy.__call__(encoder_obs, policy_obs)
                else:
                    raise ValueError("Expected dictionary observations for adapter policy")
            else:
                actions = policy.__call__(obs)
                if args_cli.dump_contract_sample and not dumped_contract_sample:
                    import numpy as np

                    np.savez(
                        args_cli.dump_contract_sample,
                        policy_obs=obs[0].detach().cpu().numpy(),
                        action=actions[0].detach().cpu().numpy(),
                    )
                    print(f"[INFO] Dumped contract sample to: {args_cli.dump_contract_sample}")
                    dumped_contract_sample = True
                
            obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            if timestep == video_length:
                break

        timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        try:
            rewritten_files = rewrite_faststart_folder_in_place(video_folder)
            for rewritten_path in rewritten_files:
                print(f"[INFO] Rewrote video for faststart compatibility: {rewritten_path}")
        except Exception as exc:
            print(f"[WARN] Failed to rewrite recorded videos for faststart compatibility: {exc}")

        if args_cli.video_compat:
            try:
                compat_files = rewrite_compat_folder(video_folder)
                for compat_path in compat_files:
                    print(f"[INFO] Re-encoded video in place for compatibility: {compat_path}")
            except Exception as exc:
                print(f"[WARN] Failed to write compatibility MP4 files: {exc}")

if __name__ == "__main__":
    main()
    simulation_app.close()
