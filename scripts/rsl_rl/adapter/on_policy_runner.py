# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 作用：Stage 3/Stage 4 的总调度器。它负责从环境取 observation groups、构建模型与算法、执行 rollout、调用 update，并记录日志。

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
# from rsl_rl.algorithms import PPO, Distillation
from .ppo import PPO
from .distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from .actor_critic import AdaptedActorCritic, ResidualActorCritic
from .student_teacher import AdaptedStudentTeacher
from rsl_rl.utils import store_code_state


# 作用：Stage 3/Stage 4 的主 runner，负责把 observation groups、模型、算法和日志系统串起来。
class AdapterOnPolicyRunner:
    """用于训练与评估的 on-policy runner。"""

    # 作用：解析环境中的 observation groups，构建 policy/algorithm，并初始化存储与日志组件。
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 检查是否启用了多卡训练
        self._configure_multi_gpu()

        # 根据算法类型确定当前训练模式
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # 解析环境返回的观测，并推断各观测组的维度
        obs_dict, extras = self.env.get_observations()
        
        # 适配式训练通常会包含三组或四组观测：
        # 
        # RL 模式（PPO + AdaptedActorCritic）：
        # - "policy": 展平后的历史观测（例如 5 步），供冻结的基础 actor MLP 使用
        # - "encoder": 时序历史观测（例如 32 步），供可训练编码器使用
        # - "critic"/"adapted_critic": 供 critic 使用的特权观测
        #
        # Distillation 模式（AdaptedStudentTeacher）：
        # - "policy": 展平后的历史观测（例如 5 步），供冻结的基础 actor MLP 使用 (shared)
        # - "student_encoder": 供可训练 student encoder 使用的有限观测
        # - "teacher": 供冻结 teacher encoder 使用的完整特权观测
        # - "critic": 供冻结 critic 使用的特权观测（蒸馏损失本身不会用到）
        
        # 1. 从 "policy" 组获取 actor 输入，即给基础 MLP 的展平历史观测
        policy_obs = obs_dict.get("policy", obs_dict)  # 如果没有 "policy"，则退化为整个观测字典
        if isinstance(policy_obs, torch.Tensor):
            num_actor_obs = policy_obs.shape[1]
        else:
            num_actor_obs = policy_obs["policy"].shape[1] if "policy" in policy_obs else list(policy_obs.values())[0].shape[1]

        # 2. 解析编码器输入；RL 与 Distillation 使用的键不同
        if self.training_type == "rl":
            # RL 模式：AdaptedActorCritic 使用 "encoder"
            encoder_obs = obs_dict.get("encoder", obs_dict.get("policy", obs_dict))
            self.student_encoder_key = "encoder"  # 为了和 learn 循环的字段命名保持一致
            self.teacher_encoder_key = None  # RL 模式下不会用到
        elif self.training_type == "distillation":
            # 蒸馏模式：student 用 "student_encoder"，teacher 用 teacher 对应观测
            encoder_obs = obs_dict.get("student_encoder", obs_dict.get("encoder", obs_dict.get("policy", obs_dict)))
            self.student_encoder_key = "student_encoder"
            # teacher encoder 的完整特权观测
            teacher_encoder_obs = obs_dict.get("teacher_encoder", obs_dict.get("encoder", obs_dict.get("policy", obs_dict)))

        # 解析编码器观测的形状
        if encoder_obs.dim() == 3:
            # 时序观测：[num_envs, seq_len, obs_dim]
            encoder_seq_len = encoder_obs.shape[1]
            num_encoder_obs = encoder_obs.shape[2]  # 每个时间步的观测维度
            encoder_obs_shape = [encoder_seq_len, num_encoder_obs]  # 存储时保留为 [seq_len, obs_dim]
        else:
            # 平面观测：[num_envs, obs_dim]，按单步输入处理
            num_encoder_obs = encoder_obs.shape[1]
            encoder_obs_shape = [num_encoder_obs]  # 存储形状为 [obs_dim]
        
        # 蒸馏模式下还需要额外解析 teacher encoder 的形状
        if self.training_type == "distillation":
            if teacher_encoder_obs.dim() == 3:
                teacher_seq_len = teacher_encoder_obs.shape[1]
                num_teacher_encoder_obs = teacher_encoder_obs.shape[2]
                teacher_encoder_obs_shape = [teacher_seq_len, num_teacher_encoder_obs]
            else:
                num_teacher_encoder_obs = teacher_encoder_obs.shape[1]
                teacher_encoder_obs_shape = [num_teacher_encoder_obs]

        # 3. 确定“特权观测”到底使用哪一组
        if self.training_type == "rl":
            # RL 模式优先使用 adapted_critic，其次才是 critic
            if "adapted_critic" in obs_dict:
                self.privileged_obs_type = "adapted_critic"
            elif "critic" in obs_dict:
                self.privileged_obs_type = "critic"  # actor-critic 强化学习，例如 PPO
            else:
                self.privileged_obs_type = None
        elif self.training_type == "distillation":
            # 蒸馏模式下，teacher encoder 的输入就视为特权观测
            if "teacher_encoder" in obs_dict:
                self.privileged_obs_type = "teacher_encoder"
            elif "encoder" in obs_dict:
                self.privileged_obs_type = "encoder"
            else:
                self.privileged_obs_type = None

        # 4. 解析特权观测的维度
        if self.training_type == "rl":
            # RL 模式下，特权观测供 critic 使用
            if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                num_critic_obs = obs_dict[self.privileged_obs_type].shape[1] if obs_dict[self.privileged_obs_type].dim() == 2 else obs_dict[self.privileged_obs_type].shape[2]
            else:
                num_critic_obs = num_actor_obs
                print(f"[WARNING] Privileged observations for '{self.privileged_obs_type}' not found. Using actor obs dim for critic.")
        elif self.training_type == "distillation":
            # 蒸馏模式下，critic 虽然冻结且不参与损失，但模型初始化时仍然需要它的输入维度
            # 优先使用 critic 观测；如果没有，再退回到 teacher encoder 的维度
            if "critic" in obs_dict:
                critic_obs_tensor = obs_dict["critic"]
                num_critic_obs = critic_obs_tensor.shape[1] if critic_obs_tensor.dim() == 2 else critic_obs_tensor.shape[2]
            elif "adapted_critic" in obs_dict:
                critic_obs_tensor = obs_dict["adapted_critic"]
                num_critic_obs = critic_obs_tensor.shape[1] if critic_obs_tensor.dim() == 2 else critic_obs_tensor.shape[2]
            else:
                # 最后退回到 teacher encoder 维度，虽然不理想，但能保证初始化安全完成
                num_critic_obs = num_teacher_encoder_obs
                print(f"[WARNING] Critic observations not found in distillation mode. Using teacher encoder obs dim ({num_teacher_encoder_obs}).")

        # 打印维度信息，便于调试
        print(f"\n{'='*70}")
        if self.training_type == "rl":
            print(f"Observation Dimensions for AdaptedActorCritic (RL mode):")
            print(f"  - Policy obs (actor input): {num_actor_obs} (flattened history)")
            print(f"  - Encoder obs (Encoder input): {num_encoder_obs} (per-timestep)")
            if encoder_obs.dim() == 3:
                print(f"    Encoder sequence length: {encoder_seq_len}")
            print(f"  - Critic obs (privileged): {num_critic_obs}")
        elif self.training_type == "distillation":
            print(f"Observation Dimensions for AdaptedStudentTeacher (Distillation mode):")
            print(f"  - Policy obs (actor input): {num_actor_obs} (flattened history)")
            print(f"  - Student encoder obs: {num_encoder_obs} (limited observations)")
            if encoder_obs.dim() == 3:
                print(f"    Student encoder sequence length: {encoder_seq_len}")
            print(f"  - Teacher encoder obs: {num_teacher_encoder_obs} (full privileged)")
            if teacher_encoder_obs.dim() == 3:
                print(f"    Teacher encoder sequence length: {teacher_seq_len}")
            print(f"  - Critic obs: {num_critic_obs} (frozen, not used in loss)")
        print(f"{'='*70}\n")

        # 根据配置中的类名解析出实际的 policy 类
        policy_class = eval(self.policy_cfg.pop("class_name"))
        
        # 根据训练模式初始化 policy
        if self.training_type == "rl":
            # RL 模式下使用 AdaptedActorCritic 或 ResidualActorCritic
            policy: AdaptedActorCritic | ResidualActorCritic = policy_class(
                num_actor_obs, num_critic_obs, self.env.num_actions, 
                num_encoder_obs=num_encoder_obs,  # 显式传入编码器输入维度
                **self.policy_cfg
            ).to(self.device)
        elif self.training_type == "distillation":
            # 蒸馏模式下使用 AdaptedStudentTeacher
            policy: AdaptedStudentTeacher = policy_class(
                num_actor_obs, num_critic_obs, self.env.num_actions,
                num_student_encoder_obs=num_encoder_obs,  # student encoder 输入维度
                num_teacher_encoder_obs=num_teacher_encoder_obs,  # teacher encoder 输入维度
                **self.policy_cfg
            ).to(self.device)

        # 如果启用了 RND，则补充它所需的 gated state 维度
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查环境里是否提供了 rnd_state
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # 读取 rnd_state 的维度
            num_rnd_state = rnd_state.shape[1]
            # 写回算法配置
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 按环境步长缩放 RND 权重，和 legged_gym 一类环境的处理方式一致
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # 如果启用了 symmetry，则把环境对象传给 symmetry 配置
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # 供 symmetry 函数在处理不同观测项时使用
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # 初始化算法对象
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 缓存训练配置
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            # 对 policy 观测做归一化
            self.obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            # 对特权观测做归一化
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # 不做归一化
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # 不做归一化

        # 初始化 rollout storage
        if self.training_type == "rl":
            # RL 模式：单编码器 PPO
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                encoder_obs_shape,  # 编码器观测形状；如果是 3D，则保留为 [seq_len, obs_dim]
                [num_critic_obs],   # critic 的特权观测形状
                [self.env.num_actions],
                [num_actor_obs],    # policy 观测形状，即给基础 actor 的展平历史观测
            )
        elif self.training_type == "distillation":
            # 蒸馏模式：同时为 student 和 teacher 准备编码器存储
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                encoder_obs_shape,          # student encoder 的观测形状（有限观测）
                teacher_encoder_obs_shape,  # teacher encoder 的观测形状（完整特权观测）
                [self.env.num_actions],
                [num_actor_obs],            # policy 观测形状，即给基础 actor 的展平历史观测
            )

        # 决定是否关闭日志记录
        # 只有 rank 0 主进程负责写日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # 日志状态
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    # 作用：执行完整训练循环；先 rollout 收集数据，再调用 PPO 或 Distillation 更新。
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # 初始化日志 writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 初始化日志后端，默认使用 TensorBoard
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # 蒸馏模式下必须先确认 teacher 权重已经成功加载
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # 随机化初始 episode 长度，增加探索多样性
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始训练前先拿到初始观测
        obs_dict, extras = self.env.get_observations()
        
        # 根据训练模式拆出需要的观测
        policy_obs = obs_dict["policy"]  # actor 输入，无论哪种模式都需要
        
        if self.training_type == "rl":
            # RL 模式：需要 encoder 和 critic 观测
            encoder_obs = obs_dict.get(self.student_encoder_key, policy_obs)  # "encoder"
            critic_obs = obs_dict.get(self.privileged_obs_type, policy_obs) if self.privileged_obs_type else policy_obs
            teacher_encoder_obs = None  # RL 模式下不会用到
        elif self.training_type == "distillation":
            # 蒸馏模式：需要 student encoder 与 teacher encoder 观测
            encoder_obs = obs_dict.get(self.student_encoder_key, policy_obs)  # "student_encoder"
            teacher_encoder_obs = obs_dict.get(self.privileged_obs_type, policy_obs)  # "teacher"
            critic_obs = None  # 蒸馏采样动作时不会用到冻结 critic
        
        # 把张量移动到训练设备
        policy_obs = policy_obs.to(self.device)
        encoder_obs = encoder_obs.to(self.device)
        if self.training_type == "rl":
            critic_obs = critic_obs.to(self.device)
        elif self.training_type == "distillation":
            teacher_encoder_obs = teacher_encoder_obs.to(self.device)
        
        self.train_mode()  # 切到训练模式，例如启用 dropout 等训练态行为

        # 训练过程中的统计缓存
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 如果启用了 RND，则额外记录外在奖励和内在奖励
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 多卡训练时，先同步各卡参数
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: 是否需要同步经验归一化器？
            # 当前默认不需要，因为它们会在训练过程中渐近收敛到相近数值。

        # 进入主训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout 收集轨迹
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 根据训练模式采样动作
                    if self.training_type == "rl":
                        # RL 模式：使用 encoder_obs、policy_obs、critic_obs
                        actions = self.alg.act(encoder_obs, policy_obs, critic_obs)
                    elif self.training_type == "distillation":
                        # 蒸馏模式：使用 student_encoder_obs、policy_obs、teacher_encoder_obs
                        actions = self.alg.act(encoder_obs, policy_obs, teacher_encoder_obs)
                    
                    # 执行一步环境交互
                    obs_dict, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    
                    # 重新拆出下一时刻观测
                    policy_obs = obs_dict["policy"]
                    
                    if self.training_type == "rl":
                        encoder_obs = obs_dict.get(self.student_encoder_key, policy_obs)  # "encoder"
                        critic_obs = obs_dict.get(self.privileged_obs_type, policy_obs) if self.privileged_obs_type else policy_obs
                    elif self.training_type == "distillation":
                        encoder_obs = obs_dict.get(self.student_encoder_key, policy_obs)  # "student_encoder"
                        teacher_encoder_obs = obs_dict.get(self.privileged_obs_type, policy_obs)  # "teacher"
                    
                    # 把张量移动到训练设备
                    if self.training_type == "rl":
                        policy_obs, encoder_obs, critic_obs, rewards, dones = (
                            policy_obs.to(self.device),
                            encoder_obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device)
                        )
                    elif self.training_type == "distillation":
                        policy_obs, encoder_obs, teacher_encoder_obs, rewards, dones = (
                            policy_obs.to(self.device),
                            encoder_obs.to(self.device),
                            teacher_encoder_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device)
                        )
                    
                    # 如有需要，对观测做归一化
                    encoder_obs = self.obs_normalizer(encoder_obs)
                    if self.privileged_obs_type is not None:
                        critic_obs = self.privileged_obs_normalizer(critic_obs)
                    else:
                        critic_obs = encoder_obs

                    # 把这一步交互结果写入算法缓存
                    self.alg.process_env_step(rewards, dones, infos)

                    # 提取内在奖励，仅用于记录日志
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 更新统计量
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # 更新奖励累计
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # 更新 episode 长度
                        cur_episode_length += 1
                        # 清空已经结束的环境对应的累计量
                        # -- 公共部分
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- 内在奖励与外在奖励
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # RL 模式下额外计算 return
                if self.training_type == "rl":
                    self.alg.compute_returns(critic_obs, encoder_obs)

            # 更新策略或蒸馏模型
            loss_dict = self.alg.update()
            
            # 如果蒸馏时启用了 DAgger，则推进一次 DAgger 调度步数
            if self.training_type == "distillation" and hasattr(self.alg, 'use_dagger') and self.alg.use_dagger:
                self.alg.increment_dagger_step()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # 写日志并按周期保存模型
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清空本轮 episode 统计
            ep_infos.clear()
            # 在训练开始时把当前代码状态存档
            if it == start_iter and not self.disable_logs:
                # 收集当前仓库的 diff 文件
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果日志后端支持，也一并上传
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后再保存一次最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # 作用：整理并输出训练日志，同时写入 TensorBoard 或 WandB。
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # 计算本轮采样到的总步数
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新累计时间步与累计训练时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode 统计信息
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 兼容标量和零维张量
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 同时写入 logger 和终端
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- 损失项
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- 策略统计
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- 性能指标
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- DAgger 统计（仅蒸馏模式）
        if self.training_type == "distillation" and hasattr(self.alg, 'use_dagger') and self.alg.use_dagger:
            dagger_stats = self.alg.get_dagger_stats()
            for key, value in dagger_stats.items():
                self.writer.add_scalar(f"DAgger/{key}", value, locs["it"])

        # -- 训练回报统计
        if len(locs["rewbuffer"]) > 0:
            # RND 模式下分别记录外在奖励和内在奖励
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # 常规训练统计
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb 不支持非整数横轴
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- 损失项
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- 奖励
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode 信息
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            # -- DAgger 统计
            if self.training_type == "distillation" and hasattr(self.alg, 'use_dagger') and self.alg.use_dagger:
                dagger_stats = self.alg.get_dagger_stats()
                if dagger_stats:
                    log_string += f"""{'DAgger beta:':>{pad}} {dagger_stats.get('dagger_beta', 0.0):.4f}\n"""
                    log_string += f"""{'DAgger teacher ratio:':>{pad}} {dagger_stats.get('dagger_teacher_ratio', 0.0):.4f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    # 作用：保存当前训练状态，包括 policy、optimizer 和 runner 元信息。
    def save(self, path: str, infos=None):
        # -- 保存模型主体
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- 如有 RND，一并保存
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- 如启用观测归一化，也一并保存
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 写入磁盘
        torch.save(saved_dict, path)

        # 如果外部日志后端支持，也上传模型文件
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    # 作用：从 checkpoint 恢复训练；Stage 4 也会在这里加载 Stage 3 teacher。
    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- 加载模型参数
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        print(f"Resumed training: {resumed_training}.")
        # -- 如有 RND，也恢复其参数
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- 如启用观测归一化，则恢复归一化器状态
        if self.empirical_normalization:
            if resumed_training:
                # 如果是继续训练，则 actor/student 与 critic/teacher 的归一化器都恢复
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果不是“继续训练”而是“载入已有模型开始新训练”，这里通常对应
                # Stage 4 接 Stage 3 的场景。此时只恢复 teacher 侧归一化器，
                # student 侧不恢复，因为它的观测空间可能已经和上一个 RL 阶段不同。
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- 如有需要且确实是恢复训练，则一并恢复优化器
        if load_optimizer and resumed_training:
            # -- 算法优化器
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND 优化器
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- 恢复当前训练迭代数
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            print(f"Resumed training from iteration {self.current_learning_iteration}.")
        return loaded_dict["infos"]

    # 作用：返回推理时使用的 policy 接口。
    def get_inference_policy(self, device=None):
        self.eval_mode()  # 切到评估模式，例如关闭 dropout
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    # 作用：把模型切换到训练模式。
    def train_mode(self):
        # -- PPO / 主策略
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- 归一化器
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    # 作用：把模型切换到评估模式。
    def eval_mode(self):
        # -- PPO / 主策略
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- 归一化器
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    # 作用：记录代码版本信息，便于实验复现。
    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """辅助函数。"""

    # 作用：初始化多卡训练需要的通信和设备配置。
    def _configure_multi_gpu(self):
        """配置多卡训练相关状态。"""
        # 检查是否启用了分布式训练
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 如果不是分布式训练，直接把本地 rank 和全局 rank 设为 0
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 读取 local rank 和 global rank
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 整理成统一的多卡配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 当前进程的全局 rank
            "local_rank": self.gpu_local_rank,  # 当前机器上的本地 rank
            "world_size": self.gpu_world_size,  # 总进程数
        }

        # 检查 device 是否和 local rank 对应
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # 校验多卡配置是否合法
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化 torch.distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # 把当前进程绑定到对应 GPU
        torch.cuda.set_device(self.gpu_local_rank)
