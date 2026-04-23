# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 作用：Stage 4 的训练算法。它负责在 rollout 中生成 teacher 监督信号，并在 update 时只优化 student encoder。

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .student_teacher import AdaptedStudentTeacher
from .rollout_storage import RolloutStorage


# 作用：定义 Stage 4 的蒸馏训练算法，用 teacher latent/action 监督 student encoder。
class Distillation:
    """
    在线蒸馏算法：训练 student encoder 去模仿 teacher encoder。

    该算法面向 AdaptedStudentTeacher 架构实现 encoder 蒸馏：
    - 只有 student_encoder 参数会被训练
    - 其余组件（actor_body、adapters、action_head、critic）全部冻结并共享
    - 蒸馏损失包含两部分：
      1. 表征损失：student 与 teacher encoder latent 之间的 MSE
      2. 动作损失：student 与 teacher 确定性动作均值之间的 MSE

    训练流程：
    ----------
    1. 使用 student encoder 与环境交互，收集 rollout 数据
    2. 计算 teacher encoder latent（仅推理，不求梯度）
    3. 计算 teacher 动作均值（仅推理，不求梯度）
    4. 优化目标：loss = λ_embed * embedding_loss + λ_action * action_loss
    5. 只更新 student_encoder 的参数

    注意：这里是纯蒸馏过程，不包含 RL（PPO）、critic loss 或 value function 学习。
    """

    policy: AdaptedStudentTeacher
    """Student-teacher 模型。"""

    # 作用：初始化蒸馏算法；最关键的是只把 student encoder 参数交给优化器。
    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        learning_rate=1e-3,
        gradient_length=1,
        max_grad_norm=1.0,
        # 蒸馏损失权重
        embedding_loss_coef=1.0,
        action_loss_coef=1.0,
        loss_type="mse",
        device="cpu",
        # DAgger 相关参数
        use_dagger=False,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_decay_steps=1000,
        dagger_schedule_type="linear",  # 可选 "linear" 或 "exponential"
        # 分布式训练相关参数
        multi_gpu_cfg: dict | None = None,
    ):
        """
        初始化蒸馏算法。

        参数：
            policy: AdaptedStudentTeacher 模型
            num_learning_epochs: 每次 update 的训练轮数
            num_mini_batches: 每轮训练的 mini-batch 数
            learning_rate: student encoder 的学习率
            max_grad_norm: 梯度裁剪的最大范数
            embedding_loss_coef: 表征蒸馏损失权重（λ_embed）
            action_loss_coef: 动作蒸馏损失权重（λ_action）
            loss_type: 损失函数类型（"mse" 或 "huber"）
            device: 运行设备
            use_dagger: 是否启用 DAgger（数据聚合）
            dagger_beta_start: 初始 teacher 动作使用概率（β）
            dagger_beta_end: 最终 teacher 动作使用概率
            dagger_decay_steps: β 衰减所经历的步数/迭代数
            dagger_schedule_type: β 的衰减方式（"linear" 或 "exponential"）
            multi_gpu_cfg: 多卡训练配置
        """
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # 多卡训练参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # 为兼容 runner 接口而保留

        # 蒸馏相关组件
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorage = None  # type: ignore  # 稍后初始化
        
        # 仅针对 student encoder 的优化器
        student_encoder_params = [p for n, p in self.policy.named_parameters() if 'student_encoder' in n and p.requires_grad]
        self.optimizer = optim.Adam(student_encoder_params, lr=learning_rate)
        
        # 缓存 student encoder 参数，供梯度裁剪使用，避免重复筛选
        self.student_encoder_params = student_encoder_params
        
        self.transition = RolloutStorage.Transition()

        # 蒸馏相关超参数
        self.num_learning_epochs = num_learning_epochs
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.embedding_loss_coef = embedding_loss_coef
        self.action_loss_coef = action_loss_coef

        # DAgger 相关参数
        self.use_dagger = use_dagger
        self.dagger_beta_start = dagger_beta_start
        self.dagger_beta_end = dagger_beta_end
        self.dagger_decay_steps = dagger_decay_steps
        self.dagger_schedule_type = dagger_schedule_type.lower()
        self.dagger_beta = dagger_beta_start  # 当前 beta 值
        self.dagger_step = 0  # 当前用于衰减调度的步数/迭代数/回合计数器
        
        # DAgger 运行时状态（按环境决定当前使用 teacher 还是 student）
        # 会在 init_storage 中按 num_envs 进行初始化
        self.using_teacher = None  # bool 张量 [num_envs]：True 表示 teacher，False 表示 student

        # 初始化损失函数
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0
        
        print(f"\n{'='*70}")
        print(f"AdapterDistillation Initialized:")
        print(f"  Embedding loss weight (λ_embed): {embedding_loss_coef}")
        print(f"  Action loss weight (λ_action): {action_loss_coef}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Trainable parameters: {sum(p.numel() for p in student_encoder_params):,}")
        if self.use_dagger:
            print(f"  DAgger enabled:")
            print(f"    - Beta schedule: {self.dagger_schedule_type}")
            print(f"    - Beta start: {dagger_beta_start}")
            print(f"    - Beta end: {dagger_beta_end}")
            print(f"    - Decay steps: {dagger_decay_steps}")
        print(f"{'='*70}\n")

    # 作用：初始化 rollout 存储，分别保存 student 输入、teacher 输入和 teacher 标签。
    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_encoder_obs_shape,
        teacher_encoder_obs_shape,
        actions_shape,
        policy_obs_shape=None,
    ):
        """
        初始化蒸馏所需的 rollout 存储。

        参数：
            training_type: 应为 "distillation"
            num_envs: 并行环境数量
            num_transitions_per_env: 每次 rollout 的 transition 数量
            student_encoder_obs_shape: student encoder 观测的形状（有限观测）
            teacher_encoder_obs_shape: teacher encoder 观测的形状（完整观测）
            actions_shape: 动作向量的形状
            policy_obs_shape: policy 观测的形状（供基础 actor MLP 使用）
        """
        # 创建 rollout 存储
        # encoder_obs_shape 对应 student encoder 观测
        # privileged_obs_shape 对应 teacher encoder 观测
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_encoder_obs_shape,  # encoder_observations（student）
            teacher_encoder_obs_shape,  # privileged_observations（teacher）
            actions_shape,
            None,  # rnd_state_shape
            policy_obs_shape,  # policy_observations
            self.device,
        )
        
        # 初始化 DAgger 的逐环境策略选择状态
        if self.use_dagger:
            self.using_teacher = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self._update_dagger_policy_selection()

    # 作用：rollout 时先生成 teacher 监督信号，再输出真正送入环境的 student 动作。
    def act(self, student_encoder_obs, policy_obs, teacher_encoder_obs):
        """
        根据 DAgger 设置，使用 student 或 teacher encoder 对应的策略来产生动作。

        参数：
            student_encoder_obs: student encoder 观测（有限观测，供 student encoder 使用）
            policy_obs: policy 观测（展平历史，供基础 actor MLP 使用）
            teacher_encoder_obs: teacher encoder 观测（完整特权观测，供 teacher encoder 使用）

        返回：
            按当前 DAgger beta 规则，从 student 或 teacher 策略中得到的动作
        """
        # ============================================================
        # 计算 teacher 输出（update 一定会用到，DAgger 模式下 rollout 也可能使用）
        # ============================================================
        with torch.no_grad():
            teacher_actions, teacher_latent = self.policy.get_teacher_action_mean(
                teacher_encoder_obs, policy_obs, return_latent=True
            )
        
        if self.use_dagger and self.dagger_beta > 0 and self.using_teacher is not None:
            # DAgger 模式：混合使用 teacher 和 student 动作
            # teacher 和 student 都使用确定性动作（均值）
            student_actions = self.policy.act_inference(
                student_encoder_obs, policy_obs, return_latent=False
            ).detach()
            
            # 按每个环境当前分配到的策略来选择动作
            # using_teacher 为 True 的环境使用 teacher，否则使用 student
            actions = torch.where(
                self.using_teacher.unsqueeze(-1),  # 形状：[num_envs, 1]
                teacher_actions,  # 使用 teacher 动作
                student_actions   # 使用 student 动作
            )
        else:
            # 纯 student 模式（未启用 DAgger，或 beta=0）
            # 使用确定性动作（均值），与蒸馏损失的定义保持一致
            actions = self.policy.act_inference(
                student_encoder_obs, policy_obs, return_latent=False
            ).detach()
        
        # 记录蒸馏所需的观测与动作
        # encoder_observations 保存 student encoder 观测
        self.transition.actions = actions  # type: ignore
        self.transition.encoder_observations = student_encoder_obs
        # policy_observations 保存 policy 观测（供 actor body 使用）
        self.transition.policy_observations = policy_obs
        # privileged_observations 保存 teacher encoder 观测
        self.transition.privileged_observations = teacher_encoder_obs
        # 缓存 teacher 输出，避免在 update 时重复计算
        # 创建新的张量，避免 "inference tensor" 相关问题
        self.transition.teacher_latent = teacher_latent.clone().detach()
        self.transition.teacher_action_mean = teacher_actions.clone().detach()
        
        return actions

    # 作用：把一步环境交互结果和缓存的 teacher 标签写入存储。
    def process_env_step(self, rewards, dones, infos):
        """处理一次环境交互结果，并将 transition 写入存储。"""
        # 记录奖励和 done 标记
        self.transition.rewards = rewards
        self.transition.dones = dones
        # 写入当前 transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        
        # 对刚结束回合的环境更新 DAgger 策略选择
        if self.use_dagger and dones.any():
            self._update_dagger_policy_selection(dones)

    # 作用：从存储中取 batch，计算 latent/action distillation loss，并只更新 student encoder。
    def update(self):
        """
        使用蒸馏损失更新 student encoder。

        这里会计算两项损失：
        1. 表征损失：student 与 teacher encoder latent 之间的 MSE
        2. 动作损失：student 与 teacher 确定性动作均值之间的 MSE

        只有 student_encoder 参数会被更新。
        """
        self.num_updates += 1
        mean_embedding_loss = 0
        mean_action_loss = 0
        mean_total_loss = 0
        num_batches = 0

        # 对所有 transition 迭代多个 epoch 进行训练
        for epoch in range(self.num_learning_epochs):
            # 使用蒸馏专用的数据生成器
            # 返回：
            # (encoder_obs, privileged_obs, policy_obs, actions, dones, teacher_latent, teacher_action_mean)
            # - encoder_obs: student encoder 观测 [num_envs, obs_dim]
            # - privileged_obs: teacher encoder 观测 [num_envs, obs_dim]（损失中不会直接使用）
            # - policy_obs: actor body 使用的 policy 观测 [num_envs, policy_obs_dim]
            # - actions: rollout 中 student 执行过的动作（损失中不会直接使用）
            # - dones: 回合终止标记（不会直接使用）
            # - teacher_latent: 缓存的 teacher encoder latent（若未缓存则为 None）
            # - teacher_action_mean: 缓存的 teacher 动作均值（若未缓存则为 None）
            for (
                student_encoder_obs_batch,
                _,  # teacher_encoder_obs（不需要，teacher 输出已缓存）
                policy_obs_batch,
                _,  # actions
                _,  # dones
                cached_teacher_latent,
                cached_teacher_action_mean
            ) in self.storage.generator():
                
                num_batches += 1
                
                # ============================================================
                # 1. 计算 student 输出，并取出 teacher 输出
                # ============================================================
                # Student：一次前向同时得到动作均值和 latent（保留梯度）
                student_action_mean, student_latent = self.policy.act_inference(
                    student_encoder_obs_batch, policy_obs_batch, return_latent=True
                )
                
                # 克隆缓存的 teacher 输出，避免直接使用 inference tensor
                teacher_latent = cached_teacher_latent.clone()
                teacher_action_mean = cached_teacher_action_mean.clone()
                
                # ============================================================
                # 2. 表征蒸馏损失
                # ============================================================
                embedding_loss = self.loss_fn(student_latent, teacher_latent)

                # ============================================================
                # 3. 动作蒸馏损失（确定性动作之间的 MSE）
                # ============================================================
                # 直接计算 student 与 teacher 动作均值之间的 MSE
                # 这会鼓励 student 产生与 teacher 一致的确定性动作
                action_loss = self.loss_fn(student_action_mean, teacher_action_mean)

                # ============================================================
                # 4. 总损失（加权组合）
                # ============================================================
                total_loss = (
                    self.embedding_loss_coef * embedding_loss +
                    self.action_loss_coef * action_loss
                )
                
                # ============================================================
                # 5. 梯度更新（只更新 student_encoder）
                # ============================================================
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 多卡梯度同步
                if self.is_multi_gpu:
                    self.reduce_parameters()
                
                # 梯度裁剪（使用预先缓存的参数列表）
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.student_encoder_params,
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # 累加损失，供日志记录使用
                mean_embedding_loss += embedding_loss.item()
                mean_action_loss += action_loss.item()
                mean_total_loss += total_loss.item()

        # 对所有 batch 的损失求平均
        mean_embedding_loss /= num_batches
        mean_action_loss /= num_batches
        mean_total_loss /= num_batches
        
        # 清空存储
        self.storage.clear()

        # 构造日志用的损失字典
        loss_dict = {
            "embedding_loss": mean_embedding_loss,
            "action_loss": mean_action_loss,
            "total_loss": mean_total_loss,
        }

        return loss_dict

    """
    DAgger 辅助函数
    """

    # 作用：更新 DAgger 中 teacher 动作被采用的概率。
    def _update_dagger_beta(self):
        """根据衰减计划更新 DAgger 的 beta。"""
        if not self.use_dagger:
            return
        
        if self.dagger_schedule_type == "linear":
            # 线性衰减：β = β_start - (β_start - β_end) * (step / decay_steps)
            progress = min(self.dagger_step / self.dagger_decay_steps, 1.0)
            self.dagger_beta = self.dagger_beta_start - (self.dagger_beta_start - self.dagger_beta_end) * progress
        
        elif self.dagger_schedule_type == "exponential":
            # 指数衰减：β = β_end + (β_start - β_end) * exp(-k * step)
            # 其中 k 的取值保证在 decay_steps 附近时 beta 接近 β_end
            k = 5.0 / self.dagger_decay_steps  # 衰减常数
            self.dagger_beta = self.dagger_beta_end + (self.dagger_beta_start - self.dagger_beta_end) * \
                torch.exp(torch.tensor(-k * self.dagger_step)).item()
        
        else:
            raise ValueError(f"未知的 DAgger 调度类型：{self.dagger_schedule_type}")
        
        # 将 beta 限制在 [0, 1] 范围内
        self.dagger_beta = max(0.0, min(1.0, self.dagger_beta))

    # 作用：为每个环境决定本轮使用 teacher 还是 student 动作。
    def _update_dagger_policy_selection(self, dones=None):
        """
        更新哪些环境在当前回合使用 teacher，哪些环境使用 student。

        该函数会在以下时机调用：
        1. 初始化时（dones=None）：为所有环境分配策略来源
        2. 回合结束后（传入 dones）：只为终止的环境重新分配

        参数：
            dones: 布尔张量 [num_envs]，表示哪些环境刚刚终止。
                   若为 None，则更新所有环境。
        """
        if not self.use_dagger or self.using_teacher is None:
            return
        
        if dones is None:
            # 初始化所有环境的 teacher/student 选择
            num_envs = self.using_teacher.shape[0]
            random_vals = torch.rand(num_envs, device=self.device)
            self.using_teacher[:] = random_vals < self.dagger_beta
        else:
            # 只更新已终止的环境
            envs_to_update = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()
            num_to_update = int(envs_to_update.sum().item())
            
            if num_to_update > 0:
                # 为需要更新的环境生成随机数
                random_vals = torch.rand(num_to_update, device=self.device)
                # 随机值小于 beta 的环境分配给 teacher
                use_teacher = random_vals < self.dagger_beta
                # 仅更新这些已终止的环境
                self.using_teacher[envs_to_update] = use_teacher

    # 作用：推进 DAgger 的调度步数。
    def increment_dagger_step(self):
        """
        推进一次 DAgger 的步数计数，并更新 beta。

        每个学习迭代调用一次，用于驱动 beta 衰减计划。
        """
        if not self.use_dagger:
            return
        
        self.dagger_step += 1
        self._update_dagger_beta()

    # 作用：返回当前 DAgger 的统计信息，便于记录日志。
    def get_dagger_stats(self):
        """
        获取当前 DAgger 的统计信息，供日志记录使用。

        返回：
            dict: 包含 beta 和 teacher 使用比例等统计信息
        """
        if not self.use_dagger or self.using_teacher is None:
            return {}
        
        teacher_ratio = self.using_teacher.float().mean().item()
        return {
            "dagger_beta": self.dagger_beta,
            "dagger_teacher_ratio": teacher_ratio,
            "dagger_step": self.dagger_step,
        }

    """
    辅助函数
    """

    # 作用：多卡训练时把参数广播到各个进程。
    def broadcast_parameters(self):
        """将模型参数广播到所有 GPU。"""
        # 获取当前 GPU 上的模型参数
        model_params = [self.policy.state_dict()]
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 在所有 GPU 上加载来自源 GPU 的模型参数
        self.policy.load_state_dict(model_params[0])

    # 作用：多卡训练时同步或归并参数更新。
    def reduce_parameters(self):
        """收集所有 GPU 的梯度并求平均。

        该函数会在 backward 之后调用，用于在多张 GPU 之间同步梯度。
        """
        # 将所有梯度拼成一个张量进行处理
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # 对所有 GPU 的梯度求平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # 用归并后的梯度回写到各参数上
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据回参数梯度
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数对应的偏移量
                offset += numel
