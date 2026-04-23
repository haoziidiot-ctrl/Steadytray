# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 作用：Stage 4 的核心模型定义。这里把 teacher encoder、student encoder、共享 actor/adapter/critic 组装成一个 student-teacher 蒸馏模型。

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .encoder import GRUEncoder, TransformerEncoder, CausalTransformerEncoder
from .adapter import FiLMAdapter, ResidualActionAdapter
from .actor_critic import AdapterSequential


# 作用：定义 Stage 4 的 student-teacher 模型；共享 teacher 的 actor/adapter/critic，只训练 student encoder。
class AdaptedStudentTeacher(nn.Module):
    """
    用于在线编码器蒸馏的 Student-Teacher 架构。

    同时支持 FiLM adapter 和 residual action adapter 两种形式。

    架构概览：
    ----------------------
    Teacher（冻结）：带有已训练编码器的完整 adapted actor-critic
        - 冻结的基础 actor MLP（或带 adapter 的冻结 actor）
        - 冻结的 FiLM adapter（若使用 AdaptedActorCritic）
        - 冻结的 residual adapter（若使用 ResidualActorCritic）
        - 冻结的 GRU/Transformer encoder（从完整观测生成 teacher embedding）
        - 冻结的 critic

    Student（仅编码器可训练）：复用 teacher 的 actor 组件和 critic
        - 冻结的基础 actor MLP（或冻结 actor，与 teacher 共享）
        - 冻结的 adapter（与 teacher 共享）
        - 可训练的 student GRU/Transformer encoder（从有限观测中学习）
        - 冻结的 critic（与 teacher 共享）

    观测类型：
    ------------------
    1. student_encoder_obs：student encoder 的有限观测
       - 形状：[num_envs, seq_len, num_student_encoder_obs]
       - 示例：仅本体感觉信息
       - 用途：student encoder（可训练）

    2. teacher_encoder_obs：teacher encoder 的完整特权观测
       - 形状：[num_envs, seq_len, num_teacher_encoder_obs]
       - 示例：本体感觉 + 外部状态
       - 用途：teacher encoder（冻结，仅推理）

    3. policy_obs：供基础 actor MLP 使用的展平历史观测
       - 形状：[num_envs, num_actor_obs]
       - 示例：冻结基础策略所用的 5 步展平历史
       - 用途：actor body（冻结）

    4. critic_obs：critic 的特权观测
       - 形状：[num_envs, num_critic_obs]
       - 用途：critic MLP（冻结）

    蒸馏损失：
    -------------------
    1. 嵌入损失：student 与 teacher encoder 输出之间的 MSE
       - Loss = ||student_encoder(student_obs) - teacher_encoder(teacher_obs)||²
       - 约束 student 产生与 teacher 相似的 latent 表示

    2. 动作分布损失：student 与 teacher 动作分布之间的 KL 散度
       - Student action dist: N(μ_s, σ)，其中 μ_s = actor(policy_obs, student_latent)
       - Teacher action dist: N(μ_t, σ)，其中 μ_t = actor(policy_obs, teacher_latent)
       - Loss = KL(N(μ_t, σ) || N(μ_s, σ))
       - 约束 student 产生与 teacher 相似的动作分布

    训练策略：
    ------------------
    - 可训练：只有 student_encoder（GRU/Transformer）参数
    - 冻结：其余所有组件（actor 组件、adapter、critic、noise_std）
    - 共享：actor 组件、adapter 和 critic 在 student 与 teacher 之间共享
    - 在线：蒸馏发生在环境交互过程中，不需要离线数据集
    """
    is_recurrent = False

    # 作用：构建 student-teacher 网络骨架，并根据 adapter_type 选择 FiLM 或 residual 路线。
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_student_encoder_obs,  # student encoder 输入维度（例如仅本体感觉）
        num_teacher_encoder_obs,  # teacher encoder 输入维度（例如完整观测）
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # 通用 adapter 参数
        ctx_dim: int = 64,
        encoder_layers: int = 1,
        use_gate: bool = True,
        # 编码器结构选择
        encoder_type: str = "gru",
        num_heads: int = 4,
        encoder_dropout: float = 0.1,
        # adapter 类型选择
        adapter_type: str = "film",  # "film" or "residual"
        # FiLM adapter 专属参数
        adapter_hidden: int = 64,
        clamp_gamma: float = 2.0,
        # residual adapter 专属参数
        residual_hidden_dims: list[int] = [128, 64],
        clamp_residual: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "AdaptedStudentTeacher.__init__ 收到了未预期的参数，这些参数会被忽略："
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation = resolve_nn_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.ctx_dim = ctx_dim
        self.num_student_encoder_obs = num_student_encoder_obs
        self.num_teacher_encoder_obs = num_teacher_encoder_obs
        self.loaded_teacher = False  # 标记 teacher 是否已经成功加载
        self.adapter_type = adapter_type.lower()
        self.encoder_type = encoder_type.lower()

        # ============================================================
        # Actor：根据 adapter 类型构建动作网络
        # ============================================================
        
        if self.adapter_type == "film":
            # FiLM adapter 架构：冻结基础 MLP + FiLM adapter
            actor_body_layers = []
            actor_body_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_body_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims) - 1):
                actor_body_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_body_layers.append(activation)
            
            # 用 FiLM adapter 包裹 actor body 中的线性层
            adapted_body_layers = []
            for layer in actor_body_layers:
                if isinstance(layer, nn.Linear):
                    # 冻结基础线性层
                    for param in layer.parameters():
                        param.requires_grad_(False)
                    # 外层包裹 FiLM adapter（同样冻结）
                    adapter = FiLMAdapter(layer, ctx_dim, hidden=adapter_hidden, 
                                         clamp_gamma=clamp_gamma, use_gate=use_gate)
                    # 冻结 adapter 参数（与 teacher 共享）
                    for param in adapter.parameters():
                        param.requires_grad_(False)
                    adapted_body_layers.append(adapter)
                else:
                    adapted_body_layers.append(layer)
            
            self.actor_body = AdapterSequential(*adapted_body_layers)
            
            # Action head：冻结的线性层（共享）
            self.action_head = nn.Linear(actor_hidden_dims[-1], num_actions)
            for param in self.action_head.parameters():
                param.requires_grad_(False)
            
            self.residual_adapter = None
            
        elif self.adapter_type == "residual":
            # residual adapter 架构：冻结完整 actor + residual adapter
            actor_layers = []
            actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims) - 1):
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
            actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
            
            self.frozen_actor = nn.Sequential(*actor_layers)
            
            # 冻结 actor 的全部参数
            for param in self.frozen_actor.parameters():
                param.requires_grad_(False)
            
            # residual action adapter（冻结，与 teacher 共享）
            # 输入：encoder latent（ctx_dim）与本体观测（num_actor_obs）的拼接
            self.residual_adapter = ResidualActionAdapter(
                num_actions, ctx_dim, proprio_dim=num_actor_obs,
                hidden_dims=residual_hidden_dims,
                use_gate=use_gate, clamp_residual=clamp_residual
            )
            for param in self.residual_adapter.parameters():
                param.requires_grad_(False)
            
            self.actor_body = None
            self.action_head = None
            
        else:
            raise ValueError(f"Unknown adapter_type: {self.adapter_type}. Must be 'film' or 'residual'")
        
        # ============================================================
        # Encoders：Student（可训练） + Teacher（冻结）
        # ============================================================
        
        # Student encoder：可训练，使用有限观测
        if self.encoder_type == "gru":
            self.student_encoder = GRUEncoder(num_student_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif self.encoder_type == "transformer":
            self.student_encoder = TransformerEncoder(
                num_student_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif self.encoder_type == "causal_transformer":
            self.student_encoder = CausalTransformerEncoder(
                num_student_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Must be 'gru', 'transformer', or 'causal_transformer'")
        
        # Teacher encoder：冻结，使用完整特权观测
        if self.encoder_type == "gru":
            self.teacher_encoder = GRUEncoder(num_teacher_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif self.encoder_type == "transformer":
            self.teacher_encoder = TransformerEncoder(
                num_teacher_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif self.encoder_type == "causal_transformer":
            self.teacher_encoder = CausalTransformerEncoder(
                num_teacher_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        
        for param in self.teacher_encoder.parameters():
            param.requires_grad_(False)
        self.teacher_encoder.eval()
        
        # ============================================================
        # Critic：冻结的 MLP（共享）
        # ============================================================
        
        # Critic 输入：critic 观测与 encoder latent 向量的拼接
        critic_input_dim = num_critic_obs + ctx_dim
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # 冻结 critic（与 teacher 共享）
        for param in self.critic.parameters():
            param.requires_grad_(False)

        # 动作噪声
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            # 冻结噪声标准差（与 teacher 共享）
            self.std.requires_grad_(False)
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            self.log_std.requires_grad_(False)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（由 update_distribution 填充）
        self.distribution = None
        # 关闭参数校验以提升速度
        Normal.set_default_validate_args(False)

        print(f"\n{'='*70}")
        print(f"AdaptedStudentTeacher 架构初始化完成：")
        print(f"{'-'*70}")
        print(f"Adapter 类型: {self.adapter_type.upper()}")
        print(f"Encoder 类型: {self.encoder_type.upper()}")
        print(f"{'-'*70}")
        print(f"共享组件（冻结）：")
        if self.adapter_type == "film":
            print(f"  - Actor body: {actor_hidden_dims}，包含 {len([l for l in self.actor_body if isinstance(l, FiLMAdapter)])} 个 FiLM adapter")
            print(f"  - Action head: {num_actions} 维动作")
        elif self.adapter_type == "residual":
            print(f"  - Frozen actor: {actor_hidden_dims} -> {num_actions} 维动作")
            print(f"  - Residual adapter: {residual_hidden_dims}")
        print(f"  - Critic: {critic_hidden_dims}")
        print(f"{'-'*70}")
        print(f"Student Encoder（可训练）：")
        print(f"  输入: {num_student_encoder_obs}（有限观测）")
        print(f"  隐层: {ctx_dim}, 层数: {encoder_layers}")
        print(f"{'-'*70}")
        print(f"Teacher Encoder（冻结）：")
        print(f"  输入: {num_teacher_encoder_obs}（完整观测）")
        print(f"  隐层: {ctx_dim}, 层数: {encoder_layers}")
        print(f"{'='*70}\n")

    # 作用：兼容 runner 接口，重置隐藏状态；当前非递归用法里基本是空操作。
    def reset(self, dones=None, hidden_states=None):
        """为兼容 runner 接口而保留的 reset。由于当前不维护隐藏状态，这里为空操作。"""
        pass
    
    # 作用：兼容 recurrent 接口，返回当前隐藏状态。
    def get_hidden_states(self):
        """为兼容 recurrent 接口而保留，当前模型非递归，因此返回 None。"""
        return None

    # 作用：占位接口；当前训练不会直接调用统一的 forward，而是分别走 student/teacher 专用接口。
    def forward(self):
        raise NotImplementedError

    @property
    # 作用：返回当前动作分布的均值。
    def action_mean(self):
        if self.distribution is None:
            # 如果分布尚未更新，则返回一个占位张量
            # 这种情况会出现在 DAgger 中使用确定性动作（act_inference）时
            return torch.zeros(self.num_actions, device=next(self.parameters()).device)
        return self.distribution.mean

    @property
    # 作用：返回当前动作分布的标准差。
    def action_std(self):
        if self.distribution is None:
            # 如果分布尚未更新，则返回基础标准差参数
            if self.noise_std_type == "scalar":
                return self.std
            elif self.noise_std_type == "log":
                return torch.exp(self.log_std)
        return self.distribution.stddev

    @property
    # 作用：返回当前动作分布的熵，用于日志或兼容接口。
    def entropy(self):
        if self.distribution is None:
            # 如果分布尚未更新，则返回零熵
            return torch.zeros(1, device=next(self.parameters()).device)
        return self.distribution.entropy().sum(dim=-1)

    # 作用：用 student encoder 更新动作分布，供采样 student 动作使用。
    def update_distribution(self, student_encoder_obs, policy_obs):
        """
        使用 student encoder 更新动作分布。

        参数：
            student_encoder_obs: student encoder 的观测（有限观测，例如仅本体感觉）
            policy_obs: 供基础 actor MLP 使用的 policy 观测
        """
        # 从 student encoder 获取上下文 embedding
        e_t = self.get_student_encoder_latent(student_encoder_obs)
        
        # 根据 adapter 类型计算动作均值
        if self.adapter_type == "film":
            # FiLM：adapted actor body + 冻结的 action head
            h = self.actor_body(policy_obs, e_t)
            mean = self.action_head(h)
        elif self.adapter_type == "residual":
            # Residual：冻结 actor + 带本体上下文的 residual adapter
            base_actions = self.frozen_actor(policy_obs)
            mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        # 计算标准差
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # 构造动作分布
        self.distribution = Normal(mean, std)

    # 作用：按当前 student 动作分布采样动作。
    def act(self, student_encoder_obs, policy_obs):
        """
        使用 student encoder 从策略中采样动作。

        参数：
            student_encoder_obs: student encoder 观测
            policy_obs: policy 观测
        """
        self.update_distribution(student_encoder_obs, policy_obs)
        return self.distribution.sample()

    # 作用：用 student encoder 输出确定性动作；Stage 4 rollout 和 update 都会用到。
    def act_inference(self, student_encoder_obs, policy_obs, return_latent=False):
        """
        使用 student encoder 计算推理时的确定性动作（均值）。

        参数：
            student_encoder_obs: student encoder 观测
            policy_obs: policy 观测
            return_latent: 若为 True，则返回 (actions_mean, latent)；否则仅返回 actions_mean。

        返回：
            actions_mean: 确定性动作均值
            latent（可选）: encoder 的 latent 向量（仅当 return_latent=True 时返回）
        """
        # 从 student encoder 获取上下文 embedding
        e_t = self.get_student_encoder_latent(student_encoder_obs)
        
        # 根据 adapter 类型计算动作均值
        if self.adapter_type == "film":
            h = self.actor_body(policy_obs, e_t)
            actions_mean = self.action_head(h)
        elif self.adapter_type == "residual":
            base_actions = self.frozen_actor(policy_obs)
            actions_mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        if return_latent:
            return actions_mean, e_t
        else:
            return actions_mean

    # 作用：结合 student latent 计算 critic 值。
    def evaluate(self, critic_observations, student_encoder_obs=None):
        """
        使用 critic 观测与 student encoder latent 共同计算 critic value。

        参数：
            critic_observations: critic 的输入观测
            student_encoder_obs: student encoder 观测；若为 None，则退回使用 critic_observations
        """
        # 使用 student encoder 观测；若未提供则退回到 critic 观测
        encoder_obs = student_encoder_obs if student_encoder_obs is not None else critic_observations
        
        # 从 student encoder 获取上下文 embedding
        e_t = self.get_student_encoder_latent(encoder_obs)
        
        # 将 critic 观测与 encoder latent 拼接
        if critic_observations.dim() == 2:
            critic_input = torch.cat([critic_observations, e_t], dim=-1)
        else:
            raise ValueError(f"Expected critic observations with 2 dims, got {critic_observations.dim()}")
        
        # 用拼接后的输入计算 critic 值
        value = self.critic(critic_input)
        return value

    # 作用：把 student 的时序观测编码成 latent 表示。
    def get_student_encoder_latent(self, observations):
        """
        根据观测得到 student encoder 的 latent 向量。

        参数：
            observations: student 的时序观测
                         形状: [num_envs, seq_len, obs_dim] - 推荐的时序输入
                                [num_envs, obs_dim] - 单步观测（会扩展成 seq_len=1）

        返回：
            torch.Tensor: encoder latent 向量 [num_envs, ctx_dim]
        """
        # 同时兼容时序输入和单步输入
        if observations.dim() == 2:
            obs_seq = observations.unsqueeze(1)
        elif observations.dim() == 3:
            obs_seq = observations
        else:
            raise ValueError(
                f"Expected observations with 2 or 3 dimensions, got shape {observations.shape}"
            )
        
        # 转置成编码器期望的格式：[seq_len, num_envs, obs_dim]
        obs_seq_t = obs_seq.transpose(0, 1)
        
        # 从 student encoder 获取上下文 embedding
        e_t = self.student_encoder(obs_seq_t)
        
        return e_t

    # 作用：把 teacher 的全量观测编码成 teacher latent，仅推理使用。
    def get_teacher_encoder_latent(self, observations):
        """
        根据观测得到 teacher encoder 的 latent 向量（仅推理使用）。

        参数：
            observations: teacher 的时序观测（完整特权观测）
                         形状: [num_envs, seq_len, obs_dim] - 推荐的时序输入
                                [num_envs, obs_dim] - 单步观测（会扩展成 seq_len=1）

        返回：
            torch.Tensor: encoder latent 向量 [num_envs, ctx_dim]
        """
        # 同时兼容时序输入和单步输入
        if observations.dim() == 2:
            obs_seq = observations.unsqueeze(1)
        elif observations.dim() == 3:
            obs_seq = observations
        else:
            raise ValueError(
                f"Expected observations with 2 or 3 dimensions, got shape {observations.shape}"
            )
        
        # 转置成编码器期望的格式：[seq_len, num_envs, obs_dim]
        obs_seq_t = obs_seq.transpose(0, 1)
        
        # 从 teacher encoder 获取上下文 embedding（不求梯度）
        with torch.no_grad():
            e_t = self.teacher_encoder(obs_seq_t)
        
        return e_t

    # 作用：用 teacher encoder 和共享 actor 计算 teacher 的确定性动作。
    def get_teacher_action_mean(self, teacher_encoder_obs, policy_obs, return_latent=False):
        """
        计算 teacher 在蒸馏时使用的确定性动作均值（仅推理）。
        相比 get_teacher_action_distribution 更高效，因为它跳过了标准差计算。

        参数：
            teacher_encoder_obs: teacher encoder 观测（完整特权观测）
            policy_obs: 供基础 actor MLP 使用的 policy 观测
            return_latent: 若为 True，则返回 (action_mean, latent)；否则仅返回 action_mean。

        返回：
            action_mean: teacher 输出的确定性动作均值
            latent（可选）: teacher encoder 的 latent 向量（仅当 return_latent=True 时返回）
        """
        with torch.no_grad():
            # 从 teacher encoder 获取上下文 embedding
            e_t = self.get_teacher_encoder_latent(teacher_encoder_obs)
            
            # 根据 adapter 类型计算动作均值
            if self.adapter_type == "film":
                h = self.actor_body(policy_obs, e_t)
                mean = self.action_head(h)
            elif self.adapter_type == "residual":
                base_actions = self.frozen_actor(policy_obs)
                mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        if return_latent:
            return mean, e_t
        else:
            return mean

    # 作用：兼容加载 Stage 3 teacher checkpoint 或 Stage 4 继续训练 checkpoint。
    def load_state_dict(self, state_dict, strict=True):
        """加载 student 与 teacher 网络的参数。

        参数：
            state_dict (dict): 模型的状态字典。
            strict (bool): 是否严格要求 state_dict 中的键与当前模块 state_dict() 返回的键完全匹配。

        返回：
            bool: 当前加载是否属于“继续训练”。该标记会被 `OnPolicyRunner` 的 `load()` 用来判断后续参数该如何恢复。
        """
        # 判断 state_dict 是 teacher checkpoint 还是继续训练的 checkpoint
        has_student_encoder = any('student_encoder' in key for key in state_dict.keys())
        has_actor_body = any('actor_body' in key for key in state_dict.keys())
        has_frozen_actor = any('frozen_actor' in key for key in state_dict.keys())
        has_residual_adapter = any('residual_adapter' in key for key in state_dict.keys())
        has_history_encoder = any('history_encoder' in key for key in state_dict.keys())
        
        if has_student_encoder:
            # 这是 AdaptedStudentTeacher checkpoint，表示继续蒸馏训练
            print("[INFO] Loading AdaptedStudentTeacher checkpoint (resuming distillation)")
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            return True
            
        elif (has_actor_body or has_frozen_actor or has_residual_adapter) and has_history_encoder:
            # 这是 adapted actor-critic checkpoint（FiLM 或 Residual），需要作为 teacher 加载
            
            # 判断 checkpoint 类型
            if has_actor_body:
                checkpoint_type = "AdaptedActorCritic (FiLM)"
            elif has_frozen_actor and has_residual_adapter:
                checkpoint_type = "ResidualActorCritic"
            else:
                checkpoint_type = "未知 adapted 类型"
            
            print(f"[INFO] Loading {checkpoint_type} checkpoint as teacher")
            print("[INFO] 映射关系：history_encoder -> teacher_encoder")
            print("[INFO] 除 student_encoder 外，其余组件全部冻结")
            
            # 构造 teacher 参数加载时所需的映射字典
            teacher_state_dict = {}
            for key, value in state_dict.items():
                # 将 history_encoder 映射到 teacher_encoder
                if 'history_encoder' in key:
                    new_key = key.replace('history_encoder', 'teacher_encoder')
                    teacher_state_dict[new_key] = value
                # 根据类型加载 actor 相关组件
                elif any(prefix in key for prefix in ['actor_body', 'action_head', 'frozen_actor', 'residual_adapter', 'critic', 'std', 'log_std']):
                    teacher_state_dict[key] = value
            
            # 加载映射后的 state_dict
            missing_keys, unexpected_keys = super().load_state_dict(teacher_state_dict, strict=False)
            
            # 预期缺失的键：student_encoder 相关参数
            expected_missing = [k for k in missing_keys if 'student_encoder' in k]
            unexpected_missing = [k for k in missing_keys if 'student_encoder' not in k]
            
            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys (not student_encoder): {unexpected_missing[:5]}...")
            
            print(f"[INFO] Student encoder 参数数量: {len(expected_missing)}（随机初始化）")
            print(f"[INFO] Teacher 加载成功！")
            
            # 设置 teacher 已成功加载的标志
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            
            # 确保除 student_encoder 外其余参数全部冻结
            for name, param in self.named_parameters():
                if 'student_encoder' not in name:
                    param.requires_grad_(False)
            
            # 打印可训练参数摘要
            self.print_trainable_parameters()

            return False  # 不是继续训练，而是加载 teacher 后开始新的蒸馏训练
            
        else:
            raise ValueError(
                "state_dict 不包含有效的 AdaptedActorCritic 或 AdaptedStudentTeacher 参数。"
                "期望包含 'actor_body' + 'history_encoder'，或 'student_encoder' 相关键。"
            )

    # 作用：兼容 recurrent 接口，按需分离隐藏状态。
    def detach_hidden_states(self, dones=None):
        """为兼容 recurrent 接口而保留的方法。"""
        pass

    # 作用：打印当前哪些参数可训练，用来确认 Stage 4 只训练 student encoder。
    def print_trainable_parameters(self):
        """打印可训练参数与冻结参数的摘要。"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        param_groups = {
            'Student Encoder': [],
            'Teacher Encoder': [],
            'Actor Body': [],
            'Critic': [],
            'Other': []
        }
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if 'student_encoder' in name:
                    param_groups['Student Encoder'].append((name, num_params))
                else:
                    param_groups['Other'].append((name, num_params))
            else:
                frozen_params += num_params
                if 'teacher_encoder' in name:
                    param_groups['Teacher Encoder'].append((name, num_params))
                elif 'actor_body' in name or 'action_head' in name:
                    param_groups['Actor Body'].append((name, num_params))
                elif 'critic' in name:
                    param_groups['Critic'].append((name, num_params))
        
        print(f"\n{'='*70}")
        print(f"模型参数总量: {total_params:,}")
        print(f"  可训练: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  冻结: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"{'-'*70}")
        
        for group_name in ['Student Encoder', 'Teacher Encoder', 'Actor Body', 'Critic', 'Other']:
            params = param_groups[group_name]
            if params:
                group_total = sum(p[1] for p in params)
                if group_name == 'Student Encoder':
                    status = "可训练"
                else:
                    status = "冻结"
                print(f"{group_name} ({status}): {group_total:,}")
        print(f"{'='*70}\n")
