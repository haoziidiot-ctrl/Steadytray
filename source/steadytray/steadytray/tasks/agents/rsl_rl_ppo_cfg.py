# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# 作用：集中定义 Stage 1 到 Stage 4 的训练配置。Stage 1 主要看 G1PPORunnerCfg，Stage 2 看 BasePPORunnerCfg，Stage 3/4 分别看 adapter 与 distillation runner 配置。

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationAlgorithmCfg

@configclass
# 作用：统一定义 adapter 类策略网络的结构超参数。
class RslRlAdaptedActorCriticCfg(RslRlPpoActorCriticCfg):
    """统一定义 adapter 类策略网络的结构超参数。"""
    
    class_name: str = "AdaptedActorCritic"
    """策略类名，可选 FiLM 版或残差版 adapter 策略。"""
    
    adapter_type: str = "film"
    """Adapter 类型，可选 FiLM 调制或残差动作修正。"""
    
    encoder_type: str = "gru"
    """时序 encoder 类型。"""
    
    ctx_dim: int = 64
    """历史 encoder 输出的上下文维度。"""
    encoder_layers: int = 1
    """Encoder 层数。"""
    use_gate: bool = False
    """是否在 adapter 中启用可学习门控。"""
    
    adapter_hidden: int = 64
    """FiLM 调制网络的隐藏层维度。"""
    clamp_gamma: float = 2.0
    """FiLM gamma 的截断范围。"""
    
    residual_hidden_dims: list[int] = [128, 64]
    """残差动作 MLP 的隐藏层维度。"""
    clamp_residual: float | None = None
    """残差动作的截断范围。"""
    
    num_heads: int = 4
    """Transformer encoder 的注意力头数。"""
    encoder_dropout: float = 0.0
    """Transformer encoder 的 dropout 比例。"""

@configclass
# 作用：定义 Stage 4 蒸馏算法的超参数。
class RslRlAdapterDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """定义 Stage 4 蒸馏算法的超参数。"""
    
    class_name: str = "Distillation"
    """算法类名。"""
    
    embedding_loss_coef: float = 1.0
    """表征蒸馏损失权重。"""
    action_loss_coef: float = 1.0
    """动作蒸馏损失权重。"""
    
    loss_type: str = "mse"
    """损失函数类型。"""
    
    max_grad_norm: float = 1.0
    """梯度裁剪的最大范数。"""
    
    use_dagger: bool = False
    """是否在蒸馏前期启用 DAgger。"""
    dagger_beta_start: float = 1.0
    """初始 teacher 动作使用概率。"""
    dagger_beta_end: float = 0.0
    """最终 teacher 动作使用概率。"""
    dagger_decay_steps: int = 1000
    """beta 从起始值衰减到终值所经历的迭代数。"""
    dagger_schedule_type: str = "linear"
    """beta 衰减方式。"""

@configclass
# 作用：在 PPO 配置上扩展上肢 curriculum 字段。
class RslRlPpoAlgorithmCurriculumCfg(RslRlPpoAlgorithmCfg):
    upper_body_cur_cfg: dict | None = None
    """上肢 curriculum 配置，为空时表示关闭。"""

@configclass
# 作用：提供基础 PPO runner 的默认训练超参数。
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # 与 task 名保持一致
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    clip_actions = 100.0  # 动作裁剪阈值

@configclass
# 作用：定义 Stage 1 预训练 locomotion 的 runner 配置。
class G1PPORunnerCfg(BasePPORunnerCfg):
    """定义 Stage 1 预训练 locomotion 的 runner 配置。"""

    algorithm = RslRlPpoAlgorithmCurriculumCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        upper_body_cur_cfg={
            "coeff": 1,
            
            "joint_indices": [5, 6, 9, 10, 13, 14, 17, 18, 21, 22],
            
            "use_curriculum": False,
            
            "curriculum_threshold": 1.0,
        },
    )

@configclass
# 作用：定义 Stage 3 adapter 强化学习训练配置。
class G1AdapterSteadyTrayRunnerCfg(RslRlOnPolicyRunnerCfg):
    """定义 Stage 3 adapter 强化学习训练配置。"""
    
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    
    policy = RslRlAdaptedActorCriticCfg(
        class_name="AdaptedActorCritic",  # FiLM 版使用 AdaptedActorCritic，残差版使用 ResidualActorCritic
        init_noise_std=0.3,
        
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        encoder_type="transformer",
        ctx_dim=64,             # encoder 输出的上下文维度
        encoder_layers=2,        # Transformer 层数
        num_heads=4,             # 注意力头数，需要整除 ctx_dim
        encoder_dropout=0.0,     # 为了稳定训练，这里不使用 dropout

        use_gate=False,           # 是否启用可学习门控参数 alpha
        
        residual_hidden_dims=[512, 256, 128],  # 残差动作 MLP 结构
        clamp_residual=None,          # 残差动作截断范围

        clamp_gamma=3.0,         # 放宽 gamma 截断范围以允许更强调制
        adapter_hidden=128,      # 增大隐藏层以提升表达能力
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    clip_actions = 100.0

@configclass
# 作用：定义 X2 stage1（站立托盘）训练 runner 配置。无 base actor 加载，从零训练 actor + Transformer encoder + critic。
class X2TrayStage1RunnerCfg(RslRlOnPolicyRunnerCfg):
    """X2 stage1 训练 runner：从零训练 actor，encoder 处理 proprio + 托盘观测时序。"""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "x2_steady_tray_stage1"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "steadytray_x2_stage1"

    policy = RslRlAdaptedActorCriticCfg(
        class_name="AdaptedActorCritic",
        adapter_type="film",  # 无 base policy 加载时 FiLM α 从 0 起步，不影响 actor 主干
        init_noise_std=1.0,

        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        encoder_type="transformer",
        ctx_dim=64,
        encoder_layers=2,
        num_heads=4,
        encoder_dropout=0.0,

        use_gate=False,
        residual_hidden_dims=[512, 256, 128],
        clamp_residual=None,
        clamp_gamma=3.0,
        adapter_hidden=128,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    clip_actions = 100.0

@configclass
# 作用：定义 Stage 4 student-teacher 蒸馏训练配置。
class G1AdapterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """定义 Stage 4 student-teacher 蒸馏训练配置。"""
    
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    
    policy = RslRlAdaptedActorCriticCfg(
        class_name="AdaptedStudentTeacher",
        adapter_type="film",  # 与默认 Stage 3 teacher 的架构保持一致
        init_noise_std=1e-3,
        
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        encoder_type="transformer",
        ctx_dim=64,              # encoder 输出的上下文维度
        encoder_layers=2,        # Transformer 层数
        num_heads=4,             # 注意力头数，需要整除 ctx_dim
        encoder_dropout=0.0,     # 为了稳定训练，这里不使用 dropout

        use_gate=False,           # 是否启用可学习门控参数 alpha

        residual_hidden_dims=[512, 256, 128],  # 残差动作 MLP 结构
        clamp_residual=None,          # 残差动作截断范围

        clamp_gamma=3.0,         # 放宽 gamma 截断范围以允许更强调制
        adapter_hidden=128,      # 增大隐藏层以提升表达能力
    )
    
    algorithm = RslRlAdapterDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=1,
        learning_rate=5.0e-5,
        gradient_length=1,
        max_grad_norm=0.5,
        embedding_loss_coef=1.0,
        action_loss_coef=1.0,
        loss_type="mse",
        use_dagger=False,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_decay_steps=500,
        dagger_schedule_type="linear",
    )
    
    clip_actions = 100.0

