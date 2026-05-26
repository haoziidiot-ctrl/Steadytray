<div align="center">

<img src="logo.png" alt="SteadyTray" width="400">

### SteadyTray


</div>

仓库包含训练和 sim2sim 部署代码，sim2real代码以及Apriltag的相机识别部署都在[Robomimic](https://github.com/haoziidiot-ctrl/RoboMimic_Deploy)

## 概述

该项目为在 Unitree 机器人上训练稳态托盘任务提供了强化学习环境，Unitree 机器人基于 IsaacLab 的自定义分支构建。训练流水线通过 RSL-RL 使用 PPO，并支持多 GPU 分布式训练。训练好的策略可以在 MuJoCo 中部署用于 sim2sim 验证。

## 安装

该项目依赖于 IsaacLab 的自定义分支，通过conda安装。

### 1. 克隆仓库

```bash
# Clone SteadyTray
git clone https://github.com/haoziidiot-ctrl/Steadytray.git

# 安装 git-lfs 和 tmux
apt-get update && apt-get install -y git-lfs tmux
git lfs install
```

### 2. 创建 conda 环境并安装 IsaacSim / PyTorch

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

python -m pip install --upgrade pip

# PyTorch 2.7 + CUDA 12.8
python -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 4.5（一次性拉全套子包）
python -m pip install "isaacsim[all,extscache]==4.5.0" \
    --extra-index-url https://pypi.nvidia.com
```

### 3. 安装 IsaacLab（项目用的是自定义分支）

```bash
cd /root/autodl-tmp/IsaacLab
./isaaclab.sh --install all
```

### 4. 安装训练依赖（RSL-RL + wandb）

```bash
python -m pip install rsl-rl-lib==2.3.3 wandb
```

### 5. 安装 SteadyTray

```bash
cd /root/autodl-tmp/steadytray
python -m pip install -e source/steadytray
```

### 6. 生成 X2 USD（X2 任务首次必跑；G1 任务可跳过）

```bash
cd /root/autodl-tmp/steadytray
python scripts/asset_tools/convert_x2_urdf.py \
    deploy/deploy_mujoco/x2t2d5_description_0525/x2_29dof_hand_simple_collision.urdf \
    source/steadytray/steadytray/assets/usds/x2_29dof_hand.usd \
    --headless
```

### 7. wandb 登录

```bash
wandb login   # 粘 API key；或 export WANDB_API_KEY=xxx
```

## 训练

培训流程由四个连续阶段组成。第 2 至 4 关每个关卡都需要加载上一关的检查点。

| Stage | Task | Description | Requires Pretrained Model |
|---|---|---|---|
| 1 | `G1-Steady-Tray-Pre-Locomotion` | 基础移动时上半身冻结，以加快训练速度 | No |
| 2 | `G1-Steady-Tray` | 通过托盘奖励微调移动 | Yes (Stage 1) |
| 3 | `G1-Steady-Object` | 托盘上物体稳定的残留教师 | Yes (Stage 2) |
| 4 | `G1-Steady-Object-Distillation` | 将特权教师简化为可部署的学生政策 | Yes (Stage 3) |

### Stage 1: Pre-train Locomotion

训练基础全身运动策略（上半身冻结）：

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Tray-Pre-Locomotion \
    --num_envs 4096 \
    --headless \
    --run_name "pretrain_loco" \
    --max_iterations 10000 \
    --logger wandb \
    --log_project_name SteadyTray_stage1
```

### Stage 2: Tray-Holding Fine-tune

通过针对托盘的奖励微调移动政策。加载第一阶段检查点：

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Tray \
    --num_envs 4096 \
	--headless \
	--resume \
    --load_run from_stage1 \
    --checkpoint <stage1_run_dir> \
    --run_name "tray_finetune" \
    --max_iterations 10000 \
    --logger wandb \
    --log_project_name SteadyTray_stage2
```

### Stage 3: Residual Object Stabilization (Teacher)

训练残留模块以稳定托盘上的物体。加载第二阶段检查点：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python scripts/rsl_rl/train.py \
  --task G1-Steady-Object \
  --num_envs 2048 \
  --headless \
  --resume \
  --load_run from_stage2 \
  --checkpoint <stage2_run_dir> \
  --run_name residual_teacher \
  --max_iterations 25000 \
  --logger wandb \
  --log_project_name SteadyTray_stage3 \
  agent.policy.class_name=AdaptedActorCritic
```

### Stage 4: Distillation (Student)

将特权教师提炼成可部署的学生政策。加载第三阶段检查点：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python scripts/rsl_rl/train.py \
  --task G1-Steady-Object-Distillation \
  --num_envs 2048 \
  --headless \
  --resume \
  --load_run from_stage3 \
  --checkpoint <stage3_run_dir> \
  --run_name distillation \
  --max_iterations 20000 \
  --logger wandb \
  --log_project_name SteadyTray_stage4
```

### 多 GPU 分布式训练

任何阶段都可以运行多 GPU 分布式训练：

```bash
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/rsl_rl/train.py \
    --task <TASK_NAME> \
    --num_envs 4096 \
    --headless \
    --distributed \
    --run_name "my_run" \
    --max_iterations 10000
```

可视化checkpoint

```bash
RUN=<model_dir>
python scripts/rsl_rl/play.py \
  --task G1-Steady-Object \
  --checkpoint "$RUN" \
  --num_envs 1 \
  --video \
  --video_seconds 30 \
  --video_compat \
  --headless \
  policy.class_name=ResidualActorCritic
```
### 推理

模型 /stage4_policy_14800.pt 中提供了预训练的学生模型。用 IsaacSim 来可视化：

```bash
conda activate g1_deploy
python deploy/deploy_mujoco/deploy_mujoco.py \
  --policy exported_23dof/stage4_policy_14800.pt \
  --config deploy/configs/g1_23dof_walk.yaml \
  --print_cmd
```

## Sim2Sim 部署 (MuJoCo)

提供轻量级 MuJoCo 部署，用于可视化训练好的策略，无需 Isaac Sim。完整说明请参见 [deploy/deploy_mujoco/README.md](deploy/deploy_mujoco/README.md) 。

快速入门：

```bash
cd deploy/deploy_mujoco
conda env create -f environment.yml
conda activate g1_deploy
```

## 把checkpoint导出成可以部署的pt文件

```bash
python deploy/scripts/batch_processing.py \
  --input_path <checkpoint_dir> \
  --output_path exported_23dof \
  --obs_dim 390 \
  --action_dim 23
```

## Sim2real 部署见[Robomimic](https://github.com/haoziidiot-ctrl/RoboMimic_Deploy)