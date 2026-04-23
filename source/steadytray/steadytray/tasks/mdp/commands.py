from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_from_euler_xyz
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
# 作用：定义基于指定机体链接的速度命令配置。
class UniformVelocityBodyCommandCfg(UniformVelocityCommandCfg):
    """定义基于指定机体链接的速度命令配置。"""
    
    body_name: str = "torso_link"
    """用于速度命令跟踪的 body 名称，默认是 torso_link。"""

# 作用：使用指定 body 而不是 root 来生成和评估速度命令。
class UniformVelocityBodyCommand(UniformVelocityCommand):
    """使用指定 body 而不是 root 来生成和评估速度命令。"""
    
    cfg: UniformVelocityBodyCommandCfg
    """命令生成器对应的配置对象。"""
    
    # 作用：初始化命令生成器并缓存目标 body 的索引。
    def __init__(self, cfg: UniformVelocityBodyCommandCfg, env):
        """初始化命令生成器并缓存目标 body 的索引。"""
        super().__init__(cfg, env)
        
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
    
    # 作用：使用目标 body 的速度来更新命令跟踪指标。
    def _update_metrics(self):
        """使用目标 body 的速度来更新命令跟踪指标。"""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        body_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.body_idx, :3]
        body_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self.body_idx, :]

        body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
        
        body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
        
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - body_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - body_ang_vel_b[:, 2]) / max_command_step
        )
    
    # 作用：计算目标 body 在世界系下的航向角。
    def _compute_body_heading_w(self) -> torch.Tensor:
        """计算目标 body 在世界系下的航向角。"""
        
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        
        forward_vec_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        
        forward_w = math_utils.quat_apply(body_quat_w, forward_vec_b)
        
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])
    
    # 作用：根据 body 航向与站立约束更新速度命令。
    def _update_command(self):
        """根据 body 航向与站立约束更新速度命令。"""
      
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            body_heading_w = self._compute_body_heading_w()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - body_heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
    
    # 作用：更新调试可视化中的目标速度与当前速度箭头。
    def _debug_vis_callback(self, event):
        """更新调试可视化中的目标速度与当前速度箭头。"""
        if not self.robot.is_initialized:
            return
        
        body_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :].clone()
        body_pos_w[:, 2] += 0.5
        
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        body_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.body_idx, :3]

        body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
        
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2], body_quat_w
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            body_lin_vel_b[:, :2], body_quat_w
        )
        
        self.goal_vel_visualizer.visualize(body_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(body_pos_w, vel_arrow_quat, vel_arrow_scale)
    
    # 作用：把平面速度解析为箭头缩放和姿态。
    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, body_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把平面速度解析为箭头缩放和姿态。"""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = quat_mul(body_quat_w, arrow_quat)
        
        return arrow_scale, arrow_quat

@configclass
# 作用：为 body 速度命令补充命令上限范围。
class UniformLevelVelocityBodyCommandCfg(UniformVelocityBodyCommandCfg):
    """为 body 速度命令补充命令上限范围。"""
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

@configclass
# 作用：定义带初始静止延迟的速度命令配置。
class DelayedUniformVelocityCommandCfg(UniformLevelVelocityBodyCommandCfg):
    """定义带初始静止延迟的速度命令配置。"""
    
    delay_time: float = 1.0

# 作用：先保持零速度一段时间再启用采样命令。
class DelayedUniformVelocityCommand(UniformVelocityBodyCommand):
    """先保持零速度一段时间再启用采样命令。"""
    
    cfg: DelayedUniformVelocityCommandCfg
    """命令生成器对应的配置对象。"""
    
    # 作用：初始化延迟计时器和缓存命令。
    def __init__(self, cfg: DelayedUniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        """初始化延迟计时器和缓存命令。"""
        super().__init__(cfg, env)
        
        self.time_since_reset = torch.zeros(self.num_envs, device=self.device)
        
        self.sampled_commands = torch.zeros(self.num_envs, 3, device=self.device)
    
    # 作用：返回命令生成器的摘要字符串。
    def __str__(self) -> str:
        """返回命令生成器的摘要字符串。"""
        msg = "DelayedUniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tDelay time: {self.cfg.delay_time}s\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    
    # 作用：重采样命令并重置延迟阶段。
    def _resample_command(self, env_ids: Sequence[int]):
        """重采样命令并重置延迟阶段。"""
        self.time_since_reset[env_ids] = 0.0
        
        super()._resample_command(env_ids)
        
        self.sampled_commands[env_ids] = self.vel_command_b[env_ids].clone()
        
        self.vel_command_b[env_ids] = 0.0
    
    # 作用：根据延迟计时切换零命令与采样命令。
    def _update_command(self):
        """根据延迟计时切换零命令与采样命令。"""
        self.time_since_reset += self._env.step_dt
        
        delay_passed_mask = self.time_since_reset >= self.cfg.delay_time
        
        self.vel_command_b[delay_passed_mask] = self.sampled_commands[delay_passed_mask]
        
        self.vel_command_b[~delay_passed_mask] = 0.0
        
        super()._update_command()