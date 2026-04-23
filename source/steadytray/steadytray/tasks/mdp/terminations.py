from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 作用：判断指定链接高度是否低于最小阈值。
def link_height_below_minimum(
    env: ManagerBasedRLEnv, 
    minimum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """判断指定链接高度是否低于最小阈值。"""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    if isinstance(asset, RigidObject):
        body_pos_w = asset.data.body_pos_w[:, 0, 2]
    elif isinstance(asset, Articulation):
        body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]

    height_violation = body_pos_w < minimum_height
    
    return height_violation
    
