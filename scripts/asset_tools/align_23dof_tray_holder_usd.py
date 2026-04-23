#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[2]
USD_PATH = REPO_ROOT / 'source/steadytray/steadytray/assets/usds/g1_23dof_side_tray_holder.usd'
REF_USD_PATH = REPO_ROOT / 'source/steadytray/steadytray/assets/usds/g1_side_tray_holder.usd'
ROBOT_PRIM_23 = '/g1_23dof_rev_1_0'
ROBOT_PRIM_29 = '/g1_29dof_rev_1_0'
NODE_ORIENT = Gf.Quatf(0.5, Gf.Vec3f(-0.5, 0.5, 0.5))
UNIT_CUBE_POINTS = [
    Gf.Vec3f(-0.5, -0.5, -0.5),
    Gf.Vec3f(0.5, -0.5, -0.5),
    Gf.Vec3f(0.5, 0.5, -0.5),
    Gf.Vec3f(-0.5, 0.5, -0.5),
    Gf.Vec3f(-0.5, -0.5, 0.5),
    Gf.Vec3f(0.5, -0.5, 0.5),
    Gf.Vec3f(0.5, 0.5, 0.5),
    Gf.Vec3f(-0.5, 0.5, 0.5),
]
UNIT_CUBE_FACE_COUNTS = [4, 4, 4, 4, 4, 4]
UNIT_CUBE_FACE_INDICES = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
]
UNIT_CUBE_EXTENT = [Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]
HOLDER_CUBES = {
    'left_tray_holder_link': [
        ('Cube', Gf.Vec3d(-0.041, -0.02, -0.02), Gf.Vec3d(0.005, 0.2, 0.12)),
        ('Cube_01', Gf.Vec3d(-0.055, -0.02, 0.037), Gf.Vec3d(0.03, 0.2, 0.005)),
        ('Cube_02', Gf.Vec3d(-0.054, 0.077, 0.02), Gf.Vec3d(0.03, 0.005, 0.04)),
    ],
    'right_tray_holder_link': [
        ('Cube', Gf.Vec3d(-0.041, -0.02, 0.02), Gf.Vec3d(0.005, 0.2, 0.12)),
        ('Cube_01', Gf.Vec3d(-0.055, -0.02, -0.037), Gf.Vec3d(0.03, 0.2, 0.005)),
        ('Cube_02', Gf.Vec3d(-0.054, 0.077, -0.02), Gf.Vec3d(0.03, 0.005, 0.04)),
    ],
}
TRAY_HOLDER_LOCAL = {
    'left_tray_holder_joint': (Gf.Vec3f(0.1254958, -0.0000038, -0.0000005), Gf.Quatf(1.0, Gf.Vec3f(-0.000030047358, -0.000027602875, 0.00009569917))),
    'right_tray_holder_joint': (Gf.Vec3f(0.1254958, 0.0000038, -0.0000005), Gf.Quatf(1.0, Gf.Vec3f(0.000030047364, -0.000027552876, -0.00009569917))),
}
REMOVED_PRIMS = [
    '/g1_23dof_rev_1_0/joints/left_wrist_pitch_joint',
    '/g1_23dof_rev_1_0/joints/left_wrist_yaw_joint',
    '/g1_23dof_rev_1_0/joints/right_wrist_pitch_joint',
    '/g1_23dof_rev_1_0/joints/right_wrist_yaw_joint',
    '/g1_23dof_rev_1_0/left_wrist_pitch_link',
    '/g1_23dof_rev_1_0/left_wrist_yaw_link',
    '/g1_23dof_rev_1_0/right_wrist_pitch_link',
    '/g1_23dof_rev_1_0/right_wrist_yaw_link',
]
XML_BOX_PREFIX = 'xml_collision_box_'


def _set_collision_enabled(stage: Usd.Stage, prim_path: str, enabled: bool) -> None:
    prim = stage.OverridePrim(prim_path)
    collision_api = UsdPhysics.CollisionAPI(prim)
    if not collision_api:
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(enabled)


def _remove_if_exists(stage: Usd.Stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim.GetPath())


def _deactivate_prim(stage: Usd.Stage, prim_path: str) -> None:
    prim = stage.OverridePrim(prim_path)
    prim.SetActive(False)


def _ensure_node(stage: Usd.Stage, holder_prim_path: str) -> None:
    node = UsdGeom.Xform.Define(stage, f'{holder_prim_path}/node_')
    xformable = UsdGeom.Xformable(node.GetPrim())
    if not xformable.GetOrderedXformOps():
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xformable.AddOrientOp().Set(NODE_ORIENT)
        xformable.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _define_cube_mesh(stage: Usd.Stage, prim_path: str, translate: Gf.Vec3d, scale: Gf.Vec3d) -> None:
    _remove_if_exists(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(UNIT_CUBE_POINTS)
    mesh.CreateFaceVertexCountsAttr(UNIT_CUBE_FACE_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(UNIT_CUBE_FACE_INDICES)
    mesh.CreateExtentAttr(UNIT_CUBE_EXTENT)
    mesh.CreateSubdivisionSchemeAttr('none')

    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.AddTranslateOp().Set(translate)
    xformable.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    xformable.AddScaleOp().Set(scale)

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision_api.CreateApproximationAttr('convexHull')


def _rewire_tray_holder_joint(stage: Usd.Stage, joint_name: str, wrist_name: str) -> None:
    joint = stage.GetPrimAtPath(f'{ROBOT_PRIM_23}/joints/{joint_name}')
    rel = joint.GetRelationship('physics:body0')
    rel.SetTargets([Sdf.Path(f'{ROBOT_PRIM_23}/{wrist_name}')])
    local_pos, local_rot = TRAY_HOLDER_LOCAL[joint_name]
    joint.GetAttribute('physics:localPos0').Set(local_pos)
    joint.GetAttribute('physics:localRot0').Set(local_rot)


def _copy_spec(src_stage: Usd.Stage, src_path: str, dst_stage: Usd.Stage, dst_path: str) -> None:
    Sdf.CopySpec(src_stage.GetRootLayer(), Sdf.Path(src_path), dst_stage.GetRootLayer(), Sdf.Path(dst_path))


def _align_holder_subtree_from_29(stage23: Usd.Stage, stage29: Usd.Stage, holder_name: str) -> None:
    holder_23 = f'{ROBOT_PRIM_23}/{holder_name}'
    holder_29 = f'{ROBOT_PRIM_29}/{holder_name}'

    _deactivate_prim(stage23, f'{holder_23}/visuals')
    _deactivate_prim(stage23, f'{holder_23}/collisions')
    _remove_if_exists(stage23, f'{holder_23}/Looks')
    _remove_if_exists(stage23, f'{holder_23}/node_/mesh_')

    _copy_spec(stage29, f'{holder_29}/Looks', stage23, f'{holder_23}/Looks')
    _copy_spec(stage29, f'{holder_29}/node_/mesh_', stage23, f'{holder_23}/node_/mesh_')

    mesh_prim = stage23.GetPrimAtPath(f'{holder_23}/node_/mesh_')
    mesh_prim.GetRelationship('material:binding').SetTargets([Sdf.Path(f'{holder_23}/Looks/DefaultMaterial')])


def main() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))
    stage.SetEditTarget(stage.GetRootLayer())
    ref_stage = Usd.Stage.Open(str(REF_USD_PATH))

    for prim_path in REMOVED_PRIMS:
        _deactivate_prim(stage, prim_path)

    _rewire_tray_holder_joint(stage, 'left_tray_holder_joint', 'left_wrist_roll_rubber_hand')
    _rewire_tray_holder_joint(stage, 'right_tray_holder_joint', 'right_wrist_roll_rubber_hand')

    for wrist in ('left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand'):
        _set_collision_enabled(stage, f'{ROBOT_PRIM_23}/{wrist}/collisions', False)

    for holder_name, cubes in HOLDER_CUBES.items():
        holder_prim_path = f'{ROBOT_PRIM_23}/{holder_name}'
        _ensure_node(stage, holder_prim_path)
        for i in range(1, 5):
            _remove_if_exists(stage, f'{holder_prim_path}/{XML_BOX_PREFIX}{i}')
        for name, translate, scale in cubes:
            _define_cube_mesh(stage, f'{holder_prim_path}/node_/{name}', translate, scale)
        _align_holder_subtree_from_29(stage, ref_stage, holder_name)

    stage.GetRootLayer().Save()


if __name__ == '__main__':
    main()
