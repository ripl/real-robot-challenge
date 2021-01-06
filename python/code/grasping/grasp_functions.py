from code.grasping.grasp_sampling import GraspSampler
from code.grasping.wholebody_planning import WholeBodyPlanner
from code.const import VIRTUAL_CUBOID_HALF_SIZE
from code.align_rotation import align_z, project_cube_xy_plane
from scipy.spatial.transform import Rotation as R
import copy
import numpy as np


def get_heuristic_grasp(env, pos, quat):
    sampler = GraspSampler(env, pos, quat)
    try:
        return sampler.get_heuristic_grasps().__next__()
    except StopIteration:
        return sampler()


def get_all_heuristic_grasps(env, pos, quat, avoid_edge_faces=True):
    return GraspSampler(
        env, pos, quat, avoid_edge_faces=avoid_edge_faces
    ).get_heuristic_grasps()


def sample_grasp(env, pos, quat):
    return GraspSampler(env, pos, quat)()


def sample_partial_grasp(env, pos, quat):
    return GraspSampler(env, pos, quat, allow_partial_sol=True)()


def get_planned_grasp(env, pos, quat, goal_pos, goal_quat, tight=False,
                      **kwargs):
    planner = WholeBodyPlanner(env)
    path = planner.plan(pos, quat, goal_pos, goal_quat, **kwargs)
    grasp = copy.deepcopy(path.grasp)
    if tight:
        path = path.tighten(env, path, coef=0.5)

    # save planned trajectory
    env.unwrapped.register_custom_log('wholebody_path', {'cube': path.cube, 'tip_path': path.tip_path})
    env.unwrapped.save_custom_logs()
    return grasp, path


def get_pitching_grasp(env, pos, quat, goal_quat):
    _, pitch_axis, pitch_angle = align_z(
        quat, project_cube_xy_plane(goal_quat)
    )
    x = np.asarray([VIRTUAL_CUBOID_HALF_SIZE[0], 0, 0])
    y = np.asarray([0, VIRTUAL_CUBOID_HALF_SIZE[1], 0])
    if pitch_angle > 0 and pitch_axis == 'x':
        cube_tip_positions = np.asarray([x,  y, -x])
    elif pitch_angle > 0 and pitch_axis == 'y':
        cube_tip_positions = np.asarray([y, -x, -y])
    elif pitch_angle < 0 and pitch_axis == 'x':
        cube_tip_positions = np.asarray([x, -y, -x])
    elif pitch_angle < 0 and pitch_axis == 'y':
        cube_tip_positions = np.asarray([y,  x, -y])
    else:
        return get_yawing_grasp(env, pos, quat, goal_quat)

    grasp_sampler = GraspSampler(env, pos, quat)
    return grasp_sampler.get_feasible_grasps_from_tips(
        grasp_sampler.T_cube_to_base(cube_tip_positions)
    ).__next__()


def get_yawing_grasp(env, pos, quat, goal_quat, step_angle=np.pi / 2):
    from code.utils import get_yaw_diff
    from code.const import COLLISION_TOLERANCE
    from scipy.spatial.transform import Rotation
    print("[get_yawing_grasp] step_angle:", step_angle * 180 / np.pi)
    angle = get_yaw_diff(quat, goal_quat)
    print('[get_yawing_grasp] get_yaw_diff:', angle * 180 / np.pi)
    angle_clip = np.clip(angle, -step_angle, step_angle)
    print('[get_yawing_grasp] clipped angle:', angle_clip * 180 / np.pi)
    goal_quat = (R.from_euler('Z', angle_clip) * R.from_quat(quat)).as_quat()
    planner = WholeBodyPlanner(env)
    try:
        path = planner.plan(pos, quat, pos, goal_quat, use_ori=True, avoid_edge_faces=True, yawing_grasp=True,
                            collision_tolerance=-COLLISION_TOLERANCE * 3, retry_grasp=0, direct_path=True)
    except RuntimeError as e:
        print(f'[get_yawing_grasp] wholebody planning failed for step_angle: {step_angle}')
        return None, None

    grasp = copy.deepcopy(path.grasp)
    # save planned trajectory
    env.unwrapped.register_custom_log('wholebody_path', {'cube': path.cube, 'tip_path': path.tip_path})
    env.unwrapped.save_custom_logs()
    return grasp, path
