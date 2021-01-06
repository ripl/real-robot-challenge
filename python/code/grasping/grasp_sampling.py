#!/usr/bin/env python3
from code.utils import Transform, keep_state
from code.const import MU, VIRTUAL_CUBOID_HALF_SIZE, INIT_JOINT_CONF
from .ik import IKUtils
from .force_closure import CuboidForceClosureTest, CoulombFriction
import itertools
import numpy as np


class Grasp(object):
    def __init__(self, cube_tip_pos, base_tip_pos, q, cube_pos, cube_quat,
                 T_cube_to_base, T_base_to_cube, valid_tips):
        self.cube_tip_pos = cube_tip_pos
        self.base_tip_pos = base_tip_pos
        self.q = q
        self.pos = cube_pos
        self.quat = cube_quat
        self.T_cube_to_base = T_cube_to_base
        self.T_base_to_cube = T_base_to_cube
        self.valid_tips = valid_tips

    def update(self, cube_pos, cube_quat):
        self.pos = cube_pos
        self.quat = cube_quat
        self.T_cube_to_base = Transform(self.pos, self.quat)
        self.T_base_to_cube = self.T_cube_to_base.inverse()
        self.base_tip_pos = self.T_cube_to_base(self.cube_tip_pos)


def sample(ax, sign, half_size, shrink_region=[0.0, 0.6, 0.0]):
    point = np.empty(3)
    for i in range(3):
        if i == ax:
            point[ax] = sign * half_size[ax]
        else:
            point[i] = np.random.uniform(-half_size[i] * shrink_region[i],
                                         half_size[i] * shrink_region[i])
    return point


def sample_side_face(n, half_size, object_ori, shrink_region=[0.0, 0.6, 0.0]):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    sample_ax = np.array([i for i in range(3) if i != axis])
    points = np.stack([
        sample(np.random.choice(sample_ax), np.random.choice([-1, 1]),
               half_size, shrink_region)
        for _ in range(n)
    ])
    return points


def sample_long_side_face(n, half_size, object_ori, shrink_region=[0.0, 0.6, 0.0]):
    '''sample from long side faces.
    long side face == x-axis or z-axis in the cube frame
    '''
    # check if z-axis (cube-frame) is in parralel with z-axis in base-frame
    R_cube_to_base = Transform(np.zeros(3), object_ori)
    base_z = R_cube_to_base(np.array([0, 0, 1]))

    if np.argmax(np.abs(base_z)) == 2:
        # z-axis (cube-frame) is in parallel with z-axis in base-frame
        # --> sample from x-axis face
        axis = 0
    else:
        # z-axis (cube-frame) is in parallel with the floor
        # --> sample from z-axis face
        axis = 2
    points = np.stack([
        sample(axis, np.random.choice([-1, 1]),
               half_size, shrink_region)
        for _ in range(n)
    ])
    return points


def get_side_face_centers(half_size, object_ori):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    points = []
    for ax in range(3):
        if ax != axis:
            points.append(sample(ax, 1, half_size, np.zeros(3)))
            points.append(sample(ax, -1, half_size, np.zeros(3)))
    return np.array(points)


def get_long_side_face_centers(half_size, object_ori):
    # check if z-axis (cube-frame) is in parralel with z-axis in base-frame
    R_cube_to_base = Transform(np.zeros(3), object_ori)
    base_z = R_cube_to_base(np.array([0, 0, 1]))

    if np.argmax(np.abs(base_z)) == 2:
        # z-axis (cube-frame) is in parallel with z-axis in base-frame
        # --> sample from x-axis face
        axis = 0
    else:
        # z-axis (cube-frame) is in parallel with the floor
        # --> sample from z-axis face
        axis = 2
    points = [sample(axis, sign, half_size, np.zeros(3)) for sign in [-1, 1]]
    vertical_points = [sample(1, sign, half_size, np.zeros(3)) for sign in [-1, 1]]  # TEMP
    return np.array(points), np.array(vertical_points)


def get_three_sided_heuristic_grasps(half_size, object_ori):
    points = get_side_face_centers(half_size, object_ori)
    grasps = []
    for ind in range(4):
        grasps.append(points[np.array([x for x in range(4) if x != ind])])
    return grasps


def get_two_sided_heurictic_grasps(half_size, object_ori):
    side_centers = get_side_face_centers(half_size, object_ori)
    tip_to_center = 0.20
    ax1 = side_centers[1] - side_centers[0]
    ax2 = side_centers[3] - side_centers[2]
    g1 = np.array([
        side_centers[0],
        side_centers[1] + tip_to_center * ax2,
        side_centers[1] - tip_to_center * ax2,
    ])
    g2 = np.array([
        side_centers[1],
        side_centers[0] + tip_to_center * ax2,
        side_centers[0] - tip_to_center * ax2,
    ])
    g3 = np.array([
        side_centers[2],
        side_centers[3] + tip_to_center * ax1,
        side_centers[3] - tip_to_center * ax1,
    ])
    g4 = np.array([
        side_centers[3],
        side_centers[2] + tip_to_center * ax1,
        side_centers[2] - tip_to_center * ax1,
    ])
    return [g1, g2, g3, g4]



def get_two_sided_long_face_heuristic_grasps(half_size, object_ori, tip_to_center=0.2):
    """
    sample a grasp from long side faces
    """
    long_side_centers, vcenters = get_long_side_face_centers(half_size, object_ori)
    ax1 = long_side_centers[1] - long_side_centers[0]
    ax2 = vcenters[1] - vcenters[0]
    g1 = np.array([
        long_side_centers[0],
        long_side_centers[1] + tip_to_center * ax2,
        long_side_centers[1] - tip_to_center * ax2,
    ])
    g2 = np.array([
        long_side_centers[1],
        long_side_centers[0] + tip_to_center * ax2,
        long_side_centers[0] - tip_to_center * ax2,
    ])
    return [g1, g2]

def get_tiny_faces_heuristic_grasps(half_size, object_ori):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    above_point = []
    for i in range(3):
        if i == axis:
            above_point.append(half_size[i] + 0.02)
        else:
            above_point.append(0)

    point = np.stack([
        [0, half_size[1], 0],
        [0, -half_size[1], 0],
        above_point
    ])
    return [point]


def get_two_long_one_tiny_heuristic_grasps(half_size, object_ori):
    side_centers = get_side_face_centers(half_size, object_ori)
    ax2 = side_centers[3] - side_centers[2]
    tip_to_center=0.20
    g1 = np.array([
        side_centers[0] + tip_to_center * ax2,
        side_centers[1] + tip_to_center * ax2,
        [0, half_size[1], 0]
    ])

    g2 = np.array([
        side_centers[0] - tip_to_center * ax2,
        side_centers[1] - tip_to_center * ax2,
        [0, -half_size[1], 0]
    ])

    return [g1, g2]


def get_clockwise_two_finger_yawing_grasp(half_size, object_ori):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    above_point = []
    for i in range(3):
        if i == axis:
            above_point.append(half_size[i] + 0.02)
        else:
            above_point.append(0)

    side_centers = get_side_face_centers(half_size, object_ori)
    tip_to_center = 0.20
    ax1 = side_centers[1] - side_centers[0]
    ax2 = side_centers[3] - side_centers[2]

    g1 = np.array([
        side_centers[0] - tip_to_center * ax2,
        side_centers[1] + tip_to_center * ax2,
        above_point
    ])
    g2 = np.array([
        side_centers[0] + tip_to_center * ax2,
        side_centers[1] - tip_to_center * ax2,
        above_point
    ])
    return [g1, g2]
    # return [g1]


def get_anticlockwise_two_finger_yawing_grasp(half_size, object_ori):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    above_point = []
    for i in range(3):
        if i == axis:
            above_point.append(half_size[i] + 0.02)
        else:
            above_point.append(0)

    side_centers = get_side_face_centers(half_size, object_ori)
    tip_to_center = 0.20
    ax1 = side_centers[1] - side_centers[0]
    ax2 = side_centers[3] - side_centers[2]
    g1 = np.array([
        side_centers[0] - tip_to_center * ax2,
        side_centers[1] + tip_to_center * ax2,
        above_point
    ])
    g2 = np.array([
        side_centers[0] + tip_to_center * ax2,
        side_centers[1] - tip_to_center * ax2,
        above_point
    ])
    # return [g1, g2]
    # return [g2]

    point = np.stack([
        #[0, half_size[1], 0],
        #[0, -half_size[1], 0],
        side_centers[0],
        side_centers[1],
        above_point
    ])

    return [point]


def get_all_heuristic_grasps(half_size, object_ori, avoid_edge_faces=True, yawing_grasp=False, is_level_1=False):
    # return get_tiny_faces_heuristic_grasps(half_size, object_ori)
    """
    if yawing_grasp and clockwise:
        return (
            get_clockwise_two_finger_yawing_grasp(half_size, object_ori)
        )
    elif yawing_grasp and not clockwise:
        return (
            get_anticlockwise_two_finger_yawing_grasp(half_size, object_ori)
        )
    """
    if yawing_grasp:
        return (
            get_two_sided_long_face_heuristic_grasps(half_size, object_ori, tip_to_center=0.25)
            + get_two_sided_long_face_heuristic_grasps(half_size, object_ori, tip_to_center=0.28)
            + get_two_sided_long_face_heuristic_grasps(half_size, object_ori, tip_to_center=0.20)
            # + get_anticlockwise_two_finger_yawing_grasp(half_size, object_ori)
        )
    else:
        if avoid_edge_faces:  # This one is ued for mpfc motion planning to carry the ojbect
            if is_level_1:
                tip_to_center = 0.20
            else:
                tip_to_center = 0.25

            return (
                get_two_sided_long_face_heuristic_grasps(half_size, object_ori, tip_to_center=tip_to_center)
                # get_two_sided_heurictic_grasps(half_size, object_ori)
                # + get_three_sided_heuristic_grasps(half_size, object_ori)
            )
        return (
            get_two_sided_heurictic_grasps(half_size, object_ori)
            + get_three_sided_heuristic_grasps(half_size, object_ori)
            + get_anticlockwise_two_finger_yawing_grasp(half_size, object_ori)
            # + get_tiny_faces_heuristic_grasps(half_size, object_ori)
        )


class GraspSampler(object):
    def __init__(self, env, pos, quat, slacky_collision=True,
                 halfsize=VIRTUAL_CUBOID_HALF_SIZE,
                 ignore_collision=False, avoid_edge_faces=True, yawing_grasp=False, allow_partial_sol=False):
        self.object_pos = pos
        self.object_ori = quat
        self.ik = env.pinocchio_utils.inverse_kinematics
        self.id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.T_cube_to_base = Transform(pos, quat)
        self.T_base_to_cube = self.T_cube_to_base.inverse()
        self.env = env
        self.ik_utils = IKUtils(env, yawing_grasp=yawing_grasp)
        self.slacky_collision = slacky_collision
        self._org_tips_init = np.array(
            self.env.platform.forward_kinematics(INIT_JOINT_CONF)
        )
        self.halfsize = halfsize
        self.tip_solver = CuboidForceClosureTest(halfsize, CoulombFriction(MU))
        self.ignore_collision = ignore_collision
        self.avoid_edge_faces = avoid_edge_faces
        self.yawing_grasp = yawing_grasp
        self.allow_partial_sol = allow_partial_sol

    def _reject(self, points_base):
        if not self.tip_solver.force_closure_test(self.T_cube_to_base,
                                                  points_base):
            return True, None
        if self.ignore_collision:
            q = self.ik_utils._sample_ik(points_base)
        elif self.allow_partial_sol:
            q = self.ik_utils._sample_ik(points_base, allow_partial_sol=True)
        else:
            q = self.ik_utils._sample_no_collision_ik(
                points_base, slacky_collision=self.slacky_collision, diagnosis=False
            )
        if q is None:
            return True, None
        return False, q

    def assign_positions_to_fingers(self, tips):
        cost_to_inds = {}
        for v in itertools.permutations([0, 1, 2]):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self._org_tips_init)
            cost_to_inds[cost] = v

        inds_sorted_by_cost = [
            val for key, val in sorted(cost_to_inds.items(), key=lambda x: x[0])
        ]
        opt_inds = inds_sorted_by_cost[0]
        opt_tips = tips[opt_inds, :]

        # verbose output
        return opt_tips, opt_inds, inds_sorted_by_cost

    def get_feasible_grasps_from_tips(self, tips):
        _, _, permutations_by_cost = self.assign_positions_to_fingers(tips)
        for perm in permutations_by_cost:
            ordered_tips = tips[perm, :]
            should_reject, q = self._reject(ordered_tips)
            if not should_reject:
                # use INIT_JOINT_CONF for tip positions that were not solvable
                valid_tips = [0, 1, 2]
                if self.allow_partial_sol:
                    for i in range(3):
                        if q[i * 3] is None:
                            valid_tips.remove(i)
                            q[i * 3:(i + 1) * 3] = INIT_JOINT_CONF[i * 3:(i + 1) * 3]

                yield Grasp(self.T_base_to_cube(ordered_tips),
                            ordered_tips, q, self.object_pos,
                            self.object_ori, self.T_cube_to_base,
                            self.T_base_to_cube, valid_tips)

    def __call__(self, shrink_region=[0.0, 0.6, 0.0], max_retries=40):
        retry = 0
        print("sampling a random grasp...")
        with keep_state(self.env):
            while retry < max_retries:
                print('[GraspSampler] retry count:', retry)
                points = sample_long_side_face(3, self.halfsize, self.object_ori,
                                               shrink_region=shrink_region)
                tips = self.T_cube_to_base(points)
                for grasp in self.get_feasible_grasps_from_tips(tips):
                    return grasp
                retry += 1

        raise RuntimeError('No feasible grasp is found.')

    def get_heuristic_grasps(self):
        grasps = get_all_heuristic_grasps(
            self.halfsize, self.object_ori,
            avoid_edge_faces=self.avoid_edge_faces,
            yawing_grasp=self.yawing_grasp,
            is_level_1=self.env.info['difficulty'] == 1
        )
        ret = []
        with keep_state(self.env):
            for points in grasps:
                tips = self.T_cube_to_base(points)
                # NOTE: we sacrifice a bit of speed by not using "yield", however,
                # context manager doesn't work as we want if we use "yield".
                # performance drop shouldn't be significant (get_feasible_grasps_from_tips only iterates 6 grasps!).
                # for grasp in self.get_feasible_grasps_from_tips(tips):
                #     yield grasp
                ret += [grasp for grasp in self.get_feasible_grasps_from_tips(tips)]
            return ret


if __name__ == '__main__':
    import pybullet as p
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    from code.const import VIRTUAL_CUBOID_HALF_SIZE
    reward_fn = 'competition_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'training_init'

    env = make_training_env(move_cube.sample_goal(-1).to_dict(), 3,
                            reward_fn=reward_fn,
                            termination_fn=termination_fn,
                            initializer=initializer,
                            action_space='torque',
                            sim=True,
                            visualization=True)

    obs = env.reset()
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    sampler = GraspSampler(env, obs['object_position'],
                           obs['object_orientation'], slacky_collision=True)
    # grasp = sampler()
    grasp = sampler.get_heuristic_grasps().__next__()

    while (p.isConnected()):
        env.platform.simfinger.reset_finger_positions_and_velocities(grasp.q)
