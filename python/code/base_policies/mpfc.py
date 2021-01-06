#!/usr/bin/env python3
import pybullet as p
import numpy as np
from code.utils import get_rotation_between_vecs, slerp, Transform
from trifinger_simulation import TriFingerPlatform
from scipy.spatial.transform import Rotation

DEBUG = False
if DEBUG:
    color_set = ((1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5))


class PlanningAndForceControlPolicy:
    def __init__(self, env, obs, fc_policy, path, action_repeat=2*2,
                 adjust_tip=True, adjust_tip_ori=False):
        self.env = env
        self.fc_policy = fc_policy
        self.obs = obs
        self.path = path.repeat(action_repeat)
        self.grasp = self.path.grasp
        self.joint_sequence = self.path.joint_conf
        self.cube_sequence = self.path.cube
        self.adjust_tip = adjust_tip
        self._step = 0
        self._actions_in_progress = False
        self.adjust_tip_ori = adjust_tip_ori
        self.executing_tip_adjust = False
        if DEBUG:
            self.visual_markers = []
            self.vis_tip_center = None

    def at_end_of_sequence(self, step):
        return step >= len(self.cube_sequence)

    def add_tip_adjustments(self, obs, num_steps=50):
        print("Appending to path....")
        # tip_pos = self.path.tip_path[-1]
        dir = 0.5 * (obs['goal_object_position'] - obs['object_position'])
        cube_pos = self.cube_sequence[-1][:3]
        cube_ori = p.getQuaternionFromEuler(self.cube_sequence[-1][3:])

        grasp = self.path.grasp
        if self.adjust_tip_ori:
            yaxis = np.array([0, 1, 0])
            goal_obj_yaxis = Rotation.from_quat(obs['goal_object_orientation']).apply(yaxis)
            obj_yaxis = Rotation.from_quat(cube_ori).apply(yaxis)
            diff_quat = get_rotation_between_vecs(obj_yaxis, goal_obj_yaxis)
            resolution = np.arange(0, 1, 1.0 / num_steps)
            interp_quat = slerp(np.array([0, 0, 0, 1]), diff_quat, resolution)

        warning_counter = 0
        warning_tips = []
        for i in range(num_steps):
            translation = cube_pos + i / num_steps * dir
            if self.adjust_tip_ori:
                rotation = (Rotation.from_quat(interp_quat[i]) * Rotation.from_quat(cube_ori)).as_quat()
            else:
                rotation = cube_ori
            goal_tip_pos = Transform(translation, rotation)(grasp.cube_tip_pos)
            q = obs['robot_position']

            for j, tip in enumerate(goal_tip_pos):
                q = self.env.pinocchio_utils.inverse_kinematics(j, tip, q)
                if q is None:
                    q = self.joint_sequence[-1]
                    # print(f'[tip adjustments] warning: IK solution is not found for tip {j}. Using the last joint conf')
                    warning_counter += 1
                    if j not in warning_tips:
                        warning_tips.append(j)
                    break
            if q is None:
                print('[tip adjustments] warning: IK solution is not found for all tip positions.')
                print(f'[tip adjustments] aborting tip adjustments (loop {i} / {num_steps})')
                break
            target_cube_pose = np.concatenate([
                translation,
                p.getEulerFromQuaternion(rotation)
            ])
            self.cube_sequence.append(target_cube_pose)
            self.joint_sequence.append(q)
            self.path.tip_path.append(goal_tip_pos)
        if warning_counter > 0:
            print(f'[tip adjustments] warning: IK solution is not found for {warning_counter} / {num_steps} times on tips {warning_tips}.')

    def __call__(self, obs):
        if not self._actions_in_progress:
            if np.linalg.norm(
                obs['robot_position'] - self.path.joint_conf[0]
            ).sum() > 0.25:
                print(
                    'large initial joint conf error:',
                    np.linalg.norm(obs['robot_position']
                                   - self.path.joint_conf[0])
                )
        self._actions_in_progress = True

        step = self._step
        if self.adjust_tip and self.at_end_of_sequence(step):
            self.add_tip_adjustments(obs)
        step = min(step, len(self.cube_sequence) - 1)
        target_cube_pose = self.cube_sequence[step]
        target_joint_conf = self.joint_sequence[step]

        torque = self.fc_policy(obs, target_cube_pose[:3],
                                p.getQuaternionFromEuler(target_cube_pose[3:]))
        action = {
            'position': np.asarray(target_joint_conf),
            'torque': torque
        }
        self._step += 1
        return self._clip_action(action)

    def _clip_action(self, action):
        tas = TriFingerPlatform.spaces.robot_torque.gym
        pas = TriFingerPlatform.spaces.robot_position.gym
        action['position'] = np.clip(action['position'], pas.low, pas.high)
        action['torque'] = np.clip(action['torque'], tas.low, tas.high)
        return action


if __name__ == '__main__':
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    from code.grasping import get_planned_grasp, execute_grasp_approach
    from code.utils import set_seed
    from .fc import ForceControlPolicy

    set_seed(0)
    viz = True
    difficulty = 4
    is_level_4 = difficulty == 4
    env = make_training_env(move_cube.sample_goal(-1).to_dict(), difficulty,
                            reward_fn='competition_reward',
                            termination_fn='no_termination',
                            initializer='training_init',
                            action_space='torque_and_position',
                            sim=True,
                            visualization=viz,
                            rank=0)
    env.reset()

    for _ in range(5):
        obs = env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0])

        grasp, path = get_planned_grasp(env, obs['object_position'],
                                        obs['object_orientation'],
                                        obs['goal_object_position'],
                                        obs['goal_object_orientation'],
                                        tight=True, use_rrt=is_level_4)
        obs, _ = execute_grasp_approach(env, obs, grasp)
        pi_fc = ForceControlPolicy(env, apply_torques=is_level_4,
                                   grasp_tip_positions=grasp.cube_tip_pos)
        pi = PlanningAndForceControlPolicy(
            env, obs, pi_fc, path, adjust_tip=not is_level_4, adjust_tip_ori=is_level_4
        )

        for _ in range(1000):
            action = pi(obs)
            obs, reward, done, info = env.step(action)
    env.close()
