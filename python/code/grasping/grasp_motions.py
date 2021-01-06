from .ik import IKUtils
from code.action_sequences import ScriptedActions
from code.const import INIT_JOINT_CONF
import numpy as np


def execute_grasp_approach(env, obs, grasp):
    action_sequence = ScriptedActions(env, obs['robot_tip_positions'], grasp)
    pregrasp_joint_conf, pregrasp_tip_pos = get_safe_pregrasp(
        env, obs, grasp
    )
    if pregrasp_joint_conf is None:
        raise RuntimeError('Feasible heuristic grasp approach is not found.')

    action_sequence.add_raise_tips()
    action_sequence.add_heuristic_pregrasp(pregrasp_tip_pos)
    action_sequence.add_grasp(coef=0.6)

    obs, done = action_sequence.execute_motion(
        frameskip=1,
        action_repeat=4 if env.simulation else 12 * 4,
        action_repeat_end=40 if env.simulation else 400
    )

    return obs, done


def get_safe_pregrasp(env, obs, grasp, candidate_margins=[1.3, 1.5, 1.8, 2.0, 2.2]):
    pregrasp_tip_pos = []
    pregrasp_jconfs = []
    ik_utils = IKUtils(env)
    init_tip_pos = env.platform.forward_kinematics(INIT_JOINT_CONF)
    mask = np.eye(3)[grasp.valid_tips, :].sum(0).reshape(3, -1)

    for margin in candidate_margins:
        tip_pos = grasp.T_cube_to_base(grasp.cube_tip_pos * margin)
        tip_pos = tip_pos * mask + (1 - mask) * init_tip_pos
        qs = ik_utils.sample_no_collision_ik(tip_pos)
        if len(qs) > 0:
            pregrasp_tip_pos.append(tip_pos)
            pregrasp_jconfs.append(qs[0])
            print('candidate margin coef {}: safe'.format(margin))
        else:
            print('candidate margin coef {}: no ik solution found'.format(margin))

    if len(pregrasp_tip_pos) == 0:
        print('warning: no safe pregrasp pose with a margin')
        tip_pos = grasp.T_cube_to_base(grasp.cube_tip_pos * candidate_margins[0])
        tip_pos = tip_pos * mask + (1 - mask) * init_tip_pos
        qs = ik_utils.sample_ik(tip_pos)
        if len(qs) == 0:
            return None, None
        else:
            pregrasp_tip_pos.append(tip_pos)
            pregrasp_jconfs.append(qs[0])
    return pregrasp_jconfs[-1], pregrasp_tip_pos[-1]


if __name__ == '__main__':
    import pybullet as p
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    from code.grasping.grasp_functions import get_heuristic_grasp

    env = make_training_env(move_cube.sample_goal(-1).to_dict(), 3,
                            reward_fn='competition_reward',
                            termination_fn='position_close_to_goal',
                            initializer='training_init',
                            action_space='torque',
                            sim=True,
                            visualization=True,
                            rank=1)

    obs = env.reset()
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    grasp = get_heuristic_grasp(env, obs['object_position'],
                                obs['object_orientation'])
    obs, done = execute_grasp_approach(env, obs, grasp)
