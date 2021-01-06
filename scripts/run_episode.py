#!/usr/bin/env python3
"""Run a single episode with our controller.

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - goal pose of the object (as JSON string) (optional)
"""
import sys
import json

from trifinger_simulation.tasks import move_cube
from code.make_env import make_training_env
from code.state_machine import TriFingerStateMachine


difficulty2config = {
    1: {'action_space': 'torque_and_position', 'adjust_tip': True},
    2: {'action_space': 'torque_and_position', 'adjust_tip': True},
    3: {'action_space': 'torque_and_position', 'adjust_tip': True},
    4: {'action_space': 'torque_and_position', 'adjust_tip': False}
}


def _init_env(goal_pose_json, difficulty):
    eval_config = {
        'action_space': difficulty2config[difficulty]['action_space'],
        'frameskip': 3,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'random_init',
        'monitor': False,
        'visualization': False,
        'sim': False,
        'rank': 0
    }

    from code.utils import set_seed
    set_seed(0)
    goal_pose_dict = json.loads(goal_pose_json)
    env = make_training_env(goal_pose_dict, difficulty, **eval_config)
    return env


def main():
    difficulty = int(sys.argv[1])
    if len(sys.argv) == 3:
        goal_pose_json = sys.argv[2]
    else:
        goal_pose = move_cube.sample_goal(difficulty)
        goal_pose_json = json.dumps({
            'position': goal_pose.position.tolist(),
            'orientation': goal_pose.orientation.tolist()
        })

    env = _init_env(goal_pose_json, difficulty)
    state_machine = TriFingerStateMachine(env)
    state_machine.run()


if __name__ == "__main__":
    main()
