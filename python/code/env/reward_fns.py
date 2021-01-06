"""Place reward functions here.

These will be passed as an arguement to the training env, allowing us to
easily try out new reward functions.
"""


import numpy as np
from trifinger_simulation.tasks import move_cube
from scipy.spatial.transform import Rotation


###############################
# Competition Reward Functions
###############################

def competition_reward(previous_observation, observation, info):
    return -move_cube.evaluate_state(
        move_cube.Pose.from_dict(observation['desired_goal']),
        move_cube.Pose.from_dict(observation['achieved_goal']),
        info["difficulty"],
    )


# For backward compatibility
task1_competition_reward = competition_reward
task2_competition_reward = competition_reward
task3_competition_reward = competition_reward
task4_competition_reward = competition_reward

##############################
# Training Reward functions
##############################


def _tip_distance_to_cube(observation):
    # calculate first reward term
    pose = observation['achieved_goal']
    return np.linalg.norm(
        observation["robot"]["tip_positions"] - pose['position']
    )


def _action_reg(observation):
    v = observation['robot']['velocity']
    t = observation['robot']['torque']
    velocity_reg = v.dot(v)
    torque_reg = t.dot(t)
    return 0.1 * velocity_reg + torque_reg


def _tip_slippage(previous_observation, observation):
    pose = observation['achieved_goal']
    prev_pose = previous_observation['achieved_goal']
    obj_rot = Rotation.from_quat(pose['orientation'])
    prev_obj_rot = Rotation.from_quat(prev_pose['orientation'])
    relative_tip_pos = obj_rot.apply(observation["robot"]["tip_positions"]
                                     - observation["achieved_goal"]["position"])
    prev_relative_tip_pos = prev_obj_rot.apply(previous_observation["robot"]["tip_positions"]
                                               - previous_observation["achieved_goal"]["position"])
    return -np.linalg.norm(relative_tip_pos - prev_relative_tip_pos)


def training_reward(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    r = competition_reward(previous_observation, observation, info)
    reg = _action_reg(observation)
    slippage = _tip_slippage(previous_observation, observation)
    return r - 0.1 * reg + 500 * shaping + 300 * slippage


def _orientation_error(observation):
    goal_rot = Rotation.from_quat(observation['desired_goal']['orientation'])
    actual_rot = Rotation.from_quat(observation['achieved_goal']['orientation'])
    error_rot = goal_rot.inv() * actual_rot
    return error_rot.magnitude() / np.pi


def match_orientation_reward(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    return -_orientation_error(observation) + 500 * shaping


def match_orientation_reward_shaped(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    ori_shaping = (_orientation_error(previous_observation)
                   - _orientation_error(observation))
    return 500 * shaping + 100 * ori_shaping
