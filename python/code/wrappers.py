"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import pybullet as p
import numpy as np
import gym
from trifinger_simulation import camera
import cv2


EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='


class NewToOldObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": env.observation_space['robot']['position'],
                "robot_velocity": env.observation_space['robot']['velocity'],
                "robot_torque": env.observation_space['robot']['torque'],
                "robot_tip_positions": env.observation_space['robot']['tip_positions'],
                "object_position": env.observation_space["achieved_goal"]["position"],
                "object_orientation": env.observation_space["achieved_goal"]["orientation"],
                "goal_object_position": env.observation_space["desired_goal"]["position"],
                "goal_object_orientation": env.observation_space["desired_goal"]["orientation"],
                "tip_force": env.observation_space["robot"]["tip_force"],
                "action_torque": env.observation_space['robot']['torque'],
                "action_position": env.observation_space['robot']['position'],
            }
        )

    def observation(self, obs):
        old_obs = {
            "robot_position": obs['robot']['position'],
            "robot_velocity": obs['robot']['velocity'],
            "robot_torque": obs['robot']['torque'],
            "robot_tip_positions": obs['robot']['tip_positions'],
            "tip_force": obs['robot']['tip_force'],
            "object_position": obs['achieved_goal']['position'],
            "object_orientation": obs['achieved_goal']['orientation'],
            "goal_object_position": obs['desired_goal']['position'],
            "goal_object_orientation": obs['desired_goal']['orientation'],
        }
        if self.action_space == self.observation_space['robot_position']:
            old_obs['action_torque'] = np.zeros_like(obs['action'])
            old_obs['action_position'] = obs['action']
        elif self.action_space == self.observation_space['robot_torque']:
            old_obs['action_torque'] = obs['action']
            old_obs['action_position'] = np.zeros_like(obs['action'])
        else:
            old_obs['action_torque'] = obs['action']['torque']
            old_obs['action_position'] = obs['action']['position']
        return old_obs


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cameras = camera.TriFingerCameras(image_size=(360, 270))
        self.metadata = {"render.modes": ["rgb_array"]}
        self._initial_reset = True
        self._accum_reward = 0
        self._reward_at_step = 0

    def reset(self):
        import pybullet as p
        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        self._accum_reward = 0
        self._reward_at_step = 0
        if self._initial_reset:
            self._episode_idx = 0
            self._initial_reset = False
        else:
            self._episode_idx += 1
        return obs

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        self._accum_reward += reward
        self._reward_at_step = reward
        return observation, reward, is_done, info

    def render(self, mode='rgb_array', **kwargs):
        assert mode == 'rgb_array', 'RenderWrapper Only supports rgb_array mode'
        images = self.cameras.cameras[0].get_image(), self.cameras.cameras[1].get_image()
        height = images[0].shape[1]
        two_views = np.concatenate((images[0], images[1]), axis=1)
        two_views = cv2.putText(two_views, 'step_count: {:06d}'.format(self.env.unwrapped.step_count), (10, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'episode: {}'.format(self._episode_idx), (10, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'reward: {:.2f}'.format(self._reward_at_step), (10, height - 130),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'acc_reward: {:.2f}'.format(self._accum_reward), (10, height - 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        return two_views


class PyBulletClearGUIWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        return obs
