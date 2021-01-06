from code.env.cube_env import ActionType
from code.const import INIT_JOINT_CONF, CONTRACTED_JOINT_CONF, AVG_POSE_STEPS
from code.utils import action_type_to, frameskip_to, Transform
from code.utils import get_yaw_diff, estimate_object_pose, get_rotation_between_vecs
from code.action_sequences import ScriptedActions
from code import grasping, base_policies
from enum import Enum, auto
from scipy.spatial.transform import Rotation
import numpy as np


class State(Enum):
    RESET = auto()
    GO_TO_INIT_POSITION = auto()
    RUN_INITIAL_MANIPULATIONS = auto()
    MOVE_TO_CENTER = auto()
    ADJUST_ORIENTATION = auto()
    EXECUTE_POLICY = auto()


class TriFingerStateMachine(object):
    def __init__(self, env, obs=None):
        self.env = env
        self.difficulty = self.env.info['difficulty']
        self.state = State.RESET
        self.done = False
        self.obs = obs
        self.grasp_check_failed_count = 0
        self.adjust_ori_count = 0

    def run(self):
        while not self.done:
            print(f"Entering state: {self.state}")
            ##################
            # RESET
            ##################
            if self.state == State.RESET:
                self.done = False
                self.obs = self.env.reset()
                print('goal position', self.obs['goal_object_position'])
                print('goal orientation', self.obs['goal_object_orientation'])
                self.state = State.GO_TO_INIT_POSITION

            ##################
            # GO_TO_INIT_POSITION
            ##################
            elif self.state == State.GO_TO_INIT_POSITION:
                self.go_to_initial_position(hold_steps=300)
                self.state = State.RUN_INITIAL_MANIPULATIONS

            ##################
            # RUN_INITIAL_MANIPULATIONS
            ##################
            elif self.state == State.RUN_INITIAL_MANIPULATIONS:
                print('adjust_ori_count', self.adjust_ori_count)
                if not self.object_centered():
                    self.state = State.MOVE_TO_CENTER
                elif self.adjust_ori_count < 2 and not self.object_oriented() and self.difficulty == 4:
                    self.state = State.ADJUST_ORIENTATION
                else:
                    self.state = State.EXECUTE_POLICY

            ##################
            # MOVE_TO_CENTER
            ##################
            elif self.state == State.MOVE_TO_CENTER:
                grasp = self.get_heuristic_grasp(execute=True)
                if grasp is None:
                    grasp = self.get_partial_grasp(execute=True)

                if grasp is None:
                    print("No heuristic pregrasp.... moving to execute policy")
                    self.state = State.EXECUTE_POLICY
                else:
                    action_sequence = ScriptedActions(
                        self.env, self.obs['robot_tip_positions'], grasp
                    )
                    action_sequence.add_grasp(coef=0.6)
                    action_sequence.add_move_to_center(coef=0.6)
                    action_sequence.add_release(4.0)
                    action_sequence.add_raise_tips(height=0.08)
                    self.obs, self.done = action_sequence.execute_motion(
                        frameskip=2,
                        action_repeat=5,
                        action_repeat_end=10
                    )
                    self.wait_for(10 if self.env.simulation else 300)
                    self.state = State.GO_TO_INIT_POSITION

            ##################
            # ADJUST_ORIENTATION
            ##################
            elif self.state == State.ADJUST_ORIENTATION:
                grasp, path = self.get_yawing_grasp(execute=True)
                if grasp is None:
                    print("Can't get grasp for yawing motion... moving to execute policy")
                    self.state = State.MOVE_TO_CENTER
                else:

                    pi = self.make_base_policy(grasp, path, adjust_tip=False, action_repeat=4)
                    while not self.done and not pi.at_end_of_sequence(pi._step):
                        action = pi(self.obs)
                        self.obs, _, self.done, _ = self.env.step(action)

                    grasp.update(self.obs['object_position'], self.obs['object_orientation'])
                    action_sequence = ScriptedActions(
                        self.env, self.obs['robot_tip_positions'], grasp
                    )
                    action_sequence.add_release2(3.0)
                    action_sequence.add_raise_tips(height=0.08)
                    self.obs, self.done = action_sequence.execute_motion(
                        frameskip=2,
                        action_repeat=6,
                        action_repeat_end=10
                    )

                    self.wait_for(10 if self.env.simulation else 100)
                    self.state = State.RUN_INITIAL_MANIPULATIONS
                    self.adjust_ori_count += 1

            ##################
            # EXECUTE_POLICY
            ##################
            elif self.state == State.EXECUTE_POLICY:
                self.adjust_ori_count = 0
                grasp_failed = False
                grasp, path = self.get_planned_grasp(execute=True)
                if grasp is None:
                    print("Grasping failed... Moving cube and trying again")
                    self.state = State.MOVE_TO_CENTER
                    grasp_failed = True

                if not grasp_failed:
                    pi = self.make_base_policy(grasp, path)
                    while self.object_grasped(grasp) and not self.done:
                        action = pi(self.obs)
                        self.obs, _, self.done, _ = self.env.step(action)

                    if not self.done:
                        print("Object has been dropped! Retrying!")
                        self.wait_for(1000)
                        self.state = State.GO_TO_INIT_POSITION

    def object_grasped(self, grasp):
        T_cube_to_base = Transform(self.obs['object_position'],
                                   self.obs['object_orientation'])
        target_tip_pos = T_cube_to_base(grasp.cube_tip_pos)
        center_of_tips = np.mean(target_tip_pos, axis=0)
        dist = np.linalg.norm(target_tip_pos - self.obs['robot_tip_positions'])
        center_dist = np.linalg.norm(center_of_tips - np.mean(self.obs['robot_tip_positions'], axis=0))
        object_is_grasped = center_dist < 0.07 and dist < 0.10
        if object_is_grasped:
            self.grasp_check_failed_count = 0
        else:
            self.grasp_check_failed_count += 1
            print('incremented grasp_check_failed_count')
            print(f'center_dist: {center_dist:.4f}\tdist: {dist:.4f}')

        return self.grasp_check_failed_count < 5

    def object_oriented(self):
        # calculate second (long) axis of object cuboid and goal cuboid
        return np.abs(get_yaw_diff(
            self.obs['object_orientation'],
            self.obs['goal_object_orientation'])
        ) < np.pi / 4

    def object_centered(self):
        dist_from_center = np.linalg.norm(
            self.obs['object_position'][:2]
        )
        return dist_from_center < 0.07

    def get_yawing_grasp(self, execute=True, avg_pose=True):
        if avg_pose:
            obj_pos, obj_ori, self.obs, self.done = estimate_object_pose(
                self.env, self.obs, steps=AVG_POSE_STEPS
            )
        else:
            obj_pos = self.obs['object_position']
            obj_ori = self.obs['object_orientation']

        candidate_step_angle = [np.pi * 2 / 3, np.pi / 2, np.pi / 3]
        # candidate_step_angle = [np.pi / 2, np.pi / 3]
        for step_angle in candidate_step_angle:
            grasp, path = grasping.get_yawing_grasp(
                self.env, obj_pos, obj_ori,
                self.obs['goal_object_orientation'],
                step_angle=step_angle
            )

            if grasp is None:
                continue

            if execute:
                try:
                    self.obs, self.done = grasping.execute_grasp_approach(
                        self.env, self.obs, grasp
                    )
                except RuntimeError:
                    print("Pregrasp motion failed for yawing grasp...")
                    continue

            # path = Path.tighten(self.env, path, coef=0.8)
            if grasp is not None:
                self.env.unwrapped.register_custom_log('target_object_pose', {'position': obj_pos, 'orientation': obj_ori})
                self.env.unwrapped.register_custom_log('target_tip_pos', grasp.T_cube_to_base(grasp.cube_tip_pos))
                self.env.unwrapped.save_custom_logs()
            return grasp, path

        return None, None

    def get_partial_grasp(self, execute=True, avg_pose=True):
        """
        sample a 'partial' grasp for object centering
        NOTE: Only use it for centering the object.
        """
        if avg_pose:
            obj_pos, obj_ori, self.obs, self.done = estimate_object_pose(
                self.env, self.obs, steps=AVG_POSE_STEPS
            )
        else:
            obj_pos = self.obs['object_position']
            obj_ori = self.obs['object_orientation']

        done = False
        while not done:
            grasp = grasping.sample_partial_grasp(self.env, obj_pos, obj_ori)
            if len(grasp.valid_tips) == 1:
                # NOTE: if only one of the tips is valid, check if
                # a. the tip is far from origin than object
                # b. angle between tip-to-origin vector and object's y-axis > 30 degree
                dist = np.linalg.norm(grasp.base_tip_pos[:, :2], axis=1)
                tip_id = np.argmax(dist)
                origin_to_tip = grasp.base_tip_pos[tip_id, :2]
                origin_to_tip /= np.linalg.norm(origin_to_tip)
                cube_yaxis = Transform(np.array([0, 0, 0]), obj_ori)(np.array([0, 1, 0]))[:2]
                cube_yaxis /= np.linalg.norm(cube_yaxis)
                # cube_yaxis = Rotation.from_quat(obj_ori).apply(np.array([0, 1, 0]))  # Is this same?
                if dist[tip_id] > np.linalg.norm(obj_pos) and np.dot(origin_to_tip, cube_yaxis) < 1 / 2:
                    print('grasp.valid_tips', grasp.valid_tips)
                    done = True
            else:
                done = True

        if grasp is not None and execute:
            self.obs, self.done = grasping.execute_grasp_approach(
                self.env, self.obs, grasp
            )

        if grasp is not None:
            self.env.unwrapped.register_custom_log('target_object_pose', {'position': obj_pos, 'orientation': obj_ori})
            self.env.unwrapped.register_custom_log('target_tip_pos', grasp.T_cube_to_base(grasp.cube_tip_pos))
            self.env.unwrapped.save_custom_logs()
        return grasp


    def get_heuristic_grasp(self, execute=True, avg_pose=True):
        if avg_pose:
            obj_pos, obj_ori, self.obs, self.done = estimate_object_pose(
                self.env, self.obs, steps=AVG_POSE_STEPS
            )
        else:
            obj_pos = self.obs['object_position']
            obj_ori = self.obs['object_orientation']

        grasps = grasping.get_all_heuristic_grasps(
            self.env, obj_pos, obj_ori, avoid_edge_faces=False
        )
        grasp = None
        for grasp in grasps:
            try:
                if execute:
                    self.obs, self.done = grasping.execute_grasp_approach(
                        self.env, self.obs, grasp
                    )
                break
            except RuntimeError:
                print("Pregrasp motion failed...")
                print("trying another grasp...")
                grasp = None

        if grasp is None:
            # This means no heuristic grasps were feasible...
            # Call sample grasp. This will throw an error if it fails
            # but that is okay because we are stuck if there are no grasps...
            print("All heuristic grasps failed... trying to find a random one.")
            try:
                grasp = grasping.sample_grasp(self.env, obj_pos, obj_ori)
            except RuntimeError:
                return None

            if execute:
                self.obs, self.done = grasping.execute_grasp_approach(
                    self.env, self.obs, grasp
                )
        if grasp is not None:
            self.env.unwrapped.register_custom_log('target_object_pose', {'position': obj_pos, 'orientation': obj_ori})
            self.env.unwrapped.register_custom_log('target_tip_pos', grasp.T_cube_to_base(grasp.cube_tip_pos))
            self.env.unwrapped.save_custom_logs()
        return grasp

    def get_planned_grasp(self, execute=True, avg_pose=True):
        if avg_pose:
            obj_pos, obj_ori, self.obs, self.done = estimate_object_pose(
                self.env, self.obs, steps=AVG_POSE_STEPS
            )
        else:
            obj_pos = self.obs['object_position']
            obj_ori = self.obs['object_orientation']

        goal_ori = self.get_closest_pitch_angle()
        if self.env.visualization:
            self.env.cube_viz.goal_viz.set_state(
                self.obs['goal_object_position'], goal_ori
            )
        try:
            grasp, path = grasping.get_planned_grasp(
                self.env,
                obj_pos,
                obj_ori,
                self.obs['goal_object_position'],
                goal_ori,
                tight=True,
                use_rrt=True,
                use_ori=self.difficulty == 4
            )
            if execute:
                self.obs, self.done = grasping.execute_grasp_approach(
                    self.env, self.obs, grasp
                )
        except RuntimeError as e:
            print('planned grasp or pregrasp motion failed...')
            print('=====')
            print(e)
            print('=====')
            return None, None

        self.env.unwrapped.register_custom_log('target_object_pose', {'position': obj_pos, 'orientation': obj_ori})
        self.env.unwrapped.register_custom_log('target_tip_pos', grasp.T_cube_to_base(grasp.cube_tip_pos))
        self.env.unwrapped.save_custom_logs()
        return grasp, path

    def make_base_policy(self, grasp, path=None, adjust_tip=True, action_repeat=12):
        return base_policies.PlanningAndForceControlPolicy(
            self.env, self.obs, base_policies.CancelGravityPolicy(self.env),
            path,
            adjust_tip=adjust_tip,
            adjust_tip_ori=False,
            action_repeat=action_repeat
        )

    def go_to_initial_position(self, hold_steps=300):
        counter = 0
        done = False
        initial_position = self.obs['robot_position']
        while not done and counter < hold_steps:
            desired_position = np.copy(initial_position)
            # close the bottom joint (joint 2) first, and then close other joints together (joint 0, joint 1)
            if counter < hold_steps / 2:
                desired_position = CONTRACTED_JOINT_CONF
            else:
                desired_position = INIT_JOINT_CONF

            with action_type_to(ActionType.POSITION, self.env):
                obs, reward, done, info = self.env.step(desired_position)
            counter += 1
        self.obs = obs
        self.done = done

    def wait_for(self, num_steps):
        step = 0
        with frameskip_to(1, self.env):
            with action_type_to(ActionType.POSITION, self.env):
                init_joint_conf = np.clip(
                    self.obs['robot_position'],
                    self.env.action_space.low,
                    self.env.action_space.high
                )
                while not self.done and step <= num_steps:
                    self.obs, _, self.done, _ = self.env.step(init_joint_conf)
                    step += 1

    def get_closest_pitch_angle(self):
        """The competition_reward doesn't care about pitch rotation,
        so we should use the smallest rotation that matches the roll and yaw
        of the goal.

        This function returns the goal orientation with that criterion.
        """
        actual_rot = Rotation.from_quat(self.obs['object_orientation'])
        goal_rot = Rotation.from_quat(self.obs['goal_object_orientation'])

        y_axis = [0, 1, 0]
        goal_direction_vector = goal_rot.apply(y_axis)
        actual_direction_vector = actual_rot.apply(y_axis)

        quat = get_rotation_between_vecs(actual_direction_vector,
                                         goal_direction_vector)
        return (Rotation.from_quat(quat) * actual_rot).as_quat()


if __name__ == '__main__':
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    import dl
    seed = 0
    dl.rng.seed(seed)

    env = make_training_env(move_cube.sample_goal(-1).to_dict(), 4,
                            reward_fn='competition_reward',
                            termination_fn='position_close_to_goal',
                            initializer='centered_init',
                            action_space='torque_and_position',
                            sim=True,
                            visualization=True,
                            episode_length=10000,
                            rank=seed)

    state_machine = TriFingerStateMachine(env)
    state_machine.run()
