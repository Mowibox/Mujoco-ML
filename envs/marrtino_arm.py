from typing import Dict, Union
from inspect import getsourcefile
from os.path import abspath,dirname

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class MARRtinoArmEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    "MARRtinoArm" is a 5-joints robot arm.
    The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.

    ## Action Space
    `Box(-inf, inf, (5,), float32)` - torques applied at the hinge joints
    
    ## Observation Space
    - *qpos (5 elements)* - angles of the 5 joints
    - *cos(qpos) (5 elements)* - cosine of the angles of the 5 joints
    - *sin(qpos) (5 elements)* - sine of the angles of the 5 joints
    - *qvel (5 elements)* - angular velocities of the 5 joints
    - *xpos (3 elements)* - 3D position of fingertip
    - *xquat (4 elements)* - 3D orientation of fingertip (quaternion)
    - *tpos (3 elements)* - 3D position of target

    ## Reward
    Weigthed sum of
    - goal: (fingertip-target distance < threshold) > 0
    - distance to target: < 0
    - high control penalty: < 0
    - high velocity penalty: < 0

    weigth parameters: 
    - reward_goal_weight: float = 1,
    - reward_dist_weight: float = 1,
    - reward_control_weight: float = 1,
    - reward_vel_weight: float = 1,

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "marrtino.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_goal_weight: float = 1,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        reward_vel_weight: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_goal_weight,
            reward_dist_weight,
            reward_control_weight,
            reward_vel_weight,
            **kwargs,
        )

        self._reward_goal_weight = reward_goal_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self._reward_vel_weight = reward_vel_weight
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64)

        dd = dirname(getsourcefile(lambda:0))  # dir of this file
        xml_file = dd+"/assets/"+xml_file

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        self.njoints = 5
        self.ndim = 3

        self.body_names = [self.data.model.body(i).name for i in range(0,self.data.model.nbody)]
        self.joint_names = [self.data.model.joint(i).name for i in range(0,self.data.model.njnt)]

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        '''
        print(f"Observation: {self.observation_space}")
        print(f"Action: {self.action_space}")
        print(f"Reward weights: goal: {self._reward_goal_weight}, " \
                f"dist: {self._reward_dist_weight}, " \
                f"ctrl: {self._reward_control_weight}, " \
                f"vel: {self._reward_vel_weight}")
        '''
        
        if self.render_mode == "human":
            # hide menu
            self.mujoco_renderer._get_viewer("human")._hide_menu = True


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
         
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")  # diff to target
        vec_norm = np.linalg.norm(vec) # dist to target
        jvel = self.data.qvel.flatten()[:self.njoints]  # joints vel
        jvel_norm = np.linalg.norm(jvel) # joints vel norm
        
        reward_goal = (1.0 if vec_norm<0.04 else 0.0) * self._reward_goal_weight 
        if vec_norm<0.04:
            reward_goal += (1.0 if jvel_norm<0.2 else 0.0) * self._reward_goal_weight
        
        reward_dist = -vec_norm * self._reward_dist_weight
        reward_vel = -jvel_norm * self._reward_vel_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        reward = reward_goal + reward_dist + reward_ctrl + reward_vel

        reward_info = {
            "dist": vec_norm,
            "reward_goal": reward_goal,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_vel": reward_vel,
        }

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos 
            #self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=3)
            self.goal[2] = 0.01 # set z=0.01
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-3:] = self.goal
        qvel = self.init_qvel + \
            self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-3:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
    

        theta = self.data.qpos.flatten()[:self.njoints]  # j0,...,j4
        
        _obs = np.concatenate(
            [
                theta,           # joint angles
                np.cos(theta),   # cos joint angles
                np.sin(theta),   # sin joint angles
                self.data.qvel.flatten()[:self.njoints],  # joint velocities
                self.data.body("fingertip").xpos,         # fingertip 3D position
                self.data.body("fingertip").xquat,        # fingertip 3D orientation
                self.data.body("target").xpos,            # target 3D position
            ]
        )
        
        return _obs


from gymnasium.envs.registration import register

def marrtino_arm(**args):
    return MARRtinoArmEnv(**args)

register(id="MARRtinoArm", 
    entry_point="gym_envs.envs.marrtino_arm:marrtino_arm",
    max_episode_steps=50,)



