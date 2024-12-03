__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Union
from inspect import getsourcefile
from os.path import abspath,dirname

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class Reacher3Env(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    "Reacher3" is a three-jointed robot arm.
    The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.

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
        xml_file: str = "reacher3.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_goal_weight: float = 1,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        reward_vel_weight: float = 0.1,
        **kwargs,
    ):

        dd = dirname(getsourcefile(lambda:0))  # dir of this file
        xml_file = dd+"/assets/"+xml_file

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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.njoints = 3
        self.ndim = 2

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
        jvel = self.data.qvel.flatten()[:2]  # joints vel
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
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.1*self.njoints, high=0.1*self.njoints, size=2)
            if np.linalg.norm(self.goal) < 0.1*self.njoints:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):       
        theta = self.data.qpos.flatten()[:self.njoints]
        return np.concatenate(
            [
                theta,           # joint angles
                np.cos(theta),   # cos joint angles
                np.sin(theta),   # sin joint angles
                self.data.qvel.flatten()[:self.njoints],   # joint velocities
                self.data.body("fingertip").xpos[:2],      # fingertip 2D position
                self.data.body("fingertip").xquat[0:1],    # fingertip 2D orientation
                self.data.body("fingertip").xquat[3:4],
                self.data.body("target").xpos[:2],         # target 2D position
            ]
        )




from gymnasium.envs.registration import register

def reacher3_v6(**args):
    return Reacher3Env(**args)

register(id="Reacher3-v6", 
    entry_point="gym_envs.envs.reacher3_v6:reacher3_v6",
    max_episode_steps=50,)



