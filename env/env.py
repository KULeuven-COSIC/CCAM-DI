import importlib
# import config
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import pandas as pd
import numpy as np
import copy
import sys
import os
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import logging
import keyboard
import threading
from tensorboardX import SummaryWriter
import math
import random
from math import sqrt
from datetime import datetime
import functions.adversary_functions as adv_func


logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.CRITICAL  # DEBUG < INFO (show step info) < WARNING (show progress bar) < CRITICAL
)

# load config as argument
if len(sys.argv) == 3:
    config = importlib.import_module(sys.argv[2])
else:
    import config

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


class IntersectionEnv(MultiAgentEnv):

    def __init__(self, env_conf):
        """
        Environment initialisation.
        """

        self.AGENT_NUM = 10
        self.MAX_SPEED = 35.0
        self.TERMINATING_AGENTS = env_conf["terminating_agents"]
        self.NON_AGENT_SWITCH = env_conf["non_agent_switch"]
        self.NON_AGENT_RATIO = env_conf["non_agent_ratio"]
        self.NON_AGENT_NUM = math.ceil(self.AGENT_NUM * self.NON_AGENT_RATIO) + 1
        self.FAKE_SWITCH = env_conf["fake_switch"]
        self.FAKE_RATIO = env_conf["fake_ratio"]
        self.FAKE_MOD_DIST = env_conf["fake_mod_dist"]
        self.sumo_random_route = env_conf["sumo_random_route"]
        self.sumo_step_length = env_conf["sumo_step_length"]
        self.sumo_traffic_insertation_period = env_conf["sumo_traffic_insertation_period"]
        self.OBS_SPACE_TYPE = env_conf["obs_space_type"]
        self.OBS_SPACE_N_CLOSEST = env_conf["obs_space_n_closest"]
        self.OBS_RANGE = env_conf["obs_range"]
        self.OBS_CLEAN_ACT_AGENT_OBS = env_conf["obs_clean_act_agent_obs"]
        self.ACTION_RANGE_FILTER = env_conf["action_range_filter"]
        self.ACTION_CONFLICT_FILTER = env_conf["action_conflict_filter"]
        self.ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER = env_conf["action_movement_vector_intersect_filter"]
        self.episode_length = env_conf["episode_length"]  # Steps in episode
        self.restart_episode_on_collision = env_conf["restart_episode_on_collision"]
        self.total_steps = config.TRAIN_TOTAL_TIMESTEPS
        self.reward_values = list(map(float, env_conf["reward_values"].split('_')))
        self.accel = env_conf["acceleration"]
        self.SORT_OBS_BY_DIST = env_conf["sort_obs_by_dist"]
        self.dist_related_inzone_reward = env_conf["dist_related_inzone_reward"]
        self.dist_related_inzone_reward_dist = env_conf["dist_related_inzone_reward_dist"]
        self.dist_related_inzone_reward_max_value = env_conf["dist_related_inzone_reward_max_value"]
        self.ADVERSARY_ACTION = env_conf["adversary_action"]
        self.PGD_ATTACK = env_conf["pgd_attack"]
        self.Q_values = None

        self.current_step = 0
        self.current_all_step = 0
        self.rl_agent_count = 0     # every agent needs new id agent_{self.rl_agent_count}

        # x: -20- 20, y: 30 - 70
        self._playground_shape = {"x1": -30.0,
                                  "x2": 30.0,
                                  "y1": 20.0,
                                  "y2": 80.0}
        self._fake_playground_shape = {"x1": self._playground_shape["x1"] - self.FAKE_MOD_DIST,
                                       "x2": self._playground_shape["x2"] + self.FAKE_MOD_DIST,
                                       "y1": self._playground_shape["y1"] - self.FAKE_MOD_DIST,
                                       "y2": self._playground_shape["y2"] + self.FAKE_MOD_DIST}

        self.playground_diagonal = math.ceil(sqrt(pow(self._playground_shape["x1"] - self._playground_shape["x2"], 2) + pow(self._playground_shape["y1"] - self._playground_shape["y2"], 2)))

        self.obs_key_correlation = dict()

        if self.OBS_SPACE_TYPE == "abs_coord":        # absolute coordinates
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=self._playground_shape["x1"], high=self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=self._playground_shape["y1"], high=self._playground_shape["y2"], shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "rel_coord":      # relative coordinates
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=2*self._playground_shape["x1"], high=2*self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=-2*(self._playground_shape["y2"] - self._playground_shape["y1"]), high=2*(self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "abs_coord_with_dest":    # absolute coordinates with from-to information
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=self._playground_shape["x1"], high=self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=self._playground_shape["y1"], high=self._playground_shape["y2"], shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "start_edge": spaces.Discrete(5, start=0),             # edge ids are 1, 2, 3, 4
                "dest_edge": spaces.Discrete(5, start=-4)              # edge ids are -1, -2, -3, -4
            })
        elif self.OBS_SPACE_TYPE == "rel_coord_with_dest":  # relative coordinates with from-to information
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "start_edge": spaces.Discrete(5, start=0),  # edge ids are 1, 2, 3, 4
                "dest_edge": spaces.Discrete(5, start=-4)  # edge ids are -1, -2, -3, -4
            })
        elif self.OBS_SPACE_TYPE == "rel_coord_with_dest_and_memory":     # relative coordinates with from-to information and 3-frame memory
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "start_edge": spaces.Discrete(5, start=0),  # edge ids are 1, 2, 3, 4
                "dest_edge": spaces.Discrete(5, start=-4),  # edge ids are -1, -2, -3, -4
                "x_1": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y_1": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed_1": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "x_2": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y_2": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed_2": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "x_3": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,), dtype=np.float32),
                "y_3": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,), dtype=np.float32),
                "speed_3": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "pol_coord_with_dest":
            vehicle_state = spaces.Dict({
                "d": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "d_diff": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "start_edge": spaces.Discrete(5, start=0),  # edge ids are 1, 2, 3, 4
                "dest_edge": spaces.Discrete(5, start=-4),  # edge ids are -1, -2, -3, -4
            })
        elif self.OBS_SPACE_TYPE == "pol_coord_with_dest_and_memory":
            vehicle_state = spaces.Dict({
                "d": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "start_edge": spaces.Discrete(5, start=0),  # edge ids are 1, 2, 3, 4
                "dest_edge": spaces.Discrete(5, start=-4),  # edge ids are -1, -2, -3, -4
                "d_1": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff_1": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_1": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_1": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "d_2": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff_2": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_2": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_2": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "d_3": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_3": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_3": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
            })
        elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec":
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,),
                                dtype=np.float32),
                "y": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]),
                                high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,),
                                dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "movement_vector": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec_and_memory":
            vehicle_state = spaces.Dict({
                "x": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"], shape=(1,),
                                dtype=np.float32),
                "y": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]),
                                high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,),
                                dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "movement_vector": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                "x_1": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"],
                                  shape=(1,), dtype=np.float32),
                "y_1": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]),
                                  high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,),
                                  dtype=np.float32),
                "speed_1": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "x_2": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"],
                                  shape=(1,), dtype=np.float32),
                "y_2": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]),
                                  high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,),
                                  dtype=np.float32),
                "speed_2": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "x_3": spaces.Box(low=2 * self._playground_shape["x1"], high=2 * self._playground_shape["x2"],
                                  shape=(1,), dtype=np.float32),
                "y_3": spaces.Box(low=-2 * (self._playground_shape["y2"] - self._playground_shape["y1"]),
                                  high=2 * (self._playground_shape["y2"] - self._playground_shape["y1"]), shape=(1,),
                                  dtype=np.float32),
                "speed_3": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec":
            vehicle_state = spaces.Dict({
                "d": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "d_diff": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "movement_vector": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })
        elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec_and_memory":
            vehicle_state = spaces.Dict({
                "d": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "movement_vector": spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                "speed": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "d_1": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff_1": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_1": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_1": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "d_2": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "d_diff_2": spaces.Box(low=-self.playground_diagonal, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_2": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_2": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32),
                "d_3": spaces.Box(low=0, high=self.playground_diagonal, shape=(1,), dtype=np.float32),
                "angle_3": spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
                "speed_3": spaces.Box(low=0.0, high=self.MAX_SPEED, shape=(1,), dtype=np.float32)
            })

        # observation space contains only the N closest vehicles data sorted by distance (0 --> all vehicles in the zone are included in the observation)
        if self.OBS_SPACE_N_CLOSEST == 0:
            obs_dict = {f"agent_{i}": vehicle_state for i in list(range(self.AGENT_NUM))}               # obs space contains AGENT_NUM vehicle data
        else:
            obs_dict = {f"agent_{i}": vehicle_state for i in list(range(self.OBS_SPACE_N_CLOSEST))}     # obs space contains closest OBS_SPACE_N_CLOSEST vehicle data
            self.SORT_OBS_BY_DIST = True                                                                # must be True to sort by distance
        obs_dict['act_agent'] = copy.deepcopy(vehicle_state)                                                           # and ego vehicle data

        # clean act_agent obs from 0.0s (e.g. x, y in rel_coord_* and pol_coord_* observation spaces)
        if self.OBS_CLEAN_ACT_AGENT_OBS:
            if "rel_coord" in self.OBS_SPACE_TYPE or "pol_coord" in self.OBS_SPACE_TYPE:
                obs_dict = self.remove_unused_act_agent_obs_spaces(obs_dict)

        self.observation_space = spaces.Dict(obs_dict)

        # Observations to return:
        # {
        #     "agent_0": {
        #         'act_agent': {
        #             'x':
        #             'y':
        #             'speed':
        #         },
        #         'agent_0': {      # agent_0 data is already in 'act_agent', so reset values here
        #             'x':
        #             'y':
        #             'speed':
        #         },
        #         'agent_1': {
        #             'x':
        #             'y':
        #             'speed':
        #         },
        #         ...
        #     },
        #     "agent_1": {...}
        #      ...
        # }

        self.action_space = spaces.Discrete(3)

        self.data = None
        self.avg_speed_agent0 = 0
        self.rewards = {}
        self.colliding_num = 0
        self.total_collisions = 0
        self.total_collisions_in_episode = 0
        self.fake_collisions = 0
        self.fake_collisions_in_episode = 0
        self.non_agent_collisions = 0
        self.non_agent_collisions_in_episode = 0
        self.total_vehicle_count = 0
        self.total_vehicle_count_in_episode = 0
        self.collision_ratio = 0
        self.collision_ratio_in_episode = 0
        self.removed_vehicles_num = 0
        self.vehicles_in_zone = 0
        self.any_action = False

        if not config.TRAIN:
            self.writer = SummaryWriter(logdir=f"runs/exp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", max_queue=10000, flush_secs=5)

        if env_conf["sumo_gui"]:
            self.sumo_command = 'sumo-gui'
        else:
            self.sumo_command = 'sumo'

        # Start the thread for monitoring key presses
        # thread = threading.Thread(target=change_logging_level)
        # thread.daemon = True
        # thread.start()

        super().__init__()


    def step(self, action_dict):
        """
        Controls the simulation by controlling all the functions below and also steps the simulation.
        :param action_dict: A dictionary of the given actions for all the agent that are in-use.
        :return: Returns the observations of the ENV, the dictionary of the calculated rewards for each agent that are
                    non-free. A terminated and truncated dictionary if the agents are done for the simulation.
                    Some additional information if needed.
        """
        self.current_step += 1
        self.current_all_step += 1

        if action_dict and self.TERMINATING_AGENTS:
            action_dict = self.action_dict_wrapper(action_dict)

        traci.simulationStep()

        in_collision, crossed, in_zone, rl_agent_db, agents_wo_any_action, non_agent_in_collision, fake_in_collision = self.manage_data_table()
        # self.take_action(action_dict, in_collision)
        if self.ADVERSARY_ACTION:
            adv_func.take_adversary_action()
        rewards = self.calculate_reward(action_dict, in_collision, crossed, in_zone, self.reward_values)

        # monitoring for tensorboard logs
        self.vehicles_in_zone = self.AGENT_NUM - self.data["is_free"].sum()
        self.total_collisions = self.total_collisions + math.ceil(len(in_collision) / 2)        # usually 2 vehicles collide, but count only 1 collision occurrence
        self.total_collisions_in_episode = self.total_collisions_in_episode + math.ceil(len(in_collision) / 2)
        self.fake_collisions = self.fake_collisions + len(fake_in_collision)                    # number of fake vehicles in collision
        self.fake_collisions_in_episode = self.fake_collisions_in_episode + len(fake_in_collision)
        self.non_agent_collisions = self.non_agent_collisions + len(non_agent_in_collision)      # number of non-agent vehicles in collision
        self.non_agent_collisions_in_episode = self.non_agent_collisions_in_episode + len(non_agent_in_collision)
        self.collision_ratio = self.total_collisions / self.total_vehicle_count
        self.collision_ratio_in_episode = self.total_collisions_in_episode / self.total_vehicle_count_in_episode

        if self.OBS_SPACE_TYPE == "abs_coord":  # absolute coordinates
            obs = self.get_observation
        elif self.OBS_SPACE_TYPE == "rel_coord":  # relative coordinates
            obs = self.get_relative_observation
        elif self.OBS_SPACE_TYPE == "abs_coord_with_dest":    # absolute coordinates with from-to information
            obs = self.get_observation_with_dest
        elif self.OBS_SPACE_TYPE == "rel_coord_with_dest":    # relative coordinates with from-to information
            obs = self.get_relative_observation_with_dest
        elif self.OBS_SPACE_TYPE == "rel_coord_with_dest_and_memory":
            obs = self.get_relative_observation_with_dest_and_memory
        elif self.OBS_SPACE_TYPE == "pol_coord_with_dest":
            obs = self.get_pol_observation_with_dest
        elif self.OBS_SPACE_TYPE == "pol_coord_with_dest_and_memory":
            obs = self.get_pol_observation_with_dest_and_memory
        elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec":
            obs = self.get_relative_observation_with_mov_vec
        elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec_and_memory":
            obs = self.get_relative_observation_with_mov_vec_and_memory
        elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec":
            obs = self.get_pol_observation_with_mov_vec
        elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec_and_memory":
            obs = self.get_pol_observation_with_mov_vec_and_memory
        else:
            obs = {}
            logging.critical(f"Observation space ({self.OBS_SPACE_TYPE}) not defined!")

        if self.ACTION_RANGE_FILTER:
            self.action_range_filtering(obs)
        if self.ACTION_CONFLICT_FILTER:
            self.action_conflict_filtering(obs)
        if self.ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER:
            self.action_movement_vector_filtering(obs)
        if self.SORT_OBS_BY_DIST:
            _, self.obs_key_correlation = self.sort_observation_dict_by_distance(self, obs=obs)       # modifies agent_id's in the observation dictionary
            if self.OBS_SPACE_N_CLOSEST > 0:
                self.filter_obs_space_n_closest(self, obs=obs)
        if self.OBS_CLEAN_ACT_AGENT_OBS:
            if "rel_coord" in self.OBS_SPACE_TYPE or "pol_coord" in self.OBS_SPACE_TYPE:
                for agent in obs:
                    obs[agent] = self.remove_unused_act_agent_obs(obs[agent])

        if config.SUMO_GUI:
            self.visualize_observed_vehicles_GUI(obs=obs, key_corr=self.obs_key_correlation)

        info = {}
        terminated = {}
        truncated = {}
        terminated["__all__"] = False
        truncated["__all__"] = False

        if self.current_step == self.episode_length-1:
            terminated["__all__"] = True
            truncated["__all__"] = True

        if self.TERMINATING_AGENTS:
            if in_collision:                  # terminate if collision happens (based on config)
                for agent in np.intersect1d(in_collision, rl_agent_db.index):       # only terminate vehicle type == "agent" (exist in rl_agent_db)
                    terminated[agent] = True
                    truncated[agent] = True
                if self.restart_episode_on_collision and len(np.intersect1d(in_collision, rl_agent_db.index)) != 0 and self.any_action:
                    terminated["__all__"] = True
                    truncated["__all__"] = True
            elif crossed:
                for agent in np.intersect1d(crossed, rl_agent_db.index):
                    if agent not in agents_wo_any_action:
                        terminated[agent] = True
                        truncated[agent] = True
        elif not self.TERMINATING_AGENTS:
            if in_collision and self.restart_episode_on_collision:  # reset if collision happens (based on config)
                for agent in in_collision:
                    terminated[agent] = True
                    truncated[agent] = True
                terminated["__all__"] = True
                truncated["__all__"] = True

        for agent in list(rewards.keys()):        # remove rewards for vehicles that never had any action
            if agent in agents_wo_any_action:
                rewards.pop(agent)

        logging.info(f"\n*******************************************************\n"
                     f"step: {self.current_step} \t\t({self.current_all_step}/{self.total_steps})\n"
                     f"action: {action_dict} \n"
                     f"reward: {rewards} \n"
                     f"observation: {list(obs.keys())}\n"
                     f"colliding: {in_collision} \t\t\t\t (total:{self.total_collisions})\n"
                     # f"speed: {(self.data[self.data['speed'].notnull()]['speed']).to_dict()} \n"
                     f"terminater/trancated: {terminated['__all__']}/{truncated['__all__']} \n")
        logging.info(''.join("\n%s (%s)\n%s\n" % (key1,
                                                  self.data.at[key1, 'vehicle_id'],
                                                  ('\n'.join("%s: %s" % (key2,
                                                                         ('\t'.join("%s: %-10.5r" % (key3, val3 if isinstance(val3, int) else val3[0]) for (key3, val3) in val2.items()))
                                                                         ) for (key2, val2) in val1.items()))
                                                  ) for (key1, val1) in obs.items()))

        # if isinstance(val3, int) else val3[0]

        # show progress bar
        if logging.getLogger().level > logging.INFO:
            self.progress_bar(count=self.current_all_step, total=self.total_steps, status=f"{self.current_all_step}/{self.total_steps}")

        if not self.data.at[f"agent_0", "is_free"]:
            self.avg_speed_agent0 = self.data.at[f"agent_0", "speed"]

        if not config.TRAIN:
            self.writer.add_scalar('avg_speed_agent0', self.avg_speed_agent0, self.current_all_step)
            self.writer.add_scalar('removed_vehicles', self.removed_vehicles_num, self.current_all_step)
            self.writer.add_scalar('reward', sum(self.rewards.values()), self.current_all_step)
            self.writer.add_scalar('collision_num', self.colliding_num, self.current_all_step)
            self.writer.add_scalar('total_collision_num', self.total_collisions, self.current_all_step)
            self.writer.add_scalar('fake_collision_num', self.fake_collisions, self.current_all_step)
            self.writer.add_scalar('non_agent_collision_num', self.non_agent_collisions, self.current_all_step)
            self.writer.add_scalar('collision_ratio', self.total_collisions/self.total_vehicle_count, self.current_all_step)
            self.writer.add_scalar('total_vehicle_count', self.total_vehicle_count, self.current_all_step)


        if self.TERMINATING_AGENTS:
            if obs:
                obs = self.obs_wrapper(obs)
            if rewards:
                rewards = self.reward_wrapper(rewards, rl_agent_db)
            if len(terminated.keys()) > 1 or len(truncated.keys()) > 1:
                terminated, truncated = self.terminated_truncate_wrapper(terminated, truncated, rl_agent_db)

        return obs, rewards, terminated, truncated, info


    def action_dict_wrapper(self, action_dict):
        """
        New rl_agent_id is needed for each new agent vehicle. If vehicles leaves the intersection or collides: terminated=True
        (Convert rl_agent_id to agent_id)
        :param action_dict:
        :return:
        """
        wrapped_action_dict = {}
        for rl_agent_id in action_dict:
            agent_id = self.data.index[self.data['rl_agent_id'] == rl_agent_id][0]
            wrapped_action_dict[agent_id] = action_dict[rl_agent_id]
            logging.info(f"Action wrapper: {rl_agent_id} --> {agent_id}")
        return wrapped_action_dict

    def obs_wrapper(self, obs):
        """
        Convert agent_id to rl_agent_id
        """
        wrapped_obs = {}
        for agent_id in obs:
            rl_agent_id = self.data.at[agent_id, 'rl_agent_id']
            wrapped_obs[rl_agent_id] = obs[agent_id]
            logging.info(f"Obs wrapper: {agent_id} --> {rl_agent_id}")
        return wrapped_obs

    def reward_wrapper(self, rewards, rl_agent_db):
        """
        Convert agent_id to rl_agent_id
        """
        wrapped_rewards = {}
        for agent_id in rewards:
            rl_agent_id = rl_agent_db[agent_id]
            wrapped_rewards[rl_agent_id] = rewards[agent_id]
            logging.info(f"Reward wrapper: {agent_id} --> {rl_agent_id}")
        return wrapped_rewards

    def terminated_truncate_wrapper(self, terminated, truncated, rl_agent_db):
        """
        Convert agent_id to rl_agent_id
        """
        wrapped_terminated = {}
        wrapped_truncated = {}
        wrapped_terminated["__all__"] = terminated["__all__"]
        wrapped_truncated["__all__"] = truncated["__all__"]
        for agent_id in terminated:
            if agent_id != '__all__':
                rl_agent_id = rl_agent_db[agent_id]
                wrapped_terminated[rl_agent_id] = terminated[agent_id]
                logging.info(f"Terminated/Truncated wrapper: {agent_id} --> {rl_agent_id}")
        for agent_id in truncated:
            if agent_id != '__all__':
                rl_agent_id = rl_agent_db[agent_id]
                wrapped_truncated[rl_agent_id] = truncated[agent_id]
                # logging.info(f"Truncated wrapper: {agent_id} --> {rl_agent_id}")
        return wrapped_terminated, wrapped_truncated


    def reset(self, *, seed=None, options=None):
        """
        Resets the simulation.
        :return: Returns the observations of the ENV and some additional information if needed.
        """
        logging.info("\nRESET\n")
        self.current_step = 0
        self.any_action = False
        self.total_collisions_in_episode = 0
        self.fake_collisions_in_episode = 0
        self.non_agent_collisions_in_episode = 0
        self.total_vehicle_count_in_episode = 0
        self.collision_ratio_in_episode = 0
        self.avg_speed_agent0 = 0
        self.rl_agent_count = 0
        self.data = self.reset_data_table()
        if config.TRAIN and config.TRAIN_WITH_TUNE:
            os.chdir(os.getenv("TUNE_ORIG_WORKING_DIR"))

        if self.sumo_random_route:
            os.system(f"{sys.executable} {os.path.join('env', 'randomTrips.py')} -n {os.path.join('env', '8_8_network.net.xml')} -e {config.EPISODE_LENGTH * config.SUMO_STEP_LENGTH} -l --random -p {self.sumo_traffic_insertation_period} -o {os.path.join('env', 'out.trips_rand.xml')}")
            os.system(f"duarouter -n {os.path.join('env', '8_8_network.net.xml')} -r {os.path.join('env', 'out.trips_rand.xml')} -o {os.path.join('env', '8_8_routes_rand.rou.xml')} --ignore-errors --no-warnings")
            if config.SUMO_TURNS_ALLOWED:
                sumocfg = os.path.join("env", "8_8_conf_randrou.sumocfg")     # turns allowed
            else:
                sumocfg = os.path.join("env", "8_8_conf_flow.sumocfg")          # turns disallowed: just WE, EW, SN, NS random traffic
        else:
            sumocfg = os.path.join("env", "8_8_conf.sumocfg")

        try:
            traci.start([self.sumo_command, "-c", sumocfg, "--step-length", str(self.sumo_step_length), "-S", "--quit-on-end", "--random", "--collision.check-junctions", "--collision.action", "remove", "--collision.mingap-factor", "0", "--no-warnings", "--no-step-log"])  # start a new simulation
        except:
            self.sumo_restart(sumocfg)

        # draw playground_shape = observation_zone
        traci.polygon.add(
            polygonID="observation_zone",
            shape=(
                (self._playground_shape["x1"], self._playground_shape["y1"]),
                (self._playground_shape["x1"], self._playground_shape["y2"]),
                (self._playground_shape["x2"], self._playground_shape["y2"]),
                (self._playground_shape["x2"], self._playground_shape["y1"])
            ),
            color=(100, 200, 0, 255),
            fill=True,
            lineWidth=0.5
        )

        # while not self.is_in_zone('0'):
        while not self.data["vehicle_id"].any():
            traci.simulationStep()
            self.manage_data_table()
            if self.OBS_SPACE_TYPE == "abs_coord":  # absolute coordinates
                obs = self.get_observation
            elif self.OBS_SPACE_TYPE == "rel_coord":  # relative coordinates
                obs = self.get_relative_observation
            elif self.OBS_SPACE_TYPE == "abs_coord_with_dest":    # absolute coordinates with from-to information
                obs = self.get_observation_with_dest
            elif self.OBS_SPACE_TYPE == "rel_coord_with_dest":  # relative coordinates with from-to information
                obs = self.get_relative_observation_with_dest
            elif self.OBS_SPACE_TYPE == "rel_coord_with_dest_and_memory":
                obs = self.get_relative_observation_with_dest_and_memory
            elif self.OBS_SPACE_TYPE == "pol_coord_with_dest":
                obs = self.get_pol_observation_with_dest
            elif self.OBS_SPACE_TYPE == "pol_coord_with_dest_and_memory":
                obs = self.get_pol_observation_with_dest_and_memory
            elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec":
                obs = self.get_relative_observation_with_mov_vec
            elif self.OBS_SPACE_TYPE == "rel_coord_with_mov_vec_and_memory":
                obs = self.get_relative_observation_with_mov_vec_and_memory
            elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec":
                obs = self.get_pol_observation_with_mov_vec
            elif self.OBS_SPACE_TYPE == "pol_coord_with_mov_vec_and_memory":
                obs = self.get_pol_observation_with_mov_vec_and_memory
            else:
                obs = {}
                logging.critical(f"Observation space ({self.OBS_SPACE_TYPE}) not defined!")

            if self.ACTION_RANGE_FILTER:
                self.action_range_filtering(obs)
            if self.ACTION_CONFLICT_FILTER:
                self.action_conflict_filtering(obs)
            if self.ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER:
                self.action_movement_vector_filtering(obs)
            if self.SORT_OBS_BY_DIST:
                _, self.obs_key_correlation = self.sort_observation_dict_by_distance(self, obs=obs)  # modifies agent_id's in the observation dictionary
                if self.OBS_SPACE_N_CLOSEST > 0:
                    self.filter_obs_space_n_closest(self, obs=obs)
            if self.OBS_CLEAN_ACT_AGENT_OBS:
                if "rel_coord" in self.OBS_SPACE_TYPE or "pol_coord" in self.OBS_SPACE_TYPE:
                    for agent in obs:
                        obs[agent] = self.remove_unused_act_agent_obs(obs[agent])

        info = {}
        return obs, info


    def close(self):
        """
        Closes the ENV.
        :return: True if it was successful.
        """
        logging.critical("\nCLOSE")
        traci.close(wait=True)
        return True

    '''
    HELPER FUNCTIONS:
    '''

    def take_action(self, action_dict, in_collision):
        """
        Based on the received actions dictionary, it carries out one of the following actions for each agent in-use:
        - slow down
        - accelerate
        - keep the speed unchanged
        :param action_dict: Dictionary of actions for each agent that are non-free.
        """
        sim_time = traci.simulation.getTime()

        # collisions = traci.simulation.getCollidingVehiclesIDList()
        # self.colliding_num = len(collisions)
        # in_collision = [agent for agent in self.data.index if self.data.at[agent, "vehicle_id"] in collisions]

        # take action for not colliding agents (colliding vehicles are removed from simulation and can not be accessed with TraCI)

        for agent, row in self.data.drop(in_collision).iterrows():  # iterate agents not in collision
            if config.SUMO_GUI and not row.is_free:
                traci.vehicle.setColor(row.vehicle_id, (120, 120, 0, 255))  # set color: vehicle in the intersection
            if agent in action_dict.keys() and not row.is_free and row.start_time != sim_time and row.vehicle_type == "agent":
                self.data.at[agent, "had_any_action"] = True
                if action_dict[agent] == 0:  # slow down
                    traci.vehicle.setAcceleration(row.vehicle_id, -self.accel, self.sumo_step_length)  # decelerate with 4.5 m/s2 for config.SUMO_STEP_LENGTH sec (4.5 m/s2 is the threshold of emergency breaking warning)
                    if config.SUMO_GUI:
                        traci.vehicle.setColor(row.vehicle_id, (255, 0, 0, 255))        # red
                elif action_dict[agent] == 1:  # keep speed
                    traci.vehicle.setAcceleration(row.vehicle_id, 0.0, self.sumo_step_length)
                    if config.SUMO_GUI:
                        traci.vehicle.setColor(row.vehicle_id, (0, 255, 0, 255))        # green
                elif action_dict[agent] == 2:  # accelerate
                    traci.vehicle.setAcceleration(row.vehicle_id, self.accel, self.sumo_step_length)  # accelerate with 4.5 m/s2 for config.SUMO_STEP_LENGTH sec
                    if config.SUMO_GUI:
                        traci.vehicle.setColor(row.vehicle_id, (0, 0, 255, 255))        # blue
            elif row.vehicle_type == "agent":           # vehicle without action, set speed to 50km/h
                traci.vehicle.setSpeed(row.vehicle_id, 13.0)
            elif self.NON_AGENT_SWITCH == True and row.vehicle_type == "non_agent":
                if config.SUMO_GUI:
                    traci.vehicle.setColor(row.vehicle_id, (255, 0, 255, 255))
            elif self.FAKE_SWITCH == True and row.vehicle_type == "fake":
                if config.SUMO_GUI:
                    traci.vehicle.setColor(row.vehicle_id, (100, 0, 32, 255))
        return


    @property
    def get_observation(self):
        """
        Takes an observation of each agent that are in-use based on the predefined observation-space.
        (X coordinates, Y coordinates, speed of the vehicle, for each agent that are in-use)
        :return: Dictionary of the observation for the non-free agents.
        """
        observations = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            # agent vehicle data is in agent_obs["act_agent"], while other vehicle data is in ["agent_0"], ["agent_1"], ...
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_obs = {}
                agent_obs["act_agent"] = {
                    'x': np.array(np.array([row['pos_x']])).astype(np.float32),
                    'y': np.array(np.array([row['pos_y']])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32)
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32),
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32)
                        }
                    agent_obs[other_agent] = agent_data
                observations[agent] = agent_obs
        return observations

    @property
    def get_relative_observation(self):
        """
        :return: Returns a relative observation. In the center (0,0) there is the car controlled by the "act_agent"
        and the other cars controlled by other agents have their (x,y) values relative to the act_agent's car.
        Similar to "get_observation".
        """
        relative_observation = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_rel_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                agent_rel_obs["act_agent"] = {
                    'x': np.array(np.array([0.0])).astype(np.float32),
                    'y': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32)
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        rel_agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        rel_agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0],
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1],
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32)
                        }
                    agent_rel_obs[other_agent] = rel_agent_data
                relative_observation[agent] = agent_rel_obs
        return relative_observation

    @property
    def get_observation_with_dest(self):
        """
        Takes an observation of each agent that are in-use based on the predefined observation-space.
        (X coordinates, Y coordinates, speed of the vehicle, for each agent that are in-use)
        :return: Dictionary of the observation for the non-free agents.
        """
        observations = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            # agent vehicle data is in agent_obs["act_agent"], while other vehicle data is in ["agent_0"], ["agent_1"], ...
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_obs = {}
                agent_obs["act_agent"] = {
                    'x': np.array(np.array([row['pos_x']])).astype(np.float32),
                    'y': np.array(np.array([row['pos_y']])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "start_edge": int(row['start_edge']),
                    "dest_edge": int(row['dest_edge'])
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "start_edge": 0,
                            "dest_edge": 0
                        }
                    else:
                        agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32),
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "start_edge": int(other_row['start_edge']),
                            "dest_edge": int(other_row['dest_edge'])
                        }
                    agent_obs[other_agent] = agent_data
                observations[agent] = agent_obs
        return observations

    @property
    def get_relative_observation_with_dest(self):
        """
        :return: Returns a relative observation. In the center (0,0) there is the car controlled by the "act_agent"
        and the other cars controlled by other agents have their (x,y) values relative to the act_agent's car.
        Similar to "get_observation".
        """
        relative_observation = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_rel_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                agent_rel_obs["act_agent"] = {
                    'x': np.array(np.array([0.0])).astype(np.float32),
                    'y': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "start_edge": int(row['start_edge']),
                    "dest_edge": int(row['dest_edge'])
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:              # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        rel_agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "start_edge": 0,
                            "dest_edge": 0
                        }
                    else:
                        rel_agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0],
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1],
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "start_edge": int(other_row['start_edge']),
                            "dest_edge": int(other_row['dest_edge'])
                        }
                    agent_rel_obs[other_agent] = rel_agent_data
                relative_observation[agent] = agent_rel_obs
        return relative_observation

    @property
    def get_relative_observation_with_dest_and_memory(self):
        """
        :return: Returns a relative observation. In the center (0,0) there is the car controlled by the "act_agent"
        and the other cars controlled by other agents have their (x,y) values relative to the act_agent's car.
        Similar to "get_observation".
        """
        relative_observation = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_rel_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                agent_rel_obs["act_agent"] = {
                    'x': np.array(np.array([0.0])).astype(np.float32),
                    'y': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "start_edge": int(row['start_edge']),
                    "dest_edge": int(row['dest_edge']),
                    "x_1": np.array(np.array([row['x_1']])).astype(np.float32),
                    "y_1": np.array(np.array([row['y_1']])).astype(np.float32),
                    "speed_1": np.array(np.array([row['speed_1']])).astype(np.float32),
                    "x_2": np.array(np.array([row['x_2']])).astype(np.float32),
                    "y_2": np.array(np.array([row['y_2']])).astype(np.float32),
                    "speed_2": np.array(np.array([row['speed_2']])).astype(np.float32),
                    "x_3": np.array(np.array([row['x_3']])).astype(np.float32),
                    "y_3": np.array(np.array([row['y_3']])).astype(np.float32),
                    "speed_3": np.array(np.array([row['speed_3']])).astype(np.float32),
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        rel_agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "start_edge": 0,
                            "dest_edge": 0,
                            "x_1": np.array(np.array([0.0])).astype(np.float32),
                            "y_1": np.array(np.array([0.0])).astype(np.float32),
                            "speed_1": np.array(np.array([0.0])).astype(np.float32),
                            "x_2": np.array(np.array([0.0])).astype(np.float32),
                            "y_2": np.array(np.array([0.0])).astype(np.float32),
                            "speed_2": np.array(np.array([0.0])).astype(np.float32),
                            "x_3": np.array(np.array([0.0])).astype(np.float32),
                            "y_3": np.array(np.array([0.0])).astype(np.float32),
                            "speed_3": np.array(np.array([0.0])).astype(np.float32),
                        }
                    else:
                        rel_agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0],
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1],
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "start_edge": int(other_row['start_edge']),
                            "dest_edge": int(other_row['dest_edge']),
                            "x_1": np.array(np.array([other_row['x_1']])).astype(np.float32),
                            "y_1": np.array(np.array([other_row['y_1']])).astype(np.float32),
                            "speed_1": np.array(np.array([other_row['speed_1']])).astype(np.float32),
                            "x_2": np.array(np.array([other_row['x_2']])).astype(np.float32),
                            "y_2": np.array(np.array([other_row['y_2']])).astype(np.float32),
                            "speed_2": np.array(np.array([other_row['speed_2']])).astype(np.float32),
                            "x_3": np.array(np.array([other_row['x_3']])).astype(np.float32),
                            "y_3": np.array(np.array([other_row['y_3']])).astype(np.float32),
                            "speed_3": np.array(np.array([other_row['speed_3']])).astype(np.float32),
                        }
                    agent_rel_obs[other_agent] = rel_agent_data
                relative_observation[agent] = agent_rel_obs
        return relative_observation


    @property
    def get_pol_observation_with_dest(self):
        # get relative observation dict to get the relative coordinates of each agent from the "act_agent"
        polar_observation = {}

        # Cartesian to Polar
        def cart2pol(x, y):
            d = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)
            return {"d": d, "angle": angle}

        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_pol_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                real_coordinates_1 = [np.array(np.array([row['x_1']])).astype(np.float32),
                                      np.array(np.array([row['y_1']])).astype(np.float32)]
                agent_pol_obs["act_agent"] = {
                    'd': np.array(np.array([0.0])).astype(np.float32),
                    'angle': np.array(np.array([0.0])).astype(np.float32),
                    'd_diff': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "start_edge": int(row['start_edge']),
                    "dest_edge": int(row['dest_edge'])
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        pol_agent_data = {
                            'd': np.array(np.array([0.0])).astype(np.float32),
                            'angle': np.array(np.array([0.0])).astype(np.float32),
                            'd_diff': np.array(np.array([0.0])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "start_edge": 0,
                            "dest_edge": 0
                        }
                    else:
                        x = np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0]
                        y = np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1]
                        x_1 = other_row['x_1'] - real_coordinates_1[0] if other_row['x_1'] != 0.0 and real_coordinates_1[0] != 0.0 else 0.0
                        y_1 = other_row['y_1'] - real_coordinates_1[1] if other_row['y_1'] != 0.0 and real_coordinates_1[1] != 0.0 else 0.0
                        pol_agent_data = {
                            'd': np.array(np.array(cart2pol(x, y)["d"])).astype(np.float32),
                            'angle': np.array(np.array(cart2pol(x, y)["angle"])).astype(np.float32),
                            'd_diff': np.array(np.array(cart2pol(x, y)["d"] - cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "start_edge": int(other_row['start_edge']),
                            "dest_edge": int(other_row['dest_edge'])
                        }
                    agent_pol_obs[other_agent] = pol_agent_data
                polar_observation[agent] = agent_pol_obs
        return polar_observation


    @property
    def get_pol_observation_with_dest_and_memory(self):
        # get relative observation dict to get the relative coordinates of each agent from the "act_agent"
        polar_observation = {}

        # Cartesian to Polar
        def cart2pol(x, y):
            d = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)
            return {"d": d, "angle": angle}

        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_pol_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32), np.array(np.array([row['pos_y']])).astype(np.float32)]
                real_coordinates_1 = [np.array(np.array([row['x_1']])).astype(np.float32), np.array(np.array([row['y_1']])).astype(np.float32)]
                real_coordinates_2 = [np.array(np.array([row['x_2']])).astype(np.float32), np.array(np.array([row['y_2']])).astype(np.float32)]
                real_coordinates_3 = [np.array(np.array([row['x_3']])).astype(np.float32), np.array(np.array([row['y_3']])).astype(np.float32)]

                agent_pol_obs["act_agent"] = {
                    'd': np.array(np.array([0.0])).astype(np.float32),
                    'angle': np.array(np.array([0.0])).astype(np.float32),
                    'd_diff': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "start_edge": int(row['start_edge']),
                    "dest_edge": int(row['dest_edge']),
                    "d_1": np.array(np.array([0.0])).astype(np.float32),
                    "d_diff_1": np.array(np.array([0.0])).astype(np.float32),
                    "angle_1": np.array(np.array([0.0])).astype(np.float32),
                    'speed_1': np.array(np.array([row['speed_1']])).astype(np.float32),
                    "d_2": np.array(np.array([0.0])).astype(np.float32),
                    "d_diff_2": np.array(np.array([0.0])).astype(np.float32),
                    "angle_2": np.array(np.array([0.0])).astype(np.float32),
                    'speed_2': np.array(np.array([row['speed_2']])).astype(np.float32),
                    "d_3": np.array(np.array([0.0])).astype(np.float32),
                    "angle_3": np.array(np.array([0.0])).astype(np.float32),
                    'speed_3': np.array(np.array([row['speed_3']])).astype(np.float32),

                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        pol_agent_data = {
                            'd': np.array(np.array([0.0])).astype(np.float32),
                            'angle': np.array(np.array([0.0])).astype(np.float32),
                            'd_diff': np.array(np.array([0.0])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "start_edge": 0,
                            "dest_edge": 0,
                            "d_1": np.array(np.array([0.0])).astype(np.float32),
                            "d_diff_1": np.array(np.array([0.0])).astype(np.float32),
                            "angle_1": np.array(np.array([0.0])).astype(np.float32),
                            'speed_1': np.array(np.array([0.0])).astype(np.float32),
                            "d_2": np.array(np.array([0.0])).astype(np.float32),
                            "d_diff_2": np.array(np.array([0.0])).astype(np.float32),
                            "angle_2": np.array(np.array([0.0])).astype(np.float32),
                            'speed_2': np.array(np.array([0.0])).astype(np.float32),
                            "d_3": np.array(np.array([0.0])).astype(np.float32),
                            "angle_3": np.array(np.array([0.0])).astype(np.float32),
                            'speed_3': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        x = np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0]
                        y = np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1]
                        x_1 = other_row['x_1'] - real_coordinates_1[0] if other_row['x_1'] != 0.0 and real_coordinates_1[0] != 0.0 else 0.0
                        y_1 = other_row['y_1'] - real_coordinates_1[1] if other_row['y_1'] != 0.0 and real_coordinates_1[1] != 0.0 else 0.0
                        x_2 = other_row['x_2'] - real_coordinates_2[0] if other_row['x_2'] != 0.0 and real_coordinates_2[0] != 0.0 else 0.0
                        y_2 = other_row['y_2'] - real_coordinates_2[1] if other_row['y_2'] != 0.0 and real_coordinates_2[1] != 0.0 else 0.0
                        x_3 = other_row['x_3'] - real_coordinates_3[0] if other_row['x_3'] != 0.0 and real_coordinates_3[0] != 0.0 else 0.0
                        y_3 = other_row['y_3'] - real_coordinates_3[1] if other_row['y_3'] != 0.0 and real_coordinates_3[1] != 0.0 else 0.0
                        pol_agent_data = {

                            'd': np.array(np.array(cart2pol(x, y)["d"])).astype(np.float32),
                            'angle': np.array(np.array(cart2pol(x, y)["angle"])).astype(np.float32),
                            'd_diff': np.array(np.array(cart2pol(x, y)["d"] - cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "start_edge": int(other_row['start_edge']),
                            "dest_edge": int(other_row['dest_edge']),
                            "d_1": np.array(np.array(cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            "d_diff_1": np.array(np.array(cart2pol(x_1, y_1)["d"] - cart2pol(x_2, y_2)["d"] if x_1 != 0.0 and x_2 != 0 else [0.0])) .astype(np.float32),
                            "angle_1": np.array(np.array(cart2pol(x_1, y_1)["angle"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'speed_1': np.array(np.array([other_row['speed_1']])).astype(np.float32),
                            "d_2": np.array(np.array(cart2pol(x_2, y_2)["d"] if x_2 != 0.0 else [0.0])).astype(np.float32),
                            "d_diff_2": np.array(np.array(cart2pol(x_2, y_2)["d"] - cart2pol(x_3, y_3)["d"] if x_2 != 0.0 and x_3 != 0 else [0.0])).astype(np.float32),
                            "angle_2": np.array(np.array(cart2pol(x_2, y_2)["angle"] if x_2 != 0.0 else [0.0])).astype(np.float32),
                            'speed_2': np.array(np.array([other_row['speed_2']])).astype(np.float32),
                            "d_3": np.array(np.array(cart2pol(x_3, y_3)["d"] if x_3 != 0.0 else [0.0])).astype(np.float32),
                            "angle_3": np.array(np.array(cart2pol(x_3, y_3)["angle"] if x_3 != 0.0 else [0.0])).astype(np.float32),
                            'speed_3': np.array(np.array([other_row['speed_3']])).astype(np.float32)
                        }
                    agent_pol_obs[other_agent] = pol_agent_data
                polar_observation[agent] = agent_pol_obs
        return polar_observation


    @property
    def get_relative_observation_with_mov_vec(self):
        """
        :return: Returns a relative observation. In the center (0,0) there is the car controlled by the "act_agent"
        and the other cars controlled by other agents have their (x,y) values relative to the act_agent's car.
        Similar to "get_observation".
        """
        relative_observation = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_rel_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                agent_rel_obs["act_agent"] = {
                    'x': np.array(np.array([0.0])).astype(np.float32),
                    'y': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    'movement_vector': np.array(np.array([row['movement_vector']])).astype(np.float32)
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:              # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        rel_agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])),
                            'movement_vector': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        rel_agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0],
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1],
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            'movement_vector': np.array(np.array([other_row['movement_vector']])).astype(np.float32)
                        }
                    agent_rel_obs[other_agent] = rel_agent_data
                relative_observation[agent] = agent_rel_obs
        return relative_observation


    @property
    def get_relative_observation_with_mov_vec_and_memory(self):
        """
        :return: Returns a relative observation. In the center (0,0) there is the car controlled by the "act_agent"
        and the other cars controlled by other agents have their (x,y) values relative to the act_agent's car.
        Similar to "get_observation".
        """
        relative_observation = {}
        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_rel_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                agent_rel_obs["act_agent"] = {
                    'x': np.array(np.array([0.0])).astype(np.float32),
                    'y': np.array(np.array([0.0])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    'movement_vector': np.array(np.array([row['movement_vector']])).astype(np.float32),
                    "x_1": np.array(np.array([row['x_1']])).astype(np.float32),
                    "y_1": np.array(np.array([row['y_1']])).astype(np.float32),
                    "speed_1": np.array(np.array([row['speed_1']])).astype(np.float32),
                    "x_2": np.array(np.array([row['x_2']])).astype(np.float32),
                    "y_2": np.array(np.array([row['y_2']])).astype(np.float32),
                    "speed_2": np.array(np.array([row['speed_2']])).astype(np.float32),
                    "x_3": np.array(np.array([row['x_3']])).astype(np.float32),
                    "y_3": np.array(np.array([row['y_3']])).astype(np.float32),
                    "speed_3": np.array(np.array([row['speed_3']])).astype(np.float32),
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        rel_agent_data = {
                            'x': np.array(np.array([self._playground_shape['x1']])).astype(np.float32),
                            'y': np.array(np.array([self._playground_shape['y1']])).astype(np.float32),
                            'speed': np.array(np.array([0.0])),
                            'movement_vector': np.array(np.array([0.0])).astype(np.float32),
                            "x_1": np.array(np.array([0.0])).astype(np.float32),
                            "y_1": np.array(np.array([0.0])).astype(np.float32),
                            "speed_1": np.array(np.array([0.0])).astype(np.float32),
                            "x_2": np.array(np.array([0.0])).astype(np.float32),
                            "y_2": np.array(np.array([0.0])).astype(np.float32),
                            "speed_2": np.array(np.array([0.0])).astype(np.float32),
                            "x_3": np.array(np.array([0.0])).astype(np.float32),
                            "y_3": np.array(np.array([0.0])).astype(np.float32),
                            "speed_3": np.array(np.array([0.0])).astype(np.float32),
                        }
                    else:
                        rel_agent_data = {
                            'x': np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0],
                            'y': np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1],
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            'movement_vector': np.array(np.array([row['movement_vector']])).astype(np.float32),
                            "x_1": np.array(np.array([other_row['x_1']])).astype(np.float32),
                            "y_1": np.array(np.array([other_row['y_1']])).astype(np.float32),
                            "speed_1": np.array(np.array([other_row['speed_1']])).astype(np.float32),
                            "x_2": np.array(np.array([other_row['x_2']])).astype(np.float32),
                            "y_2": np.array(np.array([other_row['y_2']])).astype(np.float32),
                            "speed_2": np.array(np.array([other_row['speed_2']])).astype(np.float32),
                            "x_3": np.array(np.array([other_row['x_3']])).astype(np.float32),
                            "y_3": np.array(np.array([other_row['y_3']])).astype(np.float32),
                            "speed_3": np.array(np.array([other_row['speed_3']])).astype(np.float32),
                        }
                    agent_rel_obs[other_agent] = rel_agent_data
                relative_observation[agent] = agent_rel_obs
        return relative_observation

    @property
    def get_pol_observation_with_mov_vec(self):
        # get relative observation dict to get the relative coordinates of each agent from the "act_agent"
        polar_observation = {}

        # Cartesian to Polar
        def cart2pol(x, y):
            d = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)
            return {"d": d, "angle": angle}

        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_pol_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32),
                                    np.array(np.array([row['pos_y']])).astype(np.float32)]
                real_coordinates_1 = [np.array(np.array([row['x_1']])).astype(np.float32),
                                      np.array(np.array([row['y_1']])).astype(np.float32)]

                agent_pol_obs["act_agent"] = {
                    'd': np.array(np.array([0.0])).astype(np.float32),
                    'angle': np.array(np.array([0.0])).astype(np.float32),
                    'd_diff': np.array(np.array([0.0])).astype(np.float32),
                    'movement_vector': np.array(np.array([row['movement_vector']])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32)
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        pol_agent_data = {
                            'd': np.array(np.array([0.0])).astype(np.float32),
                            'angle': np.array(np.array([0.0])).astype(np.float32),
                            'd_diff': np.array(np.array([0.0])).astype(np.float32),
                            'movement_vector': np.array(np.array([0.0])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        x = np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0]
                        y = np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1]
                        x_1 = other_row['x_1'] - real_coordinates_1[0] if other_row['x_1'] != 0.0 and real_coordinates_1[0] != 0.0 else 0.0
                        y_1 = other_row['y_1'] - real_coordinates_1[1] if other_row['y_1'] != 0.0 and real_coordinates_1[1] != 0.0 else 0.0
                        pol_agent_data = {
                            'd': np.array(np.array(cart2pol(x, y)["d"])).astype(np.float32),
                            'angle': np.array(np.array(cart2pol(x, y)["angle"])).astype(np.float32),
                            'd_diff': np.array(np.array(cart2pol(x, y)["d"] - cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'movement_vector': np.array(np.array([other_row['movement_vector']])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32)
                        }
                    agent_pol_obs[other_agent] = pol_agent_data
                polar_observation[agent] = agent_pol_obs
        return polar_observation


    @property
    def get_pol_observation_with_mov_vec_and_memory(self):
        # get relative observation dict to get the relative coordinates of each agent from the "act_agent"
        polar_observation = {}

        # Cartesian to Polar
        def cart2pol(x, y):
            d = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)
            return {"d": d, "angle": angle}

        for agent, row in self.data[self.data['is_free'] == False].iterrows():
            if self.data.at[agent, 'vehicle_type'] == "agent":
                if row['speed'] > self.MAX_SPEED:
                    logging.info("speed outside given space!!!!")
                agent_pol_obs = {}
                real_coordinates = [np.array(np.array([row['pos_x']])).astype(np.float32), np.array(np.array([row['pos_y']])).astype(np.float32)]
                real_coordinates_1 = [np.array(np.array([row['x_1']])).astype(np.float32), np.array(np.array([row['y_1']])).astype(np.float32)]
                real_coordinates_2 = [np.array(np.array([row['x_2']])).astype(np.float32), np.array(np.array([row['y_2']])).astype(np.float32)]
                real_coordinates_3 = [np.array(np.array([row['x_3']])).astype(np.float32), np.array(np.array([row['y_3']])).astype(np.float32)]

                agent_pol_obs["act_agent"] = {
                    'd': np.array(np.array([0.0])).astype(np.float32),
                    'angle': np.array(np.array([0.0])).astype(np.float32),
                    'd_diff': np.array(np.array([0.0])).astype(np.float32),
                    'movement_vector': np.array(np.array([row['movement_vector']])).astype(np.float32),
                    'speed': np.array(np.array([row['speed']])).astype(np.float32),
                    "d_1": np.array(np.array([0.0])).astype(np.float32),
                    "d_diff_1": np.array(np.array([0.0])).astype(np.float32),
                    "angle_1": np.array(np.array([0.0])).astype(np.float32),
                    'speed_1': np.array(np.array([row['speed_1']])).astype(np.float32),
                    "d_2": np.array(np.array([0.0])).astype(np.float32),
                    "d_diff_2": np.array(np.array([0.0])).astype(np.float32),
                    "angle_2": np.array(np.array([0.0])).astype(np.float32),
                    'speed_2': np.array(np.array([row['speed_2']])).astype(np.float32),
                    "d_3": np.array(np.array([0.0])).astype(np.float32),
                    "angle_3": np.array(np.array([0.0])).astype(np.float32),
                    'speed_3': np.array(np.array([row['speed_3']])).astype(np.float32)
                }
                # other agent IDs in the observation space are named "agent_0", "agent_1", ...
                other_id = 0
                for other_agent, other_row in self.data.iterrows():
                    other_id += 1
                    if self.OBS_RANGE > 0 and agent != other_agent and not other_row['is_free']:  # calculate distance if observation range is used
                        dist = sqrt((row['pos_x'] - other_row['pos_x']) ** 2 + (row['pos_y'] - other_row['pos_y']) ** 2)
                    else:
                        dist = -1
                    if other_row['is_free'] or agent == other_agent or dist > self.OBS_RANGE:
                        # observation data can not be NaN --> position is set to left-bottom corner coordinates of the intersection zone
                        pol_agent_data = {
                            'd': np.array(np.array([0.0])).astype(np.float32),
                            'angle': np.array(np.array([0.0])).astype(np.float32),
                            'd_diff': np.array(np.array([0.0])).astype(np.float32),
                            'movement_vector': np.array(np.array([0.0])).astype(np.float32),
                            'speed': np.array(np.array([0.0])).astype(np.float32),
                            "d_1": np.array(np.array([0.0])).astype(np.float32),
                            "d_diff_1": np.array(np.array([0.0])).astype(np.float32),
                            "angle_1": np.array(np.array([0.0])).astype(np.float32),
                            'speed_1': np.array(np.array([0.0])).astype(np.float32),
                            "d_2": np.array(np.array([0.0])).astype(np.float32),
                            "d_diff_2": np.array(np.array([0.0])).astype(np.float32),
                            "angle_2": np.array(np.array([0.0])).astype(np.float32),
                            'speed_2': np.array(np.array([0.0])).astype(np.float32),
                            "d_3": np.array(np.array([0.0])).astype(np.float32),
                            "angle_3": np.array(np.array([0.0])).astype(np.float32),
                            'speed_3': np.array(np.array([0.0])).astype(np.float32)
                        }
                    else:
                        x = np.array(np.array([other_row['pos_x']])).astype(np.float32) - real_coordinates[0]
                        y = np.array(np.array([other_row['pos_y']])).astype(np.float32) - real_coordinates[1]
                        x_1 = other_row['x_1'] - real_coordinates_1[0] if other_row['x_1'] != 0.0 and real_coordinates_1[0] != 0.0 else 0.0
                        y_1 = other_row['y_1'] - real_coordinates_1[1] if other_row['y_1'] != 0.0 and real_coordinates_1[1] != 0.0 else 0.0
                        x_2 = other_row['x_2'] - real_coordinates_2[0] if other_row['x_2'] != 0.0 and real_coordinates_2[0] != 0.0 else 0.0
                        y_2 = other_row['y_2'] - real_coordinates_2[1] if other_row['y_2'] != 0.0 and real_coordinates_2[1] != 0.0 else 0.0
                        x_3 = other_row['x_3'] - real_coordinates_3[0] if other_row['x_3'] != 0.0 and real_coordinates_3[0] != 0.0 else 0.0
                        y_3 = other_row['y_3'] - real_coordinates_3[1] if other_row['y_3'] != 0.0 and real_coordinates_3[1] != 0.0 else 0.0
                        pol_agent_data = {

                            'd': np.array(np.array(cart2pol(x, y)["d"])).astype(np.float32),
                            'angle': np.array(np.array(cart2pol(x, y)["angle"])).astype(np.float32),
                            'd_diff': np.array(np.array(cart2pol(x, y)["d"] - cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'movement_vector': np.array(np.array([other_row['movement_vector']])).astype(np.float32),
                            'speed': np.array(np.array([other_row['speed']])).astype(np.float32),
                            "d_1": np.array(np.array(cart2pol(x_1, y_1)["d"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            "d_diff_1": np.array(np.array(cart2pol(x_1, y_1)["d"] - cart2pol(x_2, y_2)["d"] if x_1 != 0.0 and x_2 != 0 else [0.0])) .astype(np.float32),
                            "angle_1": np.array(np.array(cart2pol(x_1, y_1)["angle"] if x_1 != 0.0 else [0.0])).astype(np.float32),
                            'speed_1': np.array(np.array([other_row['speed_1']])).astype(np.float32),
                            "d_2": np.array(np.array(cart2pol(x_2, y_2)["d"] if x_2 != 0.0 else [0.0])).astype(np.float32),
                            "d_diff_2": np.array(np.array(cart2pol(x_2, y_2)["d"] - cart2pol(x_3, y_3)["d"] if x_2 != 0.0 and x_3 != 0 else [0.0])).astype(np.float32),
                            "angle_2": np.array(np.array(cart2pol(x_2, y_2)["angle"] if x_2 != 0.0 else [0.0])).astype(np.float32),
                            'speed_2': np.array(np.array([other_row['speed_2']])).astype(np.float32),
                            "d_3": np.array(np.array(cart2pol(x_3, y_3)["d"] if x_3 != 0.0 else [0.0])).astype(np.float32),
                            "angle_3": np.array(np.array(cart2pol(x_3, y_3)["angle"] if x_3 != 0.0 else [0.0])).astype(np.float32),
                            'speed_3': np.array(np.array([other_row['speed_3']])).astype(np.float32)
                        }
                    agent_pol_obs[other_agent] = pol_agent_data
                polar_observation[agent] = agent_pol_obs
        return polar_observation


    @staticmethod
    def sort_observation_dict_by_distance(self, obs):
        """
        Reorder the agent_0-agent_n elements of the observation dictionary by distance from the ego vehicle ('act_agent'). The act_agent subdirectory is followed by the closest agent subdirectory.
        :param self:
        :param obs: original observation dictionary ordered by default as: {'act_agent', 'agent_0', 'agent_1', 'agent_2', ...}
        :return: sorted observation dictionary ordered by distance from 'act_agent', but dict keys remains 'act_agent', 'agent_0', 'agent_1', ...
        """

        def key_funct(x):
            dist = 10000000
            if "x" in obs_dict[x]:          # for coordinate observations
                if (obs_dict[x]['x'][0] != self._playground_shape["x1"] and obs_dict[x]['y'][0] != self._playground_shape["y1"]):
                    dist = sqrt((obs_dict[x]['x'][0] - obs_dict['act_agent']['x'][0]) ** 2 + (obs_dict[x]['y'][0] - obs_dict['act_agent']['y'][0]) ** 2)
                else:
                    dist = 10000000
            elif "d" in obs_dict[x]:        # for polar observation
                if obs_dict[x]["d"] > 0:
                    dist = obs_dict[x]["d"]
                else:
                    dist = 10000000
            return dist

        obs_key_correlation = {}
        for agent in obs.keys():
            obs_dict = obs[agent]
            sorted_obs_dict = dict()
            # sort mainkeys by distance calculated from x and y
            sorted_keys = sorted(obs_dict, key=key_funct)
            # move 'act_agent' to fist position
            index_to_move = sorted_keys.index('act_agent')
            sorted_keys.pop(index_to_move)
            sorted_keys.insert(0, 'act_agent')
            # use the sorted list of keys to copy original dict to sorted dict (key-val by key-val)
            for key in sorted_keys:
                sorted_obs_dict[key] = obs_dict[key]
            renamed_sorted_obs_dict = dict(zip(sorted(sorted_obs_dict.keys()), list(sorted_obs_dict.values())))  # Rllib sort the observation ordered dict by key  -> we rename keys
            obs[agent] = renamed_sorted_obs_dict

            # correlation between original obs keys and sorted keys
            orig_keys = list(obs[agent].keys())
            orig_keys.pop(0)
            sorted_keys.pop(0)
            obs_key_correlation[agent] = dict(zip(orig_keys, sorted_keys))

        return obs, obs_key_correlation


    @staticmethod
    def filter_obs_space_n_closest(self, obs):
        for agent in obs.keys():
            for i in range(self.OBS_SPACE_N_CLOSEST, self.AGENT_NUM):
                obs[agent].pop(f"agent_{i}")
        return obs


    def calculate_reward(self, action_dict, in_collision, crossed, in_zone, reward_values):
        """
        Calculates rewards for each agent that are in-use based on the following rules:
        - Gives a small negative reward for each second in the simulation to motivate the agent to do the task quicker.
        - Gives a big negative reward for each collision.
        - Gives a big positive reward for each vehicle that they get to the destination coordinates (road).
        :param
        - action_dict: Takes a dictionary of actions for each agent to do until the next step.
        - reward_values: list of rewards --> [success, in intersection, collision]
        :return: Returns a dictionary of rewards for each agent (that are in-use)
        """
        rewards = {}
        # collisions = traci.simulation.getCollidingVehiclesIDList()
        # self.colliding_num = len(collisions)
        # in_collision = [agent for agent in self.data.index if self.data.at[agent, "vehicle_id"] in collisions]

        # reward from crossing
        for agent in crossed:
            rewards[agent] = reward_values[0]

        # negative reward for collision
        for agent in in_collision:
            rewards[agent] = reward_values[2]               # e.g., -100.0

        # rewards while being in the intersection
        in_zone = list(set(action_dict.keys())-set(in_collision))
        for agent in in_zone:
            if self.dist_related_inzone_reward and self.data.at[agent, 'vehicle_type'] == "agent":
                ordered_agent_obs_all, obs_key_correlation = self.sort_observation_dict_by_distance(self, obs=self.get_relative_observation)
                ordered_agent_obs = ordered_agent_obs_all[agent]
                # the closest agent to act_agent is agent_0
                if (ordered_agent_obs["agent_0"]['x'][0] != self._playground_shape["x1"] and ordered_agent_obs["agent_0"]['y'][0] != self._playground_shape["y1"]):
                    dist = sqrt(
                        (ordered_agent_obs["agent_0"]['x'][0] - ordered_agent_obs['act_agent']['x'][0]) ** 2 + (ordered_agent_obs["agent_0"]['y'][0] - ordered_agent_obs['act_agent']['y'][0]) ** 2)
                    rewards[agent] = reward_values[1] + np.interp(dist, [0, self.dist_related_inzone_reward_dist], [self.dist_related_inzone_reward_max_value, 0])
                else:
                    # if agent_0 not exists
                    rewards[agent] = reward_values[1]

            else:
                rewards[agent] = reward_values[1]  # e.g. - 1 in [100, -1, -100]

        if config.REWARD_TYPE == "on_episode_over":     # overwrite rewards
            if in_collision:
                rewards = {f"agent_{i}": self.current_step for i in list(range(self.AGENT_NUM))}
            else:
                rewards = {f"agent_{i}": 0 for i in list(range(self.AGENT_NUM))}

        self.rewards = rewards
        return rewards


    def sumo_restart(self, sumocfg):
        """
        Restarts the SUMO.
        """
        traci.close()
        traci.start([self.sumo_command, "-c", sumocfg, "--step-length", str(self.sumo_step_length), "-S", "--quit-on-end", "--random", "--collision.check-junctions", "--collision.action", "remove", "--collision.mingap-factor", "0", "--no-warnings", "--no-step-log"])  # start a new simulation


    def manage_data_table(self):
        """
        Updates the data-table (pandas.df) with agent data. Agents are vehicles within the intersection zone.
        :return: in_collision, crossed, in_zone, rl_agent_db
        """

        # rl_agent_db = self.data["rl_agent_id"].dropna()     # get rl_agent_ids before removing from self.data
        rl_agent_db = self.data.loc[self.data["vehicle_type"] == "agent"]["rl_agent_id"].dropna()       # get rl_agent_ids before removing from self.data
        agents_wo_any_action = self.data.index[self.data['had_any_action'] == False].tolist()
        id_list = traci.vehicle.getIDList()
        for id_item in id_list:
            if id_item not in self.data.vehicle_id.tolist():
                if self.get_free_agents():  # if there is free agent
                    if self.is_in_zone(id_item):
                        self.add_vehicle(id_item)
                else:  # if there is no free agent
                    traci.vehicle.remove(id_item, reason=3)
                    logging.warning(id_item + " car was removed.")
                    self.removed_vehicles_num += 1

        sim_time = traci.simulation.getTime()

        # observe not colliding agents (colliding vehicles are removed from simulation and can not be accessed with TraCI)
        collisions = traci.simulation.getCollidingVehiclesIDList()
        self.colliding_num = len(collisions)
        in_collision = [agent for agent in self.data.index if self.data.at[agent, "vehicle_id"] in collisions]
        fake_in_collision = [agent for agent in self.data.index if self.data.at[agent, "vehicle_id"] in collisions and self.data.at[agent, "vehicle_type"] == "fake"]
        non_agent_in_collision = [agent for agent in self.data.index if self.data.at[agent, "vehicle_id"] in collisions and self.data.at[agent, "vehicle_type"] == "non_agent"]

        # remove colliding vehicles from data-table
        for agent in in_collision:
            self.data.loc[agent] = np.nan
            self.data.at[agent, "vehicle_type"] = "unknown"
            self.data.at[agent, "is_free"] = True

        # do not count collision if only fake and non_agent in collision
        if len(list(set(in_collision) - set(fake_in_collision) - set(non_agent_in_collision))) == 0:
            fake_in_collision = []
            non_agent_in_collision = []
            in_collision = []

        # iterate agents not in collision and set data-table values from SUMO (Traci)
        crossed = []
        in_zone = []
        for agent, row in self.data.drop(in_collision).iterrows():

            if not row.is_free and row.start_time != sim_time:
                # agent crossed
                if not self.is_in_zone(row.vehicle_id):
                    self.data.loc[agent] = np.nan
                    self.data.at[agent, "vehicle_type"] = "unknown"
                    self.data.at[agent, "is_free"] = True
                    traci.vehicle.setSpeedMode(row.vehicle_id, 31)  # set speed mode to default to avoid collisions after leaving the intersection zone
                    traci.vehicle.setColor(row.vehicle_id, (255, 255, 0, 255))  # set vehicle color to default yellow
                    crossed.append(agent)  # list of crossed agents

                # agent in zone
                else:
                    self.data.at[agent, "x_3"] = self.data.at[agent, "x_2"]
                    self.data.at[agent, "y_3"] = self.data.at[agent, "y_2"]
                    self.data.at[agent, "speed_3"] = self.data.at[agent, "speed_2"]

                    self.data.at[agent, "x_2"] = self.data.at[agent, "x_1"]
                    self.data.at[agent, "y_2"] = self.data.at[agent, "y_1"]
                    self.data.at[agent, "speed_2"] = self.data.at[agent, "speed_1"]

                    self.data.at[agent, "x_1"] = self.data.at[agent, "pos_x"]
                    self.data.at[agent, "y_1"] = self.data.at[agent, "pos_y"]
                    self.data.at[agent, "speed_1"] = self.data.at[agent, "speed"]

                    x_y = traci.vehicle.getPosition(row.vehicle_id)

                    self.data.at[agent, "pos_x"] = x_y[0]
                    self.data.at[agent, "pos_y"] = x_y[1]
                    self.data.at[agent, "speed"] = traci.vehicle.getSpeed(row.vehicle_id)

                    angle = traci.vehicle.getAngle(row.vehicle_id)
                    normalised_angle = angle % 360
                    self.data.at[agent, "movement_vector"] = normalised_angle

                    
                    if self.data.at[agent, "vehicle_type"] == "fake" and self.FAKE_SWITCH:
                        adv_func.compromize_data_fake(self, agent)

                    in_zone.append(agent)

        # set not agent controlled vehicles speed_mode behavior and color
        for agent, row in self.data[self.data['vehicle_type'] != "agent"].iterrows():
            if not row.is_free:
                if agent not in in_collision:
                    if not self.is_in_zone(row.vehicle_id):
                        self.data.loc[agent] = np.nan
                        self.data.at[agent, "vehicle_type"] = "unknown"
                        self.data.at[agent, "is_free"] = True
                        traci.vehicle.setSpeedMode(row.vehicle_id, 31)  # set speed mode to default to avoid collisions after leaving the intersection zone
                        traci.vehicle.setColor(row.vehicle_id, (255, 255, 0, 255))  # set vehicle color to default yellow

        return in_collision, crossed, in_zone, rl_agent_db, agents_wo_any_action, non_agent_in_collision, fake_in_collision


    def reset_data_table(self):
        """
        Resets the data-table (pandas.df) entirely. The creation of the table is also done here after the first reset.
        :return: The data-table with default values.
        """
        data = pd.DataFrame(
            columns=["vehicle_id", "pos_x", "pos_y", "speed",
                     "x_1", "y_1", "speed_1", "x_2", "y_2", "speed_2", "x_3", "y_3", "speed_3", "start_edge", "dest_edge", "movement_vector",
                     "start_time", "terminated", "truncated", "is_free", "vehicle_type", "had_any_action", "rl_agent_id"],
            index=[f"agent_{i}" for i in range(self.AGENT_NUM)]
        )
        data.is_free = True
        data.vehicle_type = "unknown"
        return data


    def add_vehicle(self, vehicle_id):
        """
        Takes a vehicle ID and fills one of the row of the data (pandas.df) with the given data of the vehicle.
        :param vehicle_id: A vehicle ID
        """
        self.total_vehicle_count += 1
        self.total_vehicle_count_in_episode += 1
        x_y = traci.vehicle.getPosition(vehicle_id)
        route = traci.vehicle.getRoute(vehicle_id)
        memory = 0.0
        angle = traci.vehicle.getAngle(vehicle_id)
        normalised_angle = angle % 360
        vehicle_type = self.add_vehicle_uniform()

        # new unique rl_agent_id for each new vehicle
        # if vehicle_type == "agent":
        rl_agent_id = f"agent_{self.rl_agent_count}"
        self.rl_agent_count += 1


        self.data.loc[self.get_free_agents()[0]] = [
            vehicle_id, x_y[0], x_y[1], traci.vehicle.getLateralSpeed(vehicle_id),
            memory, memory, memory, memory, memory, memory, memory, memory, memory,
            route[0], route[-1], normalised_angle, traci.simulation.getTime(), False, False, False, vehicle_type, False, rl_agent_id
        ]
        traci.vehicle.setSpeedMode(vehicle_id, 32)  # set speed mode to disable safety checks when entering the intersection zone


    def is_in_zone(self, vehicle_id):
        """
        Checks for each vehicle if the given one is in zone (or not yet/already not).
        :param vehicle_id: A vehicle ID
        :return: Boolean (in the zone (True) or not (False))
        """
        position = traci.vehicle.getPosition(vehicle_id)
        if (position[0] > self._playground_shape["x1"]) and (position[0] < self._playground_shape["x2"]):
            if (position[1] > self._playground_shape["y1"]) and (position[1] < self._playground_shape["y2"]):
                return True
        return False


    def get_free_agents(self):
        """
        Checks the data-rows (pandas.df) if there are any free agents.
        :return: Returns a list of agents that are free to take new vehicles.
        """
        return self.data.index[self.data['is_free'] == True].tolist()


    def add_vehicle_uniform(self):
        """
        Decides the vehicle uniform: agent driven vehicles, non-agent driven vehicles
        :param new_car_ids: List of ids of the new vehicles.
        """

        if "agent" in list(self.data.vehicle_type):
            if random.uniform(0, 1) <= self.NON_AGENT_RATIO and self.NON_AGENT_SWITCH == True:
                return "non_agent"
            if random.uniform(0, 1) <= self.FAKE_RATIO and self.FAKE_SWITCH == True:
                return "fake"
        return "agent"


    def remove_unused_act_agent_obs_spaces(self, obs_dict):
        obs_dict['act_agent'].spaces.pop("x", None)
        obs_dict['act_agent'].spaces.pop('x_1', None)
        obs_dict['act_agent'].spaces.pop('x_2', None)
        obs_dict['act_agent'].spaces.pop('x_3', None)
        obs_dict['act_agent'].spaces.pop("y", None)
        obs_dict['act_agent'].spaces.pop('y_1', None)
        obs_dict['act_agent'].spaces.pop('y_2', None)
        obs_dict['act_agent'].spaces.pop('y_3', None)
        obs_dict['act_agent'].spaces.pop('d', None)
        obs_dict['act_agent'].spaces.pop('d_1', None)
        obs_dict['act_agent'].spaces.pop('d_2', None)
        obs_dict['act_agent'].spaces.pop('d_3', None)
        obs_dict['act_agent'].spaces.pop('d_diff', None)
        obs_dict['act_agent'].spaces.pop('d_diff_1', None)
        obs_dict['act_agent'].spaces.pop('d_diff_2', None)
        obs_dict['act_agent'].spaces.pop('angle', None)
        obs_dict['act_agent'].spaces.pop('angle_1', None)
        obs_dict['act_agent'].spaces.pop('angle_2', None)
        obs_dict['act_agent'].spaces.pop('angle_3', None)
        return obs_dict


    def remove_unused_act_agent_obs(self, obs_dict):
        obs_dict['act_agent'].pop("x", None)
        obs_dict['act_agent'].pop('x_1', None)
        obs_dict['act_agent'].pop('x_2', None)
        obs_dict['act_agent'].pop('x_3', None)
        obs_dict['act_agent'].pop("y", None)
        obs_dict['act_agent'].pop('y_1', None)
        obs_dict['act_agent'].pop('y_2', None)
        obs_dict['act_agent'].pop('y_3', None)
        obs_dict['act_agent'].pop('d', None)
        obs_dict['act_agent'].pop('d_1', None)
        obs_dict['act_agent'].pop('d_2', None)
        obs_dict['act_agent'].pop('d_3', None)
        obs_dict['act_agent'].pop('d_diff', None)
        obs_dict['act_agent'].pop('d_diff_1', None)
        obs_dict['act_agent'].pop('d_diff_2', None)
        obs_dict['act_agent'].pop('angle', None)
        obs_dict['act_agent'].pop('angle_1', None)
        obs_dict['act_agent'].pop('angle_2', None)
        obs_dict['act_agent'].pop('angle_3', None)
        return obs_dict

    def action_range_filtering(self, obs):
        """
        Filter observation and thus action if there is no vehicle withing range
        :param obs:
        :return:
        """
        for agent in list(obs.keys()):
            distances = []
            other_agent_list = list(obs[agent].keys())
            other_agent_list.remove("act_agent")
            for other_agent in other_agent_list:

                if self.OBS_SPACE_TYPE.startswith("abs_"):
                    if (obs[agent][other_agent]["x"][0] != self._playground_shape["x1"] and obs[agent][other_agent]["y"][0] != self._playground_shape["y1"]):
                        dist = sqrt((obs[agent][other_agent]["x"][0] - obs[agent]["act_agent"]["x"][0]) ** 2 + (obs[agent][other_agent]["y"][0] - obs[agent]["act_agent"]["y"][0]) ** 2)
                    else:
                        dist = 10000000
                if self.OBS_SPACE_TYPE.startswith("rel_"):
                    if (obs[agent][other_agent]["x"][0] != self._playground_shape["x1"] and obs[agent][other_agent]["y"][0] != self._playground_shape["y1"]):
                        dist = sqrt((obs[agent][other_agent]["x"][0]) ** 2 + (obs[agent][other_agent]["y"][0] ** 2))
                    else:
                        dist = 10000000
                if self.OBS_SPACE_TYPE.startswith("pol_"):
                    if obs[agent][other_agent]["d"][0] > 0.0:
                        dist = obs[agent][other_agent]["d"][0]
                    else:
                        dist = 10000000

                distances.append(dist)

            if min(distances) > self.OBS_RANGE:
                obs.pop(agent)

        return obs


    def action_conflict_filtering(self, obs):
        """
        Filter observation and thus action if there is no risk of collision
        :param obs:
        :return:
        """
        for agent in list(obs.keys()):
            right_conflicts_agents = set()
            straight_no_conflicts_agents = set()
            straight_conflicts_agents = set()
            left_no_conflicts_agents = set()
            left_conflicts_agents = set()

            edges = [1, 2, 3, 4]
            # referenced to the agent's start edge, determine the conflicting/not-conflicting [start, dest] of the other_agent
            # e.g., agent start_edge=3 and goes straight: it has no conflict with
            # 1.) (start, dest) = [4,3] --> [+1, 0]
            # 2.) (start, dest) = [1,3] --> [+2, 0]
            # 3.) (start, dest) = [1,4] --> [+2, +1]
            right_conflicts = [[+1, -1], [+2, -1]]
            straight_no_conflicts = [[+1, 0], [+2, 0], [+2, +1]]
            left_no_conflicts = [[+1, 0], [-1, +2]]

            # get edge index (index makes easier to handle edge list turn around: e.g. 1-1==>4 or 4+1==>1)
            idx_start = edges.index(int(self.data.at[agent, "start_edge"]))
            idx_dest = edges.index(abs(int(self.data.at[agent, "dest_edge"])))

            # right turn (2 conflicting routes)
            if (idx_start-1) % len(edges) == idx_dest:
                for other_agent, other_row in self.data.drop(agent).iterrows():
                    if not other_row.is_free:
                        idx_other_start = edges.index(int(other_row["start_edge"]))
                        idx_other_dest = edges.index(abs(int(other_row["dest_edge"])))
                        for i in range(len(right_conflicts)):
                            if list(np.add([idx_start, idx_start], right_conflicts[i]) % len(edges)) == [idx_other_start, idx_other_dest] \
                                    or idx_start == idx_other_start \
                                    or idx_dest == idx_other_dest:
                                right_conflicts_agents.add(other_agent)

            # straight (3 no conflicting routes)
            if (idx_start + 2) % len(edges) == idx_dest:
                for other_agent, other_row in self.data.drop(agent).iterrows():
                    if not other_row.is_free:
                        idx_other_start = edges.index(int(other_row["start_edge"]))
                        idx_other_dest = edges.index(abs(int(other_row["dest_edge"])))
                        for i in range(len(straight_no_conflicts)):
                            if list(np.add([idx_start, idx_start], straight_no_conflicts[i]) % len(edges)) == [idx_other_start, idx_other_dest]:
                                straight_no_conflicts_agents.add(other_agent)
                # vehicles = set.union(set(obs.keys()), self.data.index[self.data['vehicle_type'] == 'fake'])     # fake/non_agent vehicles are not in obs, must be added
                vehicles = set.union(set(obs.keys()), self.data.index[self.data['vehicle_type'] == 'fake'], self.data.index[self.data['vehicle_type'] == 'non_agent'])
                straight_conflicts_agents = vehicles - straight_no_conflicts_agents
                straight_conflicts_agents.remove(agent)

            # lef turn (3 no conflicting routes)
            if (idx_start + 1) % len(edges) == idx_dest:
                for other_agent, other_row in self.data.drop(agent).iterrows():
                    if not other_row.is_free:
                        idx_other_start = edges.index(int(other_row["start_edge"]))
                        idx_other_dest = edges.index(abs(int(other_row["dest_edge"])))
                        for i in range(len(left_no_conflicts)):
                            if list(np.add([idx_start, idx_start], left_no_conflicts[i]) % len(edges)) == [idx_other_start, idx_other_dest]:
                                left_no_conflicts_agents.add(other_agent)
                # vehicles = set.union(set(obs.keys()), self.data.index[self.data['vehicle_type'] == 'fake'])     # fake/non_agent vehicles are not in obs, must be added
                vehicles = set.union(set(obs.keys()), self.data.index[self.data['vehicle_type'] == 'fake'], self.data.index[self.data['vehicle_type'] == 'non_agent'])
                left_conflicts_agents = vehicles - left_no_conflicts_agents
                left_conflicts_agents.remove(agent)

            # logging.info(f"{agent}:    right:{right_conflicts_agents}      straight:{straight_conflicts_agents}        left:{left_conflicts_agents}")
            if not bool(set.union(right_conflicts_agents, straight_conflicts_agents, left_conflicts_agents)):
                obs.pop(agent)
                # logging.info(f"{agent} removed\n")

        return obs


    def action_movement_vector_filtering(self, obs):
        """
        Filter observation and thus action if the movement vectors are intersecting or not. Determines if two movement vectors (defined by start points and angles) intersect.
        :param obs:
        :return:
        """
        for agent in list(obs.keys()):
            intersecting_agents = set()
            for other_agent, other_row in self.data.drop(agent).iterrows():
                if not other_row.is_free:
                    p1 = (self.data.at[agent, "pos_x"], self.data.at[agent, "pos_y"])
                    angle1 = (math.radians(-self.data.at[agent, "movement_vector"]+90))      # in 0 degree is at 12h (clock) and inverse in SUMO
                    p2 = (other_row["pos_x"], other_row["pos_y"])
                    angle2 = (math.radians(-other_row["movement_vector"]+90))
                    intersect, intersect_point = self.movement_vector_intersect_detection(p1, angle1, p2, angle2)
                    dist = math.sqrt((abs(p2[0]-p1[0])**2)+(abs(p2[1]-p1[1])**2))
                    if (intersect and dist < self.OBS_RANGE) or dist < 6.0:         # vectors intersecting or vehicles ar very close (~vehicle_length, 5m )
                        intersecting_agents.add(other_agent)
            if not bool(intersecting_agents):
                obs.pop(agent)
        return obs


    def movement_vector_intersect_detection(self, p1, angle1, p2, angle2):
        """
        Determines if two movement vectors (defined by start points and angles) intersect.
        :param angle1:
        :param p2:
        :param angle2:
        :return:
        """
        # Calculate direction vectors from angles
        d1 = np.array([np.cos(angle1), np.sin(angle1)])
        d2 = np.array([np.cos(angle2), np.sin(angle2)])

        # Convert starting points to numpy arrays
        p1, p2 = map(np.array, (p1, p2))

        # Formulate the system of equations
        # p1 + t * d1 = p2 + s * d2
        A = np.array([d1, -d2]).T  # Coefficients matrix
        b = p2 - p1  # Difference of starting points

        # Check if the determinant is zero (parallel vectors)
        det = np.linalg.det(A)
        if np.isclose(det, 0):
            return False, None  # Vectors are parallel and do not intersect

        # Solve for t and s
        t, s = np.linalg.solve(A, b)

        # If t and s are both >= 0, the vectors intersect
        if t >= 0 and s >= 0:
            intersection_point = p1 + t * d1
            return True, tuple(intersection_point)
        else:
            return False, None


    def visualize_observed_vehicles_GUI(self, obs, key_corr):
        """
        Draw line between two vehicles if observing each other
        :param obs:
        :param key_corr:
        :return:
        """
        for poly_id in traci.polygon.getIDList():
            if poly_id != "observation_zone":
                traci.polygon.remove(poly_id)
        # if len(obs) >= 2:
        for act_agent in obs.keys():
            for other_agent in obs[act_agent].keys():
                if other_agent != "act_agent":
                    orig_other_agent = key_corr[act_agent][other_agent]
                    # if self.data.vehicle_type[orig_other_agent] == "agent" and act_agent != orig_other_agent:
                    if act_agent != orig_other_agent:
                        traci.polygon.add(
                            polygonID=f"line-{act_agent}-{orig_other_agent}",
                            shape=(
                                (self.data.pos_x[act_agent], self.data.pos_y[act_agent]),
                                (self.data.pos_x[orig_other_agent], self.data.pos_y[orig_other_agent]),
                            ),
                            layer=1000,
                            color=(240, 240, 240, 255),
                            fill=False,
                            lineWidth=0.1
                        )
        return


    def progress_bar(self, count, total, status='', bar_len=60):
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        fmt = '[%s] %s%s   : %s' % (bar, percents, '%', status)
        # fmt = '[%s]' % count
        print('\b' * len(fmt), end='')  # clears the line
        sys.stdout.write(fmt)
        sys.stdout.flush()


# Define a function to change logging level
def change_logging_level():
    while True:
        # Wait for a key press
        keyboard.wait('ctrl+alt+l')

        # Change logging level
        current_level = logging.getLogger().getEffectiveLevel()
        new_level = logging.INFO if current_level == logging.WARNING else logging.WARNING
        logging.getLogger().setLevel(new_level)
        logging.info(f"Logging level changed to {logging.getLevelName(new_level)}")
