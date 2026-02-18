
TRAIN = False               # set to True for training the models, False for using the trained model
TRAIN_WITH_TUNE = True      # set how to train: 1) Basic RLlib method or 2) using Tune: https://docs.ray.io/en/latest/rllib/core-concepts.html#algorithms
MULTI_POLICY = True         # use different policies for different incoming directions; False: same policy is used for each vehicle
TERMINATING_AGENTS = True   # every new vehicle is a new agent (unique agent_id) that terminates when colliding or passing the intersection
TERMINATING_AGENTS_LIST = [True]
ALG = "DQN"                 # PPO, DQN
USE_LSTM = True             # use LSTM in PPO
LSTM_SIZE = 32
LSTM_SIZE_LIST = [32]       # [16, 32, 64]
SUMO_GUI = True            # use SUMO GUI
SUMO_RANDOM_ROUTE = True    # True: each episode has unique random generated traffic; False: use pre-generated SUMO route file
SUMO_RANDOM_ROUTE_LIST = [True]
SUMO_TURNS_ALLOWED = False              # if turns disallowed: just WE, EW, SN, NS random traffic (works if SUMO_RANDOM_ROUTE=True)
SUMO_STEP_LENGTH = 0.1                  # Duration (seconds) of one step in SUMO [0.1, 0.5, 1.0]
SUMO_STEP_LENGTH_LIST = [0.1]           # [0.1, 0.2]
SUMO_TRAFFIC_INSERTATION_PERIOD = 0.8   # generates vehicles with a constant period and arrival rate of (1/period) per second
SUMO_TRAFFIC_INSERTATION_PERIOD_LIST = [0.8]    # [0.5, 1.0] (no real impact on performance)
EPISODE_LENGTH = 5100                   # steps in episode: 1000/SUMO_STEP_LENGTH (last vehicle enters at 999 s)
RESTART_EPISODE_ON_COLLISION = True     # restart episode if collision happens, else the episode runs EPISODE_LENGTH steps
RESTART_EPISODE_ON_COLLISION_LIST = [True]
# TRAIN_BATCH_SIZE = 2048                 # steps in one train loop (RLlib default: DQN:32, PPO:4000)  --> training iterations=TRAIN_TOTAL_TIMESTEPS/TRAIN_BATCH_SIZE
TRAIN_BATCH_SIZE_LIST = [4096]          # [2048, 4096, 8192] used if TRAIN_WITH_TUNE (4096)
# ROLLOUT_FRAGMENT_LENGTH = 512           # train_batch_size >= rollout_fragment_length (RLib default: 1)
ROLLOUT_FRAGMENT_LENGTH_LIST = [512]    # [512, 1024] used if TRAIN_WITH_TUNE (128)
LEARNING_RATE = 0.0001                  # (RLlib default: 0.001)
LEARNING_RATE_LIST = [0.0001]   # [0.001, 0.0001, 0.00001]
GAMMA = 0.99                            # discount factor (RLlib default: 0.99)
GAMMA_LIST = [0.9]
PPO_LAMBDA = 0.9                            # PPO Generalized Advantage Estimator (GAE) parameter: [0.9, 0.95, 0.99, 1.0] Defines the exponential weight used between actually measured rewards vs. value function estimates over multiple time steps (default: 1.0)
PPO_LAMBDA_LIST = [0.99]
PPO_CLIP_PARAM = 0.3                    # clip parameter controls how much the policy is allowed to change in each iteration (default:0.3). A higher clip parameter can lead to more stable updates, but it can also limit the ability of the policy to explore new actions. A lower clip parameter can lead to more exploration, but it can also lead to instability.
PPO_CLIP_PARAM_LIST = [0.3]             # (0.3)
TRAIN_LOOPS = 200                       # training loops: training with train()
TRAIN_TOTAL_TIMESTEPS = 1_000_000       # training steps: training with Tune (500_000) (HPC run time: 1M steps ~ 10h)
SAVED_MODEL_PATH = "saved_model"        # trained model is loaded from this folder
REWARD_TYPE = "on_step"                 # ["on_episode_over", "on_step"]; "on_episode_over": get reward (episode length) only at the end of the episode; "on_step": get reward every step according to REWARD_VALUES parameter.
REWARD_VALUES = '10_-0.1_-20'              # reward values of an agent in a list: [success, in intersection, collision]
REWARD_VALUES_LIST = ['10_-0.1_-20']       # ['100, 0, -100', '10, -0.1, -100', '100, -1, -100', '0, 0, -100', '0, -1, -100']
DISTANCE_RELATED_INZONE_REWARD = True   # reward for being in the intersection zone depends on the distance from the closest vehicle
DISTANCE_RELATED_INZONE_REWARD_LIST = [True]
DISTANCE_RELATED_INZONE_REWARD_DIST = 20     # range radius [meter] from where the distance related reward activate
DISTANCE_RELATED_INZONE_REWARD_DIST_LIST = [20]  # [3, 5, 10]
DISTANCE_RELATED_INZONE_REWARD_MAX_VALUE = -1  # maximum negative reward when ego vehicle is very close to other vehicle (just before collision)
DISTANCE_RELATED_INZONE_REWARD_MAX_VALUE_LIST = [-1]  # [-2, -5, -10]
ACCELERATION = 20                        # accelerate/decelerate (default is 4.5 m/s2)
ACCELERATION_LIST = [20]
SORT_OBS_BY_DISTANCE = True             # order agents in the observation by distance from the ego vehicle ("act_agent")
SORT_OBS_BY_DISTANCE_LIST = [True]      # [True, False]
OBS_SPACE_TYPE = "rel_coord_with_dest"  # "abs_coord", "rel_coord", "abs_coord_with_dest", "rel_coord_with_dest" : which observation type to use
OBS_SPACE_TYPE_LIST = ["rel_coord_with_dest"]
                                        # ["abs_coord", "rel_coord", "abs_coord_with_dest", "rel_coord_with_dest", "rel_coord_with_dest_and_memory", "pol_coord_with_dest", "pol_coord_with_dest_and_memory",
                                        # "rel_coord_with_mov_vec", "rel_coord_with_mov_vec_and_memory", "pol_coord_with_mov_vec", "pol_coord_with_mov_vec_and_memory"]
OBS_SPACE_N_CLOSEST = 1                 # observation space contains only the N closest vehicles data sorted by distance (0 --> all vehicles in the zone are included in the observation)
OBS_SPACE_N_CLOSEST_LIST = [1]    # [0, 2, 4]
OBS_RANGE = 20                          # observation space contains only vehicles within range (0 --> all vehicles in the zone are included in the observation)
OBS_RANGE_LIST = [20]
OBS_CLEAN_ACT_AGENT_OBS = True
OBS_CLEAN_ACT_AGENT_OBS_LIST = [True]   # clean act_agent obs from 0.0 values (e.g. x, y in rel_coord_* and pol_coord_* observation spaces)
ACTION_RANGE_FILTER = True              # no action for agents without surrounding vehicles within OBS_RANGE
ACTION_RANGE_FILTER_LIST = [True]
ACTION_CONFLICT_FILTER = True           # no action for agents that has no conflicting path with other vehicles
ACTION_CONFLICT_FILTER_LIST = [True]
ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER = True      # no action for agents if the movement vectors are intersecting
ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER_LIST = [True]

NON_AGENT_SWITCH = False                # if True there could be non agent driven vehicles, if False there will not be
NON_AGENT_SWITCH_LIST = [False]
NON_AGENT_RATIO = 0.05                  # ratio of non agent driven vehicles
NON_AGENT_RATIO_LIST = [0.05]

FAKE_SWITCH = True                      # if True there could be non agent driven vehicles sharing fake x,y information
FAKE_SWITCH_LIST = [False]
FAKE_RATIO = 0.1                        # ratio of vehicles with fake shared information (observation)
FAKE_RATIO_LIST = [0.1]
FAKE_MOD_DIST = 1                       # modify x,y values by random.uniform(-self.FAKE_MOD_DIST, self.FAKE_MOD_DIST)
FAKE_MOD_DIST_LIST = [5]

ADVERSARY_ACTION = False                # control the speed of adversary vehicles to hit other vehicles
PGD_ATTACK = True                      # instead of random noise, add clever noise to attack learning (PGD attack)


def print_conf():
    print("CONFIG settings:")
    for key, value in globals().copy().items():
        if not(key.startswith('__') or key.startswith('print_conf')):
            print(f"\t{key} = {value}")
