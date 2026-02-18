import importlib
import argparse
# import config
import ray
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from colorama import Back, Fore
import os
from env.env import IntersectionEnv
from functions.callbacks import MyCallbacks
from functions.model_features import get_Q_values
import functions.adversary_functions as adv_func

# load config as argument (e.g. config1.py: python run.py --config config1)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    # choices=[],
    default="config",       # config file name without .py
    help="Config to use."
)
args = parser.parse_args()
config = importlib.import_module(args.config)

config.print_conf()

context = ray.init(local_mode=True, num_cpus=1)
# context = ray.init(num_cpus=1)    # num_cpus=1, (HPC: --mem-per-cpu=8000/12000)

print(f"Ray URL: {context.dashboard_url}")

env_config = {
    "terminating_agents": config.TERMINATING_AGENTS,
    "restart_episode_on_collision": config.RESTART_EPISODE_ON_COLLISION,
    "obs_space_type": config.OBS_SPACE_TYPE,
    "obs_space_n_closest": config.OBS_SPACE_N_CLOSEST,
    "obs_range": config.OBS_RANGE,
    "obs_clean_act_agent_obs": config.OBS_CLEAN_ACT_AGENT_OBS,
    "action_range_filter": config.ACTION_RANGE_FILTER,
    "action_conflict_filter": config.ACTION_CONFLICT_FILTER,
    "action_movement_vector_intersect_filter": config.ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER,
    "episode_length": config.EPISODE_LENGTH,
    "sumo_gui": config.SUMO_GUI,
    "sumo_random_route": config.SUMO_RANDOM_ROUTE,
    "sumo_step_length": config.SUMO_STEP_LENGTH,
    "sumo_traffic_insertation_period": config.SUMO_TRAFFIC_INSERTATION_PERIOD,
    "reward_values": config.REWARD_VALUES,
    "acceleration": config.ACCELERATION,
    "sort_obs_by_dist": config.SORT_OBS_BY_DISTANCE,
    "dist_related_inzone_reward": config.DISTANCE_RELATED_INZONE_REWARD,
    "dist_related_inzone_reward_dist": config.DISTANCE_RELATED_INZONE_REWARD_DIST,
    "dist_related_inzone_reward_max_value": config.DISTANCE_RELATED_INZONE_REWARD_MAX_VALUE,
    "non_agent_switch": config.NON_AGENT_SWITCH,
    "non_agent_ratio": config.NON_AGENT_RATIO,
    "fake_switch": config.FAKE_SWITCH,
    "fake_ratio": config.FAKE_RATIO,
    "fake_mod_dist": config.FAKE_MOD_DIST,
    "adversary_action": config.ADVERSARY_ACTION,
    "pgd_attack": config.PGD_ATTACK
}


if config.MULTI_POLICY:
    # use different policies for different incoming directions
    policies = {
        "policy_N": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        ),
        "policy_E": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        ),
        "policy_S": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        ),
        "policy_W": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        )
    }
else:
    # use the same policy for each vehicle
    policies = {
        "policy_0": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        ),
}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if config.MULTI_POLICY:
        if config.TRAIN:
            TERM_AGENTS = tune.grid_search(config.TERMINATING_AGENTS_LIST)['grid_search'][0]
        else:
            TERM_AGENTS = config.TERMINATING_AGENTS
        if TERM_AGENTS:
            rl_agent_id = agent_id
            data = episode.worker.env.data
            agent_id = data.index[data['rl_agent_id'] == rl_agent_id][0]

        if episode.worker.env.data.loc[agent_id]["start_edge"] == '1':
            return "policy_W"
        elif episode.worker.env.data.loc[agent_id]["start_edge"] == '2':
            return "policy_N"
        elif episode.worker.env.data.loc[agent_id]["start_edge"] == '3':
            return "policy_E"
        elif episode.worker.env.data.loc[agent_id]["start_edge"] == '4':
            return "policy_S"
    else:
        return "policy_0"


if config.ALG == 'PPO':
    rl_config = (
        PPOConfig()
        .environment(
            env=IntersectionEnv,
            env_config=env_config,
            disable_env_checking=True       # disable_env_checking=True|False
        )
        .training(
            # train_batch_size=config.TRAIN_BATCH_SIZE,
            lr=config.LEARNING_RATE,
            model=dict(use_lstm=True, lstm_cell_size=config.LSTM_SIZE, max_seq_len=20) if config.USE_LSTM else dict(use_lstm=False),   # use LSTM
        )
        .rollouts(
            num_rollout_workers=0,
            # rollout_fragment_length=config.ROLLOUT_FRAGMENT_LENGTH,
            compress_observations=False,
            batch_mode="complete_episodes",         # "complete_episodes", "truncate_episodes"
        )
        .multi_agent(
            count_steps_by="agent_steps",
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
        .callbacks(MyCallbacks)
        .reporting(
            keep_per_episode_custom_metrics=True,
            min_sample_timesteps_per_iteration=256        # default 1000
        )
    )

elif config.ALG == 'DQN':
    rl_config = (
        DQNConfig()
        .environment(
            env=IntersectionEnv,
            env_config=env_config,
            disable_env_checking=True       # disable_env_checking=True|False
        )
        .resources(
            # num_gpus=1,
        )
        .training(
            # train_batch_size=config.TRAIN_BATCH_SIZE,
            lr=config.LEARNING_RATE,
            # replay_buffer_config={
            #     "capacity": 10000
            # },
        )
        .rollouts(
            num_rollout_workers=0,
            # rollout_fragment_length=config.ROLLOUT_FRAGMENT_LENGTH,
            compress_observations=False,
            batch_mode="complete_episodes",     # "complete_episodes", default: "truncate_episodes"
        )
        .multi_agent(
            count_steps_by="agent_steps",
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
        .callbacks(MyCallbacks)
        .reporting(
            keep_per_episode_custom_metrics=True,
            min_sample_timesteps_per_iteration=512        # default 1000
        )
        .exploration(
            exploration_config={
                "epsilon_timesteps": 1.0 * config.TRAIN_TOTAL_TIMESTEPS
            }
        )
    )

elif config.ALG == 'MARWIL':
    rl_config = (
        MARWILConfig()
        .environment(
            env=IntersectionEnv,
            env_config=env_config,
            disable_env_checking=True       # disable_env_checking=True|False
        )
        .training(
            train_batch_size=config.TRAIN_BATCH_SIZE,  # number of episodes in train()
        )
        .rollouts(
            num_rollout_workers=0,
            rollout_fragment_length=config.ROLLOUT_FRAGMENT_LENGTH,
            compress_observations=False,
            batch_mode="truncate_episodes",
        )
        .multi_agent(
            count_steps_by="env_steps",
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
        .callbacks(MyCallbacks)
        .reporting(keep_per_episode_custom_metrics=True)
    )


if config.TRAIN and not config.TRAIN_WITH_TUNE:
    algo = rl_config.build()
    for i in range(config.TRAIN_LOOPS):  # set iteration number
        print(f"\n{Back.YELLOW}{Fore.BLACK}train loop: {i}/{config.TRAIN_LOOPS} {Back.RESET}{Fore.RESET}")
        print(algo.train())

    checkpoint_path = algo.save(config.SAVED_MODEL_PATH)
    print(f"{Back.GREEN}Results:{Back.RESET}\n"
          f"Checkpoint saved at: {checkpoint_path}\n"
          f"Tensorboard command: tensorboard --logdir [log_folder] --samples_per_plugin scalars=10000")


if config.TRAIN and config.TRAIN_WITH_TUNE:
    tuner = tune.Tuner(
        config.ALG,
        param_space=rl_config.to_dict() | {
            "env_config": {**env_config, **{             # add parameters with grid_search
                "terminating_agents": tune.grid_search(config.TERMINATING_AGENTS_LIST),
                "sumo_random_route": tune.grid_search(config.SUMO_RANDOM_ROUTE_LIST),
                "sumo_step_length": tune.grid_search(config.SUMO_STEP_LENGTH_LIST),
                "sumo_traffic_insertation_period": tune.grid_search(config.SUMO_TRAFFIC_INSERTATION_PERIOD_LIST),
                "restart_episode_on_collision": tune.grid_search(config.RESTART_EPISODE_ON_COLLISION_LIST),
                "obs_space_type": tune.grid_search(config.OBS_SPACE_TYPE_LIST),
                "obs_space_n_closest": tune.grid_search(config.OBS_SPACE_N_CLOSEST_LIST),
                "obs_range": tune.grid_search(config.OBS_RANGE_LIST),
                "obs_clean_act_agent_obs": tune.grid_search(config.OBS_CLEAN_ACT_AGENT_OBS_LIST),
                "action_range_filter": tune.grid_search(config.ACTION_RANGE_FILTER_LIST),
                "action_conflict_filter": tune.grid_search(config.ACTION_CONFLICT_FILTER_LIST),
                "action_movement_vector_intersect_filter": tune.grid_search(config.ACTION_MOVEMENT_VECTOR_INTERSECT_FILTER_LIST),
                "reward_values": tune.grid_search(config.REWARD_VALUES_LIST),
                "acceleration": tune.grid_search(config.ACCELERATION_LIST),
                "sort_obs_by_dist": tune.grid_search(config.SORT_OBS_BY_DISTANCE_LIST),
                "dist_related_inzone_reward": tune.grid_search(config.DISTANCE_RELATED_INZONE_REWARD_LIST),
                "dist_related_inzone_reward_dist": tune.grid_search(config.DISTANCE_RELATED_INZONE_REWARD_DIST_LIST),
                "dist_related_inzone_reward_max_value": tune.grid_search(config.DISTANCE_RELATED_INZONE_REWARD_MAX_VALUE_LIST),
                "non_agent_switch": tune.grid_search(config.NON_AGENT_SWITCH_LIST),
                "non_agent_ratio": tune.grid_search(config.NON_AGENT_RATIO_LIST),
                "fake_switch": tune.grid_search(config.FAKE_SWITCH_LIST),
                "fake_ratio": tune.grid_search(config.FAKE_RATIO_LIST),
                "fake_mod_dist": tune.grid_search(config.FAKE_MOD_DIST_LIST)
            }},
            "train_batch_size": tune.grid_search(config.TRAIN_BATCH_SIZE_LIST),
            "rollout_fragment_length": tune.grid_search(config.ROLLOUT_FRAGMENT_LENGTH_LIST),
            "gamma": tune.grid_search(config.GAMMA_LIST),
            "lambda": tune.grid_search(config.PPO_LAMBDA_LIST),
            "clip_param": tune.grid_search(config.PPO_CLIP_PARAM_LIST),
            "lr": tune.grid_search(config.LEARNING_RATE_LIST),
            "lstm_cell_size": tune.grid_search(config.LSTM_SIZE_LIST) if config.ALG == "PPO" else None,
        },
        run_config=train.RunConfig(
            stop={
                "timesteps_total": config.TRAIN_TOTAL_TIMESTEPS
            },
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1000       # training iteration
            ),
            storage_path=os.path.join(os.getcwd(), "ray_results"),
            # storage_path = "/home/c_ai5ha/c_ai5g/rl_crossroad/ray_results"
            # local_dir = "/home/c_ai5ha/c_ai5g/rl_crossroad/ray_results"

        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max"
        )
    )
    results = tuner.fit()
    best_results = results.get_best_result()
    print(f"{Back.GREEN}Results:{Back.RESET}\n")


if not config.TRAIN:
    env = IntersectionEnv(
        env_conf=env_config
    )
    checkpoint_path = config.SAVED_MODEL_PATH
    restored_algo = Algorithm.from_checkpoint(checkpoint_path)
    obs, info = env.reset()
    state = {}
    agent_policy = ''

    for i in range(config.EPISODE_LENGTH):
        action_dict = {}
        agent_ids = list(obs.keys())

        if config.ALG == "DQN" and env.FAKE_SWITCH and env.PGD_ATTACK:
            env.Q_values = adv_func.get_agents_Q_values(env=env, obs=obs, restored_algo=restored_algo, agent_policy=agent_policy, agent_ids=agent_ids)

        for agent_id in agent_ids:
            if config.TERMINATING_AGENTS:
                rl_agent_id = agent_id
                temp_agent_id = env.data.index[env.data['rl_agent_id'] == rl_agent_id][0]
            if config.MULTI_POLICY:
                if env.data.loc[temp_agent_id]["start_edge"] == '1':
                    agent_policy = "policy_W"
                elif env.data.loc[temp_agent_id]["start_edge"] == '2':
                    agent_policy = "policy_N"
                elif env.data.loc[temp_agent_id]["start_edge"] == '3':
                    agent_policy = "policy_E"
                elif env.data.loc[temp_agent_id]["start_edge"] == '4':
                    agent_policy = "policy_S"
            else:
                agent_policy = "policy_0"

            if state.get(agent_id) is None:
                state[agent_id] = restored_algo.get_policy(agent_policy).model.get_initial_state()

            if restored_algo.get_policy(agent_policy).model.model_config['use_lstm']:      # if trained with USE_LSTM
                action_dict[agent_id], state[agent_id], _ = restored_algo.compute_single_action(observation=obs[agent_id], state=state[agent_id], policy_id=agent_policy, explore=False)
            else:
                action_dict[agent_id] = restored_algo.compute_single_action(observation=obs[agent_id], policy_id=agent_policy, explore=False)

            # print(agent_id, agent_policy)

        # create reference test: all the actions are keep speed (action=1)
        CREATE_REFERENCE_RUN = False
        if CREATE_REFERENCE_RUN:
            for agent in action_dict.keys():
                action_dict[agent] = 1

        obs, rewards, terminated, truncated, infos = env.step(action_dict)

    env.close()
    print(f"{Back.GREEN}{Fore.BLACK}Tensorboard command: {Back.RESET}{Fore.RESET}\n"
          f"tensorboard --logdir {checkpoint_path} --samples_per_plugin scalars=10000")


config.print_conf()
ray.shutdown()
