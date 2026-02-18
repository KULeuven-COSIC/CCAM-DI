from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict
import numpy as np

class MyCallbacks(DefaultCallbacks):

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     "after env reset!"
        # )
        # Create lists to store angles in
        episode.user_data["total_collisions"] = []
        episode.user_data["total_collisions_in_episode"] = []
        episode.user_data["fake_collisions"] = []
        episode.user_data["fake_collisions_in_episode"] = []
        episode.user_data["non_agent_collisions"] = []
        episode.user_data["non_agent_collisions_in_episode"] = []
        episode.user_data["total_vehicle_count"] = []
        episode.user_data["total_vehicle_count_in_episode"] = []
        episode.user_data["removed_vehicles"] = []
        episode.user_data["avg_speed_agent0"] = []
        episode.user_data["vehicles_in_zone"] = []
        episode.user_data["collision_ratio"] = []
        episode.user_data["collision_ratio_in_episode"] = []


    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        total_collisions = base_env.envs[0].total_collisions
        total_collisions_in_episode = base_env.envs[0].total_collisions_in_episode
        fake_collisions = base_env.envs[0].fake_collisions
        fake_collisions_in_episode = base_env.envs[0].fake_collisions_in_episode
        non_agent_collisions = base_env.envs[0].non_agent_collisions
        non_agent_collisions_in_episode = base_env.envs[0].non_agent_collisions_in_episode
        removed_vehicles = base_env.envs[0].removed_vehicles_num
        avg_speed_agent0 = base_env.envs[0].avg_speed_agent0
        vehicles_in_zone = base_env.envs[0].vehicles_in_zone
        collision_ratio = base_env.envs[0].collision_ratio
        collision_ratio_in_episode = base_env.envs[0].collision_ratio_in_episode
        total_vehicle_count = base_env.envs[0].total_vehicle_count
        total_vehicle_count_in_episode = base_env.envs[0].total_vehicle_count_in_episode

        episode.user_data["total_collisions"].append(total_collisions)
        episode.user_data["total_collisions_in_episode"].append(total_collisions_in_episode)
        episode.user_data["fake_collisions"].append(fake_collisions)
        episode.user_data["fake_collisions_in_episode"].append(fake_collisions_in_episode)
        episode.user_data["non_agent_collisions"].append(non_agent_collisions)
        episode.user_data["non_agent_collisions_in_episode"].append(non_agent_collisions_in_episode)
        episode.user_data["total_vehicle_count"].append(total_vehicle_count)
        episode.user_data["total_vehicle_count_in_episode"].append(total_vehicle_count_in_episode)
        episode.user_data["removed_vehicles"].append(removed_vehicles)
        episode.user_data["avg_speed_agent0"].append(avg_speed_agent0)
        episode.user_data["vehicles_in_zone"].append(vehicles_in_zone)
        episode.user_data["collision_ratio"].append(collision_ratio)
        episode.user_data["collision_ratio_in_episode"].append(collision_ratio_in_episode)

        # episode.custom_metrics["total_collisions"] = np.max(episode.user_data["total_collisions"])
        # episode.custom_metrics["total_collisions_in_episode"] = np.max(episode.user_data["total_collisions_in_episode"])
        # episode.custom_metrics["total_vehicle_count"] = np.max(episode.user_data["total_vehicle_count"])
        # episode.custom_metrics["total_vehicle_count_in_episode"] = np.max(episode.user_data["total_vehicle_count_in_episode"])
        # episode.custom_metrics["removed_vehicles"] = np.max(episode.user_data["removed_vehicles"])
        # episode.custom_metrics["avg_speed_agent0"] = np.average(episode.user_data["avg_speed_agent0"])
        # episode.custom_metrics["vehicles_in_zone_max"] = np.max(episode.user_data["vehicles_in_zone"])
        # episode.custom_metrics["collision_ratio"] = np.max(episode.user_data["collision_ratio"])
        # episode.custom_metrics["collision_ratio_in_episode"] = np.max(episode.user_data["collision_ratio_in_episode"])

        episode.custom_metrics["total_collisions"] = episode.user_data["total_collisions"][-1]
        episode.custom_metrics["total_collisions_in_episode"] = episode.user_data["total_collisions_in_episode"][-1]
        episode.custom_metrics["fake_collisions"] = episode.user_data["fake_collisions"][-1]
        episode.custom_metrics["fake_collisions_in_episode"] = episode.user_data["fake_collisions_in_episode"][-1]
        episode.custom_metrics["non_agent_collisions"] = episode.user_data["non_agent_collisions"][-1]
        episode.custom_metrics["non_agent_collisions_in_episode"] = episode.user_data["non_agent_collisions_in_episode"][-1]
        episode.custom_metrics["total_vehicle_count"] = episode.user_data["total_vehicle_count"][-1]
        episode.custom_metrics["total_vehicle_count_in_episode"] = episode.user_data["total_vehicle_count_in_episode"][-1]
        episode.custom_metrics["removed_vehicles"] = episode.user_data["removed_vehicles"][-1]
        episode.custom_metrics["avg_speed_agent0"] = np.average(episode.user_data["avg_speed_agent0"])
        episode.custom_metrics["vehicles_in_zone_max"] = episode.user_data["vehicles_in_zone"][-1]
        episode.custom_metrics["collision_ratio"] = episode.user_data["collision_ratio"][-1]
        episode.custom_metrics["collision_ratio_in_episode"] = episode.user_data["collision_ratio_in_episode"][-1]

        if 'DQN' in str(episode.worker.config):
            episode.custom_metrics["epsilon"] = policies[list(worker.get_policies_to_train())[0]].exploration.get_state()['cur_epsilon']

    # def on_episode_end(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs,
    # ):
    #     # Check if there are multiple episodes in a batch, i.e.
    #     # "batch_mode": "truncate_episodes".
    #     # if worker.config.batch_mode == "truncate_episodes":
    #     #     # Make sure this episode is really done.
    #     #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
    #     #         -1
    #     #     ]["dones"][-1], (
    #     #         "ERROR: `on_episode_end()` should only be called "
    #     #         "after episode is done!"
    #     #     )
    #     episode.custom_metrics["total_collisions"] = np.max(episode.user_data["total_collisions"])
    #     episode.custom_metrics["removed_vehicles"] = np.max(episode.user_data["removed_vehicles"])
    #     episode.custom_metrics["avg_speed_agent0"] = np.average(episode.user_data["avg_speed_agent0"])


    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        # Normally, RLlib would aggregate any custom metric into a mean, max and min
        # of the given metric.
        # For the sake of this example, we will instead compute the variance and mean
        # of the pole angle over the evaluation episodes.

        total_collisions = result["custom_metrics"]["total_collisions"]
        total_collisions_in_episode = result["custom_metrics"]["total_collisions_in_episode"]
        fake_collisions = result["custom_metrics"]["fake_collisions"]
        fake_collisions_in_episode = result["custom_metrics"]["fake_collisions_in_episode"]
        non_agent_collisions = result["custom_metrics"]["non_agent_collisions"]
        non_agent_collisions_in_episode = result["custom_metrics"]["non_agent_collisions_in_episode"]
        total_vehicle_count = result["custom_metrics"]["total_vehicle_count"]
        total_vehicle_count_in_episode = result["custom_metrics"]["total_vehicle_count_in_episode"]
        removed_vehicles = result["custom_metrics"]["removed_vehicles"]
        avg_speed_agent0 = result["custom_metrics"]["avg_speed_agent0"]
        vehicles_in_zone_max = result["custom_metrics"]["vehicles_in_zone_max"]
        collision_ratio = result["custom_metrics"]["collision_ratio"]
        collision_ratio_in_episode = result["custom_metrics"]["collision_ratio_in_episode"]

        # result["custom_metrics"]["total_collisions"] = np.max(total_collisions)
        # result["custom_metrics"]["total_collisions_in_episode"] = np.max(total_collisions_in_episode)
        # result["custom_metrics"]["total_vehicle_count"] = np.max(total_vehicle_count)
        # result["custom_metrics"]["total_vehicle_count_in_episode"] = np.max(total_vehicle_count_in_episode)
        # result["custom_metrics"]["removed_vehicles"] = np.max(removed_vehicles)
        # result["custom_metrics"]["avg_speed_agent0"] = np.average(avg_speed_agent0)
        # result["custom_metrics"]["vehicles_in_zone_max"] = np.max(vehicles_in_zone_max)
        # result["custom_metrics"]["collision_ratio"] = np.max(collision_ratio)
        # result["custom_metrics"]["collision_ratio_in_episode"] = np.max(collision_ratio_in_episode)

        result["custom_metrics"]["total_collisions"] = total_collisions[-1]
        result["custom_metrics"]["total_collisions_in_episode"] = total_collisions_in_episode[-1]
        result["custom_metrics"]["fake_collisions"] = fake_collisions[-1]
        result["custom_metrics"]["fake_collisions_in_episode"] = fake_collisions_in_episode[-1]
        result["custom_metrics"]["non_agent_collisions"] = non_agent_collisions[-1]
        result["custom_metrics"]["non_agent_collisions_in_episode"] = non_agent_collisions_in_episode[-1]
        result["custom_metrics"]["total_vehicle_count"] = total_vehicle_count[-1]
        result["custom_metrics"]["total_vehicle_count_in_episode"] = total_vehicle_count_in_episode[-1]
        result["custom_metrics"]["removed_vehicles"] = removed_vehicles[-1]
        result["custom_metrics"]["avg_speed_agent0"] = np.average(avg_speed_agent0)
        result["custom_metrics"]["vehicles_in_zone_max"] = vehicles_in_zone_max[-1]
        result["custom_metrics"]["collision_ratio"] = collision_ratio[-1]
        result["custom_metrics"]["collision_ratio_in_episode"] = collision_ratio_in_episode[-1]

        if 'DQN' in str(algorithm):
            # result["custom_metrics"]["epsilon"] = np.max(result["custom_metrics"]["epsilon"])
            result["custom_metrics"]["epsilon"] = result["custom_metrics"]["epsilon"][-1]

        print(f"\n(on_train) Training iterations: {algorithm.iteration}"
              f"\n(on_train) Timestep: {result['agent_timesteps_total']}"
              f"\n(on_train) Timesteps trained in this iteration: {result['num_steps_trained_this_iter']} (total: {result['timesteps_total']})"
              f"\n(on_train) Episodes this iteration: {result['episodes_this_iter']}"
              f"\n(on_train) Custom metrics: {result['custom_metrics']}"
              "\n------------------------------------------------------------")
