from functions.model_features import get_Q_values, get_Q_values_and_grad
import random
import torch
import numpy as np
from torch.distributions import Beta
import torch.nn.functional as F

def compromize_observation(env, obs, restored_algo, agent_policy, rl_agent_id):
    """
    Compromise observations of vehicle_type=="fake" agents
    """
    agent_id = env.data.index[env.data['rl_agent_id'] == rl_agent_id][0]        # get env.data table index of the agent
    other_agent_list = list(obs[rl_agent_id].keys())
    other_agent_list.remove("act_agent")
    for vehicle in other_agent_list:
        other_agent_id = env.obs_key_correlation[agent_id][vehicle]         # get real env.data index may be renamed <-- observation agent IDs are renamed due to reordering (SORT_OBS_BY_DIST)
        veh_type = env.data.at[other_agent_id, 'vehicle_type']
        if env.FAKE_SWITCH:
            if not env.PGD_ATTACK:
                # random noise
                if 'x' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['x'] = obs[rl_agent_id][vehicle]['x'] + random.uniform(-env.FAKE_MOD_DIST, env.FAKE_MOD_DIST)
                if 'y' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['y'] = obs[rl_agent_id][vehicle]['y'] + random.uniform(-env.FAKE_MOD_DIST, env.FAKE_MOD_DIST)
                if 'd' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['d'] = obs[rl_agent_id][vehicle]['d'] + random.uniform(-env.FAKE_MOD_DIST, env.FAKE_MOD_DIST)
                if 'angle' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['angle'] = obs[rl_agent_id][vehicle]['angle'] + random.uniform(-env.FAKE_MOD_DIST, env.FAKE_MOD_DIST)
            else:
                # PGD attack               
                if 'x' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['x'] = obs[rl_agent_id][vehicle]['x'] + pgd_attack(obs=obs, env=env, restored_algo=restored_algo, agent_policy=agent_policy, rl_agent_id=rl_agent_id, vehicle = vehicle)[0]
                if 'y' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['y'] = obs[rl_agent_id][vehicle]['y'] + pgd_attack(obs=obs, env=env, restored_algo=restored_algo, agent_policy=agent_policy, rl_agent_id=rl_agent_id, vehicle = vehicle)[1]
                if 'd' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['d'] = obs[rl_agent_id][vehicle]['d'] + pgd_attack(env, obs, restored_algo, agent_policy, rl_agent_id)
                if 'angle' in obs[rl_agent_id][vehicle]:
                    obs[rl_agent_id][vehicle]['angle'] = obs[rl_agent_id][vehicle]['angle'] + pgd_attack(env, obs, restored_algo, agent_policy, rl_agent_id)
    return obs
def extract_obs_values(obs, agent):
    """Extracts only numerical values from nested obs[agent] dictionary"""
    obs_values = []

    for key, subdict in obs[agent].items():
        if isinstance(subdict, dict):  # If nested, extract numerical values
            for subkey, value in subdict.items():
                if isinstance(value, (int, float, np.ndarray)):  # Check for numerical types
                    obs_values.append(float(value))  # Convert np.float32 to float
        elif isinstance(subdict, (int, float, np.ndarray)):  # If already numeric
            obs_values.append(float(subdict))  # Convert np.float32 to float

    return np.array(obs_values, dtype=np.float32)  # Convert to numpy array

def simple_pgd_attack_random_noise( obs, env, restored_algo, agent_policy, rl_agent_id, vehicle, epsilon=0.9):
    """
    Adds simple random noise to the agent's x and y coordinates.

    Args:
        obs (dict): The observation dict containing agent data.
        rl_agent_id (str or int): The agent's ID.
        vehicle (str): Vehicle key under the agent's observation.
        epsilon (float): Maximum amount of noise to add.

    Returns:
        tuple: (noise_x, noise_y)
    """
    # Generate random noise in range [-epsilon, epsilon]
    noise_x = 50 * random.uniform(-epsilon, epsilon)
    noise_y = 50 * random.uniform(-epsilon, epsilon)

    # Optionally apply it to the position directly
    # obs[rl_agent_id][vehicle]['x'] += noise_x
    # obs[rl_agent_id][vehicle]['y'] += noise_y

    return (noise_x, noise_y)



def take_adversary_action():
    """
    Control the speed of adversary vehicles to hit other vehicles
    """
    return
