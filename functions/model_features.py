from ray.rllib.models.preprocessors import get_preprocessor
import numpy as np
import torch

def get_Q_values(algo, policy_id, observation_space, obs, agent_id):
    get_preprocessor(observation_space)(observation_space)
    pp = get_preprocessor(observation_space)(observation_space)
    flatten_obs = pp.transform(obs[agent_id])
    model = algo.get_policy(policy_id).model
    model_out = model({"obs": torch.from_numpy(np.array([flatten_obs]))})
    Q_values = model.get_q_value_distributions(model_out[0])[0]
    return Q_values