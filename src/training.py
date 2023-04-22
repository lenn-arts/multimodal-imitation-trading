from functools import partial

from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
import stable_baselines3.common.vec_env as sbve
import gym
#from stable-baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt
import torch as torch
import numpy as np

from environment import FixedHorizonCartPole

from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
)
from imitation.data import rollout
from imitation.rewards import reward_nets

def setup():
    env_creator = partial(CliffWorldEnv, height=4, horizon=8, width=7, use_xy_obs=True)
    env_single = env_creator() # creates CliffWorldEnv with above params
    env_single = gym.make('seals/CartPole-v0')
    env_single = FixedHorizonCartPole(500)
    # POMDPS = partially observable Markov Decision Process
    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator()) # this exposes full state?

    # This is just a vectorized environment because `generate_trajectories` expects one
    state_venv = sbve.DummyVecEnv([state_env_creator] * 4)

    # performs soft bellman backup for finite-horizon MDP, returns soft values, Q-values and MCE policy (V,Q,π)
    _, _, pi = mce_partition_fh(env_single) # pi = optimal policy from fully observable environment?
    print(pi)

    # calculate state visitation frequency Ds for each state s under a given policy pi. Returns probabilities D and expected discounted number of visits for the horizon Dcum  
    _, om = mce_occupancy_measures(env_single, pi=pi)

    # generate expert policy for current environment via RL
    rng = np.random.default_rng()
    expert_policy = TabularPolicy(
        state_space=env_single.state_space,
        action_space=env_single.action_space,
        pi=pi,
        rng=rng,
    )

    # generate trajectories from expert policy
    expert_trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
        rng=rng,
    )

    print("Expert stats: ", rollout.rollout_stats(expert_trajs))


    def train_mce_irl(demos, hidden_sizes, lr=0.01, **kwargs):
        reward_net = reward_nets.BasicRewardNet(
            env_single.observation_space,
            env_single.action_space,
            hid_sizes=hidden_sizes,
            use_action=False,
            use_done=False,
            use_next_state=False,
        )

        mce_irl = MCEIRL(
            demos,
            env_single,
            reward_net,
            log_interval=250,
            optimizer_kwargs=dict(lr=lr),
            rng=rng,
        )
        occ_measure = mce_irl.train(**kwargs)

        imitation_trajs = rollout.generate_trajectories(
            policy=mce_irl.policy,
            venv=state_venv,
            sample_until=rollout.make_min_timesteps(5000),
            rng=rng,
        )
        print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        env_single.draw_value_vec(occ_measure)
        plt.title("Occupancy for learned reward")
        plt.xlabel("Gridworld x-coordinate")
        plt.ylabel("Gridworld y-coordinate")
        plt.subplot(1, 2, 2)
        _, true_occ_measure = mce_occupancy_measures(env_single)
        env_single.draw_value_vec(true_occ_measure)
        plt.title("Occupancy for true reward")
        plt.xlabel("Gridworld x-coordinate")
        plt.ylabel("Gridworld y-coordinate")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        env_single.draw_value_vec(
            reward_net(torch.as_tensor(env_single.observation_matrix), None, None, None)
            .detach()
            .numpy()
        )
        plt.title("Learned reward")
        plt.xlabel("Gridworld x-coordinate")
        plt.ylabel("Gridworld y-coordinate")
        plt.subplot(1, 2, 2)
        env_single.draw_value_vec(env_single.reward_matrix)
        plt.title("True reward")
        plt.xlabel("Gridworld x-coordinate")
        plt.ylabel("Gridworld y-coordinate")
        plt.show()

        return mce_irl
    
if __name__=="__main__":
    setup()