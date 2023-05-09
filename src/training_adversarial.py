import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals  # needed to load environments

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt

from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from datetime import datetime
from alpaca.data.timeframe import TimeFrame
import datetime
from stable_baselines3.common import monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from imitation.util.util import *
import functools
import warnings
import os
from gym.wrappers import TimeLimit
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from experts import get_expert_trajectories, expert_2
from stock_env import TradingEnv
from data import get_crypto_bars
from evaluation import print_summary, plot_trading_chart, calculate_profit

def test_run(bars, env, policy, j=0):
    test_env = env
    rewards = []
    actions = []
    done = False
    obs = env.reset()
    while done is False:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        rewards.append(reward)
        actions.append(action)
    print(f"num steps: {len(actions)}, total reward:{np.sum(np.array(rewards))}, avg reward:{np.sum(np.array(rewards))/len(rewards)}")
    profit, trades = calculate_profit(bars, actions)
    print_summary(profit, trades)
    plot_trading_chart(bars, actions, ret_img=False, save=True, j=j)


def run():
    env = gym.make("seals/CartPole-v0")

    # 1) get expert by training (PPO algo)
    """
    expert = PPO(
        policy=MlpPolicy,
        env=env, # not vectorized
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64)
    expert.learn(1000)  # Note: set to 100000 to train a proficient expert
    print("done training expert")

    # 2) get expert trajectories
    rng = np.random.default_rng()
    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "seals/CartPole-v0",
            n_envs=5,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            rng=rng),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng
    )
    print(type(rollouts), len(rollouts), rollouts[0])
    """


    def make_vec_env(
        bars,
        ws,
        *,
        rng: np.random.Generator,
        n_envs: int = 8,
        parallel: bool = False,
        log_dir: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
        env_make_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> VecEnv:

        def make_env(i: int, this_seed: int, env=env, bars=bars, ws=ws) -> gym.Env:
            env = TradingEnv(bars, ws)

            # Seed each environment with a different, non-sequential seed for diversity
            # (even if caller is passing us sequentially-assigned base seeds). int() is
            # necessary to work around gym bug where it chokes on numpy int64s.
            env.seed(int(this_seed))

            if max_episode_steps is not None:
                env = TimeLimit(env, max_episode_steps)
        
            # Use Monitor to record statistics needed for Baselines algorithms logging
            # Optionally, save to disk
            log_path = None
            if log_dir is not None:
                log_subdir = os.path.join(log_dir, "monitor")
                os.makedirs(log_subdir, exist_ok=True)
                log_path = os.path.join(log_subdir, f"mon{i:03d}")

            env = monitor.Monitor(env, log_path)

            if post_wrappers:
                for wrapper in post_wrappers:
                    env = wrapper(env, i)

            return env

        env_seeds = make_seeds(rng, n_envs)
        env_fns: List[Callable[[], gym.Env]] = [
            functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
        ]
        if parallel:
            # See GH hill-a/stable-baselines issue #217
            return SubprocVecEnv(env_fns, start_method="forkserver")
        else:
            return DummyVecEnv(env_fns)

    # params
    begin_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 3, 31)
    ws = 10

    print(" GET EXPERT ")
    rollouts = get_expert_trajectories(ws=ws, num_trajs=50, begin_date=begin_date, end_date=end_date)
    print(type(rollouts), len(rollouts))
    print([f"{key} {type(val)}" for key, val in rollouts[0].__dict__.items()])
    bars = get_crypto_bars("BTC/USD", begin_date,
                        end_date, timeframe=TimeFrame.Day)
    actions = expert_2(bars)
    profit, trades = calculate_profit(bars, actions)
    print_summary(profit, trades)
    plot_trading_chart(bars, actions, ret_img=False, save=True, j=0, mode="enum")

    env = TradingEnv(bars, ws)
    rng = np.random.default_rng()
    #venv = make_vec_env(env, rng=rng)
    venv = make_vec_env(bars, ws, rng=rng, n_envs=2)

    # 3) train to get reward function
    #venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)
    learner = PPO( # actor-critic policy
        env=venv, # vectorized env
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=20) # 10

    reward_net = BasicShapedRewardNet( # discrimnator net
        venv.observation_space, venv.action_space, 
        normalize_input_layer=RunningNorm)

    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=20, # 1024
        gen_replay_buffer_capacity=100, # 2048
        n_disc_updates_per_round=4,
        venv=venv, # same vectorized env, but will have learned reward function
        gen_algo=learner, # generator = policy?
        reward_net=reward_net,) # discriminator = reward

    print("TEST RUN BEFORE")
    test_run(bars, env, learner,j=1)

    print("evaluating before")
    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 5, return_episode_rewards=True)

    airl_epochs = 3
    print("BEGINNING AIRL TRAINING")
    for i_epoch in range(airl_epochs):
        print("starting training at:",datetime.datetime.now())
        airl_trainer.train(4096)  # Note: set to 300000 for better results

        print("TEST RUN AFTER")
        test_run(bars, env, learner, j=2+i_epoch)

        print("evaluating after")
        learner_rewards_after_training, _ = evaluate_policy(
            learner, venv, 5, return_episode_rewards=True)
        

    # Z) plot
    print("before training:", np.mean(learner_rewards_before_training))
    print("after training:", np.mean(learner_rewards_after_training))

    plt.hist(
        [learner_rewards_before_training, learner_rewards_after_training],
        label=["untrained", "trained"])
    plt.legend()
    plt.savefig(os.path.dirname(__file__)+f"/test_bars.png")
    plt.close()
    #plt.show()


    print("CROSS CHECK AGAINST BC")
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=rollouts,
        rng=rng,
    )
    bc_trainer.train(n_epochs=1)
    reward, _ = evaluate_policy(bc_trainer.policy, env, 5)

if __name__=="__main__":
    run()