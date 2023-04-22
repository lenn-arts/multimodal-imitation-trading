from seals.base_envs import TabularModelPOMDP
import numpy as np
from seals.diagnostics.cliff_world import CliffWorldEnv

import gym.envs.classic_control
from gym import spaces

class CartPole(TabularModelPOMDP):
    def __init__(self, *, transition_matrix: np.ndarray, 
                 observation_matrix: np.ndarray, 
                 reward_matrix: np.ndarray, 
                 horizon = None, 
                 initial_state_dist = None):
        super().__init__(transition_matrix=transition_matrix, 
                         observation_matrix=observation_matrix, 
                         reward_matrix=reward_matrix, 
                         horizon=horizon, 
                         initial_state_dist=initial_state_dist)


class FixedHorizonCartPole(gym.envs.classic_control.CartPoleEnv):
    """Fixed-length variant of CartPole-v1.
    Reward is 1.0 whenever the CartPole is an "ok" state (i.e. the pole is upright
    and the cart is on the screen). Otherwise reward is 0.0.
    Done is always False. (Though note that by default, this environment is wrapped
    in `TimeLimit` with max steps 500.)
    """

    def __init__(self, horizon):
        """Builds FixedHorizonCartPole, modifying observation_space from Gym parent."""
        super().__init__()
        self.horizon = horizon

        high = [
            np.finfo(np.float32).max,  # x axis
            np.finfo(np.float32).max,  # x velocity
            np.pi,  # theta in radians
            np.finfo(np.float32).max,  # theta velocity
        ]
        high = np.array(high)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        """Reset for FixedHorizonCartPole."""
        return super().reset().astype(np.float32)

    def step(self, action):
        """Step function for FixedHorizonCartPole."""
        with warnings.catch_warnings():
            # Filter out CartPoleEnv warning for calling step() beyond done=True.
            warnings.filterwarnings("ignore", ".*You are calling.*")
            super().step(action)

        self.state = list(self.state)
        x, _, theta, _ = self.state

        # Normalize theta to [-pi, pi] range.
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.state[2] = theta

        state_ok = bool(
            abs(x) < self.x_threshold and abs(theta) < self.theta_threshold_radians,
        )

        rew = 1.0 if state_ok else 0.0
        return np.array(self.state, dtype=np.float32), rew, False, {}

if __name__=="__main__":
    env = CliffWorldEnv(width=5,height=5,horizon=10, use_xy_obs=True)
    env = FixedHorizonCartPole(500)
    #print(env.observation_matrix.shape)
    #print(env.transition_matrix.shape)
    #print(env.reward_matrix.shape)