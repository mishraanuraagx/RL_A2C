import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import episode_reward_plot


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    return_lst = []
    ret = next_value
    for reward in reversed(rewards):
        ret = reward + discount * ret
        return_lst.append(ret)
    return return_lst[::-1]


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):
        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.logprob_lst = []
        self.return_lst = []

        self.gamma = gamma
        self.traj_start = 0

    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""

        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)

    def get(self):
        """Get all stored transition attributes in the form of lists."""

        return self.obs_lst, self.action_lst, self.logprob_lst, self.return_lst

    def clear(self):
        """Reset the transition memory."""

        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.logprob_lst = []
        self.return_lst = []
        self.traj_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        reward_traj = self.reward_lst[self.traj_start:]
        return_traj = compute_returns(reward_traj, next_value, self.gamma)
        self.return_lst.extend(return_traj)
        self.traj_start = len(self.reward_lst)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)


class VPG:
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """

        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episodes_counter = 0

        for timestep in range(1, total_timesteps + 1):

            action, logprob = self.predict(obs, train_returns=True)
            obs_, reward, terminated, truncated, _ = self.env.step(action)
            self.memory.put(obs, action, reward, logprob)
            episode_rewards.append(reward)

            # Update current obs
            obs = obs_

            if terminated or truncated:

                obs, _ = self.env.reset()

                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

                self.memory.finish_trajectory(0.0)

                episodes_counter += 1

                if episodes_counter == self.episodes_update:
                    # Get transitions from memory
                    obs_lst, reward_lst, logprob_lst, return_lst = self.memory.get()

                    # Calculate loss
                    loss = self.calc_actor_loss(logprob_lst, return_lst)

                    # Back-propagate and optimize
                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

                    # Clear memory
                    episodes_counter = 0
                    self.memory.clear()

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1,
                                    wait=timestep == total_timesteps)

    @staticmethod
    def calc_actor_loss(logprob_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # Note: negative sign, as torch optimizers per default perform gradients descent, not ascent
        return -(torch.Tensor(return_lst) * torch.stack(logprob_lst)).mean()

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        probs = self.actor_net(torch.Tensor(obs))
        policy = Categorical(probs=probs)
        action = policy.sample()
        logprob = policy.log_prob(action)

        if train_returns:

            return action.item(), logprob
        else:

            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = gym.make(env_id)
    vpg = VPG(_env)
    vpg.learn(100000)
