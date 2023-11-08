import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt

from utils import episode_reward_plot, TIMESTAMP, save_frames_as_gif
# from vpg import compute_returns
from vpg import compute_returns, ActorNetwork


def compute_advantages(returns, values):
    """ Compute episode advantages based on precomputed episode returns.

    Parameters
    ----------
    returns : list of float
        Episode returns calculated with compute_returns.
    values: list of float
        Critic outputs for the states visited during the episode

    Returns
    -------
    list of float
        Episode advantages.
    """

    advantages = []
    for ret, value in zip(returns, values):
        advantages.append(ret - value)
    return advantages


def compute_generalized_advantages(rewards, values, next_value, discount, lamb):
    """ Compute generalized advantages (GAE) of the episode.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    values: list of float
        Episode state values.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
    discount : float
        Discount factor.
    lamb: float
        Lambda parameter of GAE.

    Returns
    -------
    list of float
        Generalized advanteges of the episode.
    """

    gae = 0
    advantages = []
    for reward, value in zip(reversed(rewards), reversed(values)):
        td_error = reward + discount * next_value - value
        gae = td_error + discount * lamb * gae
        advantages.append(gae)
        next_value = value
    return advantages[::-1]


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma, lamb, use_gae):

        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logprob_lst = []
        self.return_lst = []
        self.adv_lst = []

        self.gamma = gamma
        self.lamb = lamb
        self.use_gae = use_gae

        self.traj_start = 0

    def put(self, obs, action, reward, logprob, value):
        """Put a transition into the memory."""

        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.value_lst.append(value)
        self.logprob_lst.append(logprob)

    def get(self):
        """Get all stored transition attributes in the form of lists."""

        return self.obs_lst, self.reward_lst, self.logprob_lst, self.value_lst, self.return_lst, self.adv_lst

    def clear(self):
        """Reset the transition memory."""

        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logprob_lst = []
        self.return_lst = []
        self.adv_lst = []

        self.traj_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return and advantage or generalized advantage estimation.

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
        """

        reward_traj = self.reward_lst[self.traj_start:]
        value_traj = self.value_lst[self.traj_start:]
        traj_returns = compute_returns(reward_traj, next_value, self.gamma)
        self.return_lst.extend(traj_returns)
        self.traj_start = len(self.obs_lst)
        # traj_returns = compute_returns(self.reward_lst[self.traj_start:], next_value, self.gamma)
        # self.return_lst.extend(traj_returns)

        if self.use_gae:
            traj_adv = compute_generalized_advantages(reward_traj, value_traj, next_value, self.gamma, self.lamb)
        else:
            traj_adv = compute_advantages(traj_returns, value_traj)
        self.adv_lst.extend(traj_adv)


class CriticNetwork(nn.Module):
    """Neural Network used to learn the state-value function."""

    def __init__(self, num_observations):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCritic:
    """The Actor-Critic approach."""

    # Advantage: bs 200, lr 0.005
    # GAE: bs 500, critic lr 0.001, lr 0.01
    def __init__(self, env, batch_size=500, gamma=0.99, lamb=0.99, lr=0.01, use_gae=True):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        batch_size : int, optional
            Number of transitions to use for one opimization step.
        gamma : float, optional
            Discount factor.
        lamb : float, optional
            Lambda parameters of GAE.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.batch_size = batch_size
        self.env = env
        self.memory = TransitionMemory(gamma, lamb, use_gae)

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.critic_net = CriticNetwork(self.obs_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.critic_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the actor-critic.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episode_counter = 0
        for timestep in range(1, total_timesteps + 1):
            action, logprob, value = self.predict(obs, train_returns=True)
            obs_, reward, terminated, truncated, info = self.env.step(action)
            episode_rewards.append(reward)

            # Put into transition buffer
            self.memory.put(obs, action, reward, logprob, value)

            # Update current obs
            obs = obs_

            if terminated or truncated:
                obs, _ = self.env.reset()
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []
                self.memory.finish_trajectory(0.0)

            if (timestep - episode_counter) == self.batch_size:
                # Finish partial trajectory
                self.memory.finish_trajectory(self.critic_net(torch.Tensor(obs)).item())
                # Get transitions inside memory
                obs_lst, reward_lst, logprob_lst, value_lst, return_lst, adv_lst = self.memory.get()

                # Calculate losses
                actor_loss = self.calc_actor_loss(logprob_lst, adv_lst)
                critic_loss = self.calc_critic_loss(value_lst, return_lst)
                loss = actor_loss + critic_loss

                # Back-propagate and optimize
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward()
                self.optim_actor.step()
                self.optim_critic.step()

                # Clear memory
                self.memory.clear()
                episode_counter = timestep

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)
                # if last 10 rewards are same values, then stop training
                # if np.mean(overall_rewards[-40:]) >= 490:
                #     break

    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss for one batch of transitions."""
        return F.mse_loss(torch.Tensor(value_lst), torch.Tensor(return_lst))

    @staticmethod
    def calc_actor_loss(logprob_lst, adv_lst):
        """Calculate actor "loss" for one batch of transitions."""

        return -(torch.stack(logprob_lst) * torch.Tensor(adv_lst)).mean()

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        obs = torch.Tensor(obs)
        probs = self.actor_net(obs)
        policy = Categorical(probs=probs)
        action = policy.sample()
        logprob = policy.log_prob(action)

        value = self.critic_net(obs)

        if train_returns:
            return action.item(), logprob, value.item()
        else:
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    use_gae = True
    _env = None
    learn: bool = False
    if learn:
        _env = gym.make(env_id)
    else:
        # _env = gym.make(env_id, render_mode="rgb_array")
        _env = gym.make(env_id, render_mode="human")

    obs, _ = _env.reset()

    print("Observation space" + str(_env.observation_space))
    AC = ActorCritic(_env, use_gae=use_gae, lr=0.005)


    if learn:
        AC.learn(120000)

        print('saving the models')
        torch.save(AC.actor_net.state_dict(), "savepoint/a2c/actor_state_"+TIMESTAMP)
        torch.save(AC.critic_net.state_dict(), "savepoint/a2c/critic_state_"+TIMESTAMP)

        print('saving optims')
        torch.save(AC.optim_actor.state_dict(), "savepoint/a2c/actor_optim_"+TIMESTAMP)
        torch.save(AC.optim_critic.state_dict(), "savepoint/a2c/critic_optim_"+TIMESTAMP)

    else:
        # Load previous model
        load_timestamp: str = "1699447808"
        print("loading saved models")
        AC.actor_net.load_state_dict(torch.load("savepoint/a2c/actor_state_" + load_timestamp))
        AC.critic_net.load_state_dict(torch.load("savepoint/a2c/critic_state_" + load_timestamp))
        AC.optim_actor.load_state_dict(torch.load("savepoint/a2c/actor_optim_" + load_timestamp))
        AC.optim_critic.load_state_dict(torch.load("savepoint/a2c/critic_optim_" + load_timestamp))

        counter = 0
        terminated = False
        truncated = False
        frames = []
        while counter < 100:
            action = AC.predict(obs)
            obs_, reward, terminated, truncated, info = _env.step(action)
            counter = counter+1
            sleep(0.01)
            _env.render()
            # frames.append(_env.render())
            if terminated:
                print('terminated')
                _env.reset()
                # break

        print(frames[0].shape)
        # to save a gif of all the frames
        # save_frames_as_gif(frames)

    input("Press Enter to end...")
