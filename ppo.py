import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

from utils import episode_reward_plot
from vpg import compute_returns, ActorNetwork
from a2c import CriticNetwork, compute_advantages, compute_generalized_advantages


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma, lamb, use_gae):

        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logprob_lst = []
        self.logprob_old_lst = []
        self.return_lst = []
        self.adv_lst = []

        self.gamma = gamma
        self.lamb = lamb
        self.use_gae = use_gae

        self.traj_start = 0

    def put(self, obs, action, reward, logprob, logprob_old, value):
        """Put a transition into the memory."""

        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)
        self.logprob_old_lst.append(logprob_old)
        self.value_lst.append(value)

    def get(self):
        """Get all stored transition attributes in the form of lists."""

        return self.obs_lst, self.reward_lst, self.logprob_lst, self.logprob_old_lst, self.value_lst, self.return_lst, self.adv_lst

    def clear(self):
        """Reset the transition memory."""

        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logprob_lst = []
        self.logprob_old_lst = []
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

        if self.use_gae:
            traj_adv = compute_generalized_advantages(reward_traj, value_traj, next_value, self.gamma, self.lamb)
        else:
            traj_adv = compute_advantages(traj_returns, value_traj)
        self.adv_lst.extend(traj_adv)


class ProximalPolicyOptimization:
    """The Actor-Critic approach."""

    def __init__(self, env, batch_size=500, gamma=0.99, lamb=0.99, lr=0.01, eps=0.5, use_gae=True):
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
        self.eps = eps
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

        logprob_old = None

        episode_counter = 0
        for timestep in range(1, total_timesteps + 1):

            action, logprob, value = self.predict(obs, train_returns=True)
            obs_, reward, terminated, truncated, info = self.env.step(action)
            episode_rewards.append(reward)

            # Put into transition buffer (only if `logprob_old` exists, i.e. we are not in the first step)
            if logprob_old:
                self.memory.put(obs, action, reward, logprob, logprob_old.detach(), value)
            logprob_old = logprob

            # Update current obs
            obs = obs_

            if terminated or truncated:

                obs, _ = self.env.reset()
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []
                self.memory.finish_trajectory(0.0)
                logprob_old = None

            if (timestep - episode_counter) == self.batch_size:

                # Finish partial trajectory
                self.memory.finish_trajectory(self.critic_net(torch.Tensor(obs)).item())
                # Get transitions inside memory
                obs_lst, reward_lst, logprob_lst, logprob_old_lst, value_lst, return_lst, adv_lst = self.memory.get()

                # Calculate losses
                actor_loss = self.calc_actor_loss(logprob_lst, logprob_old_lst, adv_lst, eps=self.eps)
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

    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss for one batch of transitions."""

        return F.mse_loss(torch.Tensor(value_lst), torch.Tensor(return_lst))

    @staticmethod
    def calc_actor_loss(logprob_lst, logprob_old_lst, adv_lst, eps):
        """Calculate actor "loss" for one batch of transitions."""

        adv_list = torch.Tensor(adv_lst)
        ratio = (torch.stack(logprob_lst) - torch.stack(logprob_old_lst)).exp()
        clipped = torch.clamp(ratio, 1-eps, 1+eps) * adv_list
        return -(torch.min(ratio*adv_list, clipped)).mean()

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

    _env = gym.make(env_id)
    PPO = ProximalPolicyOptimization(_env, use_gae=use_gae)
    PPO.learn(100000)
