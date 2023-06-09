import copy
from datetime import datetime
from functional import seq
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import Adamax

from safe_explorer.core.config import Config
from safe_explorer.core.replay_buffer import ReplayBuffer
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.utils.list import for_each, select_with_predicate

class DDPG:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 action_modifier=None):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._action_modifier = action_modifier

        self.count = 0
        self.count_tem_train = 0
        self.count_tem_eval = 0
        self.total_episode = 0
        self.episodic_reward_buffer = []
        self.episodic_length_buffer = []
        self.accum_constraint_violations = []
        self.accum_lv_eval = []
        self.accum_lv_train = []
        self.soh = []

        self._config = Config.get().ddpg.trainer

        self._initialize_target_networks()

        self._models = {
            'actor': self._actor,
            'critic': self._critic,
            'target_actor': self._target_actor,
            'target_critic': self._target_critic
        }

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        if self._config.use_gpu:
            self._cuda()

        self._initialize_optimizers()


    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad

        if self._config.use_gpu:
            self._cuda()
            tensor = tensor.cuda()

        return tensor

    def _initialize_target_networks(self):
        self._target_actor = copy.deepcopy(self._actor)
        self._target_critic = copy.deepcopy(self._critic)
    
    def _initialize_optimizers(self):
        self._actor_optimizer = Adam(self._actor.parameters(), lr=self._config.actor_lr)
        self._critic_optimizer = Adam(self._critic.parameters(), lr=self._config.critic_lr)
    
    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models.values())

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models.values())

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models.values())

    def _get_action(self, observation, c, is_training=True):
        # Action + random gaussian noise (as recommended in spinning up)
        action = self._actor(self._as_tensor(self._flatten_dict(observation)))
        if is_training:
            action += self._config.action_noise_range * torch.randn(self._env.action_space.shape).cuda()

        action = action.cpu().data.numpy()        # Convert tensor to ndarray

        if self._action_modifier:
            action = self._action_modifier(observation, action, c)

        return action

    def _get_q(self, batch):
        return self._critic(self._as_tensor(batch["observation"]))

    def _get_target(self, batch):
        # For each observation in batch:
        # target = r + discount_factor * (1 - done) * max_a Q_tar(s, a)
        # a => actions of actor on current observations
        # max_a Q_tar(s, a) = output of critic
        observation_next = self._as_tensor(batch["observation_next"])
        reward = self._as_tensor(batch["reward"]).reshape(-1, 1)
        done = self._as_tensor(batch["done"]).reshape(-1, 1)

        action = self._target_actor(observation_next).reshape(-1, *self._env.action_space.shape)

        q = self._target_critic(observation_next, action)

        return reward  + self._config.discount_factor * (1 - done) * q

    def _flatten_dict(self, inp):
        if type(inp) == dict:
            inp = np.concatenate(list(inp.values()))
        return inp

    def _update_targets(self, target, main):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self._config.polyak * target_param.data + \
                                    (1 - self._config.polyak) * main_param.data)

    def _update_batch(self):
        batch = self._replay_buffer.sample(self._config.batch_size)
        # Only pick steps in which action was non-zero
        # When a constraint is violated, the safety layer makes action 0 in
        # direction of violating constraint
        # valid_action_mask = np.sum(batch["action"], axis=1) > 0
        # batch = {k: v[valid_action_mask] for k, v in batch.items()}

        # Update critic
        self._critic_optimizer.zero_grad()
        q_target = self._get_target(batch)
        q_predicted = self._critic(self._as_tensor(batch["observation"]),
                                   self._as_tensor(batch["action"]))
        # critic_loss = torch.mean((q_predicted.detach() - q_target) ** 2)
        # Seems to work better
        critic_loss = F.smooth_l1_loss(q_predicted, q_target)

        critic_loss.backward()
        self._critic_optimizer.step()

        # Update actor
        self._actor_optimizer.zero_grad()
        # Find loss with updated critic
        new_action = self._actor(self._as_tensor(batch["observation"])).reshape(-1, *self._env.action_space.shape)
        actor_loss = -torch.mean(self._critic(self._as_tensor(batch["observation"]), new_action))
        actor_loss.backward()
        self._actor_optimizer.step()

        # Update targets networks
        self._update_targets(self._target_actor, self._actor)
        self._update_targets(self._target_critic, self._critic)

        #self._train_global_step += 1

    def _update(self, episode_length):
        # Update model #episode_length times
        for_each(lambda x: self._update_batch(),
                 range(min(episode_length, self._config.max_updates_per_episode)))

    def evaluate(self):
        self.episode_rewards = []
        self.episode_lengths = []
        episode_actions = []
        self.temp = []
        self.soc = []
        self.volt = []
        self.curr = []

        observation = self._env.reset()
        c = self._env.get_constraint_values()
        episode_reward = 0
        episode_length = 0
        episode_action = 0

        self._eval_mode()

        for step in range(self._config.evaluation_steps):
            action = self._get_action(observation, c, is_training=False)
            current = np.clip(23. * (action - 1.), -46., 0.)
            episode_action += current
            observation, reward, done, soh = self._env.step(action)
            c = self._env.get_constraint_values()
            episode_reward += reward
            episode_length += 1

            self.temp.append(40*observation['agent_position'] + 5.)
            self.volt.append(observation['agent_voltage'])
            self.soc.append(observation['agent_soc'])
            self.curr.append(current)

            if done or (episode_length == self._config.max_episode_length):

                self.episodic_reward_buffer.append(episode_reward)
                self.episodic_length_buffer.append(episode_length)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_actions.append(episode_action / episode_length)

                observation = self._env.reset()
                c = self._env.get_constraint_values()
                episode_reward = 0
                episode_length = 0
                episode_action = 0

        mean_episode_reward = np.mean(self.episode_rewards)
        mean_episode_length = np.mean(self.episode_lengths)

        self._train_mode()

        print(f"Validation completed:\n"
              #f"Number of episodes: {len(episode_actions)}\n"
              f"Average episode length: {mean_episode_length}\n"
              f"Average reward: {mean_episode_reward}\n"
              f"Average action magnitude: {np.mean(episode_actions)}\n")
              #f"Constraint Violations: {self.count}\n"
              #f"Limit Violation at Training: {self.count_tem_train}\n"
              #f"Limit Violation at Evaluation: {self.count_tem_eval}\n")

    def train(self):
        self.soh = []
        self.flag_rv = False
        start_time = time.time()

        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        observation = self._env.reset()
        c = self._env.get_constraint_values()
        episode_reward = 0
        episode_length = 0

        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for step in range(number_of_steps):
            # Randomly sample episode_ for some initial steps
            action = self._env.action_space.sample() if step < self._config.start_steps \
                     else self._get_action(observation, c)

            observation_next, reward, done, soh = self._env.step(action)
            episode_reward += reward
            episode_length += 1

            self.soh.append(soh)

            self._replay_buffer.add({
                "observation": self._flatten_dict(observation),
                "action": action,
                "reward": np.asarray(reward) * self._config.reward_scale,
                "observation_next": self._flatten_dict(observation_next),
                "done": np.asarray(done),
            })

            observation = observation_next
            c = self._env.get_constraint_values()

            if observation['agent_position'][0] > 1:
                self.flag_rv = True

            # Make all updates at the end of the episode
            if done or (episode_length == self._config.max_episode_length):
                self.total_episode += 1
                if self.flag_rv:
                    self.count_tem_train = self.count_tem_train + 1
                    self.flag_rv = False
                np.save("curves/final_soh.npy", soh)
                self.accum_lv_train.append(self.count_tem_train)

                if step >= self._config.min_buffer_fill:
                    self._update(episode_length)
                # Reset episode
                observation = self._env.reset()
                c = self._env.get_constraint_values()
                episode_reward = 0
                episode_length = 0

            # Check if the epoch is over
            if step != 0 and step % self._config.steps_per_epoch == 0:
                print(f"Finished epoch {step / self._config.steps_per_epoch}.")
                print(f"Number of Episodes: {self.total_episode}")
                print(f"Constraint Violation during training: {self.count_tem_train}")
                print(f"Percentage of episodes in which restriction violation occurred: {100*(self.count_tem_train/self.total_episode)}")
                print(f"SOH: {soh}")
                print("------------------------------------------------------------")
                print("Running validation...")
                self.evaluate()
                print("===========================================================")
                print("|")
                print("|")
                print("|")
                print("===========================================================")

        print("==========================================================")
        print(f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print(f"Number of Episodes: {self.total_episode}")
        print(f"Constraint Violation during training: {self.count_tem_train}")
        print(f"Percentage of episodes in which restriction violation occurred: {100 * (self.count_tem_train / self.total_episode)}")
        print("==========================================================")