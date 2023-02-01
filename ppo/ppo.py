import gym
import torch
import numpy as np
from tqdm import tqdm
from ppo.agent import PPOAgent
from utility import logger



class PPO:

    def __init__(
        self,
        agent: PPOAgent,
        env,
        optimizer,
        steps: int,
        rollout_steps: int,
        update_epochs: int,
        update_batch_size: int,
        eps: float
    ):
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.steps = steps
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size
        self.eps = eps
        self.device = next(self.agent.parameters()).device

        obs_shape = self.env.observation_space.shape
        self.obs_buffer_shape = (rollout_steps * obs_shape[0], *obs_shape[1:])
        self.action_buffer_shape = self.obs_buffer_shape[:2]
        self.samples_per_rollout = obs_shape[0]


    def get_agent_input(self, obs):
        return obs.coords.to(self.device)

    
    def train_step(self):
        self.agent.train()
        obs, _ = self.env.reset()

        obs_buffer = torch.zeros(self.obs_buffer_shape, device=self.device, dtype=torch.float32)
        action_buffer = torch.zeros(self.action_buffer_shape, device=self.device, dtype=torch.int64)
        sum_log_probs_buffer = torch.zeros(self.obs_buffer_shape[0:1], device=self.device, dtype=torch.float32)
        reward_buffer = torch.zeros(self.obs_buffer_shape[0:1], device=self.device, dtype=torch.float32)

        # rollout
        with torch.no_grad():
            for i in range(self.rollout_steps):
                start_idx = i * self.samples_per_rollout
                end_idx = start_idx + self.samples_per_rollout

                x = self.get_agent_input(obs)
                agent_out = self.agent(x)
                action, sum_log_probs = agent_out.tour[..., :-1], agent_out.sum_log_probs
                obs, reward, _, __, ___ = self.env.step(agent_out.tour)

                obs_buffer[start_idx:end_idx] = x
                # action_buffer[start_idx:end_idx] = action
                sum_log_probs_buffer[start_idx:end_idx] = sum_log_probs
                reward_buffer[start_idx:end_idx] = reward

        logger.info(f"Rollout terminated.\nOptimizing over {len(obs_buffer)} samples with batch size {self.update_batch_size} for {self.update_epochs} epochs...")

        # optimize
        for _ in tqdm(range(self.update_epochs)):
            idxs = np.random.permutation(np.arange(len(obs_buffer)))
            start_idxs = idxs[::self.update_batch_size]

            clip_ratio = (1 - self.eps, 1 + self.eps)
            for start_idx in start_idxs:
                batch_idxs = idxs[start_idx:start_idx + self.update_batch_size]
                x = obs_buffer[batch_idxs]
                agent_out = self.agent(x)
                _, new_sum_log_probs = agent_out.tour[..., :-1], agent_out.sum_log_probs
                
                logratio = new_sum_log_probs - sum_log_probs_buffer[batch_idxs]
                ratio = logratio.exp()
                clipped_ratio = torch.clip(ratio, *clip_ratio)
                advantage = - reward_buffer[batch_idxs]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                candidate_loss = advantage * torch.stack([ratio, clipped_ratio])
                loss = torch.max(candidate_loss, dim=0)[0].mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            