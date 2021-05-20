import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from net import GaussianPolicy, QNetwork_1, QNetwork_2, DeterministicPolicy
from utils import soft_update, hard_update
from torch.optim import Adam

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/sac.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

def generate_action(env, state_list, policy, action_bound):
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()



        a, logprob, mean = policy(s_list, goal_list, speed_list)
        a, logprob = a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])

    else:
        print("env.index == 0")
        a = None
        scaled_action = None
        logprob = None

    return a, logprob, scaled_action

def sac_update_stage(policy, optimizer, critic, critic_opt, critic_target, batch_size, memory, epoch,
               replay_size, tau, alpha, gamma, updates, update_interval,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    
    # Sample a batch of transitions from replay buffer:
    obss, goals,speeds, actions, logprobs, rewards, n_obss,n_goals, n_speeds, masks = memory.sample(batch_size)
    
    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    rewards = rewards.reshape((num_step*num_env, 1))
    n_obss = n_obss.reshape((num_step*num_env, frames, obs_size))
    n_goals = n_goals.reshape((num_step*num_env, 2))
    n_speeds = n_speeds.reshape((num_step*num_env, 2))
    masks = masks.reshape((num_step*num_env, 1))
    
    for update in range(epoch):

        sampled_obs = Variable(torch.from_numpy(obss)).float().cuda()
        sampled_goals = Variable(torch.from_numpy(goals)).float().cuda()
        sampled_speeds = Variable(torch.from_numpy(speeds)).float().cuda()

        sampled_n_obs = Variable(torch.from_numpy(n_obss)).float().cuda()
        sampled_n_goals = Variable(torch.from_numpy(n_goals)).float().cuda()
        sampled_n_speeds = Variable(torch.from_numpy(n_speeds)).float().cuda()

        sampled_actions = Variable(torch.from_numpy(actions)).float().cuda()
        sampled_logprobs = Variable(torch.from_numpy(logprobs)).float().cuda()

        sampled_rewards = Variable(torch.from_numpy(rewards)).float().cuda()
        sampled_masks = Variable(torch.from_numpy(masks)).float().cuda()
       

        with torch.no_grad():
            n_actions, n_logprobs, _ = policy(sampled_n_obs, sampled_n_goals, sampled_n_speeds)
            qf1_n_target, qf2_n_target = critic_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds, n_actions)
            min_qf_n_target = torch.min(qf1_n_target, qf2_n_target) - alpha * n_logprobs
            n_q_value = sampled_rewards + (1 - sampled_masks) * gamma * (min_qf_n_target)

        qf1, qf2 = critic(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

        qf1_loss = F.mse_loss(qf1, n_q_value)
        qf2_loss = F.mse_loss(qf2, n_q_value)
        
        qf_loss = qf1_loss + qf2_loss

        critic_opt.zero_grad()
        qf_loss.backward()
        critic_opt.step()
        
        act, log_pi, _ = policy(sampled_obs, sampled_goals, sampled_speeds)

        qf1_pi, qf2_pi= critic(sampled_obs, sampled_goals, sampled_speeds, act)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        alpha_loss = torch.tensor(0.).cuda()
        alpha_tlogs = torch.tensor(alpha) # For TensorboardX logs

        if updates % update_interval == 0:
            soft_update(critic_1_target, critic_1, tau)
            soft_update(critic_2_target, critic_2, tau)

        updates = updates + 1

    return updates


### SAC

class SAC(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.device = torch.device("cuda")

        self.action_space_array = np.array(action_space)
        self.action_space = action_space
        self.critic_1 = QNetwork_1(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            #self.policy = GaussianPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space_array.shape[0], args.hidden_size, self.action_space_array).to(self.device)
            self.policy = GaussianPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):
        frame_list, goal_list, vel_list = [], [], []
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])

        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)

        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()

        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list)

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list)
        
        #print(action)
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch = memory.sample(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        '''
        print(frame_batch.shape)
        print(goal_batch.shape)
        print(speed_batch.shape)
        print(action_batch.shape)
        '''

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch)
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch)

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi)
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi)
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()



