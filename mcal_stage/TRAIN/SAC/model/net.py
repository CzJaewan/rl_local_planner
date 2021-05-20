import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Normal

from model.utils import log_normal_density


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 1,  -1)

class QNetwork_1(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_1, self).__init__()
        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # Q1 architecture
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel, action):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        '''
        print(frame.shape)
        print(goal.shape)
        print(vel.shape)
        print(action.shape)
        '''
        xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

class QNetwork_2(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_2, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # Q2 architecture
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + num_actions, hidden_dim)
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel, action):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))

        xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x2

class ValueNetwork(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))

        state = torch.cat([o1, goal, vel], 1) # observation

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(2))

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # Different from PPO

        self.mean1_linear = nn.Linear(hidden_dim, 1) # Different from PPO
        self.mean2_linear = nn.Linear(hidden_dim, 1) # Different from PPO


        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            
            scale = [0.5, 1]
            bias = [0.5, 0]

            self.action_scale = torch.FloatTensor(scale)
            self.action_bias = torch.FloatTensor(bias)

            print("self.action_scale: ", self.action_scale)
            print("self.action_bias: ", self.action_bias)

    def forward(self, frame, goal, vel):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        state = torch.cat((o1, goal, vel), dim=-1) # observation

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        #mean1 = F.sigmoid(self.mean1_linear(x))
        #mean2 = F.tanh(self.mean2_linear(x))
        
        #mean = torch.cat((mean1, mean2), dim=-1)

        mean = self.mean_linear(x)

        #log_std = self.logstd.expand_as(mean)

        log_std = self.log_std_linear(x)
        
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, frame, goal, vel):
        mean, log_std = self.forward(frame, goal, vel)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)


        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale  + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


if __name__ == '__main__':
    from torch.autograd import Variable


