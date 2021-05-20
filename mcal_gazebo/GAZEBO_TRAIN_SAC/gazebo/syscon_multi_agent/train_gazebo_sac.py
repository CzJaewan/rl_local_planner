import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
import argparse
from mpi4py import MPI

from gym import spaces

from torch.optim import Adam
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from model.utils import soft_update, hard_update

from model.net import QNetwork_1, QNetwork_2, ValueNetwork, GaussianPolicy, DeterministicPolicy
from syscon_gazebo_train_amcl_world import StageWorld
from model.sac import SAC
from model.replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stage",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust \alpha (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=50000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=500000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--laser_beam', type=int, default=512,
                    help='the number of Lidar scan [observation] (default: 512)')
parser.add_argument('--num_env', type=int, default=1,
                    help='the number of environment (default: 1)')
parser.add_argument('--laser_hist', type=int, default=3,
                    help='the number of laser history (default: 3)')
parser.add_argument('--act_size', type=int, default=2,
                    help='Action size (default: 2, translation, rotation velocity)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')                    
args = parser.parse_args()

def run(comm, env, agent, policy_path, args):

    # Training Loop
    test_interval = 10
    save_interval = 100

    total_numsteps = 0
    updates = 0

    # world reset
    if env.index == 0: # step
        env.reset_gazebo_simulation()
        #Tesnorboard
        writer = SummaryWriter('test_runs/' + policy_path)
        

    # replay_memory     
    memory = ReplayMemory(args.replay_size, args.seed)

    

    for i_episode in range(args.num_steps):

        episode_reward = 0
        episode_steps = 0
        done = False 

        # Env reset
        env.set_gazebo_pose_4agent()
        # generate goal
        env.generate_goal_point_gazebo_4agent()    
        
        # Get initial state
        frame = env.get_laser_observation()
        frame_stack = deque([frame, frame, frame])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [frame_stack, goal, speed]

        print("set_state")

        # Episode start
        while not done and not rospy.is_shutdown():    
            state_list = comm.gather(state, root=0)

            if env.index == 0:
                action = agent.select_action(state_list)
                print("select_action")

            else:
                action = None
            
            if env.index == 0:
                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1
                 
            # Execute actions
            #-------------------------------------------------------------------------            
            action_clip_bound = [[0, -1.0], [1.0, 1.0]] #### Action maximum, minimum values
            cliped_action = np.clip(action, a_min=action_clip_bound[0], a_max=action_clip_bound[1])
            real_action = comm.scatter(cliped_action, root=0)    
            
            #if real_action[0] >= 0.02 and real_action[0] <= 0.2:
            #    real_action[0] = real_action[0] / 0.6
            '''
            if real_action[0] < 0.15 and real_action[0] > 0:
                real_action[0] = 0.15
                
            '''
            if real_action[1] > 0 and real_action[1] < 0.10:
                real_action[1] = 0.0

            elif real_action[1] < 0 and real_action[1] > -0.10:
                real_action[1] = 0.0
            

            env.control_vel(real_action)

            #rospy.sleep(0.001)

            ## Get reward and terminal state
            reward, done, result = env.get_reward_and_terminate(episode_steps)
            #print("Action : [{}, {}], Distance : {}, Reward : {}".format(real_action[0], real_action[1], env.distance, reward))

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Get next state
            next_frame = env.get_laser_observation()
            left = frame_stack.popleft()
            
            frame_stack.append(next_frame)
            next_goal = np.asarray(env.get_local_goal())
            next_speed = np.asarray(env.get_self_speed())
            next_state = [frame_stack, next_goal, next_speed]

            r_list = comm.gather(reward, root=0)
            done_list = comm.gather(done, root=0)
            next_state_list = comm.gather(next_state, root=0)

            if env.index == 0:
                #meomry.list_push(state_list, action, r_list, next_state_list, done_list)
                for i in range(np.asarray(state_list).shape[0]):
                    memory.push(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i], next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i]) # Append transition to memory

            state = next_state  
        

        #if total_numsteps > args.num_steps:
        #    break
        
        if env.index == 0:
            writer.add_scalar('reward/train', episode_reward, i_episode)


        if env.index == 0:
            #if global_update != 0 and global_update % 20 == 0:
            if i_episode != 0 and i_episode % save_interval == 0:
    
                torch.save(agent.policy.state_dict(), policy_path + '/policy_epi_{}'.format(i_episode))
                print('########################## policy model saved when update {} times#########'
                            '################'.format(i_episode))
                torch.save(agent.critic_1.state_dict(), policy_path + '/critic_1_epi_{}'.format(i_episode))
                print('########################## critic model saved when update {} times#########'
                            '################'.format(i_episode))
                torch.save(agent.critic_2.state_dict(), policy_path + '/critic_2_epi_{}'.format(i_episode))
                print('########################## critic model saved when update {} times#########'
                            '################'.format(i_episode))

        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        print("Env: {}, memory_size: {}, Goal: ({} , {}), Episode: {}, steps: {}, Reward: {}, Distance: {}, {}".format(env.index, len(memory), round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), round(distance, 2), result))


if __name__ == '__main__':
    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)
    print("Ready to environment")
    
    env.reset_gazebo_simulation()

    reward = None
    if rank == 0:
        policy_path = 'policy_stage_train_1104'
        #board_path = 'runs/r2_epi_0'
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = [[0, 1], [-1, 1]] #### Action maximum, minimum values
        action_bound = spaces.Box(-1, +1, (2,), dtype=np.float32)
        agent = SAC(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        #'/w5a10_policy_epi_2000.pth'

        #file_policy = policy_path + '/slow_action_syscon_policy_epi_1500.pth' 
        #file_critic_1 = policy_path + '/slow_action_syscon_critic_1_epi_1500.pth'
        #file_critic_2 = policy_path + '/slow_action_syscon_critic_2_epi_1500.pth'

        #file_policy = policy_path + '/1023_gazebo_policy_epi_400.pth' 
        #file_critic_1 = policy_path + '/1023_gazebo_critic_1_epi_400.pth'
        #file_critic_2 = policy_path + '/1023_gazebo_critic_2_epi_400.pth'

        #file_policy = policy_path + '/syscon_6world_policy_epi_12900.pth' 
        #file_critic_1 = policy_path + '/syscon_6world_critic_1_epi_12900.pth'
        #file_critic_2 = policy_path + '/syscon_6world_critic_2_epi_12900.pth'

        #file_policy = policy_path + '/20201025_multi_policy_epi_500.pth' 
        #file_critic_1 = policy_path + '/20201025_multi_critic_1_epi_500.pth'
        #file_critic_2 = policy_path + '/20201025_multi_critic_2_epi_500.pth'


        file_policy = policy_path + '/stage_1027_policy_epi_1500.pth' 
        file_critic_1 = policy_path + '/stage_1027_critic_1_epi_1500.pth'
        file_critic_2 = policy_path + '/stage_1027_critic_2_epi_1500.pth'


        if os.path.exists(file_policy):
            print('###########################################')
            print('############Loading Policy Model###########')
            print('###########################################')
            state_dict = torch.load(file_policy)
            agent.policy.load_state_dict(state_dict)
        else:
            print('###########################################')
            print('############Start policy Training###########')
            print('###########################################')

        if os.path.exists(file_critic_1):
            print('###########################################')
            print('############Loading critic_1 Model###########')
            print('###########################################')
            state_dict = torch.load(file_critic_1)
            agent.critic_1.load_state_dict(state_dict)
            hard_update(agent.critic_1_target, agent.critic_1)
        else:
            print('###########################################')
            print('############Start critic_1 Training###########')
            print('###########################################')
    
        if os.path.exists(file_critic_2):
            print('###########################################')
            print('############Loading critic_2 Model###########')
            print('###########################################')
            state_dict = torch.load(file_critic_2)
            agent.critic_2.load_state_dict(state_dict)
            hard_update(agent.critic_2_target, agent.critic_2)
        else:
            print('###########################################')
            print('############Start critic_2 Training###########')
            print('###########################################')    

    else:
        agent = None
        policy_path = None
        
    try:
        run(comm=comm, env=env, agent=agent, policy_path=policy_path, args=args)
    except KeyboardInterrupt:
        pass
