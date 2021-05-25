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
#from syscon_gazebo_test_amcl_world import StageWorld
#from syscon_gazebo_test_amcl_world_LAH import StageWorld
from syscon_gazebo_test_amcl_world_LAH_hybrid import StageWorld
from model.sac import SAC

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
parser.add_argument('--num_steps', type=int, default=1, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=200000, metavar='N',
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
    total_numsteps = 0

    # world reset
    if env.index == 0: # step
        env.reset_world()

        #Tesnorboard
        writer = SummaryWriter('test_runs/' + policy_path)
        
    for i_episode in range(args.num_steps):

        env.control_vel([0,0])
        
	
	while not rospy.is_shutdown():
            get_goal = env.is_sub_goal
            if get_goal:
                break

        #env.set_gazebo_pose(8.0, 0.0, 3.14)
        #env.set_rviz_pose(8.0, 0.0, 3.14)

        #env.send_goal_point()
        
        #env.send_goal_point2()

        env.collision_check = False

        collision_logger = 0

        episode_reward = 0
        episode_steps = 0
        done = False        
        
        # Get initial state
        frame = env.get_laser_observation()
        frame_stack = deque([frame, frame, frame])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [frame_stack, goal, speed]
        pre_time = rospy.Time.now()

        start_time = rospy.Time.now()

        # Episode start
        while not done and not rospy.is_shutdown():    
            compute_start_time = rospy.Time.now()

            state_list = comm.gather(state, root=0)
        
            if env.index == 0:

                action = agent.select_action(state_list, evaluate=True)
            else:
                action = None

            # Execute actions
            #-------------------------------------------------------------------------            
            action_clip_bound = [[0, -0.6], [0.55, 0.6]] #### Action maximum, minimum values
            cliped_action = np.clip(action, a_min=action_clip_bound[0], a_max=action_clip_bound[1])
            #cliped_action = action
            real_action = comm.scatter(cliped_action, root=0)    
            
            
            #if real_action[0] >= 0.02 and real_action[0] <= 0.2:
            #    real_action[0] = real_action[0] / 0.6
            
            '''
            if real_action[0] < 0.1 and real_action[0] > 0.0:
                real_action[0] = 0.1
            if real_action[1] > 0 and real_action[1] < 0.1:
                real_action[1] = 0.1

            elif real_action[1] < 0 and real_action[1] > -0.1:
                real_action[1] = -0.1
            '''
            env.control_vel(real_action)

            #print("{}".format(step_time.nsecs))

            ## Get reward and terminal state
            reward, done, result = env.get_reward_and_terminate(episode_steps)
            #print("Action : [{}, {}], Distance : {}, Reward : {}".format(real_action[0], real_action[1], env.distance, reward))

            if env.collision_check == False:
                logger_step.info('Env: %d, Episode: %d, steps: %d, Action : [%05.2f, %05.2f], Speed : [%05.2f, %05.2f], state : [%05.2f, %05.2f, %05.2f], traj : [%05.2f, %05.2f, %05.2f], Distance : %05.2f, Reward : %05.2f', env.index, i_episode+1, episode_steps, real_action[0], real_action[1], env.speed_GT[0], env.speed_GT[1], env.state[0], env.state[1], env.state[2], env.state_GT[0], env.state_GT[1], env.state_GT[2], env.distance, reward)

            else:
                if collision_logger == 0:
                  
                    end_time = rospy.Time.now()
                    epi_time = end_time - start_time
            
                    print("Env: {}, Goal: ({} , {}), Episode: {}, steps: {}, Reward: {}, {}".format(env.index, round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), result))
                    logger.info('Env: %d, Goal: (%05.1f , %05.1f), Episode: %d, steps: %d, Reward: %05.2f, Distance: %05.2f, Result: %s', env.index, round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), round(env.distance, 2), result)
                    logger_time.info('Env: %d, Episode: %d, time :%d.%d, start_time: %d.%d, end_time: %d.%d',  env.index, i_episode+1, epi_time.secs, epi_time.nsecs, start_time.secs, start_time.nsecs, end_time.secs, end_time.nsecs )
                    collision_logger = 1

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

            state = next_state  


        env.control_vel([0,0])
        env.is_sub_goal = False

        if env.index == 0:
            writer.add_scalar('reward/train', episode_reward, i_episode)
        
        if env.collision_check == False:

            end_time = rospy.Time.now()
            epi_time = end_time - start_time

            print("Env: {}, Goal: ({} , {}), Episode: {}, steps: {}, Reward: {}, {}".format(env.index, round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), result))
            logger.info('Env: %d, Goal: (%05.1f , %05.1f), Episode: %d, steps: %d, Reward: %05.2f, Distance: %05.2f, Result: %s', env.index, round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), round(env.distance, 2), result)
            logger_time.info('Env: %d, Episode: %d, time :%d.%d, start_time: %d.%d, end_time: %d.%d',  env.index, i_episode+1, epi_time.secs, epi_time.nsecs, start_time.secs, start_time.nsecs, end_time.secs, end_time.nsecs )


if __name__ == '__main__':

    task_name = 'SIMUL_MCAL_ADD'
    log_path = './log/' + task_name

    # config log
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_file = log_path + '/epi_output.log'
    step_output_file = log_path + '/step_output.log'
    time_output_file = log_path + '/time_output.log'
    time_compute_output_file = log_path + '/time_compute_output.log'

    # config log
    logger = logging.getLogger('epi_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger.info('start_logging')

    logger_step = logging.getLogger('step_logger')
    logger_step.setLevel(logging.INFO)
    file_handler = logging.FileHandler(step_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_step.addHandler(file_handler)
    logger_step.addHandler(stdout_handler)

    logger_step.info('start_logging')

    logger_time = logging.getLogger('time_logger')
    logger_time.setLevel(logging.INFO)
    file_handler = logging.FileHandler(time_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_time.addHandler(file_handler)
    logger_time.addHandler(stdout_handler)

    logger_time.info('start_logging')

    logger_compute_time = logging.getLogger('time_logger')
    logger_compute_time.setLevel(logging.INFO)
    file_handler = logging.FileHandler(time_compute_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_compute_time.addHandler(file_handler)
    logger_compute_time.addHandler(stdout_handler)

    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)
    print("Ready to environment")
    
    reward = None
    if rank == 0:
        policy_path = 'policy_test'
        #board_path = 'runs/r2_epi_0'
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = [[0, 1], [-1, 1]] #### Action maximum, minimum values
        action_bound = spaces.Box(-1, +1, (2,), dtype=np.float32)
        agent = SAC(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        #'/w5a10_policy_epi_2000.pth'

        # static obstacle best policy
        file_policy = policy_path + '/syscon_6world_policy_epi_12900.pth' 
        #file_policy = policy_path + '/1028_policy_epi_2200.pth' 
        #file_policy = policy_path + '/stage_1027_policy_epi_1500.pth' 
        #file_policy = policy_path + '/20201022_policy_epi_4500.pth' 
        #file_policy = policy_path + '/1023_lsq_policy_epi_200.pth' 

        file_critic_1 = policy_path + '/syscon_6world_critic_1_epi_12900.pth'
        file_critic_2 = policy_path + '/syscon_6world_critic_2_epi_12900.pth'



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
