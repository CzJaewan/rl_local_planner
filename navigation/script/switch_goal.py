import time
import rospy
import rospkg 

import math

import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from mpi4py import MPI

class Switch_Goal():
    def __init__(self, index):

        self.index = index
    
        node_name = 'switch_Env_' + str(index)
        rospy.init_node(node_name, anonymous=None)

        pub_goal_topic = '/R_007/move_base_simple/goal'
        self.goal_pub = rospy.Publisher(pub_goal_topic, PoseStamped, queue_size=10)

        self.Goal_data = [[15, 2], [4.5,-4.5]]
        self.goal_count = 0

        switch_topic = '/R_007/switch_mode'
        self.switch_sub = rospy.Subscriber(switch_topic, Int32, self.switch_mode_callback)

        wait_topic = '/R_007/wait_mode'
        self.wait_sub = rospy.Subscriber(wait_topic, Int32, self.wait_mode_callback)

        self.goal_pub_flag = 1 
	self.epi_count = 0

    def switch_mode_callback(self, switch_data):

        switch = switch_data.data
	
        if switch == 1:

            self.goal_pub_flag = 1
            
            if self.goal_count == 1:
                self.goal_count = 0
            else:
                self.goal_count = 1
            self.epi_count = self.epi_count + 1

        elif switch == 2:
            
            self.goal_pub_flag = 0

            if self.goal_count == 1:
                self.goal_count = 0
            else:
                self.goal_count = 1
            self.epi_count = self.epi_count + 1
        
        else :
            
            self.goal_pub_flag = 0
        

    def send_goal_point(self):

        if self.goal_pub_flag == 1:
            print("goal : [{}, {}]".format(self.Goal_data[self.goal_count][0],self.Goal_data[self.goal_count][1]))
            goal_data = PoseStamped()

            goal_data.header.frame_id = "map"
            goal_data.pose.position.x = self.Goal_data[self.goal_count][0]
            goal_data.pose.position.y = self.Goal_data[self.goal_count][1]
            goal_data.pose.position.z = 0
            
            goal_data.pose.orientation.x = 0
            goal_data.pose.orientation.y = 0
            goal_data.pose.orientation.z = 0
            goal_data.pose.orientation.w = 1.0
            
            self.goal_pub.publish(goal_data)

            self.goal_pub_flag = 0

    def wait_mode_callback(self, msgs):
        if msgs.data == 1:
            self.goal_pub_flag = 1



if __name__ == '__main__':

    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = Switch_Goal(index=rank)

    while not rospy.is_shutdown():
            
        env.send_goal_point()
        rospy.sleep(1.0)

        if env.epi_count > 11:
            break
