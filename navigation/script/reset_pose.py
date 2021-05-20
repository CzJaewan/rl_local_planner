import time
import rospy
import rospkg 
import copy

import math
import tf
import numpy as np
import argparse

from std_msgs.msg import Duration
from scipy.interpolate import interp1d

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

parser = argparse.ArgumentParser(description='set_pose')
parser.add_argument('--num', type=int, default=0, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--cho', type=int, default=0, metavar='N',
                    help='batch size (default: 256)')
args = parser.parse_args()

class reset_pose():
    def __init__(self, index):

        self.index = index

        print(index)
        
        node_name = 'reset_pose'
        rospy.init_node(node_name, anonymous=None)

        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.Goal_data = [[-8.0, 0.0, 0.00], [8.0, 0.0, 3.14]]
        self.Start_data = [[8.0, 0.0, 3.14], [-8.0, 0.0, 0.00]]

        # -----------rviz----------------------
        
        ininit_amcl_topic = 'robot_' + str(index) + 'initialpose'
        self.pub_amcl_init = rospy.Publisher(ininit_amcl_topic, PoseWithCovarianceStamped, queue_size=10)
        
        self.robot_name = 'robot_' + str(index)
     
    def set_gazebo_pose(self):
        
        x = self.Start_data[self.index][0]
        y = self.Start_data[self.index][1]
        w = self.Start_data[self.index][2]

        print("set_pose")

        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, w, 'rxyz')

        state_msg.pose.orientation.x = qtn[0]
        state_msg.pose.orientation.y = qtn[1]
        state_msg.pose.orientation.z = qtn[2]
        state_msg.pose.orientation.w = qtn[3]

        self.init_pose = [x, y, w]
        self.first_pose = 1

        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            resp = self.set_state( state_msg )

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        rospy.sleep(1.)


    def set_rviz_pose(self):
        
        x = self.Start_data[self.index][0]
        y = self.Start_data[self.index][1]
        w = self.Start_data[self.index][2]

        rate = rospy.Rate(1) # 10hz

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "/map"
        pose.pose.pose.position.x= x
        pose.pose.pose.position.y= y
        pose.pose.pose.position.z=0
        pose.pose.covariance=[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        
        qtn = tf.transformations.quaternion_from_euler(0, 0, w, 'rxyz')

        pose.pose.pose.orientation.x = qtn[0]
        pose.pose.pose.orientation.y = qtn[1]
        pose.pose.pose.orientation.z = qtn[2]
        pose.pose.pose.orientation.w = qtn[3]
        
        self.pub_amcl_init.publish(pose)
        rate.sleep()
        self.pub_amcl_init.publish(pose)
        rate.sleep()
        self.pub_amcl_init.publish(pose)
        rate.sleep()


if __name__ == '__main__':
    

    env = reset_pose(args.num)

    env.set_gazebo_pose()

