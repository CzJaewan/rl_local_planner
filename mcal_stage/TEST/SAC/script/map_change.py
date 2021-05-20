import time
import rospy
import copy
import tf
import numpy as np
import random

from geometry_msgs.msg import Twist, Pose, Point32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int32


class MapChange():
    def __init__(self):

        node_name = 'map_change_node'
        rospy.init_node(node_name, anonymous=None)

        map_sub_topic = '/env_change' 
        self.map_num_sub = rospy.Subscriber(map_sub_topic, Int32, self.map_env_callback)

        map_pub_topic = '/env_num' 
        self.map_num_pub = rospy.Publisher(map_pub_topic, Int32, queue_size=10)
	
        self.map_num_index = 0

    def map_env_callback(self, num):

	    self.map_num_index = num.data
	
    def pub_map_num(self):
        map_data = Int32()
        map_data.data = self.map_num_index
        self.map_num_pub.publish(map_data)



if __name__ == '__main__':

    map_change = MapChange()

    while not rospy.is_shutdown():

        map_change.pub_map_num()
        print(map_change.map_num_index)
        rospy.sleep(1)
