import time
import rospy
import copy
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped


class PUBGOAL():
    def __init__(self, index):
        self.index = index

        self.goal_x = [-10, 10, 4, 5, 6, 7, 8, 9, 10]
        self.goal_y = [0, 0, 4, 5, 6, 7, 8, 9, 10] 

        node_name = 'goal' + str(index)
        rospy.init_node(node_name, anonymous=None)

        # -----------Publisher and Subscriber-------------

        pub_goal_topic = '/robot_' + str(index) + '/move_base_simple/goal'
        self.pub_goal = rospy.Publisher(pub_goal_topic, PoseStamped, queue_size=2)

    def pub_goal_func(self):
        goal_pub = PoseStamped()
        goal_pub.header.stamp = rospy.Time(0)
        goal_pub.header.frame_id = "map"

        goal_pub.pose.position.x = self.goal_x[self.index]
        goal_pub.pose.position.y = self.goal_y[self.index]
        goal_pub.pose.orientation.z = 0.
        goal_pub.pose.orientation.x = 0.
        goal_pub.pose.orientation.y = 0.
        goal_pub.pose.orientation.z = 0.
        self.pub_goal.publish(goal_pub)
        rospy.sleep(2.)
