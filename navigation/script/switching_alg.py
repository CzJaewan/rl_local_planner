import time
import rospy
import rospkg 

import math
import tf
import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from skimage.transform import rotate
from std_srvs.srv import Empty
from matplotlib import pyplot as plt
from PIL import Image

import cv2

class Switching_():
    def __init__(self):
        node_name = 'switching_move_base'
        rospy.init_node(node_name, anonymous=None)

        map_topic = '/map'
        map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)


        local_map_topic = '/move_base/local_costmap/costmap'
        local_map_sub = rospy.Subscriber(local_map_topic, OccupancyGrid, self.local_map_callback)


        laser_topic = '/scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        amcl_pose_topic = 'amcl_pose'
        self.amcl_pose_sub = rospy.Subscriber(amcl_pose_topic, PoseWithCovarianceStamped, self.amcl_pose_callback)

        self.clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        self.amcl_state = None 
        self.scan = None
        self.map_np = None
        self.map_local_np = None

        rospy.spin()

    def map_callback(self, map):
        map_data = map.data
        map_array = np.array(map_data)
        #self.map_np = np.reshape(map_array, (832,384))
        self.map_np = np.reshape(map_array, (384,832))

    def local_map_callback(self, map):
        map_data = map.data
        map_array = np.array(map_data)

        self.map_local_np = np.reshape(map_array, (200,200))
        self.clear_costmap()    


    def amcl_pose_callback(self, amcl_pose):
        Quaternious = amcl_pose.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.amcl_state = [amcl_pose.pose.pose.position.x, amcl_pose.pose.pose.position.y, Euler[2]]

        self.amcl_pixel_data = [ 831 - round((self.amcl_state[0]+20.7)/0.05), round((self.amcl_state[1]+10.0)/0.05)]
        #self.amcl_pixel_data = [round((self.amcl_state[0])), round((self.amcl_state[1]))]
        
    def cart2polar(self, x, y):
        """returns r, theta(degrees)
        """
        r = (x ** 2 + y ** 2) ** .5
        theta = math.degrees(math.atan2(y,x))
        return r, theta


    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        
        self.scan = np.array(scan.ranges) # 1440
        
        local_x_min = int(self.amcl_pixel_data[1]-100)
        local_x_max = int(self.amcl_pixel_data[1]+100)

        local_y_min = int(self.amcl_pixel_data[0]-100)
        local_y_max = int(self.amcl_pixel_data[0]+100)

        theta = self.amcl_state[2]

        self.local_data = np.zeros(shape=(200,200),dtype=np.int8)
        
        for i in range(200):
            for j in range(200):
                if (local_x_min + i >= 0) and local_x_min + i <= 383:
                    if (local_y_min + j >= 0) and local_y_min + j <= 831:
                        origin_data = self.map_np[local_x_min + i, local_y_min + j]

                        if origin_data <= 0:
                            origin_data = 0
                        else:
                            origin_data = 100

                        self.local_data[i,j] = origin_data

        rot = rotate(self.local_data, theta*180/3.14)
         
        self.polar_r_data = []
        self.polar_t_data = []
        
        self.map_lidar = np.zeros(shape=(360,1), dtype=np.float32)

        for i in range(360):
            rot_angle = rotate(rot, 0.25*(i-45))
            for j in range(100):
                if rot_angle[100-j,100] > 0.1:
                    r, t = self.cart2polar(100-j, 100)
                    self.map_lidar[j] = r
                    break
                else:
                    self.map_lidar[j] = 21
        print(self.map_lidar)
        print(self.map_lidar[180])
        print(self.scan[720])
        '''
        #print(self.polar_t_data)
        mmp_data = np.zeros(shape=(200,200),dtype=np.int8)

        for i in range(200):
            for j in range(200):
                if self.map_local_np[i,199-j] <= 0:
                    mmp_data[i,j] = 0
                else:
                    mmp_data[i,j] = 100
        #print("local_min : {}, local_max : {}".format(np.min(mmp_data),np.max(mmp_data)))
        #print("map  _min : {}, map  _max : {}".format(np.min(self.local_data),np.max(self.local_data)))
        cv2.imshow("src",self.local_data)
        cv2.imshow("rot", rot)
        #cv2.imshow("diff", self.local_data - mmp_data)

        cv2.waitKey(0)
        '''
        


if __name__ == '__main__':

    Switching_()