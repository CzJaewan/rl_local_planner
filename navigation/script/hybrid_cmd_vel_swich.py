import time
import rospy
import rospkg 
import copy

import math
import tf
import numpy as np

from std_msgs.msg import Duration
from scipy.interpolate import interp1d

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
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)

        laser_topic = '/scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        rl_cmd_vel_topic = '/rl_cmd_vel'
        self.rl_cmd_vel = rospy.Subscriber(rl_cmd_vel_topic, Twist, self.rl_vel_callback)

        dwa_cmd_vel_topic = '/dwa_cmd_vel'
        self.dwa_cmd_vel = rospy.Subscriber(dwa_cmd_vel_topic, Twist, self.dwa_vel_callback)
        
        cmd_vel_topic = '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        self._tf_listener = tf.TransformListener()

        self.amcl_state = None 
        self.scan = None
        self.map_np = None
        self.map_local_np = None

        self.switching_flag = 0
        plt.ion()

        self.timer = rospy.Timer(rospy.Duration(1), self.control_loop_pro)

        rospy.spin()

    def map_callback(self, map):
        map_data = map.data
        map_array = np.array(map_data)
        #self.map_np = np.reshape(map_array, (832,384))
        self.map_np = np.reshape(map_array, (384,832))


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

    def rl_vel_callback(self, rl_vel):
        
        if self.switching_flag == 1:
        
            rl_cmd_vel = Twist()
            rl_cmd_vel.linear.x = rl_vel.linear.x
            rl_cmd_vel.linear.y = 0.
            rl_cmd_vel.linear.y = 0.
            rl_cmd_vel.angular.x = 0.
            rl_cmd_vel.angular.y = 0.
            rl_cmd_vel.angular.z = rl_vel.angular.z
            
            self.cmd_vel.publish(rl_cmd_vel)
        
    def dwa_vel_callback(self, dwa_vel):

        if self.switching_flag == 0:
            
            dwa_cmd_vel = Twist()
            dwa_cmd_vel.linear.x = dwa_vel.linear.x
            dwa_cmd_vel.linear.y = 0.
            dwa_cmd_vel.linear.y = 0.
            dwa_cmd_vel.angular.x = 0.
            dwa_cmd_vel.angular.y = 0.
            dwa_cmd_vel.angular.z = dwa_vel.angular.z
            
            self.cmd_vel.publish(dwa_cmd_vel)

    def control_loop_pro(self, timer):

        max_scan_dist = 10
        virtscan_angle_limit = np.pi/2
        loss_threshold = 3
        switching_threshold = 0
        
        (pos, rot) = self._tf_listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
        
        Euler2 = tf.transformations.euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])

        self.amcl_state = [pos[0], pos[1], Euler2[2]]
        self.amcl_pixel_data = [round((self.amcl_state[0]+21.2)/0.05), round((self.amcl_state[1]+10.0)/0.05)]

        theta = self.amcl_state[2]
        #print(theta * 180/np.pi)
        angle = np.linspace(-180 + 0.25 , 180  , 1440) * np.pi/180

        lidar_pts_x = self.scan * np.cos(angle+theta)+self.amcl_state[0]
        lidar_pts_y = self.scan * np.sin(angle+theta)+self.amcl_state[1]

        lidar_pixel_x_float = (lidar_pts_x+21.2)/0.05
        lidar_pixel_y_float = (lidar_pts_y+10.0)/0.05

        lidar_pixel_x = lidar_pixel_x_float.astype('int32')
        lidar_pixel_y = lidar_pixel_y_float.astype('int32')
        
        #plt.contourf(self.map_np*255,0)
        #plt.plot(self.amcl_pixel_data[0], self.amcl_pixel_data[1],'o')
        #plt.plot(lidar_pixel_x, lidar_pixel_y)
        #plt.show()

        local_v_min = max((int(self.amcl_pixel_data[0])-200), 1)
        local_v_max = min((int(self.amcl_pixel_data[0])+200), self.map_np.shape[1])
        print("local_v_min : {}, max : {}".format(local_v_min, local_v_max))
      
        local_u_min = max((int(self.amcl_pixel_data[1])-200), 1)
        local_u_max = min((int(self.amcl_pixel_data[1])+200), self.map_np.shape[0])
        print("local_u_min : {}, max : {}".format(local_u_min, local_u_max))
        
        local_data = self.map_np[local_u_min:local_u_max, local_v_min:local_v_max]
        print("local_data shape : {}".format(local_data.shape))

        new_amcl_pixel_data = [None, None]
        print("amcl_pixel_data : [{}, {}]".format(self.amcl_pixel_data[0], self.amcl_pixel_data[1]))

        new_amcl_pixel_data[0] = min(200, int(self.amcl_pixel_data[0]))
        new_amcl_pixel_data[1] = min(200, int(self.amcl_pixel_data[1]))

        print("new_amcl_pixel_data : [{}, {}]".format(new_amcl_pixel_data[0], new_amcl_pixel_data[1]))
        
        [ir, ic] = np.where(local_data > 0)

        dis_from_robot = np.sqrt((ic-new_amcl_pixel_data[0])*(ic-new_amcl_pixel_data[0]) + (ir-new_amcl_pixel_data[1])*(ir-new_amcl_pixel_data[1]))
        
        lamda = np.arctan2((ir-new_amcl_pixel_data[1]),(ic-new_amcl_pixel_data[0])) - theta
        xwrap = np.remainder(lamda, 2*np.pi)
        mask = np.abs(xwrap)>np.pi
        xwrap[mask] -= 2*np.pi*np.sign(xwrap[mask])
        ang_from_robot = xwrap

        print("shape ir : {}, ic : {} ".format(ir.shape,ic.shape))
        print("shape dist : {}, angle : {} ".format(dis_from_robot.shape,ang_from_robot.shape))
        
        static_obs_pixel = np.vstack([ic,ir,dis_from_robot,ang_from_robot])

        print("shape static_obs_pixel : {} ".format(static_obs_pixel.shape))
        print(static_obs_pixel[:,0])

        [idxs1] =np.where(ang_from_robot < -virtscan_angle_limit*1.1) # or virtscan_angle_limit*1.1<ang_from_robot)
        [idxs2] =np.where(ang_from_robot > virtscan_angle_limit*1.1)
        idxs = np.hstack([idxs1, idxs2])

        print("shape idxs1 : {}, idxs2 : {} ".format(idxs1.shape,idxs2.shape))
        print("shape idxs : {} ".format(idxs.shape))

        static_obs_pixel[:, idxs] = None
        static_obs_pixel_cut = np.delete(static_obs_pixel, idxs, axis=1)

        static_obs_pixel_t = static_obs_pixel_cut.T

        static_obs_pixel_sort = static_obs_pixel_t[np.lexsort((static_obs_pixel_t[:,2],static_obs_pixel_t[:,3]))]
        [_, uidx] = np.unique(static_obs_pixel_sort[:,3],return_index=True)
        
        print("shape static_obs_pixel_sort : {}, uidx : {} ".format(static_obs_pixel_sort.shape,uidx.shape))

        idx_all = np.linspace(0, static_obs_pixel_sort[:,3].shape[0]-1, static_obs_pixel_sort[:,3].shape[0])

        uidx_del = np.delete(idx_all, uidx.T, axis=0) 

        static_obs_pixel_uidx = np.delete(static_obs_pixel_sort, uidx_del.T, axis=0)

        print("shape static_obs_pixel_sort : {}, static_obs_pixel_uidx : {} ".format(static_obs_pixel_sort.shape,static_obs_pixel_uidx.shape))

        vlidar_angle = np.linspace(-virtscan_angle_limit, virtscan_angle_limit, 721)

        set_interp = interp1d(static_obs_pixel_uidx[:,3], static_obs_pixel_uidx[:,2]*0.05, fill_value="extrapolate")
        vlidar_data = set_interp(vlidar_angle)
        [idx] = np.where(angle==-virtscan_angle_limit)
        
        vlidar_data[np.isnan(vlidar_data)] = max_scan_dist
        vlidar_data[np.isinf(vlidar_data)] = max_scan_dist
        vlidar_data[vlidar_data > max_scan_dist] = max_scan_dist

        croplidar_data = self.scan[idx[0]:idx[0]+vlidar_data.shape[0]]

        croplidar_data[np.isnan(croplidar_data)] = max_scan_dist
        croplidar_data[np.isinf(croplidar_data)] = max_scan_dist
        croplidar_data[croplidar_data > max_scan_dist] = max_scan_dist

        
        print("shape vlidar_angle : {} ".format(vlidar_angle.shape))
        print("shape vlidar_data : {}, croplidar_data : {} ".format(vlidar_data.shape,croplidar_data.shape))

        lidar_rmse = np.sqrt(((croplidar_data - vlidar_data)**2).mean())
        
        print("lidar_rmse : {}".format(lidar_rmse))

        loss_data =  vlidar_data - croplidar_data

        np.savetxt('reallidar.txt', croplidar_data, header='--xy save start--', footer='--xy save end--', fmt='%1.5f')
        np.savetxt('vlidar.txt', vlidar_data, header='--xy save start--', footer='--xy save end--', fmt='%1.5f')
        np.savetxt('loss_data.txt', loss_data, header='--xy save start--', footer='--xy save end--', fmt='%1.5f')

        loss_data[loss_data < loss_threshold] = 0

        np.savetxt('threshold_loss_data.txt', loss_data, header='--xy save start--', footer='--xy save end--', fmt='%1.5f')

        loss = loss_data.mean()

        if loss > switching_threshold:
            self.switching_flag = 1
            print("loss : {}, algorithm : rl".format(loss))
        else:
            self.switching_flag = 0
            print("loss : {}, algorithm : dwa".format(loss))

        '''

        plt.clf()

        plt.subplot(2,1,1)
        plt.plot(vlidar_angle, loss_data, 'r')
        plt.axis([-virtscan_angle_limit, virtscan_angle_limit, -1, 10])
        
        plt.subplot(2,1,2)
        plt.plot(vlidar_angle, vlidar_data, 'r')
        plt.plot(vlidar_angle, croplidar_data, 'b')
        plt.axis([-virtscan_angle_limit, virtscan_angle_limit, 0, 12])
        
        plt.show()
        cv2.waitKey(0)
        plt.pause(0.01)
        
        '''
        
if __name__ == '__main__':
    Switching_()
