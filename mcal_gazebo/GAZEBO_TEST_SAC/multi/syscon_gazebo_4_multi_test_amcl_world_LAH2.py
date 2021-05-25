import time
import rospy
import rospkg 

import math

import copy
import tf
import numpy as np

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from model.utils import test_init_pose, test_goal_point
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from model.respawnGoal import Respawn


class StageWorld():
    def __init__(self, beam_num, index, num_env):
        
        robot_num = 3

        self.index = index
        self.num_env = num_env

        self.robot_radius = [0.2]

        node_name = 'Gazebo_Env_' + str(index + robot_num)
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.

        pub_goal_topic = 'robot_' + str(index + robot_num) + '/move_base_simple/goal'
        self.goal_pub = rospy.Publisher(pub_goal_topic, PoseStamped, queue_size=10)

        #self.Goal_data = [[9, 4.5], [9.6, -2.2], [2.5, 0.6], [2.0, 4.3], [9.7, 7.3], [8.1, 12]]
        self.Goal_data = [[-3.0, -3.0, 0.749466], [3.0, -3.0, -0.893008],[-3.0, 3.0, 2.35619], [3.0, 3.0, 3.92] ]
        
        self.robot_name = 'robot_' + str(index + robot_num)

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index + robot_num) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        
        object_state_topic = 'robot_' + str(index + robot_num) + '/odom'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)
        
        amcl_pose_topic = 'robot_' + str(index + robot_num) + '/amcl_pose'
        self.amcl_pose_sub = rospy.Subscriber(amcl_pose_topic, PoseWithCovarianceStamped, self.amcl_pose_callback)

        laser_topic = 'robot_' + str(index + robot_num) + '/scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index + robot_num) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)
        
        # -----------Gazebo data----------------------------
        goal_topic = 'robot_' + str(index + robot_num) + '/move_base_simple/goal'
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)
        
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # -----------rviz----------------------
        
        ininit_amcl_topic = 'robot_' + str(index + robot_num) + 'initialpose'
        self.pub_amcl_init = rospy.Publisher(ininit_amcl_topic, PoseWithCovarianceStamped, queue_size=10)

        # -----------Service-------------------
        #self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)
        self.reset_gazebo = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.is_sub_goal = False
        self.collision_check = False
	# -----------Look A head --------------

        lah_topic = 'robot_' + str(index + robot_num) + '/lookAhead_point'
        self.lah_sub = rospy.Subscriber(lah_topic, PoseStamped, self.lah_callback)

  
        self.tf_listener = tf.TransformListener()

        self.PathMsg = Path()

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None

        self.is_crashed = None

        self.global_goal_point = [0, 0]
        self.goal_point = [0, 0]

        self.pre_distance = 0
        self.distance = 0

        self.is_collision = 0
        self.lidar_danger = 0.5
        self.scan_min = 6.0

        self.lah_flag = 0

        self.way_Path = []
        self.waypoint_gen_flag = 0
        self.way_goal_index = 0
        
        self.wait_mode = True

        #self.goal_model = Respawn(self.goal_point[0], self.goal_point[1], 'goal')
        #self.waypoint_model = Respawn(0, 0, 'waypoint')

        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            print(self.scan)
            print(self.speed)
            print(self.state)
            print(self.speed_GT)
            print(self.state_GT)
            
            pass

        rospy.sleep(1.)
  
    def wait_mode_callback(self, msgs):
        if msgs.data == 1:
            self.wait_mode = False
                    
    def send_goal_point(self):
        goal_data = PoseStamped()

        goal_data.header.frame_id = "map"
        goal_data.pose.position.x = self.Goal_data[3][0]
        goal_data.pose.position.y = self.Goal_data[3][1]
        goal_data.pose.position.z = 0
        
        goal_data.pose.orientation.x = 0
        goal_data.pose.orientation.y = 0
        goal_data.pose.orientation.z = 0
        goal_data.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_data)



    def set_gazebo_pose(self, x, y, w):
        
        self.control_vel([0, 0])

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


    def set_rviz_pose(self, x, y, w):
    
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
        

    def reset_gazebo_simulation(self):
        self.reset_gazebo()
    
    def lah_callback(self, Pose_msg):

        if not self.state_GT is None:
            self.goal_point[0] = Pose_msg.pose.position.x
            self.goal_point[1] = Pose_msg.pose.position.y

            #print("get sub goal : [ {}, {} ]". format(self.goal_point[0], self.goal_point[1]))

            #if self.global_goal_point[0] == self.goal_point[0] and self.global_goal_point[1] == self.goal_point[1]:
            #    self.lah_flag = 1


    def goal_callback(self, Goal):

        #self.goal_model.deleteModel()
        self.lah_flag = 0
        self.global_goal_point[0] = Goal.pose.position.x
        self.global_goal_point[1] = Goal.pose.position.y
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)

        #self.goal_model.goal_position.position.x = Goal.pose.position.x
        #self.goal_model.goal_position.position.y = Goal.pose.position.y
    
        #self.goal_model.respawnModel()

        self.is_sub_goal = True

    def get_goal(self):
        return self.is_sub_goal

    def ground_truth_callback(self, GT_odometry):
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def amcl_pose_callback(self, amcl_pose):
        Quaternious = amcl_pose.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [amcl_pose.pose.pose.position.x, amcl_pose.pose.pose.position.y, Euler[2]]


    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        
        full_scan = np.array(scan.ranges) # 1440
        #print(full_scan.shape)
        count = 0
        second_scan = full_scan[208:1232]
        scan_re = []

        for i in range(512):
            num = i * 2
            min_scan = min(second_scan[num], second_scan[num+1])
            scan_re.append(min_scan)

        self.scan_origin = np.array(scan_re) 

        pre_scan = 10

        for i in range(512):
            if self.scan_origin[i] == 21.0:
                self.scan_origin[i] = pre_scan
            else :
                pre_scan = self.scan_origin[i]

        for i in range(512):
            if self.scan_origin[i] > 10:
                self.scan_origin[i] = 10
            
        self.scan = self.scan_origin

        self.laser_cb_num += 1
        self.collision_laser_flag(self.robot_radius[0])


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]
        self.state_GT = self.state

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def collision_laser_flag(self, r):
        scan = copy.deepcopy(self.scan_origin)
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0

        scan_min = np.min(scan)

        if scan_min < 0.5:
            self.is_collision = 1
        else:
            self.is_collision = 0

        self.scan_min = scan_min

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 10.0 - 0.5

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        #self.reset_stage
        #self.reset_gazebo()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        
        #rospy.sleep(1.)

    def generate_goal_point(self):
        self.goal_point = test_goal_point(self.index)
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)

    def generate_test_goal_point(self):
        self.goal_point = [-3.00, -3.00]
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)

    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()

        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()

        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.global_goal_point[0] - x) ** 2 + (self.global_goal_point[1] - y) ** 2)

        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        reward_ct = 0
        reward_t = 0
        result = 0

        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 20
            result = 'Reach Goal'
        
        if self.is_collision == 1:
            terminate = True
            reward_c = -20.
            result = 'Crashed'
            self.collision_check = True
        
        if np.abs(w) > 1:
            reward_w = -0.1 * np.abs(w)

        if t > 100000:
            terminate = True
            result = 'Time out'
        else:
            reward_t = -0.1

        reward = reward_g + reward_c + reward_w + reward_ct + reward_t

        return reward, terminate, result

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)

    def generate_random_pose(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y * 10 + 1)
        else:
            y = -(y * 10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        theta = np.random.uniform(0, 2*np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y*10 + 1)
        else:
            y = -(y*10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        return [x, y]

    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

