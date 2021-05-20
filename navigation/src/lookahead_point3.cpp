#include <iostream>
#include <map>
#include <math.h>
#include "ros/ros.h"
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <vector>

#include <string>
#include <fstream>

using namespace std;
using std::string;

/********************/
/* CLASS DEFINITION */
/********************/
class LookAhead
{
    public:
        LookAhead();
        void initMarker();
        bool isForwardWayPt(const geometry_msgs::Point& wayPt, const geometry_msgs::Pose& carPose);
        bool isWayPtAwayFromLfwDist(const geometry_msgs::Point& wayPt, const geometry_msgs::Point& car_pos);
        double getYawFromPose(const geometry_msgs::Pose& carPose);        

        double polyeval(Eigen::VectorXd coeffs, double x);        
        Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
        
        double getCar2GoalDist();

	double getW(double _alpha);	
		
        double get_alpha(const geometry_msgs::Pose& carPose);

    private:
        

        ros::NodeHandle n_;
        ros::Subscriber odom_sub, path_sub, goal_sub, amcl_sub;
        ros::Publisher ackermann_pub, cmdvel_pub, marker_pub, lookAhead_pub;
        ros::Timer timer1, timer2;
        tf::TransformListener tf_listener;

        ros::Time tracking_stime;
        ros::Time tracking_etime;
        ros::Time tracking_time;
        int tracking_time_sec;
        int tracking_time_nsec;
        
        //time flag
        bool start_timef = false;
        bool end_timef = false;

        ros::Time maxvel_check_time;
        bool max_vel_check;

        std::ofstream file;
    
        double _waypointsDist = -1.0;
        double _pathLength = 8.0;
        bool _path_computed = false;

        string _map_frame, _odom_frame, _car_frame;
       
        unsigned int idx = 0;

        visualization_msgs::Marker points, line_strip, goal_circle;
        geometry_msgs::Point odom_goal_pos, goal_pos;
        geometry_msgs::Twist cmd_vel;
        ackermann_msgs::AckermannDriveStamped ackermann_cmd;
        nav_msgs::Odometry odom;
        nav_msgs::Path map_path, odom_path, odom_path_w, _odom_path;

        int _downSampling;

        double L, Lfw, Vcmd, lfw, steering, velocity;
        double w;
        double steering_gain, max_w, base_angle, goal_radius, speed_incremental;
        int controller_freq;
        bool foundForwardPt, goal_received, goal_reached, cmd_vel_mode, debug_mode, smooth_accel;

        void odomCB(const nav_msgs::Odometry::ConstPtr& odomMsg);
        void pathCB(const nav_msgs::Path::ConstPtr& pathMsg);
        void goalCB(const geometry_msgs::PoseStamped::ConstPtr& goalMsg);
        void amclCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& amclMsg);
        void controlLoopCB(const ros::TimerEvent&);

}; // end of class


LookAhead::LookAhead()
{
    //Private parameters handler
    ros::NodeHandle pn("~");

    //Car parameter
    pn.param("L", L, 0.26); // length of car
    pn.param("Vcmd", Vcmd, 1.0);// reference speed (m/s)
    pn.param("Lfw", Lfw, 2.0); // forward look ahead distance (m)
    pn.param("lfw", lfw, 0.13); // distance between front the center of car

    //Controller parameter
    pn.param("controller_freq", controller_freq, 20);
    pn.param("steering_gain", steering_gain, 1.0);
    pn.param("max_w", max_w, 1.0);
    pn.param("path_length", _pathLength, 4.0); // unit: m
    pn.param("goal_radius", goal_radius, 0.5); // goal radius (m)
    pn.param("base_angle", base_angle, 0.0); // neutral point of servo (rad) 
    pn.param("cmd_vel_mode", cmd_vel_mode, false); // whether or not publishing cmd_vel
    pn.param("debug_mode", debug_mode, false); // debug mode
    pn.param("smooth_accel", smooth_accel, true); // smooth the acceleration of car
    pn.param("speed_incremental", speed_incremental, 0.5); // speed incremental value (discrete acceleraton), unit: m/s

    //Parameter for topics & Frame name
    pn.param<std::string>("map_frame", _map_frame, "map" ); //*****for mpc, "odom"
    pn.param<std::string>("odom_frame", _odom_frame, "odom");
    pn.param<std::string>("car_frame", _car_frame, "base_footprint" );

    //Publishers and Subscribers
    odom_sub = n_.subscribe("/odom", 1, &LookAhead::odomCB, this);
    path_sub = n_.subscribe("/move_base/TrajectoryPlannerROS/global_plan", 1, &LookAhead::pathCB, this);
    goal_sub = n_.subscribe("/move_base_simple/goal", 1, &LookAhead::goalCB, this);
    amcl_sub = n_.subscribe("/amcl_pose", 5, &LookAhead::amclCB, this);
    marker_pub = n_.advertise<visualization_msgs::Marker>("/lookAhead/path_marker", 10);

    lookAhead_pub = n_.advertise<geometry_msgs::PoseStamped>("/lookAhead_point", 1);

    timer1 = n_.createTimer(ros::Duration((1.0)/controller_freq), &LookAhead::controlLoopCB, this); //

    //Init variables
    foundForwardPt = false;
    goal_received = false;
    goal_reached = false;
    velocity = 0.0;
    w = 0.0;
    steering = base_angle;

    //Show info
    ROS_INFO("[param] base_angle: %f", base_angle);
    ROS_INFO("[param] Vcmd: %f", Vcmd);
    ROS_INFO("[param] Lfw: %f", Lfw);

    //Visualization Marker Settings
    initMarker();

}

void LookAhead::initMarker()
{
    points.header.frame_id = line_strip.header.frame_id = goal_circle.header.frame_id = "odom";
    points.ns = line_strip.ns = goal_circle.ns = "Markers";
    points.action = line_strip.action = goal_circle.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = line_strip.pose.orientation.w = goal_circle.pose.orientation.w = 1.0;
    points.id = 0;
    line_strip.id = 1;
    goal_circle.id = 2;

    points.type = visualization_msgs::Marker::POINTS;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    goal_circle.type = visualization_msgs::Marker::CYLINDER;
    // POINTS markers use x and y scale for width/height respectively
    points.scale.x = 0.2;
    points.scale.y = 0.2;

    //LINE_STRIP markers use only the x component of scale, for the line width
    line_strip.scale.x = 0.1;

    goal_circle.scale.x = goal_radius;
    goal_circle.scale.y = goal_radius;
    goal_circle.scale.z = 0.1;

    // Points are green
    points.color.g = 1.0f;
    points.color.a = 1.0;

    // Line strip is blue
    line_strip.color.b = 1.0;
    line_strip.color.a = 1.0;

    //goal_circle is yellow
    goal_circle.color.r = 1.0;
    goal_circle.color.g = 1.0;
    goal_circle.color.b = 0.0;
    goal_circle.color.a = 0.5;
}


void LookAhead::odomCB(const nav_msgs::Odometry::ConstPtr& odomMsg)
{

    this->odom = *odomMsg;
}



void LookAhead::pathCB(const nav_msgs::Path::ConstPtr& pathMsg) //jaewan
{

    this->map_path = *pathMsg;

    if(goal_received && !goal_reached)
    {    
        nav_msgs::Path odom_path = nav_msgs::Path();
        try
        {
            double total_length = 0.0;
            int sampling = _downSampling;

            //find waypoints distance
            if(_waypointsDist <=0.0)
            {        
                double dx = pathMsg->poses[1].pose.position.x - pathMsg->poses[0].pose.position.x;
                double dy = pathMsg->poses[1].pose.position.y - pathMsg->poses[0].pose.position.y;
                _waypointsDist = sqrt(dx*dx + dy*dy);
                _downSampling = int(_pathLength/10.0/_waypointsDist);
            }            

            // Cut and downsampling the path
            for(int i =0; i< pathMsg->poses.size(); i++)
            {
                if(total_length > _pathLength)
                    break;

                if(sampling == _downSampling)
                {   
                    geometry_msgs::PoseStamped tempPose;
                    tf_listener.transformPose(_odom_frame, ros::Time(0) , pathMsg->poses[i], _map_frame, tempPose);                     
                    odom_path.poses.push_back(tempPose);  
                    sampling = 0;
                }
                total_length = total_length + _waypointsDist; 
                sampling = sampling + 1;  
            }
           
            if(odom_path.poses.size() >= 6 )
            {
                _odom_path = odom_path; // Path waypoints in odom frame
                _path_computed = true;
                // publish odom path
                odom_path.header.frame_id = _odom_frame;
                odom_path.header.stamp = ros::Time::now();
            }
            else
            {
                cout << "Failed to path generation" << endl;
                _waypointsDist = -1;
            }

        }
        catch(tf::TransformException &ex)
        {
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }
    
}


void LookAhead::goalCB(const geometry_msgs::PoseStamped::ConstPtr& goalMsg)
{

    this->goal_pos = goalMsg->pose.position;    
    try
    {
        geometry_msgs::PoseStamped odom_goal;
        tf_listener.transformPose("odom", ros::Time(0) , *goalMsg, "map" ,odom_goal);
        odom_goal_pos = odom_goal.pose.position;
        goal_received = true;
        goal_reached = false;

        /*Draw Goal on RVIZ*/
        goal_circle.pose = odom_goal.pose;
        marker_pub.publish(goal_circle);
    }
    catch(tf::TransformException &ex)
    {
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
    }
}

double LookAhead::getYawFromPose(const geometry_msgs::Pose& carPose)
{

    float x = carPose.orientation.x;
    float y = carPose.orientation.y;
    float z = carPose.orientation.z;
    float w = carPose.orientation.w;

    double tmp,yaw;
    tf::Quaternion q(x,y,z,w);
    tf::Matrix3x3 quaternion(q);
    quaternion.getRPY(tmp,tmp, yaw);

    return yaw;
}

bool LookAhead::isForwardWayPt(const geometry_msgs::Point& wayPt, const geometry_msgs::Pose& carPose)
{

    float car2wayPt_x = wayPt.x - carPose.position.x;
    float car2wayPt_y = wayPt.y - carPose.position.y;
    double car_theta = getYawFromPose(carPose);

    float car_car2wayPt_x = cos(car_theta)*car2wayPt_x + sin(car_theta)*car2wayPt_y;
    float car_car2wayPt_y = -sin(car_theta)*car2wayPt_x + cos(car_theta)*car2wayPt_y;

    if(car_car2wayPt_x >0) /*is Forward WayPt*/
        return true;
    else
        return false;
}


bool LookAhead::isWayPtAwayFromLfwDist(const geometry_msgs::Point& wayPt, const geometry_msgs::Point& car_pos)
{

    double dx = wayPt.x - car_pos.x;
    double dy = wayPt.y - car_pos.y;
    double dist = sqrt(dx*dx + dy*dy);

    if(dist < Lfw)
        return false;
    else if(dist >= Lfw)
        return true;
}

double LookAhead::get_alpha(const geometry_msgs::Pose& carPose)
{
    
    geometry_msgs::Point carPose_pos = carPose.position;
    double carPose_yaw = getYawFromPose(carPose);
    geometry_msgs::Point odom_path_wayPt;
    geometry_msgs::Point forwardPt;
    geometry_msgs::Point odom_car2WayPtVec;
    geometry_msgs::PoseStamped lookAhead_Pose;
    foundForwardPt = false;

    if(!goal_reached){

        for(int i =0; i< _odom_path.poses.size(); i++)
        {
            geometry_msgs::PoseStamped map_path_pose = _odom_path.poses[i];
            geometry_msgs::PoseStamped odom_path_pose;

            try
            {
                tf_listener.transformPose("odom", ros::Time(0) , map_path_pose, "map" ,odom_path_pose);
                odom_path_wayPt = odom_path_pose.pose.position;
                bool _isForwardWayPt = isForwardWayPt(odom_path_wayPt,carPose);

                if(_isForwardWayPt)
                {
                    bool _isWayPtAwayFromLfwDist = isWayPtAwayFromLfwDist(odom_path_wayPt,carPose_pos);
                    if(_isWayPtAwayFromLfwDist)
                    {
                        forwardPt = odom_path_wayPt;
                        lookAhead_Pose.header.frame_id = "map";
                        lookAhead_Pose.pose.position.x = odom_path_wayPt.x;
                        lookAhead_Pose.pose.position.y = odom_path_wayPt.y;
                        lookAhead_pub.publish(lookAhead_Pose);

                        foundForwardPt = true;
                        break;
                    }
                }
            }
            catch(tf::TransformException &ex)
            {
                ROS_ERROR("%s",ex.what());
                ros::Duration(1.0).sleep();
            }
        }
        
    }
    else if(goal_reached)
    {
        forwardPt = odom_goal_pos;
        lookAhead_Pose.header.frame_id = "map";

        lookAhead_Pose.pose.position.x = odom_path_wayPt.x;
        lookAhead_Pose.pose.position.y = odom_path_wayPt.y;
        lookAhead_pub.publish(lookAhead_Pose);

        foundForwardPt = false;
        //ROS_INFO("goal REACHED!");
    }

    /*Visualized Target Point on RVIZ*/
    /*Clear former target point Marker*/
    points.points.clear();
    line_strip.points.clear();
    
    if(foundForwardPt && !goal_reached)
    {
        points.points.push_back(carPose_pos);
        points.points.push_back(odom_path_wayPt);
        line_strip.points.push_back(carPose_pos);
        line_strip.points.push_back(odom_path_wayPt);
    }

    marker_pub.publish(points);
    marker_pub.publish(line_strip);
 
    double alpha_t = atan2(odom_path_wayPt.y - carPose_pos.y,odom_path_wayPt.x - carPose_pos.x) - carPose_yaw;
	
    return alpha_t;
}


double LookAhead::getW(double _alpha)
{
    return 2*sin(_alpha)/Lfw;
}

void LookAhead::amclCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& amclMsg)
{

    if(this->goal_received)
    {
        double car2goal_x = this->goal_pos.x - amclMsg->pose.pose.position.x;
        double car2goal_y = this->goal_pos.y - amclMsg->pose.pose.position.y;
        double dist2goal = sqrt(car2goal_x*car2goal_x + car2goal_y*car2goal_y);
        if(dist2goal < this->goal_radius)
        {
            if(start_timef)
            {
                tracking_etime = ros::Time::now();
                tracking_time_sec = tracking_etime.sec - tracking_stime.sec; 
                tracking_time_nsec = tracking_etime.nsec - tracking_stime.nsec;

                
                int max_vel_time_sec = maxvel_check_time.sec - tracking_stime.sec;
                int max_vel_time_nsec = maxvel_check_time.nsec - tracking_stime.nsec;
              

                file << "max_vel time"<< "," << max_vel_time_sec << "," <<  max_vel_time_nsec << "\n";
                file << "tracking time"<< "," << tracking_time_sec << "," <<  tracking_time_nsec << "\n";

                file.close();

                start_timef = false;
            }
            this->goal_reached = true;
            this->goal_received = false;
            this->_path_computed = false;

            ROS_INFO("Goal Reached !");
            cout << "tracking time: " << tracking_time_sec << "." << tracking_time_nsec << endl;

        }
    }
}


void LookAhead::controlLoopCB(const ros::TimerEvent&)
{


    geometry_msgs::Pose carPose = this->odom.pose.pose;
    geometry_msgs::Twist carVel = this->odom.twist.twist;

    tf::Pose pose;
    tf::poseMsgToTF(carPose, pose);
    double alpha = tf::getYaw(pose.getRotation());

    if(this->goal_received && !this->goal_reached && this->_path_computed)
    {
        double alpha = get_alpha(carPose);  
    }
}


/*****************/
/* MAIN FUNCTION */
/*****************/
int main(int argc, char **argv)
{
    //Initiate ROS
    ros::init(argc, argv, "LookAhead");
    LookAhead controller;
    ros::AsyncSpinner spinner(2); // Use multi threads
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
