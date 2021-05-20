
#include <iostream>
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
        void get_lookahead_point(const geometry_msgs::Pose& carPose);
        void find_lookahead_point(const geometry_msgs::Pose& carPose);

    private:
        ros::NodeHandle n_;
        ros::Subscriber odom_sub, path_sub, goal_sub, amcl_sub;
        ros::Publisher ackermann_pub, cmdvel_pub, marker_pub, lookAhead_pub, marker_lookAhead_pub;
        ros::Timer timer1, timer2;
        tf::TransformListener tf_listener;

        visualization_msgs::Marker points, line_strip, goal_circle;
        geometry_msgs::Point odom_goal_pt, goal_pos, amcl_pt;
        geometry_msgs::PoseStamped odom_goal_Pose;
        geometry_msgs::PoseStamped forwardPose;


        nav_msgs::Odometry odom;
        nav_msgs::Path map_path, odom_path;

        double Lfw;
        double goal_radius;
        int controller_freq;
        bool foundForwardPt, goal_received, goal_reached;
        int path_count;

	    std::string odom_frame, map_frame, odom_topic, plan_topic, goal_topic, amcl_topic, lah_point_topic, lah_marker_topic, lah_marker_pose_topic;

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
    pn.param("Lfw", Lfw, 3.0); // forward look ahead distance (m)
    pn.param("controller_freq", controller_freq, 30);

    pn.param("goal_radius", goal_radius, 0.5); // goal radius (m)
    
    //Show info
    odom_topic = "odom";
    plan_topic = "move_base/GlobalPlanner/plan";
    goal_topic = "move_base_simple/goal";
    amcl_topic = "amcl_pose";
    odom_frame = "odom";
    map_frame = "map";
    lah_point_topic = "/lookAhead_point";
    lah_marker_topic = "/LookAhead/path_marker";
    lah_marker_pose_topic = "/LookAhead_marker_pose";
    
    
    pn.getParam("odom_topic", odom_topic);//, "/odom");
    pn.getParam("plan_topic", plan_topic);//, "/move_base/GlobalPlanner/plan"); 
    pn.getParam("goal_topic", goal_topic);//, "/move_base_simple/goal"); 
    pn.getParam("amcl_topic", amcl_topic);//, "/amcl_pose"); 
    pn.getParam("odom_frame", odom_frame);
    pn.getParam("map_frame", map_frame);

    pn.getParam("lah_point_topic", lah_point_topic);//, "/lookAhead_point"); 
    pn.getParam("lah_marker_topic", lah_marker_topic);//, "/LookAhead/path_marker");
    pn.getParam("lah_marker_point_topic", lah_marker_pose_topic);//, "/LookAhead/path_marker");

    //Publishers and Subscribers
    odom_sub = n_.subscribe(odom_topic, 1, &LookAhead::odomCB, this);
    path_sub = n_.subscribe(plan_topic, 1, &LookAhead::pathCB, this);
    goal_sub = n_.subscribe(goal_topic, 1, &LookAhead::goalCB, this);
    amcl_sub = n_.subscribe(amcl_topic, 5, &LookAhead::amclCB, this);

    marker_pub = n_.advertise<visualization_msgs::Marker>(lah_marker_topic, 10);

    lookAhead_pub = n_.advertise<geometry_msgs::PoseStamped>(lah_point_topic, 1);

    marker_lookAhead_pub = n_.advertise<geometry_msgs::Point>(lah_marker_pose_topic, 1);
    //Timer
    timer1 = n_.createTimer(ros::Duration((1.0)/controller_freq), &LookAhead::controlLoopCB, this); // Duration(0.05) -> 20Hz

    //Init variables
    foundForwardPt = false;
    goal_received = false;
    goal_reached = false;

    path_count = 0;
   

    //Show info
    ROS_INFO_STREAM(odom_topic);
    ROS_INFO_STREAM(plan_topic);
    ROS_INFO_STREAM(goal_topic);
    ROS_INFO_STREAM(amcl_topic);
    ROS_INFO_STREAM(odom_frame);

    ROS_INFO_STREAM(lah_point_topic);
    ROS_INFO_STREAM(lah_marker_topic);
    ROS_INFO_STREAM(lah_marker_pose_topic);

    //Visualization Marker Settings
    initMarker();

}

void LookAhead::initMarker()
{
    points.header.frame_id = line_strip.header.frame_id = goal_circle.header.frame_id = odom_frame;
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


void LookAhead::pathCB(const nav_msgs::Path::ConstPtr& pathMsg)
{
    this->map_path = *pathMsg;
    path_count = 0;
}


void LookAhead::goalCB(const geometry_msgs::PoseStamped::ConstPtr& goalMsg)
{
    this->goal_pos = goalMsg->pose.position;    
    try
    {
        geometry_msgs::PoseStamped odom_goal;

        tf_listener.transformPose(odom_frame, ros::Time(0) , *goalMsg, map_frame ,odom_goal);

        odom_goal_pt = odom_goal.pose.position;
	    odom_goal_Pose = odom_goal;

        goal_received = true;
        goal_reached = false;

        /*Draw Goal on RVIZ*/
        goal_circle.pose = odom_goal.pose;
        marker_pub.publish(goal_circle);
    }
    catch(tf::TransformException &ex)
    {
        ROS_ERROR("get_goal : %s",ex.what());
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

void LookAhead::find_lookahead_point(const geometry_msgs::Pose& carPose)
{
    geometry_msgs::PoseStamped lookAhead_Pose;
    geometry_msgs::Point carPose_pos = carPose.position;

    ROS_INFO("map_path_size : %d", map_path.poses.size());
    if(!goal_reached){
        
        geometry_msgs::PoseStamped first_path_pose = map_path.poses[0];
        for(int i = path_count; i< map_path.poses.size(); i++)
        {
            geometry_msgs::PoseStamped map_path_pose = map_path.poses[i];
            geometry_msgs::PoseStamped odom_path_pose;  

            //double waytoway_dist_x = first_path_pose.pose.position.x - map_path_pose.pose.position.x;
            //double waytoway_dist_y = first_path_pose.pose.position.y - map_path_pose.pose.position.y;
            double waytoway_dist_x = amcl_pt.x - map_path_pose.pose.position.x;
            double waytoway_dist_y = amcl_pt.y - map_path_pose.pose.position.y;
            
            double waytoway_dist = sqrt(waytoway_dist_x*waytoway_dist_x + waytoway_dist_y*waytoway_dist_y);

            if(waytoway_dist >= Lfw)
            {
	            lookAhead_Pose.header.frame_id = map_frame;
		        lookAhead_Pose.pose.position.x = map_path_pose.pose.position.x;
		        lookAhead_Pose.pose.position.y = map_path_pose.pose.position.y;
		        lookAhead_pub.publish(lookAhead_Pose);
                path_count = i;
   
                break;

            }
            else if(i == map_path.poses.size() - 1)
            {
                lookAhead_Pose.header.frame_id = map_frame;
                lookAhead_Pose.pose.position.x = map_path_pose.pose.position.x;
                lookAhead_Pose.pose.position.y = map_path_pose.pose.position.y;
                lookAhead_pub.publish(lookAhead_Pose);
            }

        }
    }
}

void LookAhead::get_lookahead_point(const geometry_msgs::Pose& carPose)
{
    geometry_msgs::Point carPose_pos = carPose.position;
    double carPose_yaw = getYawFromPose(carPose);
    geometry_msgs::Point forwardPt;
    geometry_msgs::Point odom_car2WayPtVec;

    geometry_msgs::PoseStamped lookAhead_Pose;
    geometry_msgs::PoseStamped map_base_way_pose;

    foundForwardPt = false;

    if(!goal_reached){
        for(int i =0; i< map_path.poses.size(); i++)
        {
            geometry_msgs::PoseStamped map_path_pose = map_path.poses[i];
            geometry_msgs::PoseStamped odom_path_pose;

            try
            {
                tf_listener.transformPose(odom_frame, ros::Time(0) , map_path_pose, map_frame ,odom_path_pose); //map_path.pose -> odom_path.pose : waypoint odom style
                geometry_msgs::Point odom_path_wayPt = odom_path_pose.pose.position; // waypoint odom style

                bool _isWayPtAwayFromLfwDist = isWayPtAwayFromLfwDist(odom_path_wayPt,carPose_pos);
                if(_isWayPtAwayFromLfwDist)
                {

                    ROS_INFO("not goal REACHED!");
                    forwardPt = odom_path_wayPt;
                    forwardPose = odom_path_pose;

                    tf_listener.transformPose(map_frame, ros::Time(0) , forwardPose, odom_frame ,map_base_way_pose);
            
                    lookAhead_Pose.header.frame_id = map_frame;
                    lookAhead_Pose.pose.position.x = map_base_way_pose.pose.position.x;
                    lookAhead_Pose.pose.position.y = map_base_way_pose.pose.position.y;
                    lookAhead_pub.publish(lookAhead_Pose);

                    foundForwardPt = true;
                    break;
                }
            }
            catch(tf::TransformException &ex)
            {
		
                ROS_ERROR("goal resached : %s",ex.what());
                ros::Duration(1.0).sleep();
            }
        }
        
    }
    else if(goal_reached)
    {
        forwardPt = odom_goal_pt;
        forwardPose = odom_goal_Pose;
        foundForwardPt = false;
        
        ROS_INFO("goal REACHED!");
        /*
        tf_listener.transformPose(map_frame, ros::Time(0) , forwardPose, odom_frame ,map_base_way_pose);
    
        lookAhead_Pose.header.frame_id = map_frame;
        lookAhead_Pose.pose.position.x = map_base_way_pose.pose.position.x;
        lookAhead_Pose.pose.position.y = map_base_way_pose.pose.position.y;
        lookAhead_pub.publish(lookAhead_Pose);
        */
    }

    /*Visualized Target Point on RVIZ*/
    /*Clear former target point Marker*/
    points.points.clear();
    line_strip.points.clear();
    
    if(foundForwardPt && !goal_reached)
    {
        points.points.push_back(carPose_pos);
        points.points.push_back(forwardPt);
        line_strip.points.push_back(carPose_pos);
        line_strip.points.push_back(forwardPt);
        marker_lookAhead_pub.publish(forwardPt);

    }

    marker_pub.publish(points);
    marker_pub.publish(line_strip);
    /*
    geometry_msgs::PoseStamped map_base_way_pose;
    try
    {
        tf_listener.transformPose(map_frame, ros::Time(0) , forwardPose, odom_frame ,map_base_way_pose);
    
        lookAhead_Pose.header.frame_id = map_frame;
        lookAhead_Pose.pose.position.x = map_base_way_pose.pose.position.x;
        lookAhead_Pose.pose.position.y = map_base_way_pose.pose.position.y;
        lookAhead_pub.publish(lookAhead_Pose);
    }
    catch(tf::TransformException &ex)
    {
	
        ROS_ERROR("send lookAhead : %s",ex.what());
        ros::Duration(1.0).sleep();
    }
    */
}





void LookAhead::amclCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& amclMsg)
{
    amcl_pt = amclMsg->pose.pose.position;

    if(this->goal_received)
    {
        double car2goal_x = this->goal_pos.x - amclMsg->pose.pose.position.x;
        double car2goal_y = this->goal_pos.y - amclMsg->pose.pose.position.y;
        double dist2goal = sqrt(car2goal_x*car2goal_x + car2goal_y*car2goal_y);
        if(dist2goal < this->goal_radius)
        {
            this->goal_reached = true;
            this->goal_received = false;
            ROS_INFO("Goal Reached !");
        }
    }
}


void LookAhead::controlLoopCB(const ros::TimerEvent&)
{

    geometry_msgs::Pose carPose = this->odom.pose.pose;
    geometry_msgs::Twist carVel = this->odom.twist.twist;

    if(this->goal_received)
    {
        /*Estimate Steering Angle*/
        //get_lookahead_point(carPose);  
       find_lookahead_point(carPose);
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
