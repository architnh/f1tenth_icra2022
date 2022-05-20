// RRT assignment
// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
//#include "std_msgs/msg/float32multiarray.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

class OBS_DETECT : public rclcpp::Node {
public:
    OBS_DETECT();
    virtual ~OBS_DETECT();

    //Define Occupancy Grid Parameters
    int occu_grid_x_size_min = 50;
    int occu_grid_y_size_min = 70;
    float update_rate = 0.0001;//0.04;//Time between updating rrt graph. Time to execute RRT* is around 0.0001 min to 0.002 sec. Recommended to keep above 0.03
    int occu_grid_x_size=occu_grid_x_size_min;//135;//always make this an even number
    int occu_grid_y_size=occu_grid_y_size_min;//125;//always make this an even
    const float  resolution=0.04;
    int x_size=  occu_grid_x_size/0.04;  //WARNING IF YOU CHANGE RESOLUTION, ALSO CHANGE THE DIVIDE BY NUMBER IN THE TWO VARIABLES BELOW
    int y_size= occu_grid_y_size/0.04;  //WARNING IF YOU CHANGE RESOLUTION, ALSO CHANGE THE DIVIDE BY NUMBER IN THE TWO VARIABLES BELOW
    int center_y_min = occu_grid_y_size_min/2;
    int center_y = center_y_min;
    int center_x = 10;// occu_grid_x_size * 0.05; //occu_grid_x_size/2;

    //Pose information
    Eigen::Quaterniond q;
    Eigen::Matrix3d rotation_mat;
    nav_msgs::msg::Odometry current_car_pose;
    std::vector<std::vector<float>> spline_points;   

    //Drive command information 
    float current_car_speed; 

    //OBS_DETECT Stuff
    rclcpp::Time previous_time = rclcpp::Clock().now();
    bool use_coll_avoid=false;
    float collision_l;
    float collision_time_buffer = 1.0; //s
    std::vector<float> global_obs_detect_goal{0.0,0.0,0.0};
    int car_spline_idx = 0;
    int goal_spline_idx = 100;
    bool got_pose_flag = false;



private:
    //Spline points location
    std::string spline_file_name = "src/pure_pursuit_pkg/pure_pursuit_pkg/racelines/temp/spline.csv";

    //Publishers 
    std::string coll_grid_topic = "/coll_grid_pub_rviz";
    std::string coll_path_topic = "/coll_path_pub_rviz";
    std::string use_avoid_topic = "/use_obs_avoid";
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr path_pub;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr use_avoid_pub;

    //Subscribers
    std::string scan_topic = "/scan";
    std::string pose_topic_sim = "ego_racecar/odom";
    std::string pose_topic_real = "pf/pose/odom";
    std::string drive_topic = "/drive";
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_sub_;

    
    // callbacks
    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);
    void drive_callback(const ackermann_msgs::msg::AckermannDriveStamped::ConstSharedPtr drive_msg);

    //functions
    void check_to_activate_obs_avoid(std::vector<signed char> &obstacle_data);
    std::vector<std::vector<int>> bresenhams_line_algorithm(int goal_point[2], int origin_point[2]);
    int find_spline_index(float x, float y);
    int find_obs_detect_goal_idx(float l_dist, std::vector<std::vector<float>> spline_points, int car_idx);

    //Publisher functions
    void publish_grid(std::vector<signed char> &occugrid_flat);
    void publish_path(std::vector<signed char> &occugrid_flat); 
};


