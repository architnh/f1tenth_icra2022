// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well
#include "obs_detect_node/obs_detect_node.h"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <Eigen/Geometry>
#include "std_msgs/msg/bool.hpp"
#include <math.h>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
//#include "std_msgs/msg/float32multiarray.hpp"
//#include "std_msgs/msg/MultiArrayDimension.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

// Destructor of the OBS_DETECT classFalse
OBS_DETECT::~OBS_DETECT() {
    // Do something in here, free up used memory, print message, etc.
    RCLCPP_INFO(rclcpp::get_logger("OBS_DETECT"), "%s\n", "OBS_DETECT shutting down");
}
// Constructor of the OBS_DETECT class
OBS_DETECT::OBS_DETECT(): rclcpp::Node("obs_detect_node"){
    //User inputs
    bool sim = true;  // Set flag true for simulation, false for real

    // ROS publishers
    grid_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>(coll_grid_topic,1);
    path_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>(coll_path_topic,1);
    use_avoid_pub = this->create_publisher<std_msgs::msg::Bool>(use_avoid_topic,1);

    // ROS subscribers
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(scan_topic, 1, std::bind(&OBS_DETECT::scan_callback, this, std::placeholders::_1));
    drive_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 1, std::bind(&OBS_DETECT::drive_callback, this, std::placeholders::_1));
    if(sim == true){
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(pose_topic_sim, 1, std::bind(&OBS_DETECT::pose_callback, this, std::placeholders::_1));
    }
    else{
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(pose_topic_real, 1, std::bind(&OBS_DETECT::pose_callback, this, std::placeholders::_1));
    }

    //Read in spline points
    std::vector<float> row;
    std::string line, number;
    std::fstream file (spline_file_name, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
			stringstream str(line);
			while(getline(str, number, ','))
				row.push_back(std::stof(number));
			spline_points.push_back(row);
		}
	}
    else{ 
        std::cout<<"ERROR_ERROR_ERROR"<<std::endl;
        std::cout<<"OBS_DETECT.CPP Failed to open spline csv"<<std::endl;
    }

    //Initialzie pose
    q.x()= 0;
    q.y()= 0;
    q.z()= 0;
    q.w()= 1;
    rotation_mat = q.normalized().toRotationMatrix();
    current_car_speed = 0.0;
    collision_l = 3.0;
}

/// MAIN CALLBACK FUNCTIONS///
void OBS_DETECT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // Receive a scan message and update the occupancy grid
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //    listed_data: A new occupancy grid
    //    checks if we need to use gap follow

    //std::cout<<"Scan callback"<<std::endl;
    int x_scan;
    int y_scan;
    std::memset(occu_grid, 0, sizeof occu_grid);
    std::vector<signed char> occugrid_flat(occu_grid_y_size * occu_grid_x_size);

    //Build the occupancy grid
    for(int i=0; i<scan_msg->ranges.size(); i++){
        if (std::isnan(scan_msg->ranges[i])==false && std::isinf(scan_msg->ranges[i])==false && scan_msg->ranges[i]!=0){
            //Find the location of the scan in occugrid x and y coordinates
            x_scan = scan_msg->ranges[i] * cos(scan_msg->angle_increment * i + scan_msg->angle_min) / resolution;
            y_scan = scan_msg->ranges[i] * sin(scan_msg->angle_increment * i + scan_msg->angle_min) / resolution;
            //Make the scans show up larger on the occupancy grid
            for(int j=-1 + x_scan;j<1+ x_scan;j++){//8
                for(int k=-1 + y_scan;k<1 + y_scan;k++){
                    if(j+center_x >0 && j+center_x <occu_grid_x_size){
                        if(k+center_y >0 && k+center_y <occu_grid_y_size){
                            occu_grid[(j+center_x)][occu_grid_y_size-(k+center_y)] = 100;
                            occugrid_flat[((k  + center_y)* occu_grid_x_size) + (j + center_x)]=100;
                        }
                    }
                }
            }
        }
    }

    /*
    // Remove grid points surrounding the car
    for(int i=-1;i<=1;i++){
        for(int j=-1;j<=1;j++){
            occu_grid[(j+center_x)][occu_grid_y_size-(i+center_y)]=0;
            listed_data[((j  + center_y)* occu_grid_x_size) + (i + center_x)]=0;
        }
    }
    */


    publish_grid(occugrid_flat);
    check_to_activate_obs_avoid(occugrid_flat);
}

void OBS_DETECT::check_to_activate_obs_avoid(std::vector<signed char> &occugrid_flat){
    try{
        //Calculate goal point
        //Place the goal point 3m ahead of the car right now
        /*
        Eigen::Vector3d goal_local(3, 0, 0); 
        Eigen::Vector3d goal_global = rotation_mat * goal_local;
        goal_global[0] += current_car_pose.pose.pose.position.x; 
        goal_global[1] += current_car_pose.pose.pose.position.y;
        */

        int car_spline_idx = find_spline_index(current_car_pose.pose.pose.position.x, current_car_pose.pose.pose.position.y);
        int goal_spline_idx = find_obs_detect_goal_idx(collision_l, spline_points, car_spline_idx);
        std::vector<float> global_goal = spline_points[goal_spline_idx];
        int max_spline_idx = spline_points.size();

        int increment = 10;
        int iterations = 0;
        bool run_check = true;
        if (goal_spline_idx - car_spline_idx > 0){
            iterations = (goal_spline_idx - car_spline_idx)/increment;
        } else if (goal_spline_idx - car_spline_idx < 0) {
            iterations = (max_spline_idx - car_spline_idx + goal_spline_idx)/increment;
        } else{
          run_check = false;
        }

        std::vector<std::vector<int>> grid_interp_points;
        if (run_check == true){
          int origin_idx = car_spline_idx;
          int goal_idx = car_spline_idx;
          for(int b=0; b <= iterations; b+=1){
              origin_idx = goal_idx;
              goal_idx += increment;
              if (goal_idx >= max_spline_idx){
                  goal_idx = goal_idx - max_spline_idx;
              }
              if (b == iterations){
                goal_idx = goal_spline_idx;
              }
              float x_goal = spline_points[goal_idx][0] - current_car_pose.pose.pose.position.x;
              float y_goal = spline_points[goal_idx][1] - current_car_pose.pose.pose.position.y;
              float x_origin = spline_points[origin_idx][0] - current_car_pose.pose.pose.position.x;
              float y_origin = spline_points[origin_idx][1]- current_car_pose.pose.pose.position.y;


              Eigen::Vector3d shift_coords_origin(x_origin, y_origin, 0);
              Eigen::Vector3d local_origin = rotation_mat.inverse() * shift_coords_origin;

              Eigen::Vector3d shift_coords_goal(x_goal, y_goal, 0);
              Eigen::Vector3d local_goal = rotation_mat.inverse() * shift_coords_goal;


              //Convert to occu coordinates
              //If on first iteration, connect car to spline
              int goal_point[2];
              int origin_point[2];
              if(b==0){
                  goal_point[0] = (local_goal[0]/resolution)+center_x;
                  goal_point[1] = (local_goal[1]/resolution)+center_y;
                  origin_point[0] = center_x;
                  origin_point[1]= center_y;
              }else{
                  goal_point[0] = (local_goal[0]/resolution)+center_x;
                  goal_point[1] = (local_goal[1]/resolution)+center_y;
                  origin_point[0] =(local_origin[0]/resolution)+center_x;
                  origin_point[1] =(local_origin[1]/resolution)+center_y;
              }

                std::vector<std::vector<int>> temp_grid_interp_points;
                temp_grid_interp_points = bresenhams_line_algorithm(goal_point,origin_point);


                
                //Make Interp Points Wider
                //Add 9 cm to each side
                int add_val_x = abs((0.025 / resolution)); //0.1
                int add_val_y = abs((0.025 / resolution)); //0.1
                if(add_val_x==0){
                    add_val_x=1;
                }
                if(add_val_y==0){
                    add_val_y=1;
                }
                int size_val= temp_grid_interp_points.size();
                //std::cout<<"4"<<std::endl;
                for(int i=0;i<size_val;i++){
                    for(int j=-add_val_y;j<add_val_y;j++){
                        for(int k=-add_val_x;k<add_val_x;k++){
                            if(temp_grid_interp_points[i][0]+k >0 && temp_grid_interp_points[i][0]+k <occu_grid_x_size){
                                if( temp_grid_interp_points[i][1]+j >0 && temp_grid_interp_points[i][1]+j <occu_grid_y_size){
                                    int x_val = temp_grid_interp_points[i][0]+k;
                                    int y_val = temp_grid_interp_points[i][1]+j;
                                    std::vector<int> add_points{x_val,y_val};
                                    if(x_val >0 && x_val <occu_grid_x_size){
                                        if( y_val >0 && y_val <occu_grid_y_size){
                                            temp_grid_interp_points.push_back(add_points);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                

            grid_interp_points.insert(grid_interp_points.end(), temp_grid_interp_points.begin(), temp_grid_interp_points.end());
          }
        }
        std::vector<signed char> path_data(occu_grid_y_size * occu_grid_x_size);

        for(int i=0;i<grid_interp_points.size();i++){
            if(grid_interp_points[i][1] >= 0 && grid_interp_points[i][0] >= 0){
                //if( ((grid_interp_points[i][1])* occu_grid_x_size) + (grid_interp_points[i][0]) < (occu_grid_x_size * occu_grid_y_size)){
                    if(((grid_interp_points[i][1])* occu_grid_x_size) + (grid_interp_points[i][0]) <path_data.size()){
                        path_data[((grid_interp_points[i][1])* occu_grid_x_size) + (grid_interp_points[i][0])]=100;
                    }
                //}
            }
        }
        //Check if there is a collision!
        use_coll_avoid = false;
        for(int i=0;i<path_data.size();i++){
            if(path_data[i]==100 && i < occugrid_flat.size() && occugrid_flat[i]==100){
                //hit_count++;
                use_coll_avoid = true;
                break;
            }
        }

        publish_path(path_data);

        auto use_coll_avoid_msg= std_msgs::msg::Bool();
        use_coll_avoid_msg.data = use_coll_avoid;
        use_avoid_pub->publish(use_coll_avoid_msg);

    }
    catch(...){
        std::cout<<"OBS_DETECT ACTIVATION FAILED"<<std::endl;
    }
}


void OBS_DETECT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    // The pose callback when subscribed to particle filter's inferred pose
    // Check to see if we need gap follow
    // Args:msg
    
    current_car_pose = *pose_msg;
    q.x()= current_car_pose.pose.pose.orientation.x;
    q.y()= current_car_pose.pose.pose.orientation.y;
    q.z()= current_car_pose.pose.pose.orientation.z;
    q.w()= current_car_pose.pose.pose.orientation.w;
    rotation_mat = q.normalized().toRotationMatrix();

}

void OBS_DETECT::drive_callback(const ackermann_msgs::msg::AckermannDriveStamped::ConstSharedPtr drive_msg) {
    // The drive callback gets the published drive message

    ackermann_msgs::msg::AckermannDriveStamped msg = *drive_msg;
    current_car_speed = msg.drive.speed; 
    collision_l = current_car_speed * collision_time_buffer;
}

/// FUNCTIONS FOR DETECTING OBS_DETECT ON/OFF ///
std::vector<std::vector<int>> OBS_DETECT::bresenhams_line_algorithm(int goal_point[2], int origin_point[2]){
    try{
        int x1 = origin_point[0];
        int y1 = origin_point[1];
        int x2 = goal_point[0];
        int y2 = goal_point[1];

        int y_diff = y2 - y1;
        int x_diff = x2 - x1;

        bool swapped = false;

        if (abs(y_diff) >= abs(x_diff)){
            swapped = true;
            x1 = origin_point[1];
            y1 = origin_point[0];
            x2 = goal_point[1];
            y2 = goal_point[0];
        }

        int intermediate;
        if(x1 > x2){
            intermediate = x1;
            x1 = x2;
            x2 = intermediate;

            intermediate = y1;
            y1 = y2;
            y2 = intermediate;
        }

        y_diff = y2 - y1;
        x_diff = x2 - x1;

        int error = int(x_diff / 2);
        float ystep=-1;
        if(y1 < y2){
            ystep=1;
        }

        int y = y1;
        std::vector<std::vector<int>> output;
        for(int x=x1; x < x2+1 ;x++){
            std::vector<int> coords{x,y};
            if (abs(y_diff) > abs(x_diff)){
                coords[0] = y;
                coords[1] = x;
            }
            output.push_back(coords);
            error -= abs(y_diff);
            if(error < 0){
                y+=ystep;
                error+=x_diff;
            }
        }
        

        if(swapped == true){
            std::vector<std::vector<int>> newoutput;
            for(int i=0;i<output.size();i++){
                std::vector<int> newcoords{output[i][1],output[i][0]};
                newoutput.push_back(newcoords);
            }
            return newoutput;
        }
        
        else{
            return output;
        }  
        
       return output;  
    }
    catch(...){
        std::cout<<"bresenhams failed"<<std::endl;
    }
    
}


int OBS_DETECT::find_spline_index(float x, float y){
/*
Returns the index of the closest point on the spline to (x,y)
*/
    float spline_index=-10000;
    float min_val = 1000;

    for(int i=0;i<spline_points.size();i++){
    float dist = sqrt(pow(abs(x - spline_points[i][0]), 2)  + pow(abs(y - spline_points[i][1]), 2));
        if(dist < min_val){
            spline_index=i;
            min_val = dist;
        }
    }
    return spline_index;
}

int OBS_DETECT::find_obs_detect_goal_idx(float l_dist, std::vector<std::vector<float>> spline_points, int car_idx){
    float total_dist = 0.0;
    int goal_point_idx; 
    int current_idx = car_idx;
    int next_idx = car_idx + 1; 

    for(int i=0;i<spline_points.size();i++){
        if (current_idx >= spline_points.size()){
            current_idx = 0;
        }
        if (next_idx >= spline_points.size()){
            current_idx = 0;
        }

        total_dist += sqrt(pow(abs(spline_points[current_idx][0] - spline_points[next_idx][0]), 2)  + pow(abs(spline_points[current_idx][1] - spline_points[next_idx][1]), 2));
        if (total_dist > l_dist){
            goal_point_idx = i;
            break;
        }
        current_idx +=1;
        next_idx +=1; 
    }
    goal_point_idx += car_idx; 
    if (goal_point_idx >= spline_points.size()){
        goal_point_idx -= spline_points.size();
    }
    return goal_point_idx;
}

//Publishers
void OBS_DETECT::publish_grid(std::vector<signed char> &occugrid_flat){
    //Publish the occupancy grid
    auto new_grid= nav_msgs::msg::OccupancyGrid();
    new_grid.info.resolution=resolution;
    new_grid.info.width=occu_grid_x_size;
    new_grid.info.height=occu_grid_y_size;
    std::string frame_id="map";
    new_grid.header.frame_id=frame_id;
    new_grid.header.stamp=rclcpp::Clock().now();
    new_grid.info.origin = current_car_pose.pose.pose;
    Eigen::Vector3d occu_grid_shift(center_x * resolution, center_y * resolution, 0);
    Eigen::Vector3d shift_in_global_coords = rotation_mat * occu_grid_shift;
    new_grid.info.origin.position.x-= shift_in_global_coords[0];
    new_grid.info.origin.position.y-= shift_in_global_coords[1];
    new_grid.data= occugrid_flat;
    grid_pub->publish(new_grid);
}

//Publishers
void OBS_DETECT::publish_path(std::vector<signed char> &occugrid_flat){
    //Publish the occupancy grid
    auto new_grid= nav_msgs::msg::OccupancyGrid();
    new_grid.info.resolution=resolution;
    new_grid.info.width=occu_grid_x_size;
    new_grid.info.height=occu_grid_y_size;
    std::string frame_id="map";
    new_grid.header.frame_id=frame_id;
    new_grid.header.stamp=rclcpp::Clock().now();
    new_grid.info.origin = current_car_pose.pose.pose;
    Eigen::Vector3d occu_grid_shift(center_x * resolution, center_y * resolution, 0);
    Eigen::Vector3d shift_in_global_coords = rotation_mat * occu_grid_shift;
    new_grid.info.origin.position.x-= shift_in_global_coords[0];
    new_grid.info.origin.position.y-= shift_in_global_coords[1];
    new_grid.data = occugrid_flat;
    path_pub->publish(new_grid);
}