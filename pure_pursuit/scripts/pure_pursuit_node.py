#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scipy import interpolate
from scipy import signal
import scipy.ndimage
from scipy.spatial.transform import Rotation as R
import numpy as np
import csv
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')


        # Raceline and RVIZ parameters
        traj_csv = "/sim_ws/src/pure_pursuit/scripts/skir_points.csv"  #Name of csv in racelines directory
        self.sim_flag = True # Set flag true for simulation, false for real

        # Pure pursuit parameters

        self.pp_steer_L = 1.5  # steering look ahead for pure pursuit
        self.rrt_steer_L = .75 #Steering look ahead for RRT   DO NOT SET THIS GREATER THAN THE STANDARD LOOK AHEAD DISTANCE
        self.rrt_activation_L = 3.0#Spline Look Ahead point for RRT

        self.kp = 0.25##0.6
        self.kp_rrt = 0.5#

        self.v_max = 10.0#3#5 #8 #6
        self.v_min = 3.5#0.4# #2.5 #1.5  # This value is NOT used for calculations... only keeps the car above a certain value
        self.ay_max = 7#0.1#3
        self.ax_max = 7#0.1#3
        self.floor_friction_coeff = 1.0#0.2#1.0 #0.8 #0.4
        self.k = 3#3 #5  # Curvature Scaling Factor
        self.k_rrt = 3# Curvature Scaling Factor, rrt specific,  smaller makes the curves seem wider and can go faster around
        self.offset = 0  # Offset to make car brake before turn, and accelerate out of turn units in spline index steps

        # RRT Parameters
        self.use_rrt = False

        # debugging parameters
        self.print_velocity = True

        ### BEGIN CODE ###
        if self.sim_flag:
            self.pose_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 1)
        else:
            self.pose_subscriber = self.create_subscription(Odometry, 'pf/pose/odom', self.pose_callback, 1)

        #For vizualization
        self.goal_points_publisher = self.create_publisher(MarkerArray, 'pp_goal_points_rviz',1)
        self.current_goal_publisher = self.create_publisher(PointStamped, 'pp_current_goal_rviz',1)
        self.spline_publisher = self.create_publisher(Marker, 'pp_spline_rviz',1)

        #For drive control
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, 'drive', 1)

        #For talking with pure pursuit
        self.global_goal_publisher = self.create_publisher(Odometry, 'global_goal_pure_pursuit', 1)
        self.offset_dist_publisher = self.create_publisher(Float32, 'standoff_dist_pure_pursuit', 1)
        self.local_goal_subscriber = self.create_subscription(Odometry, 'rrt_local_goal', self.local_goal_callback,1)

        self.use_rrt_subscriber = self.create_subscription(Bool, 'use_rrt', self.use_rrt_callback,1)
        
        self.rrt_spline_subscriber = self.create_subscription(PoseArray, 'rrt_spline_points', self.rrt_spline_callback,1)
        self.rrt_spline_publisher = self.create_publisher(Marker, 'rrt_spline_rviz',1)
        self.current_rrt_goal_publisher = self.create_publisher(PointStamped, 'rrt_current_goal_rviz', 1)
        self.color_points_publisher = self.create_publisher(MarkerArray, 'color_points_rviz',1)
        
        self.spline_graph = Marker()

        #Open csv and read the waypoint data
        with open(traj_csv, 'r') as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(lines, delimiter=';')
        self.goal_points = data[:, 1:3]/10
        x = self.goal_points[:, 0]
        y = self.goal_points[:, 1]
        
        spline_data, m = interpolate.splprep([x, y], s=0, per=True)
        self.x_spline, self.y_spline = interpolate.splev(np.linspace(0, 1, 1000), spline_data)
        self.spline_points = np.vstack((self.x_spline, self.y_spline, np.zeros((len(self.y_spline)))))
        
        self.rrt_x_spline = self.x_spline[0:2]
        self.rrt_y_spline = self.y_spline[0:2]
        
        with open('spline.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(self.x_spline)):
                        writer.writerow([self.x_spline[i], self.y_spline[i]])

        self.rrt_spline_points = np.vstack((self.x_spline[0:2], self.y_spline[0:2], np.zeros((len(self.y_spline[0:2])))))
        
        self.spline_data = self.populate_spline_data()
        self.marker_data = self.populate_guide_points()

        #Publish Rviz Markers every 2 seconds
        self.timer = self.create_timer(2, self.create_spline)#Publish Spline
        self.timer = self.create_timer(2, self.publish_rviz_data)#Publish waypoints

        self.drive_velocity = np.roll(self.calculate_velocity_profile(self.x_spline, self.y_spline), - self.offset)

        self.local_goal = Odometry()
        
        self.color_data = self.populate_color_spline_points()
        self.timer = self.create_timer(2, self.publish_braking_plan) #Publish Speed Plan

        self.time_since_last_point = self.get_clock().now()

    def pose_callback(self, pose_msg):
        """
        This is the main pure pursuit callback loop
        All parameters are set in the init function
        """
        # parse information from pose_msg
        current_position = pose_msg.pose.pose.position
        current_quat = pose_msg.pose.pose.orientation

        # get the current spline index of the car and goal point
        self.spline_index_car = self.get_closest_point_to_car(current_position, self.spline_points)
        timediff = self.get_clock().now() - self.time_since_last_point


        # Use rrt to find a different goal point if rrt flag is set. RRT flag set means that the obstacle exists.
        if (self.use_rrt == True and timediff.nanoseconds * (1 / 1e9) < 0.2):
            #
            # DYNAMIC LOOKAHEAD NECESSARY TO KNOW BEFOREHAND. Bigger lookahead ensures a more smooth path. 
            # Short lookahead will lead to the RRT failing more often, and will return and impossible and sharp bend towards a much closer and unrealistic goal point.
            # Set v_max and v_min to zero and use the pose estimate to judge lookaheads for different scenarios

            self.rrt_spline_index_car = self.get_closest_point_to_car(current_position, self.rrt_spline_points)
            global_goal_point_rrt = self.rrt_find_goal_point(self.rrt_steer_L, current_position)
            global_goal_point = global_goal_point_rrt # Just for publishing and vizualization purposes
            goal_point_car = self.global_2_local(current_quat, current_position, global_goal_point_rrt)
            self.publish_rrt_current_goal_point(global_goal_point_rrt)
            steering_angle = self.calc_steer(goal_point_car, self.kp_rrt)
            
            try:
                self.rrt_drive_velocity_profile = self.rrt_calculate_velocity_profile(self.rrt_x_spline, self.rrt_y_spline,self.drive_velocity[self.spline_index_car])
                self.rrt_drive_velocity = self.rrt_drive_velocity_profile[self.rrt_spline_index_car]
            except:
                print("rrt drive velocity failed, using spline speed instead")
                self.rrt_drive_velocity = self.drive_velocity[self.spline_index_car]
        else:
            #
            # DYNAMIC LOOKAHEAD NECESSARY TO KNOW BEFOREHAND
            global_goal_point = self.find_goal_point()
            goal_point_car = self.global_2_local(current_quat, current_position, global_goal_point)
            steering_angle = self.calc_steer(goal_point_car, self.kp)
        
        #print("Lookahead dist. = ", self.global_L)
        # Get drive speed from the velocity spline
        if (self.use_rrt == True and timediff.nanoseconds * (1 / 1e9) < 0.2):
            
            drive_speed = self.rrt_drive_velocity
            #
            #TBD: Find curvature of found spline. If more than certain value, slow down speed for some time.
            # This prevents collision. Think about how F1 drivers slow down if the turn isnt favorable
            #drive_speed=2.0 #placeholder value of 2. Needs to be tuned
            #
            #print("RRT Drive Speed: ", drive_speed)
            #except:
            #    print("rrt drive velocity failed, using spline speed instead")
            #    drive_speed = self.drive_velocity[self.spline_index_car]
        else:
            drive_speed = self.drive_velocity[self.spline_index_car]
        
        # Don't let the car drive too slow
        if drive_speed < self.v_min:
            drive_speed = self.v_min

        #Send Global Spline Goal to the Publisher for Rviz as well as RRT*
        global_goal_point = self.find_goal_point()
        goal_point_car = self.global_2_local(current_quat, current_position, global_goal_point)

        # Publish
        self.publish_current_goal_point(global_goal_point)  # Create RVIZ Purple Dot
        msg = AckermannDriveStamped()
        
        msg.drive.speed = 1.0#float(drive_speed)###CHANGE THIS BACK, IN SIM THE CHANGING VELOCITY WAS CAUSING PROBLEMS

        msg.drive.steering_angle = float(steering_angle)
        self.drive_publisher.publish(msg)




        self.rrt_activation_index = self.get_closest_point_to_car(current_position, self.spline_points)
        global_goal_point = self.rrt_activation_find_goal_point()
        goal_point_car = self.global_2_local(current_quat, current_position, global_goal_point)


        # Global Goal for RRT
        global_goal = Odometry()
        global_goal.header.frame_id = "map"
        global_goal.header.stamp = self.get_clock().now().to_msg()
        global_goal.pose.pose.position.x = float(global_goal_point[0])
        global_goal.pose.pose.position.y = float(global_goal_point[1])
        global_goal.pose.pose.position.z = float(global_goal_point[2])
        self.global_goal_publisher.publish(global_goal)

        #if self.print_velocity:
        #    print(f"VEL: {drive_speed}")

    def calc_steer(self, goal_point_car, kp):
        """
        Returns the steering angle from the local goal point
        """
        y = goal_point_car[1]
        steer_dir = np.sign(y)
        r = self.pp_steer_L ** 2 / (2 * np.abs(y))
        gamma = 1 / r
        steering_angle = (gamma * kp * steer_dir)
        return steering_angle

    def publish_braking_plan(self):
        self.color_points_publisher.publish(self.color_data)
        
    def calculate_velocity_profile(self, x_spline_data, y_spline_data):
        #CALCULATE DERIVATIVES FOR CURVATURE
        xder1 = np.gradient(x_spline_data,edge_order=2)
        yder1 = np.gradient(y_spline_data,edge_order=2)

        xder2 = np.gradient(xder1)
        yder2 = np.gradient(yder1)

        curvature = np.abs((xder1 * yder2) - (yder1 * xder2)) / np.power(np.power(xder1,2) + np.power(yder1,2),3/2)

        xy_points = np.array((x_spline_data, y_spline_data))
        xy_offset = np.roll(xy_points,1)
        distances = np.linalg.norm(xy_points -xy_offset, axis=0)
        distances[0]=0

        #pass 1
        ux =np.sqrt((self.floor_friction_coeff * 9.81) / np.abs(curvature * self.k))#Curvature should really be curve radius
        ux = np.minimum(ux, self.v_max)

        ayi = ux**2 *(curvature * self.k)
        ayi = np.minimum(ayi,self.ay_max)

        #Forward pass
        v_forward=[]
        for i in range(len(ux)):
            li = distances[i-1]
            axi = self.ax_max * np.sqrt(1 - (ayi[i-1] / self.ay_max))
            v_forward.append(np.sqrt(ux[i-1]**2+(2*axi*li)))
        v_forward = np.array(v_forward)

        #Backward Pass
        v_backward=[]
        ux = np.flip(v_forward)
        ayi = ux**2 *(curvature * self.k)
        ayi = np.minimum(ayi,self.ay_max)
        for i in range(len(distances)):
            li = distances[i-1]
            axi = self.ax_max * np.sqrt(1 - (ayi[i-1] / self.ay_max))
            v_backward.append(np.sqrt(ux[i-1]**2+(2*axi*li)))
            
        v_backward = np.array(v_backward)
        v_backward = np.flip(np.minimum(v_backward, self.v_max))

        final_velocity_profile = v_backward
        final_velocity_profile = self.smooth_and_sharpen_velocity(final_velocity_profile)

        return final_velocity_profile

    def rrt_calculate_velocity_profile(self, x_spline_data, y_spline_data,current_velocity):
        #CALCULATE DERIVATIVES FOR CURVATURE
        xder1 = np.gradient(x_spline_data,edge_order=2)
        yder1 = np.gradient(y_spline_data,edge_order=2)

        xder2 = np.gradient(xder1)
        yder2 = np.gradient(yder1)

        curvature = np.abs((xder1 * yder2) - (yder1 * xder2)) / np.power(np.power(xder1,2) + np.power(yder1,2),3/2)


        xy_points = np.array((x_spline_data, y_spline_data))
        xy_offset = np.roll(xy_points,1)
        distances = np.linalg.norm(xy_points -xy_offset, axis=0)
        distances[0]=0

        #pass 1
        ux_friction =np.sqrt((self.floor_friction_coeff * 9.81) / np.abs(curvature * self.k))#Curvature should really be curve radius
        ux_friction = np.minimum(ux_friction, self.v_max)

        ux_current_vel = np.ones((len(x_spline_data))) * current_velocity

        ux = np.zeros((len(ux_friction)))
        #Interpolate between the current velocity and the friction model
        for i in range(len(ux_friction)):
            t = i / len(ux_friction)
            ux[i] = ux_current_vel[0] + t * (ux_friction[i] - ux_current_vel[0])

        ux = ux - 0.5#General Slow Down for RRT* Manuever

        ux = np.minimum(ux, self.v_max)
        ux = np.maximum(ux, 0.0)

        ayi = ux**2 *(curvature * self.k)
        ayi = np.minimum(ayi,self.ay_max)

        #Forward pass
        v_forward=[]
        for i in range(len(ux)):
            li = distances[i-1]
            axi = self.ax_max * np.sqrt(1 - (ayi[i-1] / self.ay_max))
            v_forward.append(np.sqrt(ux[i-1]**2+(2*axi*li)))
        v_forward = np.array(v_forward)

        #Backward Pass
        v_backward=[]
        ux = np.flip(v_forward)
        #ayi = np.flip(ayi)
        ayi = ux**2 *(curvature * self.k)
        ayi = np.minimum(ayi,self.ay_max)
        for i in range(len(distances)):
            li = distances[i-1]
            axi = self.ax_max * np.sqrt(1 - (ayi[i-1] / self.ay_max))
            v_backward.append(np.sqrt(ux[i-1]**2+(2*axi*li)))
            
        v_backward = np.array(v_backward)
        v_backward = np.flip(np.minimum(v_backward, self.v_max))

        final_velocity_profile = v_backward
        final_velocity_profile = self.smooth_and_sharpen_velocity(final_velocity_profile)

        return final_velocity_profile
        
    def rrt_spline_callback(self, points):
        all_points =[]
        try:
            point_number = int(points.poses[0].position.z)
            for i in range(int(points.poses[0].position.z)):
                all_points.append([points.poses[i].position.x,points.poses[i].position.y])
            all_points = np.flip(np.array(all_points).astype(float),axis=0)

            if(point_number > 3):
                x = all_points[:,0]
                y = all_points[:,1]
                path=[x,y]
                #CHANGE K= below to change the order of the spline generated
                tck, u= interpolate.splprep(path, w=None, u=None, ub=None, ue=None, k=2, task=0, s=0.3, t=None, full_output=0, nest=None, per=0, quiet=1)
                self.rrt_x_spline,self.rrt_y_spline = interpolate.splev( np.linspace( 0, 1, 26), tck)
                '''x = all_points[:,0]
                y = all_points[:,1]
                spline_data, m = interpolate.splprep([x, y], s=0, per=True)
                self.rrt_x_spline, self.rrt_y_spline = interpolate.splev(np.linspace(0, 1, 26), spline_data)'''
            elif(point_number >= 3):
                x = all_points[:,0]
                y = all_points[:,1]
                path=[x,y]
                tck, u= interpolate.splprep(path, w=None, u=None, ub=None, ue=None, k=2, task=0, s=0.2, t=None, full_output=0, nest=None, per=0, quiet=1)
                self.rrt_x_spline,self.rrt_y_spline = interpolate.splev( np.linspace( 0, 1, 26), tck)
            elif(point_number >1):
                x = all_points[:,0]
                y = all_points[:,1]
                path=[x,y]
                tck, u= interpolate.splprep(path, w=None, u=None, ub=None, ue=None, k=1, task=0, s=0.2, t=None, full_output=0, nest=None, per=0, quiet=1)
                self.rrt_x_spline,self.rrt_y_spline = interpolate.splev( np.linspace( 0, 1, 26), tck)

            if(abs(self.rrt_x_spline[0] - self.rrt_x_spline[-1]) < 0.1 and abs(self.rrt_y_spline[0] - self.rrt_y_spline[-1]) < 0.1 and len(self.rrt_x_spline) > 1):
                #print("in splitter")
                self.rrt_spline_points=np.vstack((self.rrt_x_spline[0:int(len(self.rrt_x_spline)/2)], self.rrt_y_spline[0:int(len(self.rrt_y_spline)/2)], np.zeros((int(len(self.rrt_x_spline)/2)))))
                self.rrt_x_spline = self.rrt_x_spline[0:int(len(self.rrt_x_spline)/2)]
                self.rrt_y_spline = self.rrt_y_spline[0:int(len(self.rrt_y_spline)/2)]
            else:
                self.rrt_spline_points=np.vstack((self.rrt_x_spline, self.rrt_y_spline, np.zeros((len(self.rrt_y_spline)))))

            #Plot Spline on RVIZ
            self.spline_graph.action = Marker.DELETEALL
            self.rrt_spline_publisher.publish(self.spline_graph)
        
            self.spline_graph = self.rrt_populate_spline_data()
            self.rrt_spline_publisher.publish(self.spline_graph)
        except:
            print("failed to update data")
        
    def use_rrt_callback(self, rrt_topic):
        self.use_rrt = rrt_topic.data
        self.time_since_last_point = self.get_clock().now()


    def local_goal_callback(self, local_goal_msg):
        #WARNING THIS VARIABLE IS MISNAMED, IT IS A GLOBAL POSE GOAL
        self.local_goal = local_goal_msg

    def global_2_local(self, current_quat, current_position, goal_point_global):
        # Construct transformation matrix from rotation matrix and position
        H_global2car = np.zeros([4, 4]) #rigid body transformation from  the global frame of referce to the car
        H_global2car[3, 3] = 1
        current_rotation_matrix = R.from_quat(np.array([current_quat.x,current_quat.y,current_quat.z,current_quat.w])).as_matrix()
        H_global2car[0:3, 0:3] = np.array(current_rotation_matrix)
        H_global2car[0:3, 3] = np.array([current_position.x, current_position.y, current_position.z])

        # Calculate point
        goal_point_global = np.append(goal_point_global, 1).reshape(4, 1)
        goal_point_car = np.linalg.inv(H_global2car) @ goal_point_global

        return goal_point_car

    def populate_color_spline_points(self):
        array_values=MarkerArray()
        for i in range(len(self.drive_velocity)):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.15
            message.scale.y=0.15
            message.scale.z=0.15
            message.color.a=1.0

            message.color.r= 2 *(1- float(self.drive_velocity[i] / max(self.drive_velocity)))
            message.color.b= 0.0 
            message.color.g= 2* float(self.drive_velocity[i] / max(self.drive_velocity))

            message.pose.position.x=float(self.x_spline[i])
            message.pose.position.y=float(self.y_spline[i])
            message.pose.position.z=0.0
            array_values.markers.append(message)
        return array_values
    
    def populate_guide_points(self):
        array_values=MarkerArray()

        for i in range(len(self.goal_points)):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.25
            message.scale.y=0.25
            message.scale.z=0.25
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(self.goal_points[i,0])
            message.pose.position.y=float(self.goal_points[i,1])
            message.pose.position.z=0.0
            array_values.markers.append(message)
        return array_values
    
    def rrt_populate_spline_data(self):
        message = Marker()
        message.header.frame_id="map"
        message.type= Marker.LINE_STRIP
        message.action = Marker.ADD
        message.scale.x= 0.150
        message.pose.position.x= 0.0
        message.pose.position.y= 0.0
        message.pose.position.z=0.0
        message.color.a=1.0
        message.color.r=0.0
        message.color.b=1.0
        message.color.g=1.0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0

        for i in range(len(self.rrt_x_spline)-1):
            message.header.stamp = self.get_clock().now().to_msg()
            message.id=i
            point1=Point()
            point1.x=float(self.rrt_x_spline[i])
            point1.y=float(self.rrt_y_spline[i])
            point1.z=0.0

            message.points.append(point1)
            point2=Point()
            point2.x=float(self.rrt_x_spline[i+1])
            point2.y=float(self.rrt_y_spline[i+1])
            point2.z=0.0

            message.points.append(point2)
            self.rrt_spline_publisher.publish(message)

        return message

    def populate_spline_data(self):
        message = Marker()
        message.header.frame_id="map"
        message.type= Marker.LINE_STRIP
        message.action = Marker.ADD
        message.scale.x= 0.150
        message.pose.position.x= 0.0
        message.pose.position.y= 0.0
        message.pose.position.z=0.0
        message.color.a=1.0
        message.color.r=1.0
        message.color.b=1.0
        message.color.g=1.0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0

        for i in range(len(self.x_spline)-1):
            message.header.stamp = self.get_clock().now().to_msg()
            message.id=i
            point1=Point()
            point1.x=float(self.x_spline[i])
            point1.y=float(self.y_spline[i])
            point1.z=0.0

            message.points.append(point1)
            point2=Point()
            point2.x=float(self.x_spline[i+1])
            point2.y=float(self.y_spline[i+1])
            point2.z=0.0

            message.points.append(point2)
            self.spline_publisher.publish(message)

        message.id=len(self.x_spline)
        message.header.stamp = self.get_clock().now().to_msg()
        point1=Point()
        point1.x=float(self.x_spline[-1])
        point1.y=float(self.y_spline[-1])
        point1.z=0.0

        message.points.append(point1)

        point2=Point()
        point2.x=float(self.x_spline[0])
        point2.y=float(self.y_spline[0])
        point2.z=0.0

        message.points.append(point2)

        return message

    def get_point_distances(self,points1, points2):
        return np.linalg.norm(points1-points2)

    def get_closest_point_to_car(self, current_position, all_points):
        try:
            current_position=np.array([current_position.x, current_position.y, current_position.z])
        except:
            current_position=np.array([current_position[0], current_position[1], current_position[2]])
        current_position=np.transpose(np.multiply(current_position,np.transpose(np.ones((all_points.shape)))))

        dist = np.linalg.norm(current_position - all_points, axis=0)

        point_index = np.argmin(dist)
        return point_index

    def get_path_distances(self,current_pos, points, current_point):

        pose=Point()
        pose.x=current_pos[0]
        pose.y=current_pos[1]
        pose.z=current_pos[2]
        closest_index = self.get_closest_point_to_car(pose, points)

        if current_point - closest_index > 4:
            a_vect=np.zeros((3,current_point - closest_index+1))
            b_vect=np.zeros((3,current_point - closest_index+1))

            a_vect[:,0]= current_pos
            a_vect[:,1:] = points[:,closest_index:current_point]

            b_vect = points[:,closest_index:current_point+1]

            distance=np.sum(np.linalg.norm(a_vect-b_vect, axis=0))
        else:
            distance = self.get_point_distances(current_pos, points[:,current_point])
        return distance

    def check_for_next_point(self,current_position, current_point_index, all_points, L):
        current_position=np.array([current_position.x, current_position.y, current_position.z])

        #Distance Along Drive Path
        dist_to_goal = self.get_path_distances(current_position, all_points, current_point_index)

        if(dist_to_goal < L):
            current_point_index+=1
            if current_point_index >= len(all_points[0,:]):
                current_point_index=0
            self.publish_current_goal_point(all_points[:,current_point_index])
        elif(dist_to_goal > 4 * L):
            current_point_index = self.get_closest_point_to_car(current_position, all_points)
            self.publish_current_goal_point(all_points[:,current_point_index])

        return current_point_index

    def publish_rviz_data(self):
        self.goal_points_publisher.publish(self.marker_data)
        
    def publish_rrt_current_goal_point(self, goal_point_position):
        message=PointStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id="map"
        message.point.x=float(goal_point_position[0])
        message.point.y=float(goal_point_position[1])
        message.point.z=0.0
        self.current_rrt_goal_publisher.publish(message)

    def publish_current_goal_point(self, goal_point_position):

        message=PointStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id="map"
        message.point.x=float(goal_point_position[0])
        message.point.y=float(goal_point_position[1])
        message.point.z=0.0
        self.current_goal_publisher.publish(message)

    def create_spline(self):
        self.spline_publisher.publish(self.spline_data)

    def rrt_find_goal_point(self, L, current_position):
        #Returns the global x,y,z position of the goal point
        try:
            current_position=np.array([current_position.x, current_position.y, current_position.z])
        except:
            current_position=np.array([current_position[0], current_position[1], current_position[2]])

        points_in = self.rrt_spline_points
        points_in_front = points_in# np.roll(points_in, -self.rrt_spline_index_car, axis=1)

        points_dist = np.linalg.norm(np.roll(points_in_front, -1, axis=1) - points_in_front, axis=0)
        points_dist = np.cumsum(points_dist)
        idx = np.argmin(np.abs(points_dist - L))
        goal_point_car = points_in_front[:, idx]

        #IF GOAL ENDS UP ON TOP OF THE CAR FOR SOME REASON
        dist = np.linalg.norm(current_position - goal_point_car)
        if abs(dist) < 0.1:
            goal_point_car = points_in[:,np.argmax(np.abs(points_dist - L))]
            
        return goal_point_car
    
    def find_goal_point(self):
        #Returns the global x,y,z position of the goal point
        points_in_front = np.roll(self.spline_points, - self.spline_index_car, axis=1)
        points_dist = np.linalg.norm(np.roll(points_in_front, 1, axis=1) - points_in_front, axis=0)
        points_dist = np.cumsum(points_dist)
        idx = np.argmin(np.abs(points_dist - self.pp_steer_L))
        goal_point_car = points_in_front[:, idx]

        return goal_point_car
    def rrt_activation_find_goal_point(self):
        #Returns the global x,y,z position of the goal point
        points_in_front = np.roll(self.spline_points, - self.rrt_activation_index, axis=1)
        points_dist = np.linalg.norm(np.roll(points_in_front, 1, axis=1) - points_in_front, axis=0)
        points_dist = np.cumsum(points_dist)
        idx = np.argmin(np.abs(points_dist - self.rrt_activation_L))
        goal_point_car = points_in_front[:, idx]

        return goal_point_car

    def smooth_velocity_profile(self, velocities, kernel_size=30):
        #Smooths the velocity profile
        k = np.ones(int(kernel_size))
        k = k / np.sum(k)
        return scipy.ndimage.convolve(velocities, k, mode='wrap')

    def smooth_forward(self, velocities, kernel_size=30):
        #Smooths the velocity profile
        k = np.hstack((np.zeros(int(kernel_size / 2)), np.ones(int(kernel_size / 2))))
        k = k / np.sum(k)
        return scipy.ndimage.convolve(velocities, k, mode='wrap')

    def smooth_backward(self, velocities, kernel_size=30):
        # Smooths the velocity profile
        k = np.hstack((np.ones(int(kernel_size / 2)), np.zeros(int(kernel_size / 2))))
        k = k / np.sum(k)
        return scipy.ndimage.convolve(velocities, k, mode='wrap')

    def derivative(self, velocities, kernel_size=30):
        k = np.hstack((np.ones(int(kernel_size / 2)), -1 * np.ones(int(kernel_size / 2))))
        k = k / np.sum(abs(k))
        return scipy.ndimage.convolve(velocities, k, mode='wrap')

    def smooth_and_sharpen_velocity(self, velocities, kernel_size=30):

        derivative_vel = self.derivative(velocities)
        speed_up = derivative_vel > 0

        vel_smooth = self.smooth_velocity_profile(velocities)
        vel_smooth_forward = self.smooth_forward(vel_smooth)
        vel_smooth_backward = self.smooth_backward(vel_smooth)

        vel_new = velocities.copy()
        vel_new[speed_up] = vel_smooth_backward[speed_up]
        vel_new[~speed_up] = vel_smooth_forward[~speed_up]
        vel_new = self.smooth_velocity_profile(vel_new)

        return vel_new

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()