#!/usr/bin/env python

"""

    RoboCup@Home Education | oc@robocupathomeedu.org
    navi.py - enable turtlebot to navigate to predefined waypoint location

"""

import rospy

import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, RecoveryStatus
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String

from planning.srv import NavigationLocation, NavigationLocationResponse







original = 0
start = 1

# navigation_points = {'bathroom': [1.3, 0.79, 0.0562],
# 					'kitchen': [-0.479, -0.439, -0.00143],
# 					'living room':[-1.3, -0.615, -0.00143]
# 					}

target = ''



class NavToPoint:
    def __init__(self):
        rospy.on_shutdown(self.cleanup)

	    # Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("Waiting for move_base action server...")

        # Wait for the action server to become available
        self.move_base.wait_for_server(rospy.Duration(120))
        rospy.loginfo("Connected to move base server")

        # A variable to hold the initial pose of the robot to be set by the user in RViz
        initial_pose = PoseWithCovarianceStamped()
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped, self.update_initial_pose)

	    # Get the initial pose from the user
        rospy.loginfo("*** Click the 2D Pose Estimate button in RViz to set the robot's initial pose...")
        rospy.wait_for_message('initialpose', PoseWithCovarianceStamped)

        
        # Make sure we have the initial pose
        while initial_pose.header.stamp == "":
        	rospy.sleep(1)
            
        rospy.loginfo("Ready to go")

    def update_initial_pose(self, initial_pose):
        print(initial_pose)
        self.initial_pose = initial_pose
        global original
        if original == 0:
            self.origin = self.initial_pose.pose.pose
            # global original
            original = 1

    def cleanup(self):
        rospy.loginfo("Shutting down navigation	....")
        self.move_base.cancel_goal()

    def nav_home(self):
        rospy.loginfo("Going back home")
        rospy.sleep(2)
        self.goal.target_pose.pose = self.origin
        self.move_base.send_goal(self.goal)
        waiting = self.move_base.wait_for_result(rospy.Duration(300))
        if waiting == 1:
            rospy.loginfo("Reached home")
            rospy.sleep(2)
            global start
            # start = 2


    def nav(self, A_x, A_y, A_theta):
        rospy.sleep(1)

        locations = dict()

        # # Location A
        # A_x = 2.12
        # A_y = -0.00393
        # A_theta = -0.00143

        # A_x = self.A_x
        # A_y = self.A_y
        # A_theta = self.A_theta

        quaternion = quaternion_from_euler(0.0, 0.0, A_theta)
        
        # z = quaternion[2]
        z = -0.4
        locations['A'] = Pose(Point(A_x, A_y, 0.000), Quaternion(quaternion[0], quaternion[1], z, quaternion[3]))

        self.goal = MoveBaseGoal()
        rospy.loginfo("Starting navigation test")


        # while not rospy.is_shutdown():
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.header.stamp = rospy.Time.now()

        # Robot will go to point A
        
        # rospy.loginfo("Going to point")
        rospy.sleep(2)
        rospy.loginfo(RecoveryStatus().pose_stamped.pose.position)
        self.goal.target_pose.pose = locations['A']
        self.move_base.send_goal(self.goal)
        waiting = self.move_base.wait_for_result(rospy.Duration(300))
        if waiting == 1:
            # rospy.loginfo("Reached point")
            rospy.sleep(2)
            # rospy.loginfo("Ready to go back")
            rospy.sleep(2)

            global start
            start = 0

        # After reached point A, robot will go back to initial position
        
            # rospy.Rate(5).sleep()

def navigate(request):
    # navigation_points = {'bathroom': [2.21, 0.000826, -0.00143],
	# 				'kitchen': [1.15, -2.24, -0.00534],
	# 				'bedroom':[0.312, 1.31, -0.00143]
	# 				}
    
    navigation_points = {'bathroom': [0.0848, -0.0903, 0.404],
					'kitchen': [-8.88, 3.32, 0.408],
					'bedroom':[-1.3, -0.615, -0.00143]
					}
					
    target = ''
    # print(request)
    target = request.locations
    print(target)
    A_x = navigation_points[target][0]
    A_y = navigation_points[target][1]
    A_theta = navigation_points[target][2]
    # print(target, A_x, A_y, A_theta)

    rospy.loginfo("Going to " + target)
    # navigation_object.nav(A_x, A_y, A_theta)
    rospy.sleep(10)
    rospy.loginfo("Reached point " + target)
    
    return NavigationLocationResponse(1)



if __name__=="__main__":
    # rospy.init_node('nav_node')
    rospy.init_node('navigation_server')
    navigation_object = NavToPoint()
    rospy.loginfo('Navigation Server Initiated')

    try:


        service = rospy.Service('send_location', NavigationLocation, navigate)

        rospy.spin()
        # NavToPoint()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass

