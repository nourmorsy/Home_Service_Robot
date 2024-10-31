#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Date: 2019/03/28
"""

import rospy, sys
# import moveit_commander
# from moveit_msgs.msg import RobotTrajectory
# from trajectory_msgs.msg import JointTrajectoryPoint
# from robot_control_msgs.msg import Mission, Feedback
from std_msgs.msg import String

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# ---------------------------------------------------------
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import time
# from tf2_ros import TransformListener, TransformBroadcaster
import tf2_ros

class GraspTargetObject:
    def __init__(self):
        # Initialize the move_group API
        # moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('grasp_target_object')
        # self.is_target_set = False
        # self.target_pose = None

        # self.gripper_OPEN = [-0.3]
        # self.gripper_CLOSED = [0.2]
        # self.gripper_NEUTRAL = [0.0]                
        # # Initialize the move group for the right self.arm
        # self.arm = moveit_commander.MoveGroupCommander('arm_group')
        # # Connect to the right_self.gripper move group
        # self.gripper = moveit_commander.MoveGroupCommander('gripper_group')
        # # Allow 5 seconds per planning attempt
        # self.arm.set_planning_time(5)
        # # Get the name of the end-effector link
        # self.end_effector_link = self.arm.get_end_effector_link()               
        # Set the reference frame for pose targets
        self.reference_frame = '/base_link'
        # Set the right self.arm reference frame accordingly
        # self.arm.set_pose_reference_frame(self.reference_frame)
        # # Allow replanning to increase the odds of a solution
        # self.arm.allow_replanning(True)
        # # Allow some leeway in position (meters) and orientation (radians)
        # self.arm.set_goal_position_tolerance(0.03)
        # self.arm.set_goal_orientation_tolerance(0.04)

        # rospy.Subscriber("/control_to_arm", Mission, self.controlCallback)
        # self.pub_control = rospy.Publisher("/arm_to_control", Feedback, queue_size=1)
        # self.start_grasp_attempt()
        # ------------------------------------------------------------------------------------------------
        self.dope_detected_objects = rospy.Subscriber('/dope/detected_objects', Detection3DArray, self.dopeDetectedObjects)
        self.dope_markers = rospy.Subscriber('/dope/markers', MarkerArray, self.dopeMarkers)
        self.poses = {}
        # Lists for publishers
        self.dope_poses = []
        self.dope_dimensions = []
        

        # For each loaded modal two topics are created by DOPE:
        # /dope/pose_{model name} outputs markers for the position
        # /dope/dimension_{model name} outputs dimensions
        all_topics = rospy.get_published_topics('/dope')
        
        for object_pose_topic in [topic[0] for topic in all_topics if topic[0].startswith('/dope/pose_')]:
            self.dope_poses.append(rospy.Subscriber(object_pose_topic, PoseStamped, self.dopeObjectPose, callback_args=object_pose_topic))

        for dimension_topic in [topic[0] for topic in all_topics if topic[0].startswith('/dope/dimension_')]:
            self.dope_dimensions.append(rospy.Subscriber(dimension_topic, String, self.dopeDimension, callback_args=dimension_topic))

        print('Subscribed to all dope topics. Waiting for results.')

    def get_tf(self):
        key = "TomatoSauce"
        while True:
            if self.poses and key in self.poses:
                tfBuffer = tf2_ros.Buffer()
                listener = tf2_ros.TransformListener(tfBuffer)
                
                camera_pose = PoseStamped()
                camera_pose.header.frame_id = "/camera_rgb_optical_frame"
                camera_pose.header.stamp = rospy.Time.now()   
                camera_pose.pose.position.x = self.poses["TomatoSauce"][0]
                camera_pose.pose.position.y = self.poses["TomatoSauce"][1]
                camera_pose.pose.position.z = self.poses["TomatoSauce"][2]
                # --------------------------------------------------------------
                transform_time = rospy.Time(0)  # Use current time for latest transform
                
                # Transform the pose from source to target frame
                transformed_pose = listener.transformPose(self.reference_frame, transform_time, camera_pose)
                print("Transformed Pose:")
                print(transformed_pose)
                rospy.loginfo("Transformation successful!")
                return            
            else:
                # Dictionary is empty or key doesn't exist yet
                print("Dictionary is empty or key not found. Waiting...")
                time.sleep(1)  # Wait for 1 second before checking again
    # def controlCallback(self, msg):
    #     if msg.action == "grasp" and msg.target == "object":
    #         self.start_grasp_attempt()
    #     elif msg.action == "release" and msg.target == "object":
    #         self.release_object()
    # ----------------------------------------------------------------------------------------------------------------------------
    def dopeObjectPose(self, poseStamped, objectType):
        """ Print the pose """
        object_name = objectType.replace("/dope/pose_", "")
        # print('Pose for %s: ' % object_name)
        self.poses[object_name] = [poseStamped.pose.position.x,poseStamped.pose.position.y,poseStamped.pose.position.z ,poseStamped.pose.orientation.x,poseStamped.pose.orientation.y,poseStamped.pose.orientation.z ]

    def dopeDetectedObjects(self, detection3DArray):
        """ Print the Detection3DArray message """
        pass
    
    def dopeMarkers(self, markers):
        pass
    def dopeDimension(self, string, objectType):
        pass
    # -------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    grasp = GraspTargetObject()
    grasp.get_tf()
    rospy.spin() 