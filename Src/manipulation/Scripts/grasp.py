#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy, sys
import moveit_commander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from robot_control_msgs.msg import Mission, Feedback
from std_msgs.msg import String

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# ---------------------------------------------------------
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import time
import tf2_ros
import tf2_geometry_msgs
import numpy as np

from detection.srv import ManipObject, ManipObjectResponse

class GraspTargetObject:
    def __init__(self):
        # Initialize the move_group API
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node('grasp_target_object')
        self.is_target_set = False
        self.target_pose = None
        self.pose = []
        self.aruco_poses = []
        self.gripper_OPEN = [-0.2]
        self.gripper_CLOSED = [0.2]
        self.gripper_NEUTRAL = [0.0]
                        
        # Initialize the move group for the right self.arm
        self.arm = moveit_commander.MoveGroupCommander('arm_group')
        # Connect to the right_self.gripper move group
        self.gripper = moveit_commander.MoveGroupCommander('gripper_group')
        
        # Allow 5 seconds per planning attempt
        self.arm.set_planning_time(20)
        # Get the name of the end-effector link
        self.end_effector_link = self.arm.get_end_effector_link()               
        # Set the reference frame for pose targets
        self.reference_frame = '/base_link'
        # Set the right self.arm reference frame accordingly
        self.arm.set_pose_reference_frame(self.reference_frame)
        # Allow replanning to increase the odds of a solution
        self.arm.allow_replanning(True)
        # Allow some leeway in position (meters) and orientation (radians)
        self.arm.set_goal_position_tolerance(0.03)
        self.arm.set_goal_orientation_tolerance(np.deg2rad(90))
        self.arm.set_planning_time(5)
        rospy.Subscriber("/control_to_arm", Mission, self.controlCallback)
        self.pub_control = rospy.Publisher("/arm_to_control", Feedback, queue_size=1)
        # ------------------------------------------------------------------------------------------------
        self.dope_detected_objects = rospy.Subscriber('/dope/detected_objects', Detection3DArray, self.dopeDetectedObjects)
        self.dope_markers = rospy.Subscriber('/dope/markers', MarkerArray, self.dopeMarkers)
        self.poses = {}
        # Lists for publishers
        self.dope_poses = []
        self.dope_dimensions = []
        all_topics = rospy.get_published_topics('/dope')
        
        for object_pose_topic in [topic[0] for topic in all_topics if topic[0].startswith('/dope/pose_')]:
            self.dope_poses.append(rospy.Subscriber(object_pose_topic, PoseStamped, self.dopeObjectPose, callback_args=object_pose_topic))

        for dimension_topic in [topic[0] for topic in all_topics if topic[0].startswith('/dope/dimension_')]:
            self.dope_dimensions.append(rospy.Subscriber(dimension_topic, String, self.dopeDimension, callback_args=dimension_topic))

        print('Subscribed to all dope topics. Waiting for results.')
        # ----------------------------------------------------------------------------------------------------------
        # rospy.Subscriber("/aruco_single/pose", PoseStamped, self.aruco_pose)
    
    # def aruco_pose(self,msg):
    #     position = msg.pose.position
    #     orientation = msg.pose.orientation
    #     pose_data = {
    #         'position': (position.x, position.y, position.z),
    #         'orientation': (orientation.x, orientation.y, orientation.z, orientation.w)
    #     }
    #     self.aruco_poses.append(pose_data)
    #     # print(self.aruco_poses)

    def get_tf(self):
        while True:
            if(len(self.pose) >0):
                current_pose = self.pose[-1][1]
                tfBuffer = tf2_ros.Buffer()
                listener = tf2_ros.TransformListener(tfBuffer)
                object_name = self.pose[-1][0]
                camera_pose = PoseStamped()
                camera_pose.header.frame_id = "camera_rgb_optical_frame"
                camera_pose.header.stamp = rospy.Time.now()
                camera_pose.pose.position = current_pose.pose.position
                camera_pose.pose.orientation = current_pose.pose.orientation
                print(camera_pose.pose.position)
                # --------------------------------------------------------------
                # Transform the pose from source to target frame
                transform = tfBuffer.lookup_transform("base_link",
                                       # source frame:
                                       camera_pose.header.frame_id,
                                       # get the tf at the time the pose was valid
                                       camera_pose.header.stamp,
                                       # wait for at most 1 second for transform, otherwise throw
                                       rospy.Duration(5.0))

                transformed_pose = tf2_geometry_msgs.do_transform_pose(camera_pose, transform)
                print("Transformed Pose:")
                print(transformed_pose)
                rospy.loginfo("Transformation successful!")
                return transformed_pose ,object_name
            else:
                # Dictionary is empty or key doesn't exist yet
                print("Dictionary is empty or key not found. Waiting...")
                time.sleep(1)  # Wait for 1 second before checking again
    # ------------------------------------------------------------------------------------------------------------------------------
    def controlCallback(self, msg):
        if msg.action == "grasp" and msg.target == "object":
            self.start_grasp_attempt()
        elif msg.action == "release" and msg.target == "object":
            self.release_object()
    def start_grasp_attempt(self,pose ,name):
        # Set the self.gripper target to neutal position using a joint value target
        self.gripper.set_joint_value_target(self.gripper_OPEN)
         
        # Plan and execute the self.gripper motion
        self.gripper.go()
        rospy.sleep(2)
        rospy.loginfo("gripper open ready to go")
        # Pause for a moment
        rospy.sleep(3)
        rospy.loginfo("ready")
        # Set the start state to the current state
        self.arm.set_start_state_to_current_state()

        # the center of the robot base.
        target_pose = PoseStamped()
        # target_pose.id = "box"
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()     
        target_pose.pose.position = pose.pose.position
        target_pose.pose.orientation = pose.pose.orientation
        target_pose.pose.orientation.w = 1.0
        

        # Set the goal pose of the end effector to the stored pose
        self.arm.set_planner_id("RRTConnect")
        self.arm.set_planning_time(10)
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        
        # Plan the trajectory to the goal
        traj = self.arm.plan()
        
        # Execute the planned trajectory
        self.arm.execute(traj)
    
        # Pause for a second
        rospy.sleep(5)

        # Set the start state to the current state
        self.arm.set_start_state_to_current_state()
        self.gripper.set_joint_value_target(self.gripper_CLOSED)
        self.gripper.go()
        self.arm.set_start_state_to_current_state()
        rospy.sleep(5)
        self.arm.set_named_target('rest')
        self.arm.go()
        self.arm.set_start_state_to_current_state()
        return
    def start_placing_attempt(self):
        self.arm.set_start_state_to_current_state()
        self.arm.set_named_target('place')
        self.arm.go()
        self.arm.set_start_state_to_current_state()
        self.gripper.set_joint_value_target(self.gripper_OPEN)
        # Plan and execute the self.gripper motion
        self.gripper.go()
        self.arm.set_start_state_to_current_state()
        rospy.sleep(5)
        self.arm.set_named_target('rest')
        self.arm.go()
        self.arm.set_start_state_to_current_state()
        return 
    # ----------------------------------------------------------------------------------------------------------------------------

    def dopeObjectPose(self, poseStamped, objectType):
        object_name = objectType.replace("/dope/pose_", "")
        # self.poses[object_name] = poseStamped
        self.pose.append((object_name,poseStamped))
    def dopeDetectedObjects(self, detection3DArray):
        """ Print the Detection3DArray message """
        pass
    
    def dopeMarkers(self, markers):
        pass
    def dopeDimension(self, string, objectType):
        pass
    def pose_reset(self):
        self.pose = []
    def grasp_detection(self,name):
        start_time = time.time()
        if(len(self.pose) > 0):
            for item in self.pose:
                if isinstance(item, tuple) and name in item:
                    print("didn't pick it up")
                    return False
                if time.time() - start_time > 5:
                    print("did pick it up")
                    return True
        else:
            return True

def main(request):
    grasp = GraspTargetObject()
    state = 0
    # while True:
    try:
        if request.manip == 'pick':
            rospy.loginfo('started picking...')
            transform , name = grasp.get_tf()
            grasp.start_grasp_attempt(transform,name)
            # grasp.pose_reset()
            # rospy.sleep(10)
                # if(grasp.grasp_detection(name)):
                    # break
            # rospy.spin() 
            state = 1
        elif request.manip == 'place':
            grasp.start_placing_attempt()
            state = 1
        else:
            pass
        return ManipObjectResponse(1)
    except Exception as e:
        rospy.loginfo('An error occurred: {}'.format(e))

        return ManipObjectResponse(0)
    # -------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    rospy.init_node('manipulation_server')
    rospy.loginfo('Manipulation Server Initiated')

    try:
        # grasp = GraspTargetObject()
        # transform , name = grasp.get_tf()
        # grasp.start_grasp_attempt(transform,name)
        service = rospy.Service('manipulate_object', ManipObject, main)
        rospy.spin()
        # grasp.start_placing_attempt()

    except rospy.ROSInterruptException:
        pass