#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time
from navigation.srv import DirectionMove, DirectionMoveResponse

# class TurtleBotController:
    # def __init__(self):
    #     rospy.init_node('turtlebot_controller', anonymous=True)
    #     self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

def movement(move):
    rate = rospy.Rate(10)  # 10 Hz
    move_duration = 1.0  # Move for 100 milli seconds
    start_time = time.time()
    # image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    global cmd_vel_pub
    cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

    direction = {'left' : 0.5, 'right' : -0.5}

    while not rospy.is_shutdown() and (time.time() - start_time < move_duration):
        # Adjust linear and angular velocities here as needed
        vel_msg = Twist()
        # vel_msg.linear.x = 0.2  # Adjust linear velocity for forward/backward motion
        vel_msg.angular.z = direction[move]  # Adjust angular velocity for turning
        cmd_vel_pub.publish(vel_msg)
        rate.sleep()
    return 1
        

# if __name__ == '__main__':
#     try:
#         controller = TurtleBotController()
#         controller.move()
#     except rospy.ROSInterruptException:
#         pass


def get(request):
    global move
    move = request.move
    rospy.loginfo('moving {}'.format(move))
    rospy.loginfo('start publishing')
    try:
        movement(move)
        # Wait for a single message
        # rospy.wait_for_message("/cmd_vel_mux/input/teleop", Twist)
        
        # Unregister the subscriber after receiving the first message
        # cmd_vel_pub.unregister()

        # rospy.loginfo('image saved in {}'.format(image_name))
        return DirectionMoveResponse(1)

    except rospy.ROSInterruptException:
        pass

if __name__=="__main__":

    rospy.init_node('move_motors_node')
    rospy.loginfo('Move Motor Server Initiated')

    try:
        
        service = rospy.Service('move_motor', DirectionMove, get)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

