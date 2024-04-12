#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import time
from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.autogen.messages import Base_pb2

import threading

def forward_kinematics(base):
    try:
        input_joint_angles = base.GetMeasuredJointAngles()
    except KServerException as ex:
        print("Unable to get joint angles")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False

    try:
        pose = base.ComputeForwardKinematics(input_joint_angles)
    except KServerException as ex:
        print("Unable to compute forward kinematics")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False

    return pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z


# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished



def cartesian_action_movement(base, pose):
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = pose[0]       # (meters)
    cartesian_pose.y = pose[1]    # (meters)
    cartesian_pose.z = pose[2]   # (meters)
    cartesian_pose.theta_x = pose[3] # (degrees)
    cartesian_pose.theta_y = pose[4] # (degrees)
    cartesian_pose.theta_z = pose[5] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout on action notification wait")
    return finished

def gripper_close(base, value = 0.7):
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    
    # Close the gripper with position increments
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    position = 0.00
    finger.finger_identifier = 1
    while position < value:
        finger.value = position
        # print("Going to position {:0.2f}...".format(finger.value))
        base.SendGripperCommand(gripper_command)
        position += 0.1
        time.sleep(1)

def gripper_open(base):
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    
    # Set speed to open gripper
    gripper_command.mode = Base_pb2.GRIPPER_SPEED
    finger.value = 0.1
    base.SendGripperCommand(gripper_command)
    gripper_request = Base_pb2.GripperRequest()

    # Wait for reported position to be opened
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    while True:
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        if len (gripper_measure.finger):
            # print("Current position is : {0}".format(gripper_measure.finger[0].value))
            if gripper_measure.finger[0].value < 0.01:
                break
        else: # Else, no finger present in answer, end loop
            break