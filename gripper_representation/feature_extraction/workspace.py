#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:45:03 2022

@author: kelin
"""

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics import ruthModel
import time

def return_robot_states():
        # motor_positions = np.array([pybullet.readUserDebugParameter(self.Rm1Id), 
        #                             pybullet.readUserDebugParameter(self.Rm2Id), 
        #                             pybullet.readUserDebugParameter(self.RfId)]) 
        # # RUTH motor 1 [-1.57 , 3.14], motor 2 [-3.14 , 1.57],  RUTH finger bend [-0.5, 0.5]
        # ur5_values = np.array([pybullet.readUserDebugParameter(self.U1Id), 
        #         pybullet.readUserDebugParameter(self.U2Id), 
        #         pybullet.readUserDebugParameter(self.U3Id), 
        #         pybullet.readUserDebugParameter(self.U4Id), 
        #         pybullet.readUserDebugParameter(self.U5Id), 
        #         pybullet.readUserDebugParameter(self.U6Id)])
        ur5_values = [p.getJointState(1,1)[0],p.getJointState(1,2)[0],
                        p.getJointState(1,3)[0],p.getJointState(1,4)[0],
                        p.getJointState(1,5)[0],p.getJointState(1,6)[0]]
        motor_positions = [p.getJointState(1,10)[0],p.getJointState(1,15)[0],
                        -(p.getJointState(1,13)[0]+0.12745044+0.55)]
        return motor_positions, ur5_values 
    
class FingerAngles:
    def __init__(self, robot, linkNameToID):
        self.robot = robot
        self.linkNameToID = linkNameToID
        Phal_1A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
        P2 = np.array(Phal_1A_info[0][0:2])
        P3 = np.array(Phal_2A_info[0][0:2])
        P4 = np.array(Phal_3A_info[0][0:2])
        C = (P2+P3+P4)/3
        P2C = C-P2
        P3C = C-P3
        P4C = C-P4
        P2P3 = P3-P2
        P3P4 = P4-P3
        angle1 = np.arctan2(P2P3[0]*P2C[1]-P2P3[1]*P2C[0],P2P3[0]*P2C[0]+P2P3[1]*P2C[1])
        angle2 = np.arctan2(P2P3[0]*P3C[1]-P2P3[1]*P3C[0],P2P3[0]*P3C[0]+P2P3[1]*P3C[1])
        angle3 = np.arctan2(P3P4[0]*P4C[1]-P3P4[1]*P4C[0],P3P4[0]*P4C[0]+P3P4[1]*P4C[1])
        self.init_angles = [angle1, angle2, angle3]

    def calc_finger_angles(self):
        Phal_1A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
        P2 = np.array(Phal_1A_info[0][0:2])
        P3 = np.array(Phal_2A_info[0][0:2])
        P4 = np.array(Phal_3A_info[0][0:2])
        C = (P2+P3+P4)/3
        P2C = C-P22
        P3C = C-P3
        P4C = C-P4
        P2P3 = P3-P2
        P3P4 = P4-P3
        angle1 = np.arctan2(P2P3[0]*P2C[1]-P2P3[1]*P2C[0],P2P3[0]*P2C[0]+P2P3[1]*P2C[1])
        angle2 = np.arctan2(P2P3[0]*P3C[1]-P2P3[1]*P3C[0],P2P3[0]*P3C[0]+P2P3[1]*P3C[1])
        angle3 = np.arctan2(P3P4[0]*P4C[1]-P3P4[1]*P4C[0],P3P4[0]*P4C[0]+P3P4[1]*P4C[1])
        angles = [angle1-self.init_angles[0], angle2-self.init_angles[1], angle3-self.init_angles[2]]
        return angles
    
def set_ruth_position(_robot, _RUTH_jointNameToID, _ruth_base_motors):
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, _ruth_base_motors[0], force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, _ruth_base_motors[1], force=100)  

    # Calculate relative individual finger joint movements
    finger_angles = [0, 0, 0]
    finger_pos_plus = 0.12745044
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1A'], p.POSITION_CONTROL, -finger_angles[0], force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2A'], p.POSITION_CONTROL, finger_angles[1], force=100)  
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3A'], p.POSITION_CONTROL, finger_angles[2], force=100)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1B'], p.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2B'], p.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=100)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3B'], p.POSITION_CONTROL, 0.65+_ruth_base_motors[2]+finger_pos_plus, force=100) 
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1C'], p.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=100)   
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2C'], p.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3C'], p.POSITION_CONTROL, -1+_ruth_base_motors[2]+finger_pos_plus, force=100)

rM = ruthModel()

def main():
    physics_client = p.connect(p.GUI)  # for p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)

    urdf_directory = "/home/kelin/workspace_kelin/previous_work/feature_extraction/fo_t1o_t2o.urdf"
    # urdf_root = pybullet_data.getDataPath()

    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
      floor = p.loadURDF("plane.urdf")

    robotPos = [-0.13, 0.02, -0.1]
    robotScale = 1
    robot = p.loadURDF(urdf_directory,
                        robotPos,
                        p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=1,
                        globalScaling=robotScale)
    ur5JointNameToID = {}
    ur5LinkNameToID = {}
    ur5RevoluteID = []
    num_of_ur5_joints = 6
    for j in range(num_of_ur5_joints+2):
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        ur5JointNameToID[jointName] = info[0]
        ur5LinkNameToID[info[12].decode('UTF-8')] = info[0]
        ur5RevoluteID.append(j)

    jointNameToID = {}
    linkNameToID = {}
    revoluteID = []
    for j in range(0,p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        jointNameToID[jointName] = info[0]
        linkNameToID[info[12].decode('UTF-8')] = info[0]
        revoluteID.append(j)

    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * robotScale)
    visualShapeId = -1
    
    # Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0)

    link2CoM = [-0.0138504507471105, 0.00298913196612882, 0.0239892494327788]
    link2CoM2 = np.array([link2CoM[0], 0, link2CoM[2]])
    link2_Joint = -link2CoM2 + 0.07*link2CoM2/np.linalg.norm(link2CoM2)

    link4CoM = [0.0139032099348594, -0.00302765430432597, 0.0240810659954725]
    link4CoM2 = np.array([link4CoM[0], 0, link4CoM[2]])
    link4_Joint = -link4CoM2 + 0.07*link4CoM2/np.linalg.norm(link4CoM2)

    link2_info = p.getLinkState(robot, linkNameToID['Link_2'])
    link4_info = p.getLinkState(robot, linkNameToID['Link_4'])
    CoM_diff = np.array(link4_info[0]) - np.array(link2_info[0])
    link2_Joint = link2_Joint + np.array([0, CoM_diff[2], 0])

    con1 = p.createConstraint(parentBodyUniqueId=robot,
                       parentLinkIndex=linkNameToID['Link_2'],
                       childBodyUniqueId=robot,
                       childLinkIndex=linkNameToID['Link_4'],
                       jointType=p.JOINT_POINT2POINT,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
                       childFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]])


    p.stepSimulation()
    time.sleep(1./240.)

    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, 0.0, force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, 0.0, force=1000)

    fov_rad = np.pi*45.0/180.
    image_height = 1000
    image_width = image_height
    image_cx = image_height/2
    image_cy = image_width/2
    f = image_height/(2*np.tan(fov_rad/2))
    far = 3.1
    near = 0.1
    aspect_ratio = 1.

    viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1, 0, 0.6],
    cameraTargetPosition=[0.5, 0, 0],
    cameraUpVector=[0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=near,
    farVal=far)

    top = np.tan(fov_rad/2)*near
    bottom = -top
    right = top*aspect_ratio
    left = -top*aspect_ratio
    proj_mat = np.array([[2*near/(right-left), 0, (right+left)/(right-left), 0], [0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0], [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)], [0, 0, -1, 0]])
    
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)

    p.setRealTimeSimulation(0)
    targetPos1 = 0.0
    targetPos2 = 0.0
    FA = FingerAngles(robot, linkNameToID)  # Finger angles class

    # Loop just to initialise robot
    for i in range(1000):
        targetPos1 = -i*np.pi/100000
        targetPos2 = i*np.pi/100000
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=1000)
        p.stepSimulation()

    i = 0
    approach_step_count = 0
    ur5_values = [0]*6
    finger_position = 0
    flag = 1
    ruth_init = [0,0,-0.41]
    
    while True:
        for i in range(round((np.pi/2+1)/0.08)):
 
          for j in range(round((np.pi/2+1)/0.08)):
          
            for k in range(round((0.12+0.5)/0.1)):
            
              ruth_1 = -1+0.08*i
              ruth_2 = -np.pi/2+0.08*j
              ruth_3 = -0.41+0.1*k
              set_ruth_position(robot,jointNameToID,[ruth_1,ruth_2,ruth_3])
              for l in range(10):
                 p.stepSimulation()
                 time.sleep(1/20)
        break
    p.disconnect()

if __name__ == "__main__":
    main()