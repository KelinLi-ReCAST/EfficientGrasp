# Set up UR5 and RUTH for simulation
# This file simply tests UR5 and RUTH movement control
# 2021-Oct-23 @Xian Zhang
#=========================READ ME==========================
# When calling main(mode, timer), first parameter takes 0 or 1 for control mode:
# (0) for manual motor control, (1) to start automated testing
# timer parameter is how long you want the simulation to run for.
# Edit automated testing in function auto_joint_test(), by specifying which joints to start auto testing
# Final UR5 and RUTH joint states, final fingertip position and orientation are saved in array output_motor_fingertip




from numpy import random
import pybullet 
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
from gym.envs.kelin.pybullet_object_models import ycb_objects
import os
# import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from random import choice



#======= Visualize contact points as spheres
def visualize_contact_pt(point_pos, rgba, _robotScale):
    colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.01 * _robotScale)
    visualShapeId = -1
    sphereA = pybullet.createMultiBody(0.01, colSphereId, visualShapeId, point_pos)

    # Disable Collisions between links
    pybullet.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    con2 = pybullet.createConstraint(parentBodyUniqueId=sphereA,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=point_pos)

    pybullet.changeVisualShape(sphereA, -1, rgbaColor=rgba)
    

#===== Joint motor control on Robotiq
def motor_control_rtq(_robot, _Bhand_jointNameToID, _Bhand_values):
    Bhand_joints = ['wam/bhand/finger_1/prox_joint', 'wam/bhand/finger_1/med_joint', 'wam/bhand/finger_1/dist_joint', 'wam/bhand/finger_2/prox_joint', 'wam/bhand/finger_2/med_joint', 'wam/bhand/finger_2/dist_joint', 'wam/bhand/finger_3/med_joint', 'wam/bhand/finger_3/dist_joint']
    for index, Bhand in enumerate(_Bhand_values):
        pybullet.setJointMotorControl2(_robot, _Bhand_jointNameToID[Bhand_joints[index]], pybullet.POSITION_CONTROL, Bhand, force=1000)

#===== Joint motor control on UR5
def motor_control_ur5(_robot, _ur5JointNameToID, _ur5_joint_name, _ur5_values):
    for index, ur5_control in enumerate(_ur5_values):   #ur5_control = ur5_values[index]
        pybullet.setJointMotorControl2(_robot, _ur5JointNameToID[_ur5_joint_name[index+1]], pybullet.POSITION_CONTROL, _ur5_values[index], force=1000)

#======= Config parameter panel
class pybulletDebug:
    def __init__(self, control_type, robot, ur5LinkNameToID):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw=90
        self.cpitch=-7
        self.cdist=0.66
        time.sleep(0.5)
        self.init_state = pybullet.getLinkState(robot, ur5LinkNameToID['ee_link'])
        self.init_pos = np.array(self.init_state[0])
        self.init_ori = np.array( pybullet.getEulerFromQuaternion(self.init_state[1]))

       
        if control_type == 'motors':
            self.control_type = 'motors'
            self.U1Id = pybullet.addUserDebugParameter("UR5 Shoulder Pan" , -3.14 , 3.14 , 0.)
            self.U2Id = pybullet.addUserDebugParameter("UR5 Shoulder Lift" , -3.14 , 3.14 , 0.)
            self.U3Id = pybullet.addUserDebugParameter("UR5 Elbow" , -3.14 , 3.14 , 0.)
            self.U4Id = pybullet.addUserDebugParameter("UR5 Wrist 1" , -3.14 , 3.14 , 0.)
            self.U5Id = pybullet.addUserDebugParameter("UR5 Wrist 2" , -3.14 , 3.14 , 0.)
            self.U6Id = pybullet.addUserDebugParameter("UR5 Wrist 3" , -3.14 , 3.14 , 0.)
    
        if control_type == 'xyz':
            self.control_type = 'xyz'
        #=== End effector pose xyz and orientation
            # self.U1Id = p.addUserDebugParameter("UR5_EE X" , -1.0 , 1.0 , self.init_pos[0]) 
            # self.U2Id = p.addUserDebugParameter("UR5_EE Y" , -1.0 , 1.0 , self.init_pos[1])
            # self.U3Id = p.addUserDebugParameter("UR5_EE Z" , 0.0 , 1.0 , self.init_pos[2])
            # self.U4Id = p.addUserDebugParameter("UR5_EE Rx" , -3.14 , 3.14 , self.init_ori[0])
            # self.U5Id = p.addUserDebugParameter("UR5_EE Ry" , -3.14 , 3.14 , self.init_ori[1])
            # self.U6Id = p.addUserDebugParameter("UR5_EE Rz" , -3.14 , 3.14 , self.init_ori[2])


        #=== Robotiq parameters
        # Base Motor2 - decrease value for clockwise rotation
        # When both base links are near the center of the palm, the rotation is about |1 rad|
        # Fingers most closed position at tendon motor = 0.12 rad
        self.f1_0 = pybullet.addUserDebugParameter("Finger1-palm" , 0 , 1.57 , 0.) 
        self.f1_1 = pybullet.addUserDebugParameter("Finger1-1" , 0 , 1.57 , 0.) 
        self.f1_2 = pybullet.addUserDebugParameter("Finger1-2" , 0 , 0.8 , 0.) 
        self.f2_0 = pybullet.addUserDebugParameter("Finger2-palm" , 0 , 1.57 , 0.) 
        self.f2_1 = pybullet.addUserDebugParameter("Finger2-1" , 0 , 1.57 , 0.) 
        self.f2_2 = pybullet.addUserDebugParameter("Finger2-2" , 0 , 0.8 , 0.) 
        self.f3_1 = pybullet.addUserDebugParameter("Finger3-1" , 0 , 1.57 , 0.) 
        self.f3_2 = pybullet.addUserDebugParameter("Finger3-2" , 0 , 0.8 , 0.) 

    
    def return_robot_states(self):
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
        ur5_values = [pybullet.getJointState(1,1)[0],pybullet.getJointState(1,2)[0],
                        pybullet.getJointState(1,3)[0],pybullet.getJointState(1,4)[0],
                        pybullet.getJointState(1,5)[0],pybullet.getJointState(1,6)[0]]
        motor_positions = [pybullet.getJointState(1,11)[0],pybullet.getJointState(1,12)[0],
                        pybullet.getJointState(1,13)[0],pybullet.getJointState(1,15)[0],pybullet.getJointState(1,16)[0],
                        pybullet.getJointState(1,17)[0],pybullet.getJointState(1,19)[0],pybullet.getJointState(1,20)[0]]
        
        return motor_positions, ur5_values 
    
import gym
from gym import spaces 
from gym.utils import seeding
import numpy as np

class MoveUr5RuthEnv(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, version):
       
        #========= Initialize environment
        if version == 'GUI':
            physicsClient = pybullet.connect(pybullet.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        elif version == 'DIRECT':
            physicsClient = pybullet.connect(pybullet.DIRECT) #non-graphical version

        
        # 
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        pybullet.setGravity(0,0,-9.81)
        pybullet.setRealTimeSimulation(0) #0: disable

        #===== Initialize plane and import robot
        FixedBase = False #if fixed no plane is imported
        if (FixedBase == False):
            floor = pybullet.loadURDF("plane.urdf")

        urdfDirectory = "/gym/envs/kelin/urdf/ur5_plus_barrett.urdf" 
        robotPos = [0, 0, 0]
        robotScale = 1
        robot = pybullet.loadURDF(urdfDirectory,
                           robotPos,
                           pybullet.getQuaternionFromEuler([0, 0, 0]),
                           useFixedBase=0,
                           globalScaling=robotScale)

        #===== Check joint information from URDF file
        jointNum = pybullet.getNumJoints(robot)
        # print('\n\n', jointNum)
        for jt in range(jointNum):
            jointInfo = pybullet.getJointInfo(robot, jt)
        # print('\n\n',jointInfo)

        #===== Save UR5 joint and link information
        ur5JointNameToID = {}
        ur5LinkNameToID = {}
        ur5_joint_names = []
        ur5RevoluteID = []

        num_of_ur5_joints = 6

        for j in range(num_of_ur5_joints+2):
            info = pybullet.getJointInfo(robot, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            ur5JointNameToID[jointName] = info[0]
            ur5LinkNameToID[info[12].decode('UTF-8')] = info[0]
            ur5RevoluteID.append(j)
            ur5_joint_names.append(jointName)



    # ===== Save Robotiq joint and link information
        Bhand_jointNameToID = {} # Dictionary of RUTH joint and respective ID
        Bhand_linkNameToID = {} # Dictionary of RUTH link and respective ID
        Bhand_revoluteID = [] # array list of ID of revolute type joints
        Bhand_rev_num = 0 # count number of revolute joints
        Bhand_joint_names = []
        for j in range(num_of_ur5_joints+2, pybullet.getNumJoints(robot)): #range 8-25
            info = pybullet.getJointInfo(robot, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            Bhand_jointNameToID[jointName] = info[0]
            Bhand_linkNameToID[info[12].decode('UTF-8')] = info[0]
            Bhand_joint_names.append(jointName)
            if (jointType == pybullet.JOINT_REVOLUTE):
                Bhand_jointNameToID[jointName] = info[0]
                Bhand_rev_num += 1
                Bhand_revoluteID.append(j)

        print(Bhand_joint_names)

    #===== Disable Collisions between links
        for link in Bhand_linkNameToID:
            pybullet.setCollisionFilterGroupMask(robot, Bhand_linkNameToID[link], 1, 0)
      
        for link in ur5LinkNameToID:
            pybullet.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    # Disable collision between environments
        pybullet.setCollisionFilterGroupMask(robot, -1, 1, 0) 
        pybullet.setCollisionFilterGroupMask(floor, -1, 1, 0)


        #======= Visualize contact points as spheres
        # contact_points = load_contact_pt(".\Contact_Points\Ruth\Ruth_Banana.npy", 1)
        # visualize_contact_pt(contact_points, robotScale)

        #=======Start simulation
        self.robot = robot
        self.RUTH_linkNameToID = Bhand_linkNameToID
        self.RUTH_jointNameToID = Bhand_jointNameToID
        self.ur5JointNameToID = ur5JointNameToID
        self.ur5_joint_names = ur5_joint_names 
        self.ur5LinkNameToID = ur5LinkNameToID

        self.pybulletDebug1 = pybulletDebug('motors', self.robot, self.ur5LinkNameToID)
        self._robotiq_init = [0. , 0. , 0. , 0. , 0, 0, 0, 0]
        self._ur5_init = [0.00017069140745350854, 1.5540618886045958, -1.5886830943681458, -1.6043969354088452, 1.5707558854894739, -0.0009489847876093962 ]
        
        self.version = version 

        # self.action_space = spaces.Box(np.array([-1.0,]*9), np.array([1.0,]*9))
        # self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.action_space = spaces.Box(np.array([-1.0,]*9), np.array([1.0,]*9))
        self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.goal_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        
        self.max_delta_action = 0.5
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, flag=True, count=0):
        r = 0
        action = np.clip(action, -1, 1) 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        action = action * self.max_delta_action 
        
        RUTH_motors, ur5_values = self.pybulletDebug1.return_robot_states() 
        RUTH_motors = action[1:] + RUTH_motors
        
        if RUTH_motors[0]>np.pi/2:
            RUTH_motors[0] = np.pi/2
            r -= 100
        if RUTH_motors[0]<0:
            RUTH_motors[0] = 0
            r -= 100
        if RUTH_motors[1]>np.pi/2:
            RUTH_motors[1] = np.pi/2
            r -= 100
        if RUTH_motors[1]<0:
            RUTH_motors[1] = 0
            r -= 100
        if RUTH_motors[2]>0.8:
            RUTH_motors[2] = 0.8
            r -= 100
        if RUTH_motors[2]<0:
            RUTH_motors[2] = 0
            r -= 100
            
        if RUTH_motors[3]>np.pi/2:
            RUTH_motors[3] = np.pi/2
            r -= 100
        if RUTH_motors[3]<0:
            RUTH_motors[3] = 0
            r -= 100    
        if RUTH_motors[4]>np.pi/2:
            RUTH_motors[4] = np.pi/2
            r -= 100    
        if RUTH_motors[4]<0:
            RUTH_motors[4] = 0
            r -= 100
        if RUTH_motors[5]>0.8:
            RUTH_motors[5] = 0.8
            r -= 100    
        if RUTH_motors[5]<0:
            RUTH_motors[5] = 0
            r -= 100
        
        if RUTH_motors[6]>np.pi/2:
            RUTH_motors[6] = np.pi/2
            r -= 100    
        if RUTH_motors[6]<0:
            RUTH_motors[6] = 0
            r -= 100
        if RUTH_motors[7]>0.8:
            RUTH_motors[7] = 0.8
            r -= 100    
        if RUTH_motors[7]<0:
            RUTH_motors[7] = 0
            r -= 100
            
        ur5_values[5] = ur5_values[5] + action[0]
        # ur5_values = self.pybulletDebug1.return_robot_states() 
        # ur5_values = action + ur5_values
        target = np.array([ur5_values[5],RUTH_motors[0],RUTH_motors[1],RUTH_motors[2],RUTH_motors[3],RUTH_motors[4]])  
        pybullet.setJointMotorControl2(self.robot, self.ur5JointNameToID['wrist_3_joint'], pybullet.POSITION_CONTROL, ur5_values[5], force=1000)
        motor_control_rtq(self.robot, self.RUTH_jointNameToID, RUTH_motors)
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          # real = np.asarray(real[3:])
          real = np.asarray(real[5:11])
          err = np.sum((real-target)*(real-target))
          count = count+1

          if err < 1e-3 or count==100:
              flag = False
        # time.sleep(1./20.)
        #print(target)
        obs = self.get_obs() 
        obs = obs[5:]
        pos = self.get_pos()
        
        for index, ur5_control in enumerate(ur5_values):
            if ur5_control>np.pi or ur5_control<-np.pi:
                r -= 100

        #======== Define reward function
        dist = []
        d = False 
        for j in range(3):
            temp = np.linalg.norm(self.get_pos()[ j*3 : (j+1)*3 ] - self.goal_b[ j*3 : (j+1)*3 ]) *1000
            dist.append(temp)
        #dist2 = np.linalg.norm(self.get_pos()[3:6] - self.goal[3:6])    *1000
        #dist3 = np.linalg.norm(self.get_pos()[6:9] - self.goal[6:9])    *1000
            
            if dist[j]<15:
                r+=100
            else:
                r-=100
        if sum(dist) <15:
            r+=1000
            d=True
        r-=sum(dist)
        distance = sum(dist)
        
        
        #if dist1 < 15 or dist2<15 or dist3<15:
        #    r += 100
           
        #elif dist1 >= 15 or dist2>=15 or dist3>=15:
        #    r -= 100
        #elif dist1 < 15 and dist2<15 and dist3<15:
        #    r+=500
        #    d=True
        
        # reward = - np.linalg.norm(self.get_pos() - self.goal) + penalty
        
        return obs, distance, d, {}  

    def reset(self, flag=True, count=0):
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, self._ur5_init)
        motor_control_rtq(self.robot, self.RUTH_jointNameToID, self._robotiq_init)
        target=np.array([0.00017069140745350854, 1.5540618886045958, -1.5886830943681458, -1.6043969354088452, 1.5707558854894739, -0.0009489847876093962,0,0,0,0,0 ])
        self.goal, idx = self.reset_goal()
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          real = np.asarray(real[:11])
          err = np.sum((real-target)*(real-target))
          count=count+1
          #print(err)
          if err < 1e-3 or count==100:
              flag = False
        pybullet.stepSimulation()
        # self.point_transform()
        state = self.get_obs() 
        return state[5:],idx
    
    def reset_goal(self, goal=None): 
        workspace = np.load('contact_points.npy')
        a = np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/contact_list.npy')
        contact_idx = choice(a)
        #np.random.randint(0,len(workspace))
        #print(contact_idx)
        goal_b = workspace[contact_idx,:]
        goal_b=goal_b+[0,0,0.1,0,0,0.1,0,0,0.1]
        
        goal_b1 = np.array([[goal_b[0]],[goal_b[1]],[goal_b[2]],[1]])
        goal_b2 = np.array([[goal_b[3]],[goal_b[4]],[goal_b[5]],[1]])
        goal_b3 = np.array([[goal_b[6]],[goal_b[7]],[goal_b[8]],[1]])
        Q = pybullet.getQuaternionFromEuler([0,np.pi/2,0])
        R = pybullet.getMatrixFromQuaternion(Q)
        #p = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[0]
        p = np.array([0.464,-0.11,0.66])
        T_eb = np.array([[R[0],R[1],R[2],p[0]],[R[3],R[4],R[5],p[1]],[R[6],R[7],R[8],p[2]],[0,0,0,1]])
        T_be = np.linalg.inv(T_eb)       
        goal_e1 = np.dot(T_be,goal_b1)
        goal_e2 = np.dot(T_be,goal_b2)
        goal_e3 = np.dot(T_be,goal_b3)
        
        goal = np.zeros(9)
        goal[0] = goal_e1[0]
        goal[1] = goal_e1[1]
        goal[2] = goal_e1[2]
        goal[3] = goal_e2[0]
        goal[4] = goal_e2[1]
        goal[5] = goal_e2[2]
        goal[6] = goal_e3[0]
        goal[7] = goal_e3[1]
        goal[8] = goal_e3[2]                                               
        # if goal is None:
        #     r1 = random.uniform(0,0.05)
        #     theta1 = random.uniform(0,2*np.pi)
        #     x1 = r1*np.cos(theta1)+0.4576751227292945
        #     y1 = r1*np.sin(theta1)-0.1098084577936703
        #     r2 = random.uniform(0,0.05)
        #     theta2 = random.uniform(0,2*np.pi)
        #     x2 = r2*np.cos(theta2)+0.4576751227292945
        #     y2 = r2*np.sin(theta2)-0.1098084577936703
        #     r3 = random.uniform(0,0.05)
        #     theta3 = random.uniform(0,2*np.pi)
        #     x3 = r3*np.cos(theta3)+0.4576751227292945
        #     y3 = r3*np.sin(theta3)-0.1098084577936703
        #     z = 0.352
        #     xyz = np.array([x1,y1,z,x2,y2,z,x3,y3,z])

        #     goal = xyz
        if goal.shape != (9,):
            print("the shape of goal should be (9,)")
        
        self.goal = goal 
        self.goal_b = goal_b
        if self.version == "GUI":
            visualize_contact_pt(self.goal_b[:3], [1,0,0,1], 2)
            visualize_contact_pt(self.goal_b[3:6], [0,1,0,1], 2)
            visualize_contact_pt(self.goal_b[6:], [0,0,1,1], 2)
        
        return self.goal,contact_idx
    
    def point_transform(self):
        p = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[0]
        a, ur_joints = self.pybulletDebug1.return_robot_states()
        ur_joints[3] = ur_joints[3] + random.uniform(-np.pi/6, np.pi/6)
        ur_joints[4] = ur_joints[4] + random.uniform(-np.pi/6, np.pi/6)
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, ur_joints)
        for i in range(50):
            pybullet.stepSimulation()
        p = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[0]
        Q = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[1]
        R = pybullet.getMatrixFromQuaternion(Q)
        T_eb = np.array([[R[0],R[1],R[2],p[0]],[R[3],R[4],R[5],p[1]],[R[6],R[7],R[8],p[2]],[0,0,0,1]])
        goal_e1 = np.array([[self.goal[0]],[self.goal[1]],[self.goal[2]],[1]])
        goal_e2 = np.array([[self.goal[3]],[self.goal[4]],[self.goal[5]],[1]])
        goal_e3 = np.array([[self.goal[6]],[self.goal[7]],[self.goal[8]],[1]])
        goal_b1 = np.dot(T_eb,goal_e1).T[0,:3]
        goal_b2 = np.dot(T_eb,goal_e2).T[0,:3]
        goal_b3 = np.dot(T_eb,goal_e3).T[0,:3]
        cp_list = np.vstack((goal_b1,goal_b2,goal_b3))
        cp12 = (cp_list[1] - cp_list[0]) / np.linalg.norm(cp_list[1] - cp_list[0])
        cp13 = (cp_list[2] - cp_list[0]) / np.linalg.norm(cp_list[2] - cp_list[0])
        cp_normal = np.cross(cp12, cp13) / np.linalg.norm(np.cross(cp12, cp13))
        if cp_normal[2] < 0 : # if normal vector is downward
          cp_normal *= -1
        goal_b1 = goal_b1 + 0.1*cp_normal
        goal_b2 = goal_b2 + 0.1*cp_normal
        goal_b3 = goal_b3 + 0.1*cp_normal
        self.goal_b = np.hstack((goal_b1,goal_b2,goal_b3))
        if self.version == "GUI":
            visualize_contact_pt(self.goal_b[:3], [1,0,0,1], 2)
            visualize_contact_pt(self.goal_b[3:6], [0,1,0,1], 2)
            visualize_contact_pt(self.goal_b[6:], [0,0,1,1], 2) 
        cp_center = np.empty([3]) #center of the 3 contact points
        for i in range (3):
          cp_center[i] = sum(cp_list[:,i])/3
        cp_center = cp_center + 0.1*cp_normal
        orientation = np.array(Q)
        self.visualize_frame(orientation,cp_center)
            
    def visualize_frame(self, orien, shape_CoM):
	    boxID = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents = [0.005,0.2,0.15])
	    visualShapeId = -1

	    bound_box = pybullet.createMultiBody(0.01, boxID, visualShapeId, [0.15,0.2,0.01])

	    # Disable Collisions with external objects
	    pybullet.setCollisionFilterGroupMask(bound_box, -1, 0, 0)
	    pybullet.changeVisualShape(bound_box, -1, rgbaColor=[0,0,1,0.2])
	    constraint = pybullet.createConstraint(parentBodyUniqueId=bound_box,
				parentLinkIndex=-1,
				childBodyUniqueId=-1,
				childLinkIndex=-1,
				jointType=pybullet.JOINT_FIXED,
				jointAxis=[0,0,0],
				parentFramePosition=[0,0,0],
				childFramePosition= shape_CoM,
                childFrameOrientation=orien)        
            
    def get_obs(self):
        RUTH_motors, ur5_values = self.pybulletDebug1.return_robot_states()
        return np.concatenate((ur5_values, RUTH_motors, self.goal)).copy()
    
    # def get_obs(self):
    #     ur5_values = self.pybulletDebug1.return_robot_states()
    #     return np.concatenate((ur5_values, self.goal)).copy()
    
    def get_pos(self):
        #===========Get final fingertip position
        # getLinkState returns: 0. CoM coordinates, 1. CoM orientation
        fingertip_state = {}
        fingertip_pos = []
        fingertip_ori = [] #orientation of CenvoM of fingertip
        fingertip_links = ['wam/bhand/finger_1/tip_link','wam/bhand/finger_2/tip_link','wam/bhand/finger_3/tip_link']
        for link_name in fingertip_links:
            fingertip_state[link_name] = pybullet.getLinkState(self.robot, self.RUTH_linkNameToID[link_name])
            fingertip_pos.append(fingertip_state[link_name][0])
            fingertip_ori.append(fingertip_state[link_name][1])
        return np.array(fingertip_pos).ravel() 
    


#    def render(self, mode='human', **kwargs):
#        return self.env.render(mode, **kwargs)

    def render_frame(self, hw=15, imagesize=(300,300), camera_id=0, heatmap=None): 
        return None  

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

#======= Automate individual joint testing
def auto_joint_test( _ur5_joint_index=None, _ur5_init = 0, \
                    _ruth_motor_index = None, _ruth_init = 0, \
                    _ur5_motor_limit=[-np.pi, np.pi], _ruth_motor_limit = [[-1.0, np.pi/2], [-np.pi/2, 1.0], [-0.5, 0.12]]):
    total_step = 100
    _RUTH_motors = [0. , 0. , 0. ]
    _ur5_values = [0. , 0. , 0. , 0. , 0. , 0. ]
    # _ruth_motor_limit = [[-1.0, np.pi/2], [-np.pi/2, 1.0], [-0.5, 0.12]]
    # _ur5_motor_limit=[-np.pi, np.pi]

    #==== Start from lower limit if motor limit is given
    if _ur5_init is None and _ruth_init is None:
        if _ur5_motor_limit is not None:
            _ur5_init = _ur5_motor_limit[0]
        if _ruth_motor_limit is not None:
            _ruth_init = _ruth_motor_limit[0]

    #==== Test given individual joint
    if _ur5_joint_index is not None:
        _ur5_values[_ur5_joint_index] = _ur5_init
        for i in range(total_step):
            _ur5_values[_ur5_joint_index] = _ur5_values[_ur5_joint_index] + i*_ur5_motor_limit[1]/total_step

    if _ruth_motor_index is not None:
        _RUTH_motors[_ruth_motor_index] = _ruth_init
        for j in range(total_step):
            _RUTH_motors[_ruth_motor_index] = _RUTH_motors[_ruth_motor_index] + j*_ruth_motor_limit[_ruth_motor_index][1]/total_step


    return _RUTH_motors, _ur5_values


if __name__ == "__main__":
    import gym
    ro = gym.make("kelin-v0", version="DIRECT")
    sta = ro.reset() 
    ob = ro.get_obs()
    print(ob)
    for i in range(500):
        a = ro.action_space.sample() #* 0 + 0.1
        o, r, d, _ = ro.step(a)
        # print(o)
        if i % 20 == 0:
            ro.reset() 
        time.sleep(1./20.)
    sta = ro.reset() 
    ob = ro.get_obs()
    print(ob)
    pass 