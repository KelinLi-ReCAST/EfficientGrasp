# Set up UR5 and Robotiq for simulation
# This file simply tests UR5 and RUTH movement control
# 2021-Oct-23 @Xian Zhang
#=========================READ ME==========================
# When calling main(mode, timer), first parameter takes 0 or 1 for control mode:
# (0) for manual motor control, (1) to start automated testing
# timer parameter is how long you want the simulation to run for.
# Edit automated testing in function auto_joint_test(), by specifying which joints to start auto testing
# Final UR5 and RUTH joint states, final fingertip position and orientation are saved in array output_motor_fingertip




import pybullet as p
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import pandas as pd


#======= Load contact points
def load_contact_pt(contact_point_fpath, row_index):
    cp = np.load(contact_point_fpath)  #npy file [10x9] with 10 sets of possible contact points combinations (3 points with xyz)
    cp_chosen = cp[row_index]

    cp_list = np.array(cp_chosen).reshape(-1,3)
    return(cp_list)

#======= Visualize contact points as spheres
def visualize_contact_pt(point_pos, rgba=[1,0,0,1], _robotScale=1):
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * _robotScale)
    visualShapeId = -1

    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, point_pos)

    # Disable Collisions between links
    p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=point_pos)

    p.changeVisualShape(sphereA, -1, rgbaColor=rgba)

#===== Joint motor control on UR5
def motor_control_ur5(_robot, _ur5JointNameToID, _ur5_joint_name, _ur5_values):
    for index, ur5_control in enumerate(_ur5_values):   #ur5_control = ur5_values[index]
        p.setJointMotorControl2(_robot, _ur5JointNameToID[_ur5_joint_name[index+1]], p.POSITION_CONTROL, _ur5_values[index], force=1000)

#===== Joint motor control on Robotiq
def motor_control_Bhand(_robot, _Bhand_jointNameToID, _Bhand_values):
    Bhand_joints = ['wam/bhand/finger_1/prox_joint', 'wam/bhand/finger_1/med_joint', 'wam/bhand/finger_1/dist_joint', 'wam/bhand/finger_2/prox_joint', 'wam/bhand/finger_2/med_joint', 'wam/bhand/finger_2/dist_joint', 'wam/bhand/finger_3/med_joint', 'wam/bhand/finger_3/dist_joint']
    for index, Bhand in enumerate(_Bhand_values):
        p.setJointMotorControl2(_robot, _Bhand_jointNameToID[Bhand_joints[index]], p.POSITION_CONTROL, Bhand, force=1000)
  

  
#======= Config parameter panel
class pybulletDebug:
    def __init__(self, control_type, robot, ur5LinkNameToID):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw=90
        self.cpitch=-7
        self.cdist=0.66
        time.sleep(0.5)
        self.init_state = p.getLinkState(robot, ur5LinkNameToID['ee_link'])
        self.init_pos = np.array(self.init_state[0])
        self.init_ori = np.array(p.getEulerFromQuaternion(self.init_state[1]))
        self.ur5_values = np.zeros(5)

       
        if control_type == 'motors':
            self.control_type = 'motors'
            # self.U1Id = p.addUserDebugParameter("UR5 Shoulder Pan" , -3.14 , 3.14 , 0.)
            # self.U2Id = p.addUserDebugParameter("UR5 Shoulder Lift" , -3.14 , 3.14 , 0.)
            # self.U3Id = p.addUserDebugParameter("UR5 Elbow" , -3.14 , 3.14 , 0.)
            # self.U4Id = p.addUserDebugParameter("UR5 Wrist 1" , -3.14 , 3.14 , 0.)
            # self.U5Id = p.addUserDebugParameter("UR5 Wrist 2" , -3.14 , 3.14 , 0.)
            # self.U6Id = p.addUserDebugParameter("UR5 Wrist 3" , -3.14 , 3.14 , 0.)
    
        if control_type == 'xyz':
            self.control_type = 'xyz'
        #=== End effector pose xyz and orientation
            # self.U1Id = p.addUserDebugParameter("UR5_EE X" , -1.0 , 1.0 , self.init_pos[0]) 
            # self.U2Id = p.addUserDebugParameter("UR5_EE Y" , -1.0 , 1.0 , self.init_pos[1])
            # self.U3Id = p.addUserDebugParameter("UR5_EE Z" , 0.0 , 1.0 , self.init_pos[2])
            # self.U4Id = p.addUserDebugParameter("UR5_EE Rx" , -3.14 , 3.14 , self.init_ori[0])
            # self.U5Id = p.addUserDebugParameter("UR5_EE Ry" , -3.14 , 3.14 , self.init_ori[1])
            # self.U6Id = p.addUserDebugParameter("UR5_EE Rz" , -3.14 , 3.14 , self.init_ori[2])


        #=== RUTH parameters
        # Base Motor2 - decrease value for clockwise rotation
        # When both base links are near the center of the palm, the rotation is about |1 rad|
        # Fingers most closed position at tendon motor = 0.12 rad
        self.f1_0 = p.addUserDebugParameter("Finger1-palm" , 0 , 1.57 , 0.) 
        self.f1_1 = p.addUserDebugParameter("Finger1-1" , 0 , 1.57 , 0.) 
        self.f1_2 = p.addUserDebugParameter("Finger1-2" , 0 , 0.8 , 0.) 
        self.f2_0 = p.addUserDebugParameter("Finger2-palm" , 0 , 1.57 , 0.) 
        self.f2_1 = p.addUserDebugParameter("Finger2-1" , 0 , 1.57 , 0.) 
        self.f2_2 = p.addUserDebugParameter("Finger2-2" , 0 , 0.8 , 0.) 
        self.f3_1 = p.addUserDebugParameter("Finger3-1" , 0 , 1.57 , 0.) 
        self.f3_2 = p.addUserDebugParameter("Finger3-2" , 0 , 0.8 , 0.) 
    
    def return_robot_states(self):

        Bhand_finger_values = np.array([p.readUserDebugParameter(self.f1_0), p.readUserDebugParameter(self.f1_1), p.readUserDebugParameter(self.f1_2), p.readUserDebugParameter(self.f2_0), p.readUserDebugParameter(self.f2_1), p.readUserDebugParameter(self.f2_2), p.readUserDebugParameter(self.f3_1), p.readUserDebugParameter(self.f3_2)]) 
        # self.ur5_values = np.array([p.readUserDebugParameter(self.U1Id),p.readUserDebugParameter(self.U2Id),p.readUserDebugParameter(self.U3Id),p.readUserDebugParameter(self.U4Id),p.readUserDebugParameter(self.U5Id),p.readUserDebugParameter(self.U6Id)])
        
        return Bhand_finger_values, self.ur5_values

#======= Automate individual joint testing
def auto_joint_test( _ur5_joint_index = None, _ur5_init = 0, \
                    _ruth_motor_index = None, _ruth_init = 0, \
                    _ur5_motor_limit = [-np.pi, np.pi], _ruth_motor_limit = [[-1.0, np.pi/2], [-np.pi/2, 1.0], [-0.5, 0.12]]):
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


#==================================MAIN()=================================
def main(mode, timer):
    #========= Initialize environment
    physicsClient = p.connect(p.GUI)
    # physicsClient = p.connect(p.DIRECT) #non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0) #0: disable

    #===== Initialize plane and import robot
    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    urdfDirectory = "./urdf/ur5_plus_barrett.urdf"
    robotPos = [0, 0, 0]
    robotScale = 1
    robot = p.loadURDF(urdfDirectory,
                        robotPos,
                        p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=0,
                        globalScaling=robotScale)

    #===== Check joint information from URDF file
    jointNum = p.getNumJoints(robot)
    # print('\n\n', jointNum)
    for jt in range(jointNum):
        jointInfo = p.getJointInfo(robot, jt)
        # print('\n\n',jointInfo)

    #===== Save UR5 joint and link information
    ur5JointNameToID = {}
    ur5LinkNameToID = {}
    ur5_joint_names = []
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
        ur5_joint_names.append(jointName)



    # ===== Save Robotiq joint and link information
    Bhand_jointNameToID = {} # Dictionary of RUTH joint and respective ID
    Bhand_linkNameToID = {} # Dictionary of RUTH link and respective ID
    Bhand_revoluteID = [] # array list of ID of revolute type joints
    Bhand_rev_num = 0 # count number of revolute joints
    Bhand_joint_names = []
    for j in range(num_of_ur5_joints+2, p.getNumJoints(robot)): #range 8-25
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        Bhand_jointNameToID[jointName] = info[0]
        Bhand_linkNameToID[info[12].decode('UTF-8')] = info[0]
        Bhand_joint_names.append(jointName)
        if (jointType == p.JOINT_REVOLUTE):
            Bhand_jointNameToID[jointName] = info[0]
            Bhand_rev_num += 1
            Bhand_revoluteID.append(j)

    print(Bhand_joint_names)

    #===== Disable Collisions between links
    for link in Bhand_linkNameToID:
        p.setCollisionFilterGroupMask(robot, Bhand_linkNameToID[link], 1, 0)
    
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    # Disable collision between environments
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0)



    #======= Visualize contact points as spheres
    # contact_points = load_contact_pt(".\Contact_Points\Ruth\Ruth_Banana.npy", 1)
    # visualize_contact_pt(contact_points, robotScale)

    #=======Start simulation
    pybulletDebug1 = pybulletDebug('motors',robot,ur5LinkNameToID)
    _RUTH_init = [0. , 0. , 0. ]
    _ur5_init = [0. , 0. , 0. , 0. , 0. , 0. ]
    # timer = 200
    # mode = 1

    # for i in range (timer):    
    while True:
        if mode == 0: #User input motor values from GUI
            Bhand_motors, ur5_values = pybulletDebug1.return_robot_states() 
            motor_control_ur5(robot, ur5JointNameToID, ur5_joint_names, ur5_values)
            motor_control_Bhand(robot, Bhand_jointNameToID, Bhand_motors)
            p.stepSimulation()
            time.sleep(1./20.)

        elif mode ==1: #Automate motor controls
            RUTH_motors, ur5_values = auto_joint_test(_ur5_joint_index= 0, _ur5_init = -1.57, _ruth_motor_index = None, _ruth_init = 0)

            if i<timer/2 :
                #indicate which joints to test by indicating '_ur5_joint_index' and '_ruth_motor_index'
                motor_control_ur5(robot, ur5JointNameToID,ur5_joint_names, ur5_values)
                p.stepSimulation()
                time.sleep(1./20.)

            #return to init state
            else:
                motor_control_ur5(robot, ur5JointNameToID,ur5_joint_names, _ur5_init)
                p.stepSimulation()
                time.sleep(1./20.)


        


        #===========Get final fingertip position
        # getLinkState returns: 0. CoM coordinates, 1. CoM orientation
        fingertip_state = {}
        fingertip_pos = []
        fingertip_ori = [] #orientation of CoM of fingertip
        # fingertip_links = ['Phal_1C','Phal_2C','Phal_3C']
        # for link_name in fingertip_links:
        #     fingertip_state[link_name] = p.getLinkState(robot, RUTH_linkNameToID[link_name])
        #     fingertip_pos.append(fingertip_state[link_name][0])
        #     fingertip_ori.append(fingertip_state[link_name][1])

        # visualize_contact_pt(fingertip_pos[0], rgba=[1,0,0,1],  _robotScale=robotScale)
        # visualize_contact_pt(fingertip_pos[1], rgba=[0,1,0,1], _robotScale=robotScale)
        # visualize_contact_pt(fingertip_pos[2], rgba=[0,0,1,1], _robotScale=robotScale)
        


        # output_motor_fingertip = [RUTH_motors , ur5_values, fingertip_pos, fingertip_ori]    
        
        # print(output_motor_fingertip)
 
    

if __name__ == "__main__":
    main(0, 10000)