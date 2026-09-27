import os, sys
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
#------ Some custom library
import pybullet_data
# from center_point_1 import compute_base_rotation

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)



#====== Set up physics client and environment
def env_setup(mode='GUI'):
    if mode=='GUI':
        physics_client = p.connect(p.GUI, options="--width=1200 --height = 800--mp4=test2.mp4")  # for p.DIRECT for non-graphical version
    else:
        physics_client = p.connect(p.DIRECT) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)

    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    p.setCollisionFilterGroupMask(floor, -1, 1, 0)
  

def set_ur5_position(robot, ur5JointNameToID, ur5_joints):  # set positions of ur5 joints
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, ur5_joints[0],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, ur5_joints[1],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, ur5_joints[2], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, ur5_joints[3], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, ur5_joints[4], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, ur5_joints[5], force=1000)


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

def motor_control_rtq(_robot, _RTQ_jointNameToID, _RTQ_values):
    RTQ_joints = ['finger_1_joint_1', 'palm_finger_1_joint', 'finger_2_joint_1', 'palm_finger_2_joint', 'finger_middle_joint_1']
    for index, rtq in enumerate(_RTQ_values):
        p.setJointMotorControl2(_robot, _RTQ_jointNameToID[RTQ_joints[index]], p.POSITION_CONTROL, rtq, force=1000)

#======= get robot joint and link info
def get_robot_info(robot):
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
    for j in range(num_of_ur5_joints+2,p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        jointNameToID[jointName] = info[0]
        linkNameToID[info[12].decode('UTF-8')] = info[0]
        revoluteID.append(j)
    # Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 


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

    # Form closed loop for RUTH 5 bar linkage
    con1 = p.createConstraint(parentBodyUniqueId=robot,
                       parentLinkIndex=linkNameToID['Link_2'],
                       childBodyUniqueId=robot,
                       childLinkIndex=linkNameToID['Link_4'],
                       jointType=p.JOINT_POINT2POINT,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
                       childFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]])

    return ur5JointNameToID, ur5LinkNameToID, jointNameToID, linkNameToID



#======= Visualize points as spheres
def visualise_point(point_pos, rgba=[1,0,0,1], _robotScale=1):
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * _robotScale)
    visualShapeId = -1

    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, point_pos)

    # Disable Collisions between links
    p.setCollisionFilterGroupMask(sphereA, -1, 0, 0)
    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=point_pos)

    p.changeVisualShape(sphereA, -1, rgbaColor=rgba)



def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



#============ Rearrange contact points
# Define contact points (CP1 is middle one, CP3 left, CP2 right) 
def point_rearrange(_contactPoint_ls):
    CP1a = _contactPoint_ls[0]
    CP2a = _contactPoint_ls[1]
    CP3a = _contactPoint_ls[2]
    sep12 = np.linalg.norm(CP2a-CP1a)
    sep13 = np.linalg.norm(CP3a-CP1a)
    sep23 = np.linalg.norm(CP3a-CP2a)

    side_diff = np.array([ sep12-sep13, sep12-sep23, sep13-sep23 ])
    vertex = np.argmin(side_diff) # The two sides with least difference is the leg of isoceles triangle

    # Swap order of points so vertex is always the first in the list
    new_cp_list = np.copy(_contactPoint_ls)

    if vertex == 1:
        new_cp_list[0,:] = CP2a
    elif vertex ==2:
        new_cp_list[0,:] = CP3a
    new_cp_list[vertex,:] = CP1a

    return new_cp_list


#======= Load contact points. Input file path, Int number indicating which set to be used
def load_contact_pt(obj_pos, contact_point_fpath, set_index = 0):
    cp = np.load(contact_point_fpath)
    cp_chosen = cp[set_index]
    cp_list = np.array(cp_chosen).reshape(-1,3) #save each point's xyz coordinate into nested array. 
    cp_list[:,-1] = -cp_list[:,-1] 

    obj_pos_0 = np.array([0.3, 0, 0.01]) #contact points originally calculated based on this position
    obj_pos_offset = obj_pos - obj_pos_0

    cp_list += obj_pos_offset
    new_cp_list = point_rearrange(cp_list)
    return new_cp_list


#===========Get final fingertip position
# getLinkState returns: 0. CoM coordinates, 1. CoM orientation
def get_fingertip_state():
    fingertip_state = {}
    fingertip_pos = []
    fingertip_ori = [] #orientation of CoM of fingertip
    fingertip_links = ['Phal_3C', 'Phal_2C', 'Phal_1C'] #3C: middle fingertip, 1C: left finger, 2C: right finger
    for link_name in fingertip_links:
        fingertip_state[link_name] = p.getLinkState(robot, RUTH_linkNameToID[link_name])
        fingertip_pos.append(fingertip_state[link_name][0])
        fingertip_ori.append(fingertip_state[link_name][1])

    return fingertip_pos





def calc_target_pos(cp_list, distance = 0.3):
    cp_center = np.empty([1,3]) #center of the 3 contact points
    for i in range (3):
        cp_center[0,i] = sum(cp_list[:,i])/3

    #===== Normal vector to the plane of 3 given points
    cp12 = (cp_list[1] - cp_list[0]) / np.linalg.norm(cp_list[1] - cp_list[0])
    cp13 = (cp_list[2] - cp_list[0]) / np.linalg.norm(cp_list[2] - cp_list[0])
    cp_normal = np.cross(cp12, cp13) / np.linalg.norm(np.cross(cp12, cp13))
    if cp_normal[2] < 0 : # if normal vector is downward
        cp_normal *= -1

    # Calculate target UR5 end effector position
    target_pos_ = np.array(cp_center) + distance * cp_normal # vertical axis is x at the end effector
    target_pos = target_pos_[0]

    def calc_target_ori(cp_list):
        #====== Get ee_link quaternion
        # ur5_LinkNameToID = get_robot_info(robot)[1]
        # end_link_state = p.getLinkState(robot, ur5_LinkNameToID['ee_link'])
        # ee_quat = end_link_state[1]
        #ee_quat = np.array([0.0013869110, 0.6754627730, -0.000763620, 0.7373923])
        # Calculate target approach orientation
        x_ax = (cp_list[0]-cp_list[1])/np.linalg.norm(cp_list[0]-cp_list[1])
        z_ax = np.cross(x_ax, cp_list[0]-cp_list[2]) / np.linalg.norm( np.cross(x_ax, cp_list[0]-cp_list[2]) )
        z_ax = cp_normal
        y_ax = np.cross(z_ax,x_ax)
        R_CP = np.column_stack((x_ax, y_ax, z_ax))
        R_CP_xyz = np.column_stack((-R_CP[:, 2], R_CP[:, 1], R_CP[:, 0]))
        
        #ee_rot_matrix = np.asarray(p.getMatrixFromQuaternion(ee_quat)).reshape(-1,3) # rotation matrix to get from world coordinates to ee_link
       # ee_rot_matrix[:,1] *=-1
        target_rot_matrix = R_CP_xyz
        target_ori = R.from_matrix(target_rot_matrix).as_quat()

        return target_ori
    target_ori = calc_target_ori(cp_list)
    

    return cp_normal, target_pos, target_ori





