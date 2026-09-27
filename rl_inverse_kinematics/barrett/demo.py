import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pybullet as p
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics import ruthModel
from pybullet_object_models import ycb_objects
import open3d as o3d
import matplotlib.pyplot as plt
from show3d_balls import showpoints

import move_ur

import RL_gripper

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
        motor_positions = [p.getJointState(1,11)[0],p.getJointState(1,12)[0],
                        p.getJointState(1,13)[0],p.getJointState(1,15)[0],p.getJointState(1,16)[0],
                        p.getJointState(1,17)[0],p.getJointState(1,19)[0],p.getJointState(1,20)[0]]
        
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


def visualise_contact_points(contact_point1, contact_point2, contact_point3):  # Create spheres to visualise positions of contact points
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
    visualShapeId = -1
    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point1)
    sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point2)
    sphereC = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point3)
    p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    p.setCollisionFilterGroupMask(sphereB, -1, 1, 0)
    p.setCollisionFilterGroupMask(sphereC, -1, 1, 0)
    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
              parentLinkIndex=-1,
              childBodyUniqueId=-1,
              childLinkIndex=-1,
              jointType=p.JOINT_FIXED,
              jointAxis=[0, 0, 0],
              parentFramePosition=[0, 0, 0],
              childFramePosition=contact_point1)
    con3 = p.createConstraint(parentBodyUniqueId=sphereB,
              parentLinkIndex=-1,
              childBodyUniqueId=-1,
              childLinkIndex=-1,
              jointType=p.JOINT_FIXED,
              jointAxis=[0, 0, 0],
              parentFramePosition=[0, 0, 0],
              childFramePosition=contact_point2)
    con4 = p.createConstraint(parentBodyUniqueId=sphereC,
              parentLinkIndex=-1,
              childBodyUniqueId=-1,
              childLinkIndex=-1,
              jointType=p.JOINT_FIXED,
              jointAxis=[0, 0, 0],
              parentFramePosition=[0, 0, 0],
              childFramePosition=contact_point3)


def set_ur5_position(robot, ur5JointNameToID, ur5_joints):  # set positions of ur5 joints
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, ur5_joints[0],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, ur5_joints[1],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, ur5_joints[2], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, ur5_joints[3], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, ur5_joints[4], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, ur5_joints[5], force=1000)


def motor_control_rtq(_robot, _Bhand_jointNameToID, _Bhand_values):
    Bhand_joints = ['wam/bhand/finger_1/prox_joint', 'wam/bhand/finger_1/med_joint', 'wam/bhand/finger_1/dist_joint', 'wam/bhand/finger_2/prox_joint', 'wam/bhand/finger_2/med_joint', 'wam/bhand/finger_2/dist_joint', 'wam/bhand/finger_3/med_joint', 'wam/bhand/finger_3/dist_joint']
    for index, Bhand in enumerate(_Bhand_values):
        pybullet.setJointMotorControl2(_robot, _Bhand_jointNameToID[Bhand_joints[index]], pybullet.POSITION_CONTROL, Bhand, force=1000)



def main(obj_pos, obj_ori, obj_size, point_nos, depth, rl_step, obj_name,gripper):

    if obj_ori == '000':
        obj_ori1 = p.getQuaternionFromEuler([0,0,0])
    elif obj_ori == 'pi00':
        obj_ori1 = p.getQuaternionFromEuler([np.pi/2,0,0])
    elif obj_ori == 'pi0pi':
        obj_ori1 = p.getQuaternionFromEuler([np.pi/2,0,np.pi/2])
    elif obj_ori == 'pipipi':
        obj_ori1 = p.getQuaternionFromEuler([np.pi/2,np.pi/2,np.pi/2])
    physics_client = p.connect(p.GUI)  # for p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(cameraDistance=1.2,cameraYaw=45,cameraPitch=-40,cameraTargetPosition=[0.5,0,0])
    urdf_directory = "/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/gym/envs/kelin/urdf/ur5_plus_barrett.urdf"
    # urdf_root = pybullet_data.getDataPath()

    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    robotPos = [0, 0., 0]
    robotScale = 1
    robot = p.loadURDF(urdf_directory,
                       robotPos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=1,
                       globalScaling=robotScale)

    flags = p.URDF_USE_INERTIA_FROM_FILE
    # obj_pos = [0.5, 0, 0.01]
    # obj_ori = p.getQuaternionFromEuler([np.pi/2,np.pi/2,np.pi/2])
    # obj_size = 1
    # point_nos =3
    # depth = 0.24
    # rl_step = 1
    # obj_name = 'YcbMustardBottle'
    contact_points = np.load(os.path.join('/home/kelin/Downloads/contact_points_U','BH_'+'Ycb'+obj_name+'_'+obj_ori+'.npy'))
    
#    p.setCollisionFilterGroupMask(obj, -1, 1, 0)  # Disable object collisions for now
    
    
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

    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * robotScale)
    visualShapeId = -1

    # Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0)


    p.stepSimulation()
    time.sleep(1./240.)


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

    # Loop just to initialise robot
    for i in range(1000):
        targetPos1 = -i*np.pi/100000
        targetPos2 = i*np.pi/100000
        set_ur5_position(robot, ur5JointNameToID, [0, np.pi/2, 0, 0, 0, 0])
        p.stepSimulation()

    i = 0
    approach_step_count = 0
    ur5_values = [0]*6
    finger_position = 0
    flag = 1
    ruth_init = [0,0,-0.41]
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/kelin/workspace_kelin/RAL-IROS2022/unigrasp_videos/"+gripper+"_"+obj_name+'_'+obj_ori+".mp4")
    agent = RL_gripper.run_policy()
    
    while i<=1000:

        # Step1: Capture object RGBD image and transform into pointcloud
        if i == 0:
            # flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            # width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            #     width=image_width,
            #     height=image_height,
            #     viewMatrix=viewMatrix,
            #     projectionMatrix=projectionMatrix, flags=flags)
            # depthImgb = far * near / (far - (far-near)*depthImg)
            # rgbImg2 = o3d.geometry.Image((np.array(rgbImg)[:, :, :3]).astype(np.uint8))
            # depthImg2 = o3d.geometry.Image(depthImgb)
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbImg2, depthImg2)
        
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(image_width, image_height, f, f, image_width/2, image_height/2))

            # pcd_points = np.asarray(pcd.points)
            # ii = 0
            # pcd2 = np.empty((0, 3))
            # rgbd_acc = np.zeros((image_width, image_height))
            # print(np.asarray(rgbImg).shape)
            # rgb_test = np.zeros((image_width, image_height, 3))
            # for j in range(image_width):
            #     for k in range(image_height):
            #         rgbd_acc[j][k] = ii/image_width**2
            #         ii += 1
            #         if np.asarray(segImg[j][k]) == obj:
            #             rgb_test[j][k][:] = 255
            #             pcd2 = np.vstack((pcd2, pcd_points[ii,:]))  # Collect point cloud oordinates which correspond to object

            # pcd3 = np.vstack((np.transpose(-pcd2*1000), np.ones(len(pcd2))))  # Homogenous coordinates and scale up by depth scale (1000)
            # view_matrix2 = np.transpose(np.asarray(viewMatrix).reshape((4,4)))
            # pcd4 = np.matmul(np.linalg.inv(view_matrix2), pcd3)  # convert from point cloud coordinates to world coordinates
            # pcd4[1, :] = -pcd4[1, :]
            # pcd5 = o3d.geometry.PointCloud()
            # pcd5.points = o3d.utility.Vector3dVector(pcd4.T[:,:3])

            # sample_rate = int(np.floor(pcd4.shape[1]/2048))
            # pcd6 = o3d.geometry.PointCloud.uniform_down_sample(pcd5, sample_rate)

            # sample_rate = 2048/np.asarray(pcd6.points).shape[0]
            # pcd6 = pcd6.random_down_sample(sample_rate)
            # #o3d.visualization.draw_geometries([pcd6])  #  Show point cloud of object
            # showpoints(np.asarray(pcd6.points),ballradius=6,freezerot=False)
            # np.save(os.path.join('/home/kelin/workspace_kelin/RL_IK_2/obj_pc',obj_name+'.npy'),np.asarray(pcd6.points))
            
            # Step 2: PSSN generate contact points on the object
            # contact_point1 = np.array([0.5,0.47,0.043])  # Test points for banana
            # contact_point2 = np.array([0.515,0.486,0.044])
            # contact_point3 = np.array([0.505,0.525,0.029])
            # contact_point2 = np.array([0.514,0.024,0.039])
            # contact_point1 = np.array([0.517,0.01,0.039])
            # contact_point3 = np.array([0.46,-0.024,0.039])
           
            pp = np.array([0,0,0])
            contact_point3 = contact_points[point_nos,:3]
            contact_point1 = contact_points[point_nos,3:6]
            contact_point2 = contact_points[point_nos,6:9]
            # pc = np.asarray(pcd6.points)
            # pc = np.vstack((pc,contact_point3,contact_point1,contact_point2))
            # pc_color = np.zeros((2051,3))
            # for ii in range(2048):
            #     for jj in range(3):
            #         pc_color[ii,jj] = 255
            # pc_color[2048,0] = 255
            # pc_color[2049,1] = 255
            # pc_color[2050,2] = 255
            # showpoints(pc,c_gt=pc_color, ballradius=6,freezerot=False)
            # move_ur.visualise_point(contact_point1+pp,rgba=[1,0,0,1])
            # move_ur.visualise_point(contact_point2+pp,rgba=[0,1,0,1])
            # move_ur.visualise_point(contact_point3+pp,rgba=[0,0,1,1])
            
            # Step 3: Move ur5 to desired position and orientation (ee perpendicular to plane)
        if i > 0 and i <200:
            normal, pos, ori = move_ur.calc_target_pos(np.vstack((contact_point1,contact_point2,contact_point3)))
            move_ur.visualise_point(pos,rgba=[0.5,0.5,0.5,1])
            ik = p.calculateInverseKinematics(robot, ur5LinkNameToID['ee_link'], pos, ori)
            ur_pos = list(ik[:6])
            move_ur.set_ur5_position(robot, ur5JointNameToID, ur_pos)
            move_ur.motor_control_rtq(robot, jointNameToID, ruth_init)
            p.stepSimulation()

            # Step 4: RL policy to find gripper configuration
        if i == 200:
            obj = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'Ycb'+obj_name, "model.urdf"), obj_pos, baseOrientation=obj_ori1, useFixedBase=0, flags=flags, globalScaling=obj_size)
            Q = p.getLinkState(robot,ur5LinkNameToID['ee_link'])[1]
            R = p.getMatrixFromQuaternion(Q)
            T_eb = np.array([[R[0],R[1],R[2],pos[0]],[R[3],R[4],R[5],pos[1]],[R[6],R[7],R[8],pos[2]],[0,0,0,1]])
            T_be = np.linalg.inv(T_eb) 
            cp1 = np.dot(T_be,np.append(contact_point1,1))
            cp2 = np.dot(T_be,np.append(contact_point2,1))
            cp3 = np.dot(T_be,np.append(contact_point3,1))
            for j in range(rl_step):
                ruth, ur = return_robot_states()
                obs = np.concatenate((ur, ruth, np.hstack((cp1[:3],cp2[:3],cp3[:3])))).copy()[5:]
                
                a = agent.select_action(obs)
             
                ur_pos[5] = ur[5] + a[0]
                ruth[0] = ruth[0] + a[1]
                ruth[1] = ruth[1] + a[2]
                ruth[2] = ruth[2] + a[3]
                ruth[3] = ruth[3] + a[4]
                ruth[4] = ruth[4] + a[5]
                ruth[5] = ruth[5] + a[6]
                ruth[6] = ruth[6] + a[7]
                ruth[7] = ruth[7] + a[8]
                print(ur_pos[5])
                move_ur.set_ur5_position(robot, ur5JointNameToID, ur_pos)
                move_ur.motor_control_rtq(robot, jointNameToID, ruth)
                for k in range(50):
                  p.stepSimulation()
                  time.sleep(1/20)
            move_ur.motor_control_rtq(robot, jointNameToID, [ruth[0],ruth[1],ruth[2],ruth[3],ruth[4],ruth[5],ruth[6],ruth[7]])
            for k in range(50):
               p.stepSimulation()
            move_ur.motor_control_rtq(robot, jointNameToID, [ruth[0],0,0,ruth[3],0,0,0,0])
            time.sleep(1/20)
              
           # Step 5: Grasp
        if i == 201:
            ori_new = p.getLinkState(robot,ur5LinkNameToID['ee_link'])[1]
            ppp = pos-normal*depth
            ik = p.calculateInverseKinematics(robot, ur5LinkNameToID['ee_link'], ppp, ori_new)
            ur_pos = list(ik[:6])
            move_ur.set_ur5_position(robot, ur5JointNameToID, ur_pos)
            for k in range(50):
              p.stepSimulation()
              time.sleep(1/20)
            move_ur.motor_control_rtq(robot, jointNameToID, [ruth[0],np.pi/2,0.8,ruth[3],np.pi/2,0.8,np.pi/2,0.8])
            for k in range(50):
              p.stepSimulation()
              time.sleep(1/20)
            ppp1 = pos+normal*0.15
            ik = p.calculateInverseKinematics(robot, ur5LinkNameToID['ee_link'], ppp1, ori)
            ur_pos = list(ik[:6])
            move_ur.set_ur5_position(robot, ur5JointNameToID, ur_pos)
            for k in range(50):
              p.stepSimulation()
              time.sleep(1/10)
              
        time.sleep(1./240.)
        i = i+1
    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()



