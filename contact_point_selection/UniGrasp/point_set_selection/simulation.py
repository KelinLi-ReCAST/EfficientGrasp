#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 00:38:39 2021

@author: kelin
"""

import sys
import os

# sys.path.append('/home/kelin/Downloads/Pybullet/RUTH_Gripper')
# import grasping_with_RUTH2
import numpy as np
import unigrasp
from scipy import io

mat=io.loadmat('/home/robin-lab/Kelin/UniGrasp-master/data/ObjectPointClouds/11.mat')
ROOT_DIR = "/home/robin-lab/Kelin/UniGrasp-master/data/ObjectPointClouds"
DIR = "/home/robin-lab/Kelin/UniGrasp-master/data/ObjectPointClouds/"
unigrasp.restore_stage3(220)
obj_pc_file_list = [line for line in os.listdir(ROOT_DIR) if line.startswith("Ycb")]
obj_pc = mat['ppp']
# np.load("/home/robin-lab/Kelin/UniGrasp-master/data/ObjectPointClouds/11.npy")
[p1,p2,p3] = unigrasp.test(0,obj_pc)
# for idx, env_i in enumerate(obj_pc_file_list):
#     obj_dir = os.path.join(DIR,env_i)
#     save_dir = obj_dir[:61]+"Robotiq_"+obj_dir[64:(len(obj_dir)-15)]+".npy"
#     obj_pc = np.load(obj_dir)
#     [p1,p2,p3] = unigrasp.test(0,obj_pc)
#     data=np.zeros((10,9))
#     data[:,:3]=p1
#     data[:,3:6]=p2
#     data[:,6:9]=p3
#     np.save(save_dir,data)
# p1 = np.array([19.7367,22.6189,-77.5134])
# p3 = np.array([-3.55916,29.8302,-47.3982])
# p2 = np.array([-25.9376,15.1235,-51.9902])
# p1 = np.array([26.0083,23.1629,-23.7707])
# p2 = np.array([12.5613,-18.8186,-71.8465])
# p3 = np.array([-12.3196,-32.7359,-48.3075])

# p1 = p1/500
# p1[2]=-p1[2]
# p2 = p2/500
# p2[2]=-p2[2]
# p3 = p3/500
# p3[2]=-p3[2]

# grasping_with_RUTH2.main(p1,p2,p3)