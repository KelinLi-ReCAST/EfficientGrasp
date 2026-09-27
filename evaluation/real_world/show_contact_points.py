#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:22:07 2022

@author: kelin
"""

from show3d_balls import showpoints
import numpy as np
import os

obj_name = 'drill-lay'
pc = np.load(os.path.join('/home/kelin/Downloads/revision_sparse_real_pc','Ycb'+obj_name+'.npy'))
contact_points = np.load(os.path.join('/home/kelin/Downloads/contact_points_revision','Ruth_Ycb'+obj_name+'.npy'))
pc = pc/1000
contact_point1 = contact_points[0,:3]/1000+np.array([-0.02,-0.023,0.01])
contact_point2 = contact_points[0,3:6]/1000+np.array([0.001,0.001,0.001])
contact_point3 = contact_points[0,6:9]/1000+np.array([0.04,-0.02,-0.005])

pc = np.vstack((pc,contact_point1,contact_point2,contact_point3))
pc_color = np.zeros((2051,3))
for ii in range(2048):
   for jj in range(3):
       pc_color[ii,jj] = 255
pc_color[2048,0] = 255
pc_color[2049,1] = 255
pc_color[2050,2] = 255
showpoints(pc,c_gt=pc_color, ballradius=6,freezerot=False)



############
#banana [0.001,0.001,0.001] [0.001,0.001,0.001] [-0.045,+0.015,-0.005]
#bowl [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#bowl-bottom [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#clip-L [0.01,-0.01,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#clip-M [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#drrill-lay [-0.02,-0.023,0.01] [0.001,0.001,0.001] [0.04,-0.02,-0.005]
#drill-stand 
#football [0.025,-0.1,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#mug [-0.005,0.04,0.003] [0.001,0.001,0.001] [0.001,0.001,0.001]
#mug-bottom [0.001,0.001,0.001] [-0.03,0.02,-0.01] [-0.03,-0.032,-0.02]
#screwdriver [0.001,0.001,0.001] [-0.01,-0.014,0.003] [0.001,0.001,0.001]
#soup-can [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#soup-can-lay [0.001,0.001,0.001] [0.01,0.04,-0.02] [-0.05,0.03,0.001]
#soup-can-open [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]
#spam-can [0.025,-0.03,-0.008] [0.001,0.001,0.001] [-0.02,0.01,-0.02]
#spam-can-lay [0.03,-0.01,-0.008] [0.001,0.001,0.001] [0.001,0.001,0.001]
#spam-can-open [0.001,0.001,0.001] [0.001,0.001,0.001] [0.001,0.001,0.001]