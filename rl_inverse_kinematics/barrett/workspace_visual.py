#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:21:01 2022

@author: kelin
"""
import numpy as np
import open3d as o3d
from open3d import utility as o3du
from show3d_balls import showpoints

a=np.load('/home/kelin/workspace_kelin/RL_IK/contact_points.npy')
print(a.shape)
b=np.zeros((576,9))

vertices_index = np.linspace(0,len(a)-1,576,dtype=int)
for i in range(576):
    b[i,:]=a[vertices_index[i],:]
np.save('ws1',b)    
pcd = o3d.geometry.PointCloud()
pcd.points = o3du.Vector3dVector(b[:,:3])
pcd.paint_uniform_color([1,0,0])

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3du.Vector3dVector(b[:,3:6])
pcd1.paint_uniform_color([0,1,0])

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3du.Vector3dVector(b[:,6:9])
pcd2.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([pcd,pcd1,pcd2])
pc_color = np.zeros((576*3,3))
for ii in range(576):
    pc_color[ii,0] = 255
for ii in range(576,576*2):
    pc_color[ii,1] = 255
for ii in range(576*2,576*3):
    pc_color[ii,2] = 255
contact_point1 = b[:,:3]
contact_point2 = b[:,3:6]
contact_point3 = b[:,6:9]
pc = np.vstack((contact_point1,contact_point2,contact_point3))
showpoints(pc,c_gt=pc_color, ballradius=6,freezerot=False)