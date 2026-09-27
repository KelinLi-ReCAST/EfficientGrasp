#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:48:27 2022

@author: kelin
"""

import move_ur
import numpy as np
import os

obj_name = 'spam-can-open'
contact_points = np.load(os.path.join('/home/kelin/Downloads/contact_points_revision','Ruth_Ycb'+obj_name+'.npy'))
contact_point1 = contact_points[0,:3]/1000
contact_point2 = contact_points[0,3:6]/1000
contact_point3 = contact_points[0,6:9]/1000
center = (contact_point1+contact_point2+contact_point3)/3
normal, pos, ori = move_ur.calc_target_pos(np.vstack((contact_point1,contact_point2,contact_point3)))
np.save(os.path.join('/home/kelin/Downloads/target_pos',obj_name+'.npy'),np.hstack((pos,ori,center)))