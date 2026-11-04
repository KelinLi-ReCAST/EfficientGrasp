#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:14:04 2022

@author: kelin
"""

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

import os
import numpy as np

obj_name = 'spam-can-open'
pc = np.load(os.path.join(os.path.join(REPO_ROOT, 'evaluation', 'real_world', 'raw_pointclouds'),obj_name+'.npy'))

vertices_index = np.linspace(0,len(pc)-1,2048,dtype=int)
sparse_pc = np.zeros((2048,3))
for i in range(2048):
    sparse_pc[i,:]=pc[vertices_index[i],:]

np.save(os.path.join(os.path.join(REPO_ROOT, 'evaluation', 'real_world', 'object_pointclouds'),'Ycb'+obj_name+'.npy'),sparse_pc)