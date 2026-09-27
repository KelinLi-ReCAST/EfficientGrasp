#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:14:04 2022

@author: kelin
"""

import os
import numpy as np

obj_name = 'spam-can-open'
pc = np.load(os.path.join('/home/kelin/Downloads/revision_real_pc',obj_name+'.npy'))

vertices_index = np.linspace(0,len(pc)-1,2048,dtype=int)
sparse_pc = np.zeros((2048,3))
for i in range(2048):
    sparse_pc[i,:]=pc[vertices_index[i],:]

np.save(os.path.join('/home/kelin/Downloads/revision_sparse_real_pc','Ycb'+obj_name+'.npy'),sparse_pc)