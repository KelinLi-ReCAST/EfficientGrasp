#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:43:52 2022

@author: kelin
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import argparse
import pybullet as p
import numpy as np

def argsparser():
    parser = argparse.ArgumentParser("EfficientGrasp Simulation")
    parser.add_argument('--gripper', help='Gripper name', default='barrett')
    return parser.parse_args()
        
def main(args):
    gripper_dirs = {'ruth': 'ruth', 'robotiq': 'robotiq_3f', 'barrett': 'barrett'}
    sys.path.insert(0, os.path.join(REPO_ROOT, 'rl_inverse_kinematics', gripper_dirs[args.gripper]))
    import demo
      
    idx = ['Banana','ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 'MasterChefCan',
                           'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 'Scissors', 'Strawberry',
                           'TennisBall', 'TomatoSoupCan']
    size_list = [1,0.8,0.7,1,1,0.8,0.8,1,0.8,1,0.7,0.9,1,1,1,1]
    
    ori_list = ['000','pi00','pi0pi','pipipi']
    
    object_names = ['Banana','ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 'MasterChefCan',
                           'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 'Scissors', 'Strawberry',
                           'TennisBall', 'TomatoSoupCan']
   
    for object_name in object_names:
        for obj_ori in ori_list:
            obj_idx = idx.index(object_name)
            obj_pos = [0.5, 0, 0.01]
            obj_size = size_list[obj_idx]
            point_nos =0
            depth = 0.24
            rl_step = 3
            obj_name = object_name
            demo.main(obj_pos, obj_ori, obj_size, point_nos, depth, rl_step, obj_name,args.gripper)
        
    
    
if __name__ == '__main__':
    args = argsparser()
    main(args)    