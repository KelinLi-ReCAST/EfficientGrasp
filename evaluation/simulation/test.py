#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:43:52 2022

@author: kelin
"""

import sys
sys.path.append('/home/kelin/workspace_kelin/RAL-IROS2022')

import os
import argparse
import pybullet as p
import numpy as np

def argsparser():
    parser = argparse.ArgumentParser("EfficientGrasp Simulation")
    parser.add_argument('--gripper', help='Gripper name', default='barrett')
    return parser.parse_args()
        
def main(args):
    if args.gripper == 'ruth':
        sys.path.append('/home/kelin/workspace_kelin/RAL-IROS2022/'+'train_ruth')
        import train_ruth.demo as demo
    elif args.gripper == 'robotiq':
        sys.path.append('/home/kelin/workspace_kelin/RAL-IROS2022/'+'train_robotiq')
        import train_robotiq.demo as demo 
    elif args.gripper == 'barrett':
        sys.path.append('/home/kelin/workspace_kelin/RAL-IROS2022/'+'train_barrett')
        import train_barrett.demo as demo
      
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