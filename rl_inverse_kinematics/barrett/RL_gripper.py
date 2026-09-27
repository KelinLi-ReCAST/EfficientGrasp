#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:46:21 2022

@author: kelin
"""

import time
import joblib
import os
import cv2
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F 

#from fireup import EpochLogger
from utils.logx import EpochLogger, colorize

import json
import random

from utils_ import get_SREPS, get_srep, get_transitionse

import matplotlib.pyplot as plt

def load_envs_agent(fpath, itr='last'):
    policy = 'Gaussian'
    agent = joblib.load(osp.join(fpath, 'agent.pkl'))['agent']
    itr = agent.load_policy(fpath, itr, sord=policy)
    agent.change_device2cpu() 
    agent.change_device2device()
    return agent


def run_policy():
    fpath = '/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/log/kelin-v0/seed--1'
    agent = load_envs_agent(fpath)
    return agent
    