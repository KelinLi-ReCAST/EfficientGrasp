#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 22:53:57 2021

@author: kelin
"""

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import os
import time
import sys
import argparse
import tflearn
from show3d_balls import showpoints

import open3d as o3d
time_start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR,'../')

# Basic model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--saver_dir',default='./saved_models/',help='Directory to save the trained model')
parser.add_argument('--learning_rate',type=float,default=0.0005,help='Initial learning rate')
parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
parser.add_argument('--batch_size',type=int,default=2,help='Number of examples within a batch')
parser.add_argument('--max_model_to_keep',type=int,default=400,help='max saved models')
parser.add_argument('--log_dir',default='./logging/',help='folder to save logging info')
FLAGS = parser.parse_args()

sys.path.insert(0,"../vis_3d")
#from show3d_balls import showpoints

if not os.path.exists(FLAGS.saver_dir):
  os.mkdir(FLAGS.saver_dir)

if not os.path.exists(FLAGS.log_dir):
  os.mkdir(FLAGS.log_dir)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

in_gripper_tf = tf.placeholder(tf.float32,[None,576,9],'gripper_in')
gt_gripper_tf = tf.placeholder(tf.float32,[None,576,9],'gripper_gt')

def pc_encoder(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'encode'):
      pc_feat = tflearn.layers.conv.conv_1d(inputs,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      tf.summary.histogram('hidden_layer/weight', pc_feat.W)
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,64,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,128,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,256,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.conv_1d(pc_feat,256,filter_size=1,strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.conv.max_pool_1d(pc_feat,576,strides=576,padding='valid')
      return pc_feat  
def pc_decoder(inputs,scope=None,reuse=None):
    with tf.variable_scope(scope,'decode'):
      pc_feat = tflearn.layers.core.fully_connected(inputs,512,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc_feat = tflearn.layers.core.fully_connected(pc_feat,1024,activation='relu',weight_decay=1e-5,regularizer='L2')
      pc_feat = tflearn.layers.normalization.batch_normalization(pc_feat)
      pc      = tflearn.layers.core.fully_connected(pc_feat,576 * 9,activation='linear',weight_decay=1e-3,regularizer='L2')
      pc = tf.reshape(pc,(-1,576,9))
      return pc 
  
with tf.variable_scope('gripper_encoder'):
  gripper_feat_tf = pc_encoder(in_gripper_tf)

with tf.variable_scope('gripper_decoder'):
  out_gripper_tf = pc_decoder(gripper_feat_tf)

init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init_op)

SAVER = tf.train.Saver(max_to_keep=1000)

def restore(epoch):
  save_top_dir = os.path.join("/home/kelin/workspace_kelin/previous_work/feature_extraction/saved_models/workspace")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)


def test(gripper_dir=None,gripper_name="robotiq_3f"):
  in_gripper_file_list = [line for line in os.listdir(gripper_dir) if line.startswith(gripper_name)]
  in_gripper_list = []
  for idx, env_i in enumerate(in_gripper_file_list):
      env_dir = os.path.join(gripper_dir, env_i)
      obj_pcs = np.load(env_dir)
      in_gripper_list.append(obj_pcs)

  in_gripper = np.array(in_gripper_list)
  
  
  # a=0.1272667684079048*np.ones((2048,1))
  # b=-0.02735879066260683*np.ones((2048,1))
  # c = 0.10399799999999998*np.ones((2048,1))
  # d=np.append(a,b,axis=1)
  # e = np.append(d,c,axis=1)
  # in_gripper = in_gripper - e
  
  
  gt_gripper = in_gripper
  out_gripper, gripper_feat = sess.run([out_gripper_tf, gripper_feat_tf],feed_dict={in_gripper_tf: in_gripper})
  time_end = time.time()
  print(time_end-time_start)
  print(gripper_feat.shape)
  gripper_mean = np.mean(gripper_feat,axis=0)
  gripper_max =  np.max(gripper_feat,axis=0)
  gripper_min = np.min(gripper_feat,axis=0)
  print(gripper_mean.shape)

  recon_dir = gripper_dir
  mean_feat_file = os.path.join(recon_dir,'mean.npy')
  max_feat_file = os.path.join(recon_dir,'max.npy')
  min_feat_file = os.path.join(recon_dir,'min.npy')

  print(mean_feat_file)
  print(max_feat_file)
  print(min_feat_file)
  np.save(mean_feat_file,gripper_mean)
  np.save(max_feat_file,gripper_max)
  np.save(min_feat_file,gripper_min)


  if 1:
        for gj in range(len(in_gripper)):

         pred_gripper = np.copy(out_gripper[gj])
        
         gt__gripper = np.copy(gt_gripper[gj])
         gripper_two = np.zeros((576*3,3))
         gripper_two[:576,:] = pred_gripper[:,:3]
         gripper_two[576:576*2,:] = pred_gripper[:,3:6]
         gripper_two[576*2:576*3,:] = pred_gripper[:,6:9]
         color = np.zeros((576*3,3))
         for i in range(576):
             color[i,0] = 255
         for i in range(576,576*2):
             color[i,1] = 255
         for i in range(576*2,576*3):
             color[i,2] = 255
         showpoints(gripper_two,c_gt=color, ballradius=6,freezerot=False)
         # gripper_two[576:,:] = gt__gripper
         # txt_data = np.savetxt('top1_f1_index_1.txt', gripper_two[:576,:])
         # pcd = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
         # txt_data = np.savetxt('top1_f1_index_1.txt', gripper_two[576:,:])
         # pcd1 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
         # pcd.paint_uniform_color([0, 1, 0])
         # pcd1.paint_uniform_color([1, 0, 0])

         # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
         # o3d.visualization.draw_geometries([pcd, pcd1]+[axis_pcd], width=1200, height=600)



if __name__ == "__main__":
  #### restore the model of auto encoder
  restore(9999)

  #### specify the folder path of point clouds of the gripper
  gripper_dir = "/home/kelin/workspace_kelin/previous_work/feature_extraction/ruthArrays"
  gripper_name = "ws"
  test(gripper_dir,gripper_name)
# import os
# import tensorflow as tf

# reader = tf.pywrap_tensorflow.NewCheckpointReader('/home/kelin/Downloads/UniGrasp-master/saved_models/workspace/2model.ckpt')
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key))