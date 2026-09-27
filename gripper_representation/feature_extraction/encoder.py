#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:36:05 2021

@author: kelin
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import tflearn
import sys
import os
# import open3d as o3d
import matplotlib.pyplot as plt

# Parameter
learning_rate = 0.001
training_epochs = 10000
batch_size = 1
display_step = 1
# examples_to_show = 10
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
in_gripper_list = []
# for i in range(4):
#     if i == 0:
#       ROOT_DIR = os.path.join(BASE_DIR,'WorkspaceArrays/')
#     elif i == 1:
#       ROOT_DIR = os.path.join(BASE_DIR,'coupRotWorkspaceArrays/')
#     elif i == 2:
#       ROOT_DIR = os.path.join(BASE_DIR,'fourbarWorkspaceArrays/')
#     elif i == 3:
#       ROOT_DIR = os.path.join(BASE_DIR,'singRotWorkspaceArrays/')

#     in_gripper_file_list = [line for line in os.listdir(ROOT_DIR) if line.startswith("ws")]
    
#     for idx, env_i in enumerate(in_gripper_file_list):
#         print(idx)
#         env_dir = os.path.join(ROOT_DIR, env_i)
#         obj_pcs = np.load(env_dir)
#         if len(obj_pcs!=0):
#            in_gripper_list.append(obj_pcs)
#         if len(in_gripper_list)%2000 == 0:
#            break
ROOT_DIR = os.path.join(BASE_DIR,'ruthArrays/')
in_gripper_file_list = [line for line in os.listdir(ROOT_DIR) if line.startswith("ws")]
    
for idx, env_i in enumerate(in_gripper_file_list):
        print(idx)
        env_dir = os.path.join(ROOT_DIR, env_i)
        obj_pcs = np.load(env_dir)
        if len(obj_pcs!=0):
           in_gripper_list.append(obj_pcs)
        if len(in_gripper_list)%2000 == 0:
           break
in_gripper_tf = tf.placeholder(tf.float32,[None,576,9],'gripper_in')
gt_gripper_tf = tf.placeholder(tf.float32,[None,576,9],'gripper_gt')
def restore(epoch):
  save_top_dir = os.path.join("/home/robin-lab/Kelin/saved_models/workspace")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  print("restoring from %s" % ckpt_path)
  SAVER.restore(sess, ckpt_path)
  
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
  
def save_model_stage(epoch):
  save_top_dir = os.path.join('/home/kelin/workspace_kelin/previous_work/feature_extraction/saved_models',"workspace")
  ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
  if epoch == 0:
    SAVER.save(sess, ckpt_path, write_meta_graph=True)
  else:
    SAVER.save(sess, ckpt_path, write_meta_graph=False)
  print("Saving model at epoch %d to %s" % (epoch, ckpt_path))
    
def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    finger, num_point, num_features = array1.shape
    #finger = int(num_features/3)
    distances = tf.zeros((num_point*num_point, ), dtype=tf.float32)
    for i in range(finger):        
      expanded_array1 = tf.tile(array1[i,:,:], (num_point, 1))
      expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2[i,:,:], 1), 
                    (1, num_point, 1)),
            (-1, num_features))
      distances = tf.add(distances,tf.norm(expanded_array1-expanded_array2, axis=1))
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances

def av_dist_sum(array1,array2):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    # array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2

def chamfer_distance_tf(array1, array2):
    batch_size1, num_point, num_features = array1.shape
    dist = av_dist_sum(array1, array2)
    return dist

with tf.variable_scope('gripper_encoder'):
  gripper_feat_tf = pc_encoder(in_gripper_tf)
  
with tf.variable_scope('gripper_decoder'):
  out_gripper_tf = pc_decoder(gripper_feat_tf)

in_gripper_tf_resh = tf.reshape(in_gripper_tf,(576,9))
out_gripper_tf_resh = tf.reshape(out_gripper_tf,(576,9))
split_in = tf.split(in_gripper_tf_resh,axis=1,num_or_size_splits=3)
split_out = tf.split(out_gripper_tf_resh,axis=1,num_or_size_splits=3)
aaa = tf.stack([split_in[0],split_in[1],split_in[2]],axis=0)
bbb = tf.stack([split_out[0],split_out[1],split_out[2]],axis=0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
SAVER = tf.train.Saver(max_to_keep=1000)

cost = chamfer_distance_tf(aaa,bbb)
tf.summary.scalar('cost',cost)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())


    
data1 = np.zeros((batch_size,576,9))
max = np.zeros(9)
min = np.zeros(9)
# txt_data = np.savetxt('top1_f1_index_1.txt', data[:,:3])
# pcd = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
# pcd.paint_uniform_color([1, 0, 0])
# txt_data = np.savetxt('top1_f1_index_1.txt', data[:,3:6])
# pcd1 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
# pcd1.paint_uniform_color([0,1, 0])
# txt_data = np.savetxt('top1_f1_index_1.txt', data[:,6:9])
# pcd2 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
# pcd2.paint_uniform_color([0, 0, 1])
with tf.Session() as sess:
  sess.run(init_op)  
  #restore(1000)
  #out_gripper, gripper_feat = sess.run([out_gripper_tf, gripper_feat_tf],feed_dict={in_gripper_tf: data, gt_gripper_tf:data})
  for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(len(in_gripper_list) / batch_size)
      for i in range(total_batch):
          for j in range(batch_size):
              data = np.array(in_gripper_list[j+batch_size*i])
              data1[j,:,:] = data
              # for m in range(data.shape[1]):
              #     max[m]=np.max(data[:,m])
              #     min[m]=np.min(data[:,m])
              #     for n in range(data.shape[0]):
              #         data1[j,n,m] = 2/(max[m]-min[m])*data[n,m]-(max[m]+min[m])/(max[m]-min[m])
              data2 = np.zeros((1,576,9))
              data2[0,:,:] = data1[j,:,:]
              _, c, y_pred  = sess.run([optimizer, cost, out_gripper_tf], feed_dict={in_gripper_tf: data2})
              avg_cost += c / total_batch
          # y_pred1 = np.zeros((576,9))
          # y_pred1 = y_pred[0,:,:]
          # for j in range(data.shape[1]):
          #     for i in range(data.shape[0]):
          #         y_pred1[i,j] = ((max[j]-min[j])*y_pred1[i,j]+max[j]+min[j])/2
          # txt_data = np.savetxt('top1_f1_index_1.txt', y_pred1[:,:3])
          # pcd3 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
          # pcd3.paint_uniform_color([1, 0, 0])
          # txt_data = np.savetxt('top1_f1_index_1.txt', y_pred1[:,3:6])
          # pcd4 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
          # pcd4.paint_uniform_color([0, 1, 0])
          # txt_data = np.savetxt('top1_f1_index_1.txt', y_pred1[:,6:9])
          # pcd5 = o3d.io.read_point_cloud('top1_f1_index_1.txt', format='xyz')
          # pcd5.paint_uniform_color([0, 0, 1])
          # print("Epoch:", '%04d' % (epoch+1))
          # result = sess.run(merged, feed_dict={in_gripper_tf: data1})
          # writer.add_summary(result,epoch)
      if epoch % display_step == 0:          
              save_model_stage(epoch)
              print("Epoch:", '%04d' % (epoch),
                    "cost=", "{:.9f}".format(avg_cost))
      #plt.plot(epoch,avg_cost)   
      #plt.show()
              # o3d.visualization.draw_geometries([pcd, pcd1, pcd2], width=1200, height=600)
              # o3d.visualization.draw_geometries([pcd3, pcd4, pcd5], width=1200, height=600)