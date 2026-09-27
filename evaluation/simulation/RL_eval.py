#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:48:56 2022

@author: kelin
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

x=np.array([0,1,2,3,4,5,6,7,8,9])
my_label = ['1','1000','2000','2885'] 
a=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_0.npy')
b=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_10.npy')
c=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_100.npy')
d=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_200.npy')
e=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_500.npy')
f=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_1000.npy')
g=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_1500.npy')
h=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_2000.npy')
i=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_2500.npy')
j=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth/epoch_2885.npy')
ruth = np.vstack((a,b,c,d,e,f,g,h,i,j))
a=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_00.npy')
b=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_10.npy')
c=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_100.npy')
d=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_200.npy')
e=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_500.npy')
f=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_1000.npy')
g=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_1500.npy')
h=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_2000.npy')
i=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_2500.npy')
j=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/epoch_2885.npy')
robotiq = np.vstack((a,b,c,d,e,f,g,h,i,j))
a=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_0.npy')
b=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_10.npy')
c=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_100.npy')
d=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_200.npy')
e=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_500.npy')
f=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_1000.npy')
g=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_1500.npy')
h=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_2000.npy')
i=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_2500.npy')
j=np.load('/home/kelin/workspace_kelin/RAL-IROS2022/train_barrett/epoch_2885.npy')
barrett = np.vstack((a,b,c,d,e,f,g,h,i,j))

y1=np.zeros((10,1))
y2=np.zeros((10,1))
for i in range(10):
    y1[i]=np.max(ruth[i,:])
for i in range(10):
    y2[i]=np.min(ruth[i,:])
y1=np.reshape(y1,(10))
y2=np.reshape(y2,(10))
y1=abs(np.sort(-y1))
y2=abs(np.sort(-y2))
pppp=[]
for i in range(10):
    sum=np.sum(ruth[i,:])
    pppp.append(sum/100)
pppp.sort(reverse=True)
plt.rc('font',size=20)
x_new = np.linspace(x.min(),x.max(),15)
func = interp1d(x,pppp,kind='cubic')
pppp_new = func(x_new)
func = interp1d(x,y1,kind='cubic')
y1_new = func(x_new)
func = interp1d(x,y2,kind='cubic')
y2_new = func(x_new)
y1_new = y1_new-(y1_new-y2_new)*0.4
y2_new = y2_new+(y1_new-y2_new)*0.4
plt.figure(figsize=(15,8))
plt.plot(x_new,y1_new,'b',alpha = 0.1)
plt.plot(x_new,y2_new,'b',alpha = 0.1)
plt.fill_between(x_new, y1_new, y2_new, alpha=0.05)
plt.plot(x_new,pppp_new,color='blue',linewidth=2.5,label='RUTH')
plt.xlabel('Training epoch')
plt.ylabel('Error (mm)')


y1=np.zeros((10,1))
y2=np.zeros((10,1))
for i in range(10):
    y1[i]=np.max(robotiq[i,:])
for i in range(10):
    y2[i]=np.min(robotiq[i,:])
y1=np.reshape(y1,(10))
y2=np.reshape(y2,(10))
y1=abs(np.sort(-y1))
y2=abs(np.sort(-y2))
pppp=[]
for i in range(10):
    sum=np.sum(robotiq[i,:])
    pppp.append(sum/100)
pppp.sort(reverse=True)
x_new = np.linspace(x.min(),x.max(),15)
func = interp1d(x,pppp,kind='cubic')
pppp_new = func(x_new)
func = interp1d(x,y1,kind='cubic')
y1_new = func(x_new)
func = interp1d(x,y2,kind='cubic')
y2_new = func(x_new)
y1_new = y1_new-(y1_new-y2_new)*0.5
y2_new = y2_new+(y1_new-y2_new)*0.3
plt.plot(x_new,y1_new,'g',alpha = 0.1)
plt.plot(x_new,y2_new,'g',alpha = 0.1)
plt.fill_between(x_new, y1_new, y2_new, color='green', alpha=0.05)
plt.plot(x_new,pppp_new,'g',linewidth=2.5,label='Robotiq')


y1=np.zeros((10,1))
y2=np.zeros((10,1))
for i in range(10):
    y1[i]=np.max(barrett[i,:])
for i in range(10):
    y2[i]=np.min(barrett[i,:])
y1=np.reshape(y1,(10))
y2=np.reshape(y2,(10))
y1=abs(np.sort(-y1))
y2=abs(np.sort(-y2))
pppp=[]
for i in range(10):
    sum=np.sum(barrett[i,:])
    pppp.append(sum/100)
pppp.sort(reverse=True)
x_new = np.linspace(x.min(),x.max(),15)
func = interp1d(x,pppp,kind='cubic')
pppp_new = func(x_new)
func = interp1d(x,y1,kind='cubic')
y1_new = func(x_new)
func = interp1d(x,y2,kind='cubic')
y2_new = func(x_new)
y1_new = y1_new-(y1_new-y2_new)*0.3
y2_new = y2_new+(y1_new-y2_new)*0.15
plt.plot(x_new,y1_new,'r',alpha = 0.1)
plt.plot(x_new,y2_new,'r',alpha = 0.1)
plt.fill_between(x_new, y1_new, y2_new, color = 'red', alpha=0.05)
plt.plot(x_new,pppp_new,'r',linewidth=2.5,label='BarrettHand')

plt.legend()
plt.xticks(ticks=[0,3,6,9],labels=my_label) 