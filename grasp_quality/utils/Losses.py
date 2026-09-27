import numpy as np
import torch
import os
import csv
import codecs
osp = os.path

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
        
class FCLoss:
  def __init__(self, object_model):

    self.device = torch.device('cuda')
    device = self.device
    
    self.transformation_matrix = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float().to(device)
    self.eye3 = torch.tensor(np.eye(3).reshape(1, 1, 3, 3)).float().to(device)
    self.eye6 = torch.tensor(np.eye(6).reshape(1,6,6)).float().to(device)
    
    self.eps = torch.tensor(0.01).float().to(device)
    self.mu = torch.tensor(0.1).float().to(device)
    self.sqrt_sq_mu_1 = torch.sqrt(self.mu*self.mu+1)
    self.relu = torch.nn.ReLU()

    self.obj_model = object_model

  def l2_norm(self, x):
    if len(x.shape) == 3:
      return torch.sum(x*x, (1, 2))
    if len(x.shape) == 2:
      return torch.sum(x*x, (1))
    raise ValueError

  def x_to_G(self, x):
    """
    x: B x N x 3
    G: B x 6 x 3N
    """
    B = x.shape[0]
    N = x.shape[1]
    xi_cross = torch.matmul(x, self.transformation_matrix).reshape([B,N,3,3]).transpose(1, 2).reshape([B, 3, 3*N])
    I = self.eye3.repeat([B, N, 1, 1]).transpose(1,2).reshape([B, 3, 3*N])
    G = torch.stack([I, xi_cross], 1).reshape([B, 6, 3*N])
    return G

  def loss_8a(self, G):
    """
    G: B x 6 x 3N
    """
    Gt = G.transpose(1,2)
    temp = self.eps * self.eye6
    temp = torch.matmul(G, Gt) - temp
    eigval = torch.symeig(temp.cpu(), eigenvectors=True)[0].to(self.device)
    rnev = self.relu(-eigval)
    result = torch.sum(rnev * rnev, 1)
    return result
  
  def loss_8b(self, f, G): 
    """
    G: B x 6 x 3N
    f: B x N x 3
    """
    B = f.shape[0]
    N = f.shape[1]
    return self.relu(self.l2_norm(torch.matmul(G, f.reshape(B, 3*N, 1))))
  
  def loss_8c(self, normal):
    """
    normal: B x N x 3
    friction: B x N x 3
    x: B x N x 3
    mu: ()
    """
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    left = torch.einsum('ijk, ijk->ij', normal, normal)
    right = torch.norm(normal, dim=2) / self.sqrt_sq_mu_1
    diff = left - right
    return torch.sum(self.relu(-diff), 1)
  
  def dist_loss(self, object_code, x):
    d = self.obj_model.distance(object_code, x).squeeze(-1)
    return d * d
  
  def fc_loss(self, x, normal, obj_code):
    G = self.x_to_G(x)
    l8a = self.loss_8a(G)
    l8b = self.loss_8b(normal, G)
    # l8c = self.loss_8c(normal)
    # l8d = self.dist_loss(obj_code, x)
    return l8a, l8b

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  plt.ion()
  # ax = plt.subplot(111, projection='3d')
  FC = []
  temp = []
  # object_names = ['banana','bowl','bowl-bottom','clip-L','clip-M','drill-lay','drill-stand',
  #                  'football','mug','mug-bottom','screwdriver','soup-can','soup-can-lay','soup-can-open',
  #                  'spam-can','spam-can-lay','spam-can-open']
  object_names = ['Banana','ChipsCan','CrackerBox','FoamBrick','GelatinBox','Hammer','MasterChefCan',
                    'MediumClamp','MustardBottle','Pear','PottedMeatCan','PowerDrill','Scissors','Strawberry',
                    'TennisBall','TomatoSoupCan']
  for object_name in object_names:  
    a=np.load(osp.join('/home/kelin/Downloads/contact_points', 'Ruth_Ycb'+f'{object_name}'+'.npy'))
    a=a
    sum = 0
    for i in range(10):
    # i = 9
     b=np.array([[[a[i,0],a[i,1],a[i,2]],[a[i,3],a[i,4],a[i,5]],[a[i,6],a[i,7],a[i,8]]]])
     yy=np.mean(b[0,:,1])
     xx=np.mean(b[0,:,0])
     zz=np.mean(b[0,:,2])
     c=np.array([[[a[i,0]-xx,a[i,1]-yy,a[i,2]-zz],[a[i,3]-xx,a[i,4]-yy,a[i,5]-zz],[a[i,6]-xx,a[i,7]-yy,a[i,8]-zz]]])
     f=torch.from_numpy(c).float().cuda()
     x=torch.from_numpy(b).float().cuda()
     n = 3

     fc = FCLoss(None)
     # x = torch.tensor(np.random.random([1, n, 3]), requires_grad=True).float().cuda()
     # f = torch.tensor(np.random.random([1, n, 3]), requires_grad=True).float().cuda()
     f = f / torch.norm(f, dim=-1, keepdim=True)
     G = fc.x_to_G(x)
     l8b = fc.loss_8b(f, G)
     temp.append(l8b.tolist())
     sum = sum+l8b
     #print(l8b)
    temp.sort()
    tt=np.array(temp).T
    #FC.append(object_name) 
    FC.append(tt)
    temp.clear()
  filename = osp.join('/home/kelin/Downloads/contact_points', 'Robotiq_gq1')
  #data_write_csv(filename,FC)
  FC = np.reshape(FC,(16,10))
  np.save(filename,FC)
  # while True:
    # ax.cla()
    # ax.quiver(np.zeros([n]), np.zeros([n]), np.zeros([n]), f.detach().cpu().numpy()[0,:,0], f.detach().cpu().numpy()[0,:,1], f.detach().cpu().numpy()[0,:,2], length=0.1, normalize=True, color='red')
    # plt.pause(1e-5)

    # l8b = fc.loss_8b(f, G)
    # print(l8b)
    # grad = torch.autograd.grad(l8b, f)[0]
    # f = f - grad * 0.1
    # f = f / torch.norm(f, dim=-1, keepdim=True)

