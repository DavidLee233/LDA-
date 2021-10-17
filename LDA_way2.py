import numpy as np
import matplotlib.pyplot as plt
from scipy import io
X=io.loadmat('data.mat')['data']
pos0=np.where(X[:,2]==1)
pos1=np.where(X[:,2]==0)
X1=X[pos0,0:2]
X1=X1[0,:,:]
X2=X[pos1,0:2]
X2=X2[0,:,:]

#画出原始点所在位置
plt.figure(0)
ax = plt.subplot(1,1,1)
ax.scatter(X1[:,0], X1[:,1], c='red', s=10,label='标签为1的点')
ax.scatter(X2[:,0], X2[:,1], c='blue', s=10,label='标签为0的点')
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 黑体:SimHei  仿宋:FangSong  楷体:KaiTi  微软雅黑体:Microsoft YaHei  宋体:SimSun
plt.rcParams['axes.unicode_minus']=False  #使得坐标轴可以显示负号
plt.title('初始点所在位置坐标')        #设置标题

#第一步，求各个类别的均值
mu_old1=np.mean(X1,axis = 0) #axis = 0表示按列求平均 axis = 1表示按行求平均
mu_old1=np.array([mu_old1])
mu_old2=np.mean(X2,axis = 0)
mu_old2=np.array([mu_old2])
mu_old=np.mean(X[:,0:2],0)
mu_old=np.array([mu_old])
p=np.size(X1,0)
q=np.size(X2,0)

#第二步，求类内散度矩阵
S1=np.dot((X1-mu_old1).transpose(),(X1-mu_old1))
S2=np.dot((X2-mu_old2).transpose(),(X2-mu_old2))
Sw=(p*S1+q*S2)/(p+q)
# Sw = S1 + S2

#第三步，求类间散度矩阵
Sb1=np.dot((mu_old1-mu_old).transpose(),(mu_old1-mu_old)) #。transpose函数表示转置（用.T也可表示转置）
Sb2=np.dot((mu_old2-mu_old).transpose(),(mu_old2-mu_old))
Sb=(p*Sb1+q*Sb2)/(p+q)
# Sb=np.dot((mu_old1-mu_old2).transpose(),(mu_old1-mu_old2))

#判断Sw是否可逆
bb=np.linalg.det(Sw) #求Sw的特征值，若为0则不可逆
if bb==0:
    print('不能继续计算下去，因为Sw不可逆')
    
#第四步，求最大特征值和特征向量，求解出最佳投影方向，下面为第一种方法
[V,L]=np.linalg.eig(np.dot(np.linalg.inv(Sw),np.array(Sb))) #V为特征值 L为特征向量
list1=[]
a=V
list1 = a.tolist()
b=list1.index(max(list1))
W = L[:,b]

#根据求得的投影向量W画出投影线
k=W[1]/W[0]
b=0;
x=np.arange(2,10)
yy=k*x+b
plt.plot(x,yy,color='green')

#计算第一类样本在直线上的投影点
xi=[]
yi=[]
for i in range(0,p):
    y0=X1[i,1]
    x0=X1[i,0]
    x1=(k*(y0-b)+x0)/(k**2+1)
    y1=k*x1+b
    xi.append(x1)
    yi.append(y1)
    
#计算第二类样本在直线上的投影点
xj=[]
yj=[]
for i in range(0,q):
    y0=X2[i,1]
    x0=X2[i,0]
    x1=(k*(y0-b)+x0)/(k**2+1)
    y1=k*x1+b
    xj.append(x1)
    yj.append(y1)
    
#画出投影后的点
plt.plot(xi,yi,'r+',label='投影后的第一类点')
plt.plot(xj,yj,'b>',label='投影后的第二类点')
plt.legend()
plt.show()

