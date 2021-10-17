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
ax = plt.subplot(1,2,1)
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
# [V,L]=np.linalg.eig(np.dot(np.linalg.inv(Sw),np.array(Sb))) #V为特征值 L为特征向量
# list1=[]
# a=V
# list1.extend(a)
# b=list1.index(max(list1))
# print(a[b])
# W=L[:,b]
#求最佳投影方向W的第二种方法
w = np.zeros((2, 1))
w = np.dot((np.linalg.inv(Sw)),np.array(mu_old1-mu_old2).transpose())
w_1 = w.T
#计算原始样本投影后的坐标(均变为一维点，因此用点乘)
u1 = []
for i in range(p):
    u1.append(np.dot(w_1, X1[i]))
u1 = np.array(u1)
u2 = []
for i in range(q):
    u2.append(np.dot(w_1, X2[i]))
u2 = np.array(u2)
#计算最佳分割阈值b
mu_new1 = sum(u1)/len(u1)
mu_new2 = sum(u2)/len(u2)
b = -(mu_new1+mu_new2)/2
#计算第一类样本在直线上的投影点
X_new1 = []
for i in range(len(u1)):
    X_new1.append(u1[i] + b)
X_new1 = np.array(X_new1).T
#计算第二类样本在直线上的投影点
X_new2 = []
for i in range(len(u2)):
    X_new2.append(u2[i] + b)
X_new2 = np.array(X_new2).T

X_new1 = X_new1[0].tolist()
X_new2 = X_new2[0].tolist()

#给出测试点并判断测试点的类别
X_test_old = [4.7, 2.1]
X_test_new = np.dot(w_1, np.array(X_test_old).T) + b
X_test_new = X_test_new.tolist()
ax.scatter(X_test_old[0], X_test_old[1], c='black', s=10, label='测试点')
ax.set_xlabel('X',fontsize=14)             #设置x，y轴的标签
ax.set_ylabel('Y',fontsize=14)
plt.legend()

#画出投影后的点并给出决策平面（一维）
bx = plt.subplot(1,2,2)
bx.scatter(X_new1, [0, 0, 0, 0], c='red', s=10, label='投影后的第一类点')
bx.scatter(X_new2, [0, 0, 0], c='blue', s=10, label='投影后的第二类点')
bx.scatter(X_test_new, 0, c='black', s=10, label='投影后的测试点')
bx.set_xlabel('g(x)', fontsize = 14)
plt.title('投影后一维点坐标数据')
Y = np.linspace(-1, 1, 100)
X = np.linspace(0, 0, 100)
plt.plot(X, Y, color="purple", linewidth=1.0, linestyle="-")
plt.legend()
plt.show()


