import numpy as np
import math
from numpy import *
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib import patches  # 导入作图路径添加库
from scipy import interpolate  # 导入插值函数库

#  生成数据点
#  设置高斯随机抽样的 相关参数
mean1 = np.array([3, 3])
mean2 = np.array([10, 10])
mean3 = [19, 19]
cov3 = np.array([[8, -1], [-1, 1]])
cov2 = np.array([[1, 2], [2, 10]])
cov1 = np.eye(2)  # 用numpy.eye()生成对角矩阵。协方差cov1(X,Y)为0，方差D(y)=D(X)为1
dot_num = 500  # 生成的高斯分布矩阵维度为dot_num*len（mean1）=dot_num*2
# 利用函数np.random.multivariate_normal从多元正态分布中生成随机抽样
x1 = np.random.multivariate_normal(mean1, cov1, dot_num)  # 组成一个N维的数组。并返回该数组。要进行转置
x2 = np.random.multivariate_normal(mean2, cov2, dot_num)
x3 = np.random.multivariate_normal(mean3, cov3, dot_num)
data1 = np.append(np.append(x1, x2, axis=0), x3, axis=0)  # 将矩阵拼接成N*2
N, D = data1.shape  # 输出样本数,维度数

######################################################
# 参数初始化
######################################################
K = 3
mu_ini = np.zeros([K, D])
cov = np.zeros([K, D, D])
for i in range(0, K):
    cov[i] = [[1, 0], [0, 1]]
    mu_ini[i] = [8 + i, 3 + i]
mu = mu_ini  # 用中间变量保存初始均值
omega = np.array([0.333, 0.333, 0.334])
time = 30  # 设置迭代次数
pro_mat = np.zeros((N, K))  # 初始化贝叶斯概率矩阵 N*K，
time_mu = np.zeros((K, time, D))  # 初始化一个均值矩阵，用以存放每轮迭代的值，用以绘制均值迭代运动轨迹 NK*
time0 = 0  # 初始化均值矩阵索引变量

######################################################
# 进行EM估计
######################################################
while (time > 0):   # 迭代条件
    ######################################################
    # E-Step
    ######################################################
    # 概率密度函数
    for i in range(0, N):
        sum = 0
        # 第i个点属于 第k(k=0、1、2)个高斯分布的概率之和
        for k in range(0, K):
            sum = sum + omega[k] * multivariate_normal.pdf(data1[i, :], mu[k, :], cov[k, :, :])
        # 贝叶斯概率，即第i个点属于第k的个高斯分布的概
        for k in range(0, K):
            pro_mat[i, k] = omega[k] * multivariate_normal.pdf(data1[i, :], mu[k, :], cov[k, :, :]) / sum

    ######################################################
    # M-Step，更新参数
    ######################################################
    for k in range(0, K):
        pro_sum = np.sum(pro_mat[:, k])  # 所有点分别属于第k的高斯分布的概率 之和
        # 计算更新后的omega
        omega[k] = pro_sum / N  # 或者omega[k] = np.sum(pro_mat,0)[k] / N
        # 计算更新后的协方差矩阵cov
        # operands could not be broadcast together with shapes (1800,) (900,2)
        A = pro_mat[:, k].reshape(-1, 1)  # 实现转置step1:改变行列数
        B = A.transpose()  # 实现转置step2:改变行列数
        # print(B.shape)#打印矩阵的行列，从而判断是否转置成功
        C = np.append(B, B, axis=0)

        # operands could not be broadcast together with shapes 违反了广播机制，矩阵乘法列表形式？？？
        # cov[k,:,:]= (data1 - mu[k]).T * np.multiply(C.T,(data1 - mu[k])) / pro_sum
        # 利用.mat函数将数组类型转化为矩阵类型，从而使用"*"相乘
        D = mat((data1 - mu[k]).T)
        E = mat(np.multiply(C.T, (data1 - mu[k])))  # np.multiply((data1 - mu[k]),pro_mat[:,k])
        F = mat(np.multiply(C.T, data1))  # np.multiply(data1, pro_mat[:, k])
        # cov[k, :, :] = (data1 - mu[k]).T * np.multiply(C.T, (data1 - mu[k])) / pro_sum
        cov[k, :, :] = D * E / pro_sum
        mu[k, :] = np.sum(F, axis=0) / pro_sum
        time_mu[k, time0, :] = mu[k, :]
        # 可视化每轮更新均值
        # plt.scatter(mu[k, 0], mu[k, 1], c='r', marker='2')
        # plt.plot(mu[k, 0], mu[k, 1])
    time = time - 1  # 迭代次数减1
    time0 = time0 + 1  # 均值矩阵索引+1

######################################################
# 绘制协方差误差椭圆--即等高线
######################################################
s = 7.378  # (置信度为97.5%对应的s)
# 第一步：创建绘图对象
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
area = np.pi * 2 ** 2  # 点面积
alpha = 0.4  # 散点透明度
color_ell = ['#00FF7F', '#1E90FF', '#FFA07A']  # 设置等高线颜色数组
#  添加散点图
ax.scatter(x1[:, 0], x1[:, 1], s=area, c='g', marker='.')
ax.scatter(x2[:, 0], x2[:, 1], s=area, c='c', marker='.')
ax.scatter(x3[:, 0], x3[:, 1], s=area, c='y', marker='.')

for k in range(0, K):
    # 第二步：求解椭圆相关参数
    # 可视化初值均值
    ax.scatter(8 + k, 3 + k, c='#FF00FF', marker='x')
    x0 = np.array((8 + k, time_mu[k, 0, 0]))
    y0 = np.array((3 + k, time_mu[k, 0, 1]))
    ax.plot(x0, y0, color='#FF00FF', ls="--", lw=1)
    # 可视化每轮更新均值、轨迹
    ax.scatter(time_mu[k, :, 0], time_mu[k, :, 1], marker='o', facecolors='none', edgecolors='r')
    ax.plot(time_mu[k, :, 0], time_mu[k, :, 1], color='r', ls="-", lw=1)  # 看能不能粗略地画出等高线
    # 求协方差矩阵的特征值（python默认降序排列！！）及对应特征向量
    w, v = np.linalg.eig(cov[k, :, :])  # w为特征值，v为对应的已归一的特征向量（列向量）
    # 确定椭圆长、短轴
    hei = 2 * (s * w[1]) ** 0.5  # 特别注意！！！！w[0]对应长短轴
    wid = 2 * (s * w[0]) ** 0.5
    # 确定椭圆角度
    alpha0 = arctan2(v[1, 0], v[0, 0]) / math.pi * 180
    # 第三步：绘制椭圆
    ell = Ellipse(xy=mu[k, :], width=wid, height=hei, angle=alpha0, color=color_ell[k], ls="--", fill=False,
                  linewidth=2)
    # 第四步：将图形添加到图中
    ax.add_patch(ell)
    # patches.append(ell)  # 将创建的形状全部放进去
plt.show()
# #打印最终参数
print("迭代30次后返回的权重为：", omega)
print("迭代30次后返回的均值为：", mu)
print("迭代30次后返回的协方差为：", cov)
# print("迭代30次后返回的权重、均值、协方差分别为：",omega,mu,cov)
