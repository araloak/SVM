import numpy as np
import matplotlib.pyplot
from sklearn import svm
import matplotlib.pyplot as plt

np.random.seed(8) # 保证随机的唯一性

array = np.random.randn(20,2) #产生随机数
print("array",array)
X = np.r_[array-[3,3],array+[3,3]]
print("X",X)
y = [0]*20+[1]*20
print(y)

clf = svm.SVC(C=10000,kernel='linear') #使用线性核SVM分类
clf.fit(X,y)

x1_min, x1_max = X[:,0].min(), X[:,0].max() #获得X矩阵每一行第一个元素中的最大值、最小值
print("x1_min",x1_min)
print("x1_max",x1_max)
x2_min, x2_max = X[:,1].min(), X[:,1].max() #获得X矩阵每一行第二个元素中的最大值、最小值
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

print("np.linspace(x1_min, x1_max)",np.linspace(x1_min, x1_max)) # np.linspace(start,end , nums)获得start与end之间nums个等差数列
#print("xx1",xx1)
#print("xx2",xx2)
# 得到向量w  : w_0x_1+w_1x_2+b=0
w = clf.coef_[0]
print('w',w)
f = w[0]*xx1 + w[1]*xx2 + clf.intercept_[0]+1  # 加1后才可绘制 -1 的等高线 [-1,0,1] + 1 = [0,1,2]
plt.contour(xx1, xx2, f, [0,1,2], colors = 'r') # 绘制分隔超平面、H1、H2
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],color='k') # 绘制支持向量点
plt.show()
