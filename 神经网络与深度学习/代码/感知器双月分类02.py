############################################调用库（根据自己编程情况修改）
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
############################################训练数据集
from sklearn.model_selection import train_test_split

train = pd.read_excel(io=r'data1.xlsx', sheet_name="Sheet1",header=0, usecols="A:C")
train_data = train.values
############################################X、D每一列是一个样本
X = train_data[:, 0:2].T
D = train_data[:, 2].T
print(X)
print(D)
margin = list(D).index(-1)
print(margin)
############################################超参数定义 迭代次数、学习率
iter = 100
learning_rate = 1
############################################感知器模型权值随机初始化
wn = (np.random.rand(3,) - 0.5) * 0.2
############################################中文字体设置
font = font_manager.FontProperties(fname='font/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
############################################训练数据集可视化
plt.scatter(X[0, 0:margin], X[1, 0:margin], marker='x', color="red", label="预测类1", linewidth=2)
plt.scatter(X[0, margin:len(D)], X[1,margin:len(D)], marker='*', color="blue", label="预测类-1", linewidth=2)
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.title("感知器神经网络分类训练数据集！", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.show()
##############################################在下面代码段中实现感知器学习算法（补充）
##############################################感知器学习算法（补充）

##############################################激活函数:符号函数sgn(x)
def sgn(x):
    if x >= 0:
        return 1
    return -1
##############################################感知器权值更新公式 分类错误就更新权值
def update(xn,wn,dn,learning_rate):
    if np.dot(wn, xn) * dn <= 0:
        wn = wn + learning_rate * xn * dn
    return wn
##############################################感知器输出 模型分类结果
def actual_func(xn,wn):
    yn = sgn(np.dot(wn.T,xn))
    return yn
##############################################均方误差损失函数
def loss_func(dn,yn):
    return (dn-yn)**2
##############################################训练模型学习参数
def train(data,dn,wn,iterations,learning_rate):
    loss = []
    for j in range(iterations):
        i = 0
        loss_temp = []
        for xn in data:
            yn = actual_func(xn,wn)
            wn = update(xn,wn,dn[i],learning_rate)
            loss_temp.append(loss_func(dn[i],yn))
            i = i+1
        loss.append(sum(loss_temp)/len(loss_temp))
    return loss,wn
##############################################验证模型
def test(xn,dn):
    right_count = wrong_count = i = 0
    for x in xn:
        yn = actual_func(x,wn)
        if yn==dn[i]:
            right_count += 1
        else:
            wrong_count += 1
        right_count = right_count/(right_count+wrong_count)
        wrong_count = wrong_count/(right_count+wrong_count)
        i += 1
    return wrong_count,right_count
##############################################将数据集分为训练集和测试集 训练集1599个样本  测试集400个样本
data_train,data_test, dn_train, dn_test = train_test_split(X.T,D,test_size=0.2,random_state=None)
train_data = np.ones((1599,3))
train_data[:,1:3] = data_train
train_dn = dn_train
##############################################调用训练模型函数
loss,wn = train(train_data,train_dn,wn,iter,learning_rate)
print(wn)
test_data = np.ones((400,3))
test_data[:,1:3] = data_test
test_dn = dn_test
##############################################调用验证模型函数
err,acc = test(test_data,test_dn)
print('测试结果 错误率：','%.2f'%err)
print('测试结果 正确率：','%.2f'%acc)
##############################################描点 分类直线
x = np.array(range(-15,25))
y = -x*wn[1]/wn[2]-wn[0]/wn[2]
##############################################画图
plt.figure()
plt.scatter(X[0, 0:margin], X[1, 0:margin], marker='x', color="red", label="预测类1", linewidth=2)
plt.scatter(X[0, margin:len(D)], X[1,margin:len(D)], marker='*', color="blue", label="预测类-1", linewidth=2)
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.title("感知器分类结果", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.plot(x,y,'k--')
x = [i for i in range(len(loss))]
plt.figure()
plt.title('损失函数曲线图', fontproperties=font)
plt.xlabel('迭代次数', fontproperties=font)
plt.ylabel('均方误差', fontproperties=font)
plt.plot(x,loss,c='r',linewidth=2,linestyle='-',marker='o')
plt.show()