import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from sklearn.model_selection import train_test_split

############################################训练数据集
train = pd.read_excel(io=r'data.xlsx', sheet_name="Sheet1", header=0, usecols="A:C")
train_data = train.values
############################################X、D每一列是一个样本
# 样本(x,y)点坐标
X = train_data[:, 0:2].T
# 类别
D = train_data[:, 2].T
margin = list(D).index(-1)
print(margin)
############################################中文字体设置
font = font_manager.FontProperties(fname='font/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
############################################训练数据集可视化
plt.scatter(X[0, 0:margin], X[1, 0:margin], marker='x', color="red", label="预测类1", linewidth=2)
plt.scatter(X[0, margin:len(D)], X[1, margin:len(D)], marker='*', color="blue", label="预测类-1", linewidth=2)
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.title("BP神经网络分类训练数据集", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.show()
############################################超参数定义
# 迭代次数
iter = 1000
# 学习率(梯度下降的步长)
learning_rate = 0.1
# 判断阈值
c = 0.5
###########################################类别数据处理的处理
for i in range(len(D)):
    if D[i] == -1:
        D[i] = 0
############################################参数随机初始化
# 初始化权重，随机设置，范围为（-1,1）
#输入层3个节点，隐藏层50个节点
V = np.random.random((2, 50)) * 2 - 1
W = np.random.random((50, 1)) * 2 - 1
#偏置
b1=0.5*np.ones((2000,50))  #输入层到隐藏层偏置
b2=0.5*np.ones((2000,1))  #隐藏层到输出层偏置

############################################数据集划分->训练集+测试集
data_train, data_test, dn_train, dn_test = train_test_split(X.T, D, test_size=0.2, random_state=None)
############################################训练
index = [i for i in range(len(train_data))]
np.random.shuffle(index)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# y_p_old = 0
# a2_list_1 = []
# a2_list_2 = []
for i in range(iter):
    print(f"第{i + 1}次迭代")
    for n in range(len(X.T)):
        # 向前传播
        L1 = sigmoid(np.dot(X.T, V) + b1)  # L1：输入层传递给隐藏层的值；
        L2 = sigmoid(np.dot(L1, W) + b2)  # L2：隐藏层传递到输出层的值；输出层1个节
        L2_delta = L2 * (1 - L2) * (D.T - L2)  # L2_delta：输出层的误差信号
        L1_delta = L1 * (1 - L1) * L2_delta.dot(W) # L1_delta：隐藏层的误差信号
        print(L2)
        # W_C：输出层对隐藏层的权重改变量
        # V_C：隐藏层对输入层的权重改变量
        W_C = learning_rate * L1.T.dot(L2_delta)
        V_C = learning_rate * X.T.dot(L1_delta)
        # 权重更新
        W = W + W_C
        V = V + V_C

# test_x = []
# test_y = []
# test_p = []
# print(a2_list_1)
# print(len(a2_list_1))
# print(a2_list_2)
# print(len(a2_list_2))


# 找决策曲线
def predict(X):
    L1 = sigmoid(np.dot(X, V) + b1)  # L1：输入层传递给隐藏层的值；
    L2 = sigmoid(np.dot(L1, W) + b2)  # L2：隐藏层传递到输出层的值；输出层1个节
    return L2


for x in np.arange(-15., 20., 0.1):
    for y in np.arange(-10., 15., 0.1):
        test_m = np.array([x, y])
        y_p = predict(test_m)
        print(y_p)
        # if y_p == 0.5:
        #     test_x.append(x)
        #     test_y.append(y)

############################################训练数据集可视化
plt.scatter(X[0, 0:margin], X[1, 0:margin], marker='x', color="red", label="预测类1", linewidth=2)
plt.scatter(X[0, margin:len(D)], X[1, margin:len(D)], marker='*', color="blue", label="预测类-1", linewidth=2)
# plt.plot(test_x, test_y, 'g--')
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.title("BP神经网络分类结果", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.show()