import numpy
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
# BP网络第一层(隐藏层)神经元个数
M1 = 20
# BP网络第二层(输出层)神经元个数
M2 = 1
# 判断阈值
c = 0.5
###########################################类别数据处理的处理
for i in range(len(D)):
    if D[i] == -1:
        D[i] = 0
############################################参数随机初始化
# 第一层(2*2)
W1 = (np.random.rand(M1, 2) - 0.5) * 2
print("第一层权值矩阵:", W1)
b1 = (np.random.rand(M1) - 0.5) * 2
print("第一层偏置矩阵:", b1)
# 第二层(输出层)(1,2)
W2 = (np.random.rand(M1) - 0.5) * 2
print("第二层权值矩阵:", W2)
print("第二层权值矩阵:", W2.shape)
# (2,)
b2 = round((np.random.random() - 0.5) * 2, 8)
print("第二层偏置矩阵:", b2)

index = [i for i in range(len(train_data))]
np.random.shuffle(index)


############################################激活函数sigmoid
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


############################################训练
for i in range(iter):
    print(f"第{i + 1}次迭代")
    for n in range(len(X.T)):
        # 前向计算输出
        Z1 = np.dot(W1, X[:, n]) + b1
        a1 = sigmoid(Z1)
        Z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(Z2)
        # 反向计算误差
        delt2 = a2 * (1 - a2) * (a2 - D[n])
        delt1 = a1 * (1 - a1) * (np.dot(W2.T, delt2))
        W2 = W2 - learning_rate * (np.dot(delt2, a1.T))
        b2 = b2 - learning_rate * delt2
        W1 = W1 - learning_rate * (np.outer(delt1, X[:, n].T))
        b1 = b1 - learning_rate * delt1



# 找决策曲线
def predict(X):
    a1 = sigmoid(np.dot(W1, X.T) + b1)
    a2 = sigmoid(np.dot(W2, a1) + b2)
    return a2


print("开始计算损失函数")

test_x = []
test_y = []
# 损失函数
def loss_func():
    loss = []
    loss_n = []
    for i in range(100):
        for n in range(len(X.T)):
            out1 = sigmoid(np.dot(W1, X[:, n]) + b1)
            out2 = sigmoid(np.dot(W2, out1) + b2)
            if out2 < 0.5:
                out2 = 0
            else:
                out2 = 1
            loss_n.append((out2 - D[n]) ** 2)
        loss.append(sum(loss_n) / len(X.T))
    return loss


for x in np.arange(-15., 20., 0.1):
    for y in np.arange(-10., 15., 0.1):
        test_m = np.array([x, y])
        y_p = predict(test_m)
        if round(y_p, 2) == c:
            test_x.append(x)
            test_y.append(y)

loss = loss_func()

############################################训练数据集可视化
plt.scatter(X[0, 0:margin], X[1, 0:margin], marker='x', color="red", label="预测类1", linewidth=2)
plt.scatter(X[0, margin:len(D)], X[1, margin:len(D)], marker='*', color="blue", label="预测类-1", linewidth=2)
plt.plot(test_x, test_y, 'g--')
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.title("BP神经网络分类结果", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.show()

'''损失函数'''
x = [i for i in range(len(loss))]
plt.figure()
plt.title('损失函数曲线图', fontproperties=font)
plt.xlabel('迭代次数', fontproperties=font)
plt.ylabel('均方误差', fontproperties=font)
plt.plot(x, loss, c='r', linewidth=2, linestyle='-', marker='o')
plt.show()
