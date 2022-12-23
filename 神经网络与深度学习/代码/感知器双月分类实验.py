import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ##################### 添加图片文字 ###########################################
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ############################################################################

def moon(N, w, r, d):
    '''
    # :param w: 半月宽度
    # :param r: x轴偏移量
    # :param d: y轴偏移量
    # :param N: 半月散点数量
    :return: data (2*N*3) 月亮数据集
             data_dn (2*N*1) 标签
    '''
    data = np.ones((2 * N, 4))
    # 半月1的初始化
    r1 = 10  # 半月1的半径,圆心
    w1 = np.random.uniform(-w / 2, w / 2, size=N)  # 半月1的宽度范围
    theta1 = np.random.uniform(0, np.pi, size=N)  # 半月1的角度范围
    x1 = (r1 + w1) * np.cos(theta1)  # 行向量
    y1 = (r1 + w1) * np.sin(theta1)
    label1 = [1 for i in range(1, N + 1)]  # label for Class 1

    # 半月2的初始化
    r2 = 10  # 半月1的半径,圆心
    w2 = np.random.uniform(-w / 2, w / 2, size=N)  # 半月1的宽度范围
    theta2 = np.random.uniform(np.pi, 2 * np.pi, size=N)  # 半月1的角度范围
    x2 = (r2 + w2) * np.cos(theta2) + r
    y2 = (r2 + w2) * np.sin(theta2) - d
    label2 = [-1 for i in range(1, N + 1)]  # label for Class 2

    data[:, 1] = np.concatenate([x1, x2])
    data[:, 2] = np.concatenate([y1, y2])
    data[:, 3] = np.concatenate([label1, label2])
    return data


def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1


def actual_func(xn, wn):
    '''实际响应'''
    yn = sgn(np.dot(wn.T, xn))
    return yn


def updata(xn, wn, dn, learning_rate):
    '''更新权值'''
    if np.dot(wn, xn) * dn <= 0:
        wn = wn + learning_rate * xn * dn
    return wn


def loss_func(dn, yn):
    '''均方误差'''
    return (dn - yn) ** 2


def train(data, dn, wn, iterations, learning_rate):
    loss = []
    for j in range(iterations):
        i = 0
        loss_temp = []
        for xn in data:
            yn = actual_func(xn, wn)
            wn = updata(xn, wn, dn[i], learning_rate)
            loss_temp.append(loss_func(dn[i], yn))
            i = i + 1
        loss.append(sum(loss_temp) / len(loss_temp))
    return loss, wn


def test(xn, dn):
    right_count = wrong_count = i = 0
    for x in xn:
        yn = actual_func(x, wn)
        if yn == dn[i]:
            right_count += 1
        else:
            wrong_count += 1
        right_count = right_count / (right_count + wrong_count)
        wrong_count = wrong_count / (right_count + wrong_count)
        i += 1
    return wrong_count, right_count


if __name__ == '__main__':
    # 初始化参数
    N = 1000  # 单个月亮数据集数据量
    w = 2
    r = 8
    d = 1
    iter = 100  # 迭代次数
    learning_rate = 1  # 学习率
    # wn = np.array([1, 0, 0])
    wn = (np.random.rand(3,) - 0.5) * 0.2
    data = moon(N, w, r, d)
    print((data[:, 1:3]))
    print((data[:, 3]))
    margin = list(data[:, 3]).index(-1)
    print(data.shape)
    print(margin)
    '''随机采用20%的数据用于测试，剩下的80%用于构建训练集合'''
    data_train, data_test, dn_train, dn_test = train_test_split(data[:, 1:3], data[:, 3], test_size=0.20,
                                                                random_state=None)
    '''训练数据调整'''
    train_data = np.ones((1600, 3))
    print(data_train.shape)
    print(data_test.shape)
    train_data[:, 1:3] = data_train
    # print(train_data[:, 1:3].shape)
    train_dn = dn_train
    # print(train_dn.shape)
    '''迭代 训练 寻找最优解wn'''
    loss, wn = train(train_data, train_dn, wn, iter, learning_rate)
    # print(loss)
    # print(wn.shape)   # 最优解wn
    print(wn)
    '''测试数据调整'''
    test_data = np.ones((400, 3))
    test_data[:, 1:3] = data_test
    test_dn = dn_test
    '''测试'''
    err, acc = test(test_data, test_dn)
    print('测试结果 错误率：', '%.2f' % err)
    print('测试结果 正确率：', '%.2f' % acc)
    '''决策平面 y+wx+b=0'''
    x = np.array(range(-15, 25))
    y = -x * wn[1] / wn[2] - wn[0] / wn[2]
    '''月亮数据集'''
    plt.figure()
    plt.title('月亮数据集', size=14)
    plt.xlabel('x 轴', size=14)
    plt.ylabel('y 轴', size=14)
    plt.grid(ls=':', color='gray', alpha=0.5)  # alpha是透明度
    plt.scatter(data[0:N, 1], data[0:N, 2], c='b', s=20, marker='+')
    plt.scatter(data[N:2 * N, 1], data[N:2 * N, 2], c='r', s=20, marker='+')
    plt.plot(x, y, 'k--')
    plt.savefig('./月亮数据集.png')

    '''损失函数'''
    x = [i for i in range(len(loss))]
    plt.figure()
    plt.title('损失函数曲线图', size=14)
    plt.xlabel('迭代次数', size=14)
    plt.ylabel('均方误差', size=14)
    plt.plot(x, loss, c='r', linewidth=2, linestyle='-', marker='o')
    plt.savefig('./损失函数.png')
    plt.show()
