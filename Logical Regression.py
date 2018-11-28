# 三大件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random
import time
import pdb


# 定义一个字典，把字符串转成数字
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


def load_data(path):
    data = np.loadtxt(path, dtype=float,
                      delimiter=',', converters={4: iris_type})  # 数据读取
    data = data[:100, :]
    # np.delete(data, [2, 3], axis=1)
    data = data[:100, [0, 1, 4]]
    return data


# 洗牌！！！打乱原有顺序!!!
def shuffledata(data):
    np.random.shuffle(data)
    x = data[:, :3]
    y = data[:, 3]    # Y 变成了行向量啊 100 * 1  ?????????
    y = y.reshape(-1, 1)
    return x, y

def scatter_figure(x1):
    plt.scatter(x1[:50, 0], x1[:50, 1], s=30, c='r', marker='*', label='Iris-setosa')
    plt.scatter(x1[50:100, 0], x1[50:100, 1], s=30, c='b', marker='o', label='Iris-versicolor')
    # plt.legend()
    # plt.xlabel('Width')
    # plt.ylabel('Length')
    # plt.grid()
    # plt.show()


'''
目标：
建立分类器，求解出参数（本例中为三个 theta_0, theta_1, theta_2）
设定阈值，根据阈值结果判断分类情况，是0还是1！
'''
'''
要完成的模块：
1.sigmoid 函数，从线性值映射到概率（0，1）之间的函数
2.model：返回预测结果值
3.cost：根据参数计算损失
4.gradient：计算每个参数的梯度方向
5.descent：进行参数更新
6.accuracy：计算精度
'''


def sigmoid(z):
    return 1/(1+np.exp(-z))


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))  # 要记得X中多加了一列 1


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1-y, np.log(1-model(X, theta)))   # 都是列向量了,和样本个数一样
    return np.sum(left-right)/len(y)


def gradient(X, y, theta):
    error = (model(X, theta)-y)  # ravel() 函数将多维数组降为一维
    # X的转置，m*n  ----> n*m  m*1    ===n*1
    X1 = X.T   # n * m
    term = np.dot(X1, error)   #
    term = term.transpose()/len(y)
    return term    # 要记得取平均呀

# 这个地方有问题，目标是输出 1*n 维的向量，输出的确是 m*n 维！！

# 不同的梯度下降方法
'''
1.批量梯度下降方法
2.随机梯度下降方法
3.小批量梯度下降方法
下面是比较不同的停止策略，即什么时候迭代终止呢？
STOP_ITER 是指给定的迭代次数
STOP_COST 是根据损失函数的值来确定
STOP_GRAD 是根据梯度值来确定
'''
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


# X,y数据，theta 系数，batchSize 为采样个数，=1 则为随机采样，
# =len（X）为批量，在两者之间为小批量。stopType 为停止策略，
# thresh 为阈值，alpha 为学习率，也叫步长
def desecnt(data, theta, batchSize, stopType, thresh, alpha):
    # 初始化
    init_time = time.time()
    i = 0   # 迭代次数
    k = 0   # 第几个 batch
    X, Y = shuffledata(data)
    grad = np.zeros(theta.shape)  # 梯度占位
    costs = [cost(X, Y, theta)]   # 损失
    while True:
        grad = gradient(X[k:k+batchSize], Y[k:k+batchSize], theta)

        k += batchSize

        if k >= len(Y):
            k = 0
            X, Y = shuffledata(data)

        theta = theta - alpha * grad  # 参数更新

        costs.append(cost(X, Y, theta))  # 计算新的损失，并储存在数组中
        i += 1
        # 判断是否停止
        if stopType == STOP_ITER:
            value1 = i
        elif stopType == STOP_COST:
            value1 = costs
        elif stopType == STOP_GRAD:
            value1 = grad
        if stopCriterion(stopType, value1, thresh):
            break
    return theta, i-1, costs, grad, time.time()-init_time


# 画图！
def runexpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iteration, costs, grad, dur = desecnt(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original"if (data[:, 1]).sum() > 1 else "Scaled"
    name += "data - learning rate: {}--".format(alpha)
    if batchSize == len(data):
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent -Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {}- Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".
          format(name, theta, iteration, costs[-1], dur))
    plt.plot(np.arange(len(costs)), costs, 'r')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(name.upper() + ' -Error vs Iteration')
    plt.show()
    return theta


def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]  #  111111111


#  x, y = np.split(data, (4,), axis=1)
if __name__ == '__main__':
    # 获得初始样本数据, 分为两个类。
    path = 'iris.data'
    data_init = load_data(path)  # 获取数据，记住这个方法
    X_normal = data_init[:, :2]
    # scatter_figure(X_normal)  # 散点图，可视化
    # 在数据X中多加一列1
    one = np.ones(100)
    data = np.insert(data_init, 0, one, axis=1)

    data_train = data[:80]
    data_test = data[80:]

    X, Y = shuffledata(data_train)
    # 参数初始化
    theta = np.zeros([1, 3])
    batchSize = 15
    # 1.
    # stopType = STOP_ITER
    # thresh = 5000
    # alpha = 0.01
    # 2.
    # stopType = STOP_COST
    # thresh = 0.0000001
    # alpha = 0.001

    # 3.
    stopType = STOP_GRAD
    thresh = 0.001
    alpha = 0.01

    theta1 = runexpe(data_train, theta, batchSize, stopType, thresh, alpha)
    # theta, a, costs, grad, b = desecnt(data, theta, batchSize, stopType, thresh, alpha)
    # print(costs)
    # scatter_figure(X_normal)
    # y_hat = model(X, theta1)

    predictions = predict(X, theta1)

    correct = [1 if ((a == 1 and b == 1)or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
    accuracy = (sum(map(int, correct)) / len(correct)) *100
    print('Accuracy in data_train = {}%'.format(accuracy))

    X_test, y_test = shuffledata(data_test)
    predictions_2 = predict(X_test, theta1)
    correct_2 = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions_2, y_test)]
    # accuracy_2 = s

    accuracy_2 = (sum(map(int, correct_2)) / len(correct_2)) * 100
    print('Accuracy in data_test = {}%'.format(accuracy_2))


    # 设定阈值
    # print(y_hat)
    #
    # # plt.plot(X[:, 1], y, 'r-', linewidth=2, label='LR')
    # plt.legend()
    # plt.show()


