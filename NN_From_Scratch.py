import copy
import math
import os
import pickle

import numpy as np

import struct

import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm

# 数据集处理
# train 训练集（随便用）丨valid 验证集（不能调参）丨text 测试集（不能调参）丨(real world)外界数据，拿到就丢训练集
# 数据集路径
dataset_path = Path('./MNIST')
# 训练图片集路径
train_img_path = dataset_path / 'train-images.idx3-ubyte'  # 训练集图片
train_lab_path = dataset_path / 'train-labels.idx1-ubyte'  # 训练集标签
test_img_path = dataset_path / 't10k-images.idx3-ubyte'  # 测试集图片
test_lab_path = dataset_path / 't10k-labels.idx1-ubyte'  # 测试集标签


# 激活函数
def bypass(x):
    return x


def tanh(x):  # 双曲正切
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x - np.max(x))  # 求指数,给所有数减去最大值，防止指数爆炸
    return exp / exp.sum()  # 分子分母同除一个数，所以没影响


# softmax导数函数
def d_softmax(data):
    sm = softmax(data)
    # diag:对角矩阵  outer：第一个参数挨个乘以第二个参数得到矩阵
    return np.diag(sm) - np.outer(sm, sm)


# tanh导数函数
# def d_tanh(data):
# 	return np.diag(1/(np.cosh(data))**2)
# tanh导数函数优化：
def d_tanh(data):
    return 1 / (np.cosh(data)) ** 2


def d_bypass(x):
    return 1


# 求偏导算符
differential = {softmax: d_softmax, tanh: d_tanh, bypass: d_bypass}

d_type = {bypass: 'times', softmax: 'dot', tanh: 'times'}

dimensions = [28 * 28, 100, 10]  # 每层神经网络的神经元个数，第一层输入层28*28个，中间层（隐藏层）100个，输出层10个
activation = [bypass, tanh, softmax]

distribution = [
    {},  # leave it empty!!
    {'b': [0, 0],
     'w': [-math.sqrt(6 / (dimensions[0] + dimensions[1])), math.sqrt(6 / (dimensions[0] + dimensions[1]))]},
    {'b': [0, 0],
     'w': [-math.sqrt(6 / (dimensions[1] + dimensions[2])), math.sqrt(6 / (dimensions[1] + dimensions[2]))]},
]


# 初始化参数b
def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer]) * (dist[1] - dist[0]) + dist[0]  # 生成N个随机数 给出从dist[0]~dist[1]的随机数


# 初始化参数w
def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer - 1], dimensions[layer]) * (dist[1] - dist[0]) + dist[
        0]  # 生成N个随机数 给出从dist[0]~dist[1]的随机数


# 初始化参数方法
def init_parameters():
    parametre = []
    for i in range(len(distribution)):
        layer_parametre = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parametre['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parametre['w'] = init_parameters_w(i)
                continue
        parametre.append(layer_parametre)
    return parametre


parameters = init_parameters()


# 预测函数
def predict(img, parameters):
    # 神经网络第0层处理
    l_in = img
    l_out = activation[0](l_in)
    for layer in range(1, len(dimensions)):  # 神经网络总层个数，并从1开始循环
        l_in = np.dot(l_out, parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
    return l_out


# 原始训练集6W，1W测试集，将训练集分为5W训练集和1W验证集
train_num = 50000  # train 训练集数量
valid_num = 10000  # valid 验证集数量
test_num = 10000  # text 测试集数量

# 读入训练图片集和验证图片集
with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255  # -1自动计算该维度size 28*28像素,除以255是防止二值化
    train_img = tmp_img[:train_num]  # 实际用来训练集图片
    valid_img = tmp_img[train_num:]  # 实际验证集图片

# 读入测试图片集
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]  # 实际用来训练集标签
    valid_lab = tmp_lab[train_num:]  # 实际验证集标签

# 读入训练标签和验证标签
with open(test_img_path, 'rb') as f:  # 测试集图片
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255  # -1自动计算该维度size 28*28像素

# 读入测试标签
with open(test_lab_path, 'rb') as f:  # 测试集标签
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)


# 展示训练图片
def show_train(index):  # 随机取出训练集图片
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label：{}'.format(train_lab[index]))
    plt.show()


# 展示验证图片
def show_valid(index):  # 随机取出验证集图片
    plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
    print('label：{}'.format(valid_lab[index]))
    plt.show()


# 展示测试图片
def show_test(index):  # 随机取出测试集图片
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label：{}'.format(test_lab[index]))
    plt.show()


# print(predict(train_img[0], init_parameters()))  # 对predict的测试

# 对softmax的一些测试
# h = 0.000001
# func = softmax
# input_len = 4
# for i in range(input_len):
#     test_input = np.random.rand(input_len)
#     derivative = differential[func](test_input)
#     value1 = func(test_input)
#     test_input[i] += h
#     value2 = func(test_input)
#     # print((value2 - value1) / h)
#     # print(derivative[i])
#     print(derivative[i]-(value2 - value1) / h)


# lab解析函数
# 将数解析为某一位置为1的一维矩阵
onehot = np.identity(dimensions[-1])


# loss函数的作用就是描述模型的预测值与真实值之间的差距大小
# 求平方差函数
def sqr_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff, diff)


# 计算梯度
def grad_parameters(img, lab, parameters):
    # 处理第0层
    l_in_list = [img]
    l_out_list = [activation[0](l_in_list[0])]
    for layer in range(1, len(dimensions)):  # 神经网络总层个数，并从1开始循环
        l_in = np.dot(l_out_list[layer - 1], parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
        l_in_list.append(l_in)
        l_out_list.append(l_out)

    d_layer = -2 * (onehot[lab] - l_out_list[-1])

    grad_result = [None] * len(dimensions)

    # 反向传播
    for layer in range(len(dimensions) - 1, 0, -1):  # 左闭右开
        if d_type[activation[layer]] == 'times':
            d_layer = differential[activation[layer]](l_in_list[layer]) * d_layer
        if d_type[activation[layer]] == 'dot':
            d_layer = np.dot(differential[activation[layer]](l_in_list[layer]), d_layer)
        grad_result[layer] = {}
        grad_result[layer]['b'] = d_layer
        grad_result[layer]['w'] = np.outer(l_out_list[layer - 1], d_layer)
        d_layer = np.dot(parameters[layer]['w'], d_layer)

    return grad_result


# 验证部分
parameters = init_parameters()


# # b2
# h=0.001
# layer=2
# pname='b'
# grad_list = []
# for i in range(len(parameters[layer][pname])):
#     img_i = np.random.randint(train_num)  # 图片
#     test_parametes = init_parameters()  # 初始化
#     derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parametes)[layer][pname]
#     value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parametes)
#     test_parametes[layer][pname][i] += h
#     value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parametes)
#     grad_list.append(derivative[i] - (value2 - value1) / h)
# print(np.abs(grad_list).max())


# # b1
# h = 0.00001
# layer = 1
# pname = 'b'
# grad_list = []
# for i in range(len(parameters[layer][pname])):
#     img_i = np.random.randint(train_num)  # 图片
#     test_parametes = init_parameters()  # 初始化
#     derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parametes)[layer][pname]
#     value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parametes)
#     test_parametes[layer][pname][i] += h
#     value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parametes)
#     grad_list.append(derivative[i] - (value2 - value1) / h)
# print(np.abs(grad_list).max())


# # w2
# h=0.00001
# layer = 2
# pname = 'w'
# grad_list = []
# for i in range(len(parameters[layer][pname])):  # 行
#     for j in range(len(parameters[layer][pname][0])):  # 列
#         img_i=np.random.randint(train_num)  # 图片
#         test_parametes=init_parameters()  # 初始化
#         derivative=grad_parameters(train_img[img_i],train_lab[img_i],test_parametes)[layer][pname]
#         value1=sqr_loss(train_img[img_i],train_lab[img_i],test_parametes)
#         test_parametes[layer][pname][i][j] += h
#         value2=sqr_loss(train_img[img_i],train_lab[img_i],test_parametes)
#         grad_list.append(derivative[i][j] - (value2 - value1) / h)
# print(np.abs(grad_list).max())


# # w1
# h=0.00001
# layer = 1
# pname = 'w'
# grad_list = []
# for i in tqdm(range(len(parameters[layer][pname]))):  # 行
#     for j in range(len(parameters[layer][pname][0])):  # 列
#         img_i=np.random.randint(train_num)  # 图片
#         test_parametes=init_parameters()  # 初始化
#         derivative=grad_parameters(train_img[img_i],train_lab[img_i],test_parametes)[layer][pname]
#         value1=sqr_loss(train_img[img_i],train_lab[img_i],test_parametes)
#         test_parametes[layer][pname][i][j] += h
#         value2=sqr_loss(train_img[img_i],train_lab[img_i],test_parametes)
#         grad_list.append(derivative[i][j] - (value2 - value1) / h)
# print(np.abs(grad_list).max())

# 判断神经网络精确度部分
def valid_loss(parameters):
    loss_accu = 0
    for img_i in range(valid_num):
        loss_accu += sqr_loss(valid_img[img_i], valid_lab[img_i], parameters)
    return loss_accu / (valid_num / 10000)


# valid 集的精确度
def valid_accuracy(parameters):
    correct = [predict(valid_img[img_i], parameters).argmax() == valid_lab[img_i] for img_i in range(valid_num)]
    return correct.count(True) / len(correct)


def train_loss(parameters):
    loss_accu = 0
    for img_i in range(train_num):
        loss_accu += sqr_loss(train_img[img_i], train_lab[img_i], parameters)
    return loss_accu / (train_num / 10000)


# train 集的精确度
def train_accuracy(parameters):
    correct = [predict(train_img[img_i], parameters).argmax() == train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True) / len(correct)


def grad_add(grad1, grad2):
    for layer in range(1, len(grad1)):  # 从第一层开始
        for pname in grad1[layer].keys():
            grad1[layer][pname] += grad2[layer][pname]
    return grad1


def grad_divide(grad, denominator):
    for layer in range(1, len(grad)):  # 从第一层开始
        for pname in grad[layer].keys():
            grad[layer][pname] /= denominator
    return grad


def combvine_parametres(parameters, grad, learn_rate):
    parameter_tmp = copy.deepcopy(parameters)  # 因为引用类型的数据共用一个存储地址，所以深拷贝
    for layer in range(1, len(parameter_tmp)):
        for pname in parameter_tmp[layer].keys():
            parameter_tmp[layer][pname] -= learn_rate * grad[layer][pname]
    return parameter_tmp


def train_batch(current_batch, parameters, batch_size):
    grad_accu = grad_parameters(train_img[current_batch * batch_size + 0], train_lab[current_batch * batch_size + 0],
                                parameters)
    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(train_img[current_batch * batch_size + img_i],
                                   train_lab[current_batch * batch_size + img_i],
                                   parameters)
        grad_add(grad_accu, grad_tmp)
    grad_divide(grad_accu, batch_size)
    return grad_accu


# 训练
def train_final(parameters_data):
    # 读入数据
    parameters = parameters_data[0]['parameters']
    batch_size = parameters_data[3]['batch_size']
    epoch_num = parameters_data[2]['epoch_num']
    learn_rate = parameters_data[1]['learn_rate']
    current_epoch = parameters_data[4]['current_epoch']
    train_loss_list = parameters_data[5]['train_loss_list']
    train_accu_list = parameters_data[7]['train_accu_list']
    valid_loss_list = parameters_data[6]['valid_loss_list']
    valid_accu_list = parameters_data[8]['valid_accu_list']

    for epoch in tqdm(range(epoch_num)):
        for i in range(train_num // batch_size):
            # if i%100==99:
            #     print('running batch {}/{}'.format(i+1,train_num//batch_size))
            grad_tmp = train_batch(i, parameters, batch_size)
            parameters = combvine_parametres(parameters, grad_tmp, learn_rate)
        current_epoch += 1
        train_loss_list.append(train_loss(parameters))
        train_accu_list.append(train_accuracy(parameters))
        valid_loss_list.append(valid_loss(parameters))
        valid_accu_list.append(valid_accuracy(parameters))

    print(valid_accuracy(parameters))  # 训练后正确率

    lower = 0
    if epoch_num > 1:
        get_img_list(valid_loss_list, 'validation loss', train_loss_list, 'train loss', lower=0)
        get_img_list(valid_accu_list, 'valid accu', train_accu_list, 'train accu', lower=0)
    else:
        print(valid_loss_list, valid_accu_list, valid_accu_list, train_accu_list)  # 训练后正确率

    parameters_data = parameters_data_encoded(parameters, learn_rate, epoch_num, batch_size, current_epoch,
                                              train_loss_list, valid_loss_list, train_accu_list, valid_accu_list)

    return parameters_data


# 获取不同的learn_rate对应的loss图像
def get_loss(lr_list, lower, upper, step, grad_lr):
    for lr_pow in tqdm(np.linspace(lower, upper, num=step)):
        learn_rate = 10 ** lr_pow  # 学习率（10^lr_pow）
        parameters_tmp = combvine_parametres(parameters, grad_lr, learn_rate)
        train_loss_tmp = train_loss(parameters_tmp)
        lr_list.append([lr_pow, train_loss_tmp])

    plt.plot(np.array(lr_list)[:, 0], np.array(lr_list)[:, 1], color='black', label='lr list', marker='.')
    plt.legend()  # 显示label
    plt.grid()  # 添加网格
    plt.show()


# 获取对应数据的图片
def get_img_list(list1, label1, list2, label2, lower=0):
    plt.plot(list1[lower:], color='black', label=label1, marker='.')
    plt.plot(list2[lower:], color='red', label=label2, marker='.')
    plt.legend()  # 显示label
    plt.grid()  # 添加网格
    plt.show()


# parameters数据编码
def parameters_data_encoded(parameters, learn_rate, epoch_num, batch_size, current_epoch, train_loss_list,
                            valid_loss_list, train_accu_list, valid_accu_list):
    parameters_data = [
        {'parameters': parameters},
        {'learn_rate': learn_rate},
        {'epoch_num': epoch_num},
        {'batch_size': batch_size},
        {'current_epoch': current_epoch},
        {'train_loss_list': train_loss_list},
        {'valid_loss_list': valid_loss_list},
        {'train_accu_list': train_accu_list},
        {'valid_accu_list': valid_accu_list}
    ]
    return parameters_data


# parameters模型保存
def parameters_data_save(parameters_data, path):
    try:
        # 以二进制的形式打开文件
        with open(path.replace('\\', "/"), "wb") as f:
            # 将列表a序列化后写入文件
            pickle.dump(parameters_data, f)
        return "保存成功，路径为" + path
    except:
        return "保存失败"


# parameters模型读入
def parameters_data_open(path):
    try:
        with open(path.replace('\\', "/"), 'rb') as f:
            parameters_data = pickle.load(f)
        return parameters_data
    except:
        return "读入失败"
