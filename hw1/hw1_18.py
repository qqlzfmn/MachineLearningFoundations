import random
import time


# 导入数据集  输入文件路径，输出一个5×400的列表
def load_data_set(path):
    f = open(path)
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split()
        lines[i] = list(map(float, lines[i]))
    f.close()
    return lines


# 数据集预处理  输入一个列表，输出5×400的列表x和400×1的列表y
def data_set_processing(data):
    for i in range(len(data)):
        data[i].insert(0, 1.0)  # 在每个x向量插入一个x0=1
    n = len(data)  # n=|D|
    x, y = [], []
    for i in range(n):
        x.append(data[i][:5])
        y.append(data[i][-1:][0])
    return x, y


# sign()函数  输入一个数字，输出它的符号（令0的符号为负）
def sign(a):
    if a > 0:
        return 1
    else:
        return -1


# dot()函数  实现向量点乘
def dot(a, b):
    ans = 0
    for i in range(len(a)):
        ans = ans + a[i] * b[i]
    return ans


# plus()函数  实现向量相加
def plus(a, b):
    ans = []
    for i in range(len(a)):
        ans.append(a[i] + b[i])
    return ans


# mul()函数  实现数字乘向量
def mul(a, b):
    ans = []
    for i in range(len(a)):
        ans.append(a[i] * b)
    return ans


# Pocket算法  输入预处理后的数据x、y，输出学习后的w和运行次数t
def pocket(x_list, y_list):
    weight = [0] * len(x_list[0])  # w0 = [0, 0, 0, 0, 0]
    row = len(x_list)  # 测量x矩阵的行列数
    update = 0  # PLA算法执行次数
    seed = list(range(row))  # 创建range(400)的列表
    random.shuffle(seed)  # 将seed列表变为随机种子
    mistake = row
    w_pocket = []
    while True:
        # 找到错误并修正weight
        for i in seed:  # 用随机种子执行PLA算法
            h = sign(dot(weight, x_list[i]))
            if h != int(y_list[i]):
                weight = plus(weight, mul(x_list[i], y_list[i]))
                update = update + 1
                # 遍历D并求出新weight在D上的错误数
                new_mistake = 0
                for j in range(row):
                    new_h = sign(dot(weight, x_list[j]))
                    if new_h != int(y_list[j]):
                        new_mistake = new_mistake + 1
                # 当错误数比更新前少，则保留新的weight
                if new_mistake < mistake:
                    w_pocket = weight
                    mistake = new_mistake
                if update >= 50:  # 50次更新
                    return w_pocket


# 测试错误率  输入训练后的w和测试集的x、y，输出错误率
def test_error_rate(weight, x_list, y_list):
    errors = 0
    for i in range(len(y_list)):
        h = sign(dot(weight, x_list[i]))
        if h != int(y_list[i]):
            errors = errors + 1
    error_rate = float(errors) / len(y_list)
    return error_rate


# 主函数
if __name__ == '__main__':
    data_set = load_data_set("./data_set/hw1_18_train.dat")
    x_data, y_data = data_set_processing(data_set)
    test_set = load_data_set("./data_set/hw1_18_test.dat")
    x_test, y_test = data_set_processing(test_set)
    sum_rate = 0
    start_time = time.time()
    for t in range(2000):  # 求2000次错误率，取均值
        w = pocket(x_data, y_data)
        rate = test_error_rate(w, x_test, y_test)
        sum_rate = sum_rate + rate
        print('The ' + str(t + 1) + 'th error rate is ' + str(rate))
    end_time = time.time()
    print('Average error rate is ' + str(float(sum_rate) / 2000))
    print('Algorithm cost ' + str(end_time - start_time) + ' seconds.')
