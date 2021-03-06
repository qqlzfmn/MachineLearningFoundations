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
        data[i].insert(0, 1)  # 在每个x向量插入一个x0=1
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


# PLA算法  输入预处理后的数据x、y，输出学习后的w和运行次数t
def pla(x_list, y_list):
    weight = [0] * len(x_list[0])  # w0 = [0, 0, 0, 0, 0]
    row = len(x_list)  # 测量x矩阵的行列数
    update = 0  # PLA算法执行次数
    while True:
        # 找到错误并修正weight
        for i in range(row):
            h = sign(dot(weight, x_list[i]))
            if h != int(y_list[i]):
                weight = plus(weight, mul(x_list[i], y_list[i]))
                update = update + 1
        # 遍历D并求出weight在D上的错误数
        mistake = 0
        for j in range(row):
            new_h = sign(dot(weight, x_list[j]))
            if new_h != int(y_list[j]):
                mistake = mistake + 1
        # 当错误数降到0时停止算法
        if mistake == 0:
            break
    return weight, update


# 主函数
if __name__ == '__main__':
    data_set = load_data_set("./data_set/hw1_15_train.dat")
    x_data, y_data = data_set_processing(data_set)
    w, t = pla(x_data, y_data)
    print('PLA run ' + str(t) + ' updates.')
    print('Learned w is ' + str(w))
