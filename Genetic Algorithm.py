import numpy as np
import math
import matplotlib.pyplot as plt
import random

pos_list = np.array([[530.0, 209.0], [1126.0, 1229.0], [1492.0, 901.0], [692.0, 1332.0],
                     [1485.0, 163.0], [1960.0, 710.0], [1602.0, 922.0], [1692.0, 1134.0],
                     [733.0, 1444.0], [570.0, 1280.0], [1243.0, 705.0], [664.0, 1318.0],
                     [217.0, 156.0], [1319.0, 579.0], [933.0, 1323.0], [375.0, 93.0],
                     [295.0, 1368.0], [1417.0, 1243.0], [1833.0, 933.0], [1980.0, 238.0],
                     [1136.0, 1320.0], [123.0, 999.0], [1022.0, 496.0], [19.0, 492.0],
                     [261.0, 1022.0], [846.0, 115.0], [216.0, 612.0], [570.0, 905.0],
                     [1854.0, 516.0], [1041.0, 332.0], [17.0, 825.0], [1790.0, 1203.0],
                     [1715.0, 307.0], [1357.0, 8.0], [348.0, 20.0], [371.0, 1176.0],
                     [171.0, 1067.0], [1230.0, 1052.0], [368.0, 549.0], [23.0, 1329.0],
                     [1823.0, 1343.0], [1178.0, 893.0], [590.0, 842.0], [339.0, 766.0],
                     [1826.0, 1437.0], [1751.0, 849.0], [322.0, 1379.0], [1858.0, 232.0],
                     [348.0, 1112.0], [1136.0, 898.0], [242.0, 1154.0], [1598.0, 680.0],
                     [288.0, 530.0], [1808.0, 328.0], [1742.0, 249.0], [1935.0, 65.0],
                     [503.0, 1199.0], [36.0, 452.0], [1400.0, 970.0], [128.0, 1388.0],
                     [976.0, 230.0], [666.0, 1444.0], [1430.0, 1456.0], [1643.0, 425.0]])

city_num = pos_list.shape[0]
city_list = list()
colony_size = 300
last_generation = list()
next_generation = list()

max_iter = 481

iter_aver_list = np.zeros((max_iter))
iter_min_list = np.zeros((max_iter))
best_path_history = np.zeros((max_iter, city_num))

switch_possibility =1
mutate_possibility = 0.5


class City():
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.phero_density = np.zeros((city_num))
        self.dis_count = np.zeros((city_num))


class Path():
    def __init__(self):
        self.id = id
        self.path = np.zeros((city_num))
        self.fitness = 0
        self.length = 0

    def gen_random_path(self):
        self.path = np.random.permutation(range(city_num))

    def cal_length(self):
        self.length = 0
        for i in range(city_num - 1):
            m, n = int(self.path[i]), int(self.path[i + 1])
            self.length += city_list[m].dis_count[n]
        m, n = int(self.path[city_num - 1]), int(self.path[0])
        self.length += city_list[m].dis_count[n]
        self.fitness = 1 / self.length
        return self.length


def cal_dis(c1: City, c2: City) -> float:
    dis = math.sqrt(math.pow(c1.x - c2.x, 2) + math.pow(c1.y - c2.y, 2))
    return dis


def crossover(p1: Path, p2: Path):
    start = random.randint(0, city_num)
    finish = random.randint(0, city_num)
    if finish < start:
        start, finish = finish, start
    new_path1 = np.zeros((city_num))
    new_path2 = np.zeros((city_num))
    swap1 = p1.path[start:finish]
    swap2 = p2.path[start:finish]
    new_path1[start:finish] = swap2
    new_path2[start:finish] = swap1
    j = 0
    for i in p1.path:
        if i not in swap2:
            if j == start:
                j = finish
            new_path1[j] = i
            j += 1
    j = 0
    for i in p2.path:
        if i not in swap1:
            if j == start:
                j = finish
            new_path2[j] = i
            j += 1
    c1 = Path()
    c2 = Path()
    c1.path = new_path1
    c2.path = new_path2
    return c1, c2


def mutate(p: Path):
    m = random.randint(0, city_num - 1)
    n = random.randint(0, city_num - 1)
    new_path = p.path
    new_path[m], new_path[n] = new_path[n], new_path[m]
    c = Path()
    c.path = new_path
    return c


if __name__ == "__main__":
    best_iter = 0
    global_min_lenth = 30000
    # initialize cities

    for i in range(city_num):
        new_city = City(pos_list[i][0], pos_list[i][1], i)
        dis_map = np.zeros(city_num)
        city_list.append(new_city)
    for i in range(city_num):
        for j in range(city_num):
            city_list[i].dis_count[j] = cal_dis(city_list[i], city_list[j])
            city_list[j].dis_count[i] = cal_dis(city_list[i], city_list[j])

    # initialize the colony
    for i in range(colony_size):
        new_path = Path()
        new_path.gen_random_path()
        last_generation.append(new_path)

    for iter in range(max_iter):
        iter_len = np.zeros((colony_size))
        iter_best_path = np.zeros((city_num))

        next_generation.clear()

        for i in range(int(colony_size / 2 - 1)):
            if random.random() < switch_possibility:
                c1, c2 = crossover(last_generation[i], last_generation[-i])
                next_generation.append(c1)
                next_generation.append(c2)

        for i in range(colony_size):
            next_generation.append(last_generation[i])
            if random.random() < mutate_possibility:
                c = mutate(last_generation[i])
                next_generation.append(c)
        # choose
        iter_min_len = last_generation[0].cal_length()
        gen_size = len(next_generation)
        lenth_list = np.zeros((gen_size))
        for i in range(gen_size):
            length = next_generation[i].cal_length()
            lenth_list[i] = length
            if length < iter_min_len:
                iter_best_path = next_generation[i].path
                iter_min_len = length
        lenth_list.sort()
        iter_len = lenth_list[0:colony_size]
        threshold = lenth_list[colony_size - 1]
        last_generation.clear()
        for p in next_generation:
            if p.length <= threshold:
                last_generation.append(p)
        last_generation = last_generation[0:colony_size]

        iter_aver_len = np.average(iter_len)
        iter_aver_list[iter] = iter_aver_len
        iter_min_list[iter] = iter_min_len
        best_path_history[iter] = iter_best_path
        if iter_min_len < global_min_lenth:
            global_min_lenth = iter_min_len
            best_iter = iter
        print("iteration: ", iter, "aver:", iter_aver_len, "min:", iter_min_len)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    axes[0].plot(iter_aver_list, 'k', marker='*')
    axes[0].set_title('Average Length')
    axes[0].set_xlabel(u'iteration')

    axes[1].plot(iter_min_list, 'k', marker='<')
    axes[1].set_title('Best Length')
    axes[1].set_xlabel(u'iteration')
    fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
    plt.close()
    plt.show()

    k = 0
    while k < max_iter:
        finalpath = best_path_history[k]
        plt.plot(pos_list[:, 0], pos_list[:, 1], 'r.', marker='>')
        plt.xlim([-100, 2000])
        plt.ylim([-100, 1500])
        for i in range(city_num - 1):
            m, n = int(finalpath[i]), int(finalpath[i + 1])
            plt.plot([pos_list[m][0], pos_list[n][0]], [pos_list[m][1], pos_list[n][1]], "k")
        m, n = int(finalpath[city_num - 1]), int(finalpath[0])
        plt.plot([pos_list[m][0], pos_list[n][0]], [pos_list[m][1], pos_list[n][1]], "k")
        ax = plt.gca()
        ax.set_title("Best Path")
        ax.set_xlabel('X_axis')
        ax.set_ylabel('Y_axis')

        plt.savefig(str(k) + 'path.png', dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()
        if k <= 40:
            k += 20
        elif k<300:
            k+=60
        else:
            k += 180

    finalpath = best_path_history[best_iter]
    print("Best iter:", best_iter, "minimum length:", global_min_lenth)
    print("best path:", finalpath)
    plt.plot(pos_list[:, 0], pos_list[:, 1], 'r.', marker='>')
    plt.xlim([-100, 2100])
    plt.ylim([-100, 1600])
    for i in range(city_num - 1):
        m, n = int(finalpath[i]), int(finalpath[i + 1])
        plt.plot([pos_list[m][0], pos_list[n][0]], [pos_list[m][1], pos_list[n][1]], "k")
    m, n = int(finalpath[city_num - 1]), int(finalpath[0])
    plt.plot([pos_list[m][0], pos_list[n][0]], [pos_list[m][1], pos_list[n][1]], "k")
    ax = plt.gca()
    ax.set_title("Best Path " + " iter: " + str(best_iter) + " minlen: " + str(global_min_lenth))
    ax.set_xlabel('X_axis')
    ax.set_ylabel('Y_axis')

    plt.savefig('Bestpath.png', dpi=500, bbox_inches='tight')
    plt.close()
    plt.show()
