import random
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

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
print(city_num)
city_list = list()
ant_num = 60
ant_list = list()
alpha = 1
beta = 10

max_iter = 181
vol_rate = 0.1
stddis = 10

iter_aver_list = np.zeros((max_iter))
iter_min_list = np.zeros((max_iter))
best_path_history = np.zeros((max_iter, city_num))


class City():
    def __init__(self, x, y, id):
        self.id = id
        self.x = x
        self.y = y
        self.phero_density = np.zeros((city_num))
        self.dis_count = np.zeros((city_num))


class Ant():
    def __init__(self, id):
        self.id = id
        self.visiting = -1
        self.visited = list()
        self.unvisited = list()

    def reset(self, start: int):
        self.visiting = start
        self.visited.clear()
        self.unvisited.clear()
        self.unvisited = list(range(city_num))
        self.unvisited.remove(start)

    def walk(self):
        while self.unvisited:
            eta_list = np.zeros(city_num)
            next_city = -1
            for cid in self.unvisited:
                dis = city_list[self.visiting].dis_count[cid]
                eta = math.pow(city_list[self.visiting].phero_density[cid], alpha) * math.pow(stddis / dis, beta)
                eta_list[cid] = eta
            if sum(eta_list) == 0:
                min_dis = 0
                for cid in self.unvisited:
                    if city_list[self.visiting].dis_count[cid] > min_dis:
                        min_dis = city_list[self.visiting].dis_count[cid]
                        next_city = cid
            else:
                prob_list_cumsum = (eta_list / sum(eta_list)).cumsum()
                prob_list_cumsum -= np.random.rand()
                next_city = list(prob_list_cumsum > 0).index(True)
            self.unvisited.remove(next_city)
            self.visited.append(self.visiting)
            self.visiting = next_city
        self.visited.append(self.visiting)

    def set_phero(self):
        for i in range(len(self.visited) - 1):
            next = self.visited[i + 1]
            dis = city_list[self.visited[i]].dis_count[next]
            city_list[self.visited[next]].phero_density[next] += stddis / dis
            city_list[self.visited[i]].phero_density[i] += stddis / dis
        first = self.visited[0]
        last = self.visited[city_num - 1]
        dis = city_list[last].dis_count[first]
        city_list[self.visited[last]].phero_density[first] += stddis / dis
        city_list[self.visited[first]].phero_density[last] += stddis / dis

    def tot_len(self):
        tot = 0
        for i in range(len(self.visited) - 1):
            next = self.visited[i + 1]
            tot += city_list[self.visited[i]].dis_count[next]
        next = self.visited[0]
        tot += city_list[self.visited[len(self.visited) - 1]].dis_count[next]
        return tot


def cal_dis(c1: City, c2: City) -> float:
    dis = math.sqrt(math.pow(c1.x - c2.x, 2) + math.pow(c1.y - c2.y, 2))
    return dis


if __name__ == "__main__":
    best_iter = 0
    min_lenth = 30000
    # initialize cities

    for i in range(city_num):
        new_city = City(pos_list[i][0], pos_list[i][1], i)
        dis_map = np.zeros(city_num)
        city_list.append(new_city)
    for i in range(city_num):
        for j in range(city_num):
            city_list[i].dis_count[j] = cal_dis(city_list[i], city_list[j])
            city_list[j].dis_count[i] = cal_dis(city_list[i], city_list[j])


    # initialize ants

    for i in range(ant_num):
        new_ant = Ant(i)
        ant_list.append(new_ant)

    # main loop
    for iter in range(max_iter):
        iter_len = np.zeros((ant_num))
        iter_best_path = np.zeros((city_num))
        # put ants in cities
        perm = np.random.permutation(range(city_num))
        for ant in ant_list:
            ant.reset(perm[ant.id % city_num])

        # ant walks
        for ant in ant_list:
            ant.walk()

        # ant leaves pheromone,calculate total length
        iter_min_len = ant_list[0].tot_len()
        for ant in ant_list:
            ant.set_phero()
            length = ant.tot_len()
            iter_len[ant.id] = length
            if length < iter_min_len:
                iter_min_len = length
                iter_best_path = ant.visited

        iter_aver_len = np.average(iter_len)
        iter_aver_list[iter] = iter_aver_len
        iter_min_list[iter] = iter_min_len
        best_path_history[iter] = iter_best_path
        if iter_min_len < min_lenth:
            min_lenth = iter_min_len
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
        else:
            k += 60

    finalpath = best_path_history[best_iter]
    print("Best iter:", best_iter, "minimum length:", min_lenth)
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
    ax.set_title("Best Path " + " iter: " + str(best_iter) + " minlen: " + str(min_lenth))
    ax.set_xlabel('X_axis')
    ax.set_ylabel('Y_axis')

    plt.savefig('Bestpath.png', dpi=500, bbox_inches='tight')
    plt.close()
    plt.show()
