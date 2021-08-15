import numpy as np
import matplotlib.pyplot as plt

city_num = 64
pos_list = np.zeros((city_num, 2))

for i in range(city_num):
    pos_list[i][0] = np.random.randint(0, 2000)
    pos_list[i][1] = np.random.randint(0, 1500)

for i in range(int(city_num / 4)):
    print("[", pos_list[4 * i][0], ",", pos_list[4 * i][1], "],[", pos_list[4 * i + 1][0], ",", pos_list[4 * i + 1][1],
          "],[", pos_list[4 * i + 2][0], ",", pos_list[4 * i + 2][1], "],[", pos_list[4 * i + 3][0], ",",
          pos_list[4 * i + 3][1], "],")

plt.plot(pos_list[:, 0], pos_list[:, 1], 'r.', marker='>')
plt.xlim([-100, 2100])
plt.ylim([-100, 1600])
ax = plt.gca()
ax.set_title("Origin Graph")
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

plt.savefig('Origin Graph.png', dpi=500, bbox_inches='tight')
plt.close()
plt.show()