import math, random

GAOSS_SIGMA = 4
def get_dist(pos1, pos2):
    return math.pow((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]), 0.5)

def get_random_gauss_list( bottom, top, size):
    mean = (bottom + top) // 2
    val_list = []
    while len(val_list) < size:
        val = random.gauss(mean, GAOSS_SIGMA)
        while val < bottom or val > top:
            val = random.gauss(mean, GAOSS_SIGMA)
        val_list.append(val)
    return val_list

def get_random_pos(x_bottom, x_top, y_bottom, y_top, size):
    pos_list = []
    while len(pos_list) < size:
        x = random.randint(x_bottom, x_top)
        y = random.randint(y_bottom, y_top)
        pos_list.append([x, y])
    return pos_list