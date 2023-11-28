import math
import random
import utils

ALPHA = 0.5


class TaskManager:  # 任务发布者
    def __init__(self, pos):
        self.pos = pos


class Task:  # 任务
    def __init__(self, K, L, H, O, tn: TaskManager):
        self.TN = tn
        self.data_size_K = K  # 数据量
        self.locations_L = L  # 地点集合
        self.compute_res_H = H  # 计算量
        self.output_size_O = O  # 输出数据量


class Robot:
    def __init__(self, pos, fn, wn, pn, vn, id):
        self.id = id  # 标识
        self.pos = pos  # 当前位置
        self.fn = fn  # 计算能力
        self.wn = wn  # 带宽
        self.pn = pn  # 功率
        self.vn = vn  # 速度

        self.kalfa = 0.002
        self.N0 = 0.01
        self.hn_lamda = 100
        self.theta = 10
        self.mulT = 50

    def cpt_rn(self, task: Task):
        # hn = 0.5
        hn = 50 * random.expovariate(self.hn_lamda)
        if not len(task.TN.pos) == 2:
            pass
        SINRn = self.pn * hn * math.pow(utils.get_dist(self.pos, task.TN.pos), -ALPHA) / self.N0
        return self.wn * math.log2(1 + SINRn)

    def cpt_t_delivery(self, task: Task):
        return task.data_size_K / self.cpt_rn(task)

    def cpt_snl(self, task: Task):
        snl = 0
        cur_pos = self.pos
        for np in task.locations_L:
            snl += utils.get_dist(cur_pos, np)
            # cur_pos = np
        return snl

    def cpt_En(self, task: Task):
        Ec = self.kalfa * task.data_size_K * task.compute_res_H * self.fn * self.fn
        Et = self.pn * self.cpt_t_delivery(task)
        Em = self.theta * self.cpt_snl(task)
        return Ec + Et + Em

    def cpt_Tn(self, task: Task):
        Tc = task.compute_res_H / self.fn
        Td = self.cpt_t_delivery(task)
        Tm = 10 * self.cpt_snl(task) / self.vn
        return (Tc + Td + Tm) * self.mulT

    def get_cost(self, task: Task):
        return self.cpt_En(task) + self.cpt_Tn(task)

    def get_single_value(self, task, price, price_sigma):
        return price - price_sigma * self.get_cost(task)

