import math
import random


ALPHA = 0.5


class TaskManager:
    def __init__(self, pos):
        self.pos = pos
class Task:
    def __init__(self, K, L, H, O, tn: TaskManager):
        self.TN = tn
        self.data_size_K = K
        self.locations_L = L
        self.compute_res_H = H
        self.output_size_O = O



class Robot:
    def __init__(self, pos, fn, wn, pn, vn, id):
        self.id = id
        self.pos = pos
        self.fn = fn
        self.wn = wn
        self.pn = pn
        self.vn = vn

        self.kalfa = 0.002
        self.N0 = 0.01
        self.hn_lamda = 4
        self.theta = 10
        self.mulT = 50


    def cpt_rn(self, task:Task):
        hn = 0.5
        # hn = random.expovariate(self.hn_lamda)
        if not len(task.TN.pos) == 2:
            pass
        SINRn = self.pn * hn * math.pow(get_dist(self.pos, task.TN.pos), -ALPHA) / self.N0
        return self.wn * math.log2(1 + SINRn)
    def cpt_t_delivery(self, task: Task):
        return task.data_size_K / self.cpt_rn(task)

    def cpt_snl(self, task:Task):
        snl = 0
        cur_pos = self.pos
        for np in task.locations_L:
            snl += get_dist(cur_pos, np)
            # cur_pos = np
        return snl


    def cpt_En(self, task: Task):
        Ec = self.kalfa * task.data_size_K * task.compute_res_H * self.fn * self.fn
        Et = self.pn * self.cpt_t_delivery(task )
        Em = self.theta * self.cpt_snl(task)
        return 1 * (Ec + Et + Em)

    def cpt_Tn(self, task: Task):
        Tc = task.compute_res_H / self.fn
        Td = self.cpt_t_delivery(task)
        Tm = 10 * self.cpt_snl(task) / self.vn
        return (Tc + Td + Tm) * self.mulT

    def get_cost(self, task: Task):
        return self.cpt_En(task) + self.cpt_Tn(task)

    def get_single_value(self, task, price, price_sigma):
        return price - price_sigma * self.get_cost(task)


def get_dist(pos1, pos2):
    return math.pow((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]), 0.5)