import os.path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import agents, utils
import math
import scipy.io as sio

plt.rcParams['font.sans-serif'] = ['SimHei']
default_R = 50  # 机器人数量
default_T = 5  # 任务的locations_L长度


class Opm:
    def __init__(self, config):
        tn_num = 1  # 默认一个 TaskManager
        if not 'R' in config: config['R'] = default_R
        if not 'T' in config: config['T'] = default_T

        # 初始化TN
        t_pos_list = [[0, 0]]
        self.tns = [agents.TaskManager(t_pos_list[i]) for i in range(tn_num)]

        # 初始化robots 和 tasks
        file_name = "models/hist_R" + str(config['R']) + "_T" + str(config['T']) + ".pkl"
        if not os.path.exists(file_name):
            f_list = utils.get_random_gauss_list(9, 11, config['R'])
            w_list = utils.get_random_gauss_list(4, 6, config['R'])
            p_list = [(f_list[i] + w_list[i]) for i in range(config['R'])]
            v_list = utils.get_random_gauss_list(5, 10, config['R'])
            pos_list = utils.get_random_pos(1, 10, 1, 10, config['R'])

            robots = [agents.Robot(pos_list[i], f_list[i], w_list[i], p_list[i], v_list[i], i) for i in
                      range(config['R'])]
            K_list = utils.get_random_gauss_list(50, 100, len([config['T']]))
            L_list = [utils.get_random_pos(1, 10, 1, 10, step) for step in [config['T']]]
            H_list = utils.get_random_gauss_list(10, 50, len([config['T']]))
            O_list = utils.get_random_gauss_list(1, 5, len([config['T']]))
            tasks = [agents.Task(K_list[i], L_list[i], H_list[i], O_list[i], self.tns[0]) for i in
                     range(len([config['T']]))]
            with open(file_name, 'wb') as f:
                pickle.dump([robots, tasks], f)
                f.close()
        else:
            print("load history robots and tasks")
            with open(file_name, 'rb') as f:
                robots, tasks = pickle.load(f)
                f.close()

        self.robots = robots
        self.tasks = tasks

        # 其他参数
        self.price_sigma = 1
        self.group_costs_belta = 10
        self.price_iyta = 10000 * (config['T'] / 5) * (config['R'] / 50)
        self.price_p0 = 2500 * config['T'] / 5

    def divide_task(self, task: agents.Task, coal: [agents.Robot]):
        if len(coal) == 1:
            return [task]
        f_list = [robot.fn for robot in coal]
        r_list = [robot.cpt_rn(task) for robot in coal]
        sum_f, sum_r = sum(f_list), sum(r_list)

        H_divided = [task.compute_res_H * f_val / sum_f for f_val in f_list]
        K_divided = [task.data_size_K * r_val / sum_r for r_val in r_list]
        O_divided = [task.output_size_O / len(coal) for i in range(len(coal))]
        L_divided = [[] for _ in range(len(coal))]

        for pos in task.locations_L:
            best_dist, best_idx = 10000000, -1
            for i in range(len(coal)):
                d = utils.get_dist(coal[i].pos, pos)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            L_divided[best_idx].append(pos)

        mul_tasks = [agents.Task(K_divided[i], L_divided[i], H_divided[i], O_divided[i], task.TN) for i in
                     range(len(coal))]
        return mul_tasks

    def should_join_in(self, task: agents.Task, rob, coal_value, coal, last_price):
        if not rob:
            new_coal = coal
        else:
            new_coal = coal + [rob]

        v_respective = 0
        for robot in new_coal:
            v_respective += coal_value[robot.id]

        mul_tasks = self.divide_task(task, new_coal)
        mul_T = []
        total_energy_cost = 0
        for i in range(len(new_coal)):
            cur_rob = new_coal[i]
            cur_task = mul_tasks[i]
            mul_T.append(cur_rob.cpt_Tn(cur_task))
            total_energy_cost += cur_rob.cpt_En(cur_task)

        if len(new_coal) > 1:
            group_cost = self.group_costs_belta * len(new_coal)
        else:
            group_cost = 0

        v_together = last_price - self.price_sigma * (max(mul_T) + total_energy_cost) - group_cost
        if v_respective < 0:
            v_respective = 0
        return [v_together - v_respective, total_energy_cost, max(mul_T)]

    def run(self, task: agents.Task, y_labels, limit_group_size=0, reserve=False, algo_type="TRC-HG"):
        if algo_type == "TRC-HG":
            pass
        elif algo_type == "IO":
            limit_group_size = 1
        elif algo_type == "WP":
            limit_group_size = 2
        elif algo_type == "RR":
            reserve = True
        else:
            raise ValueError
        if reserve:
            for r in self.robots:
                r.fn /= 2
                r.wn /= 2
                r.pn /= 2
                r.vn /= 2

        step_values = [[] for _ in range(len(y_labels))]

        last_price = self.price_p0
        coals = [[robot] for robot in self.robots]
        robot_dict = {robot.id: robot for robot in self.robots}
        which_coal = {self.robots[i].id: i for i in range(len(self.robots))}
        every_robot_value = {robot.id: robot.get_single_value(task, last_price, self.price_sigma) for robot in
                             self.robots}
        coal_value = [self.should_join_in(task, False, every_robot_value, coal, last_price) for coal in coals]

        for i in range(len(coal_value)):  # 初始化联盟价值为个体收益
            if len(coals[i]) == 1:
                coal_value[i][0] += every_robot_value[coals[i][0].id]

        step = 0
        while step < len(self.robots) * 2:
            energy, latency, price, best_coal_idx = self.cpt_best(coal_value, task)

            step_values[0].append(energy)
            step_values[1].append(latency)
            step_values[2].append(price)
            step_values[3].append(len(coals))
            step_values[4].append(len(coals[best_coal_idx]))
            step_values[5].append(coal_value[best_coal_idx][0])

            step += 1
            moves = []
            for id in range(len(self.robots)):
                robot = robot_dict[id]
                best_coal = -1
                best_reward = 0

                for i, coal in enumerate(coals):
                    if which_coal[id] == i: continue
                    comp_val, _, _ = self.should_join_in(task, robot, every_robot_value, coal, last_price)
                    if comp_val > best_reward:
                        best_coal = i
                        best_reward = comp_val
                if best_reward > 0 and best_reward > coal_value[best_coal][0] \
                        and best_reward / (1 + len(coals[best_coal])) > coal_value[which_coal[id]][0] / len(
                    coals[which_coal[id]]):
                    if not (limit_group_size > 0 and len(coals[best_coal]) >= limit_group_size):  # 结伴策略（coal最多只容纳2个）
                        moves.append((id, best_coal, best_reward))
            if len(moves) == 0:
                break
            moves.sort(key=lambda x: x[2], reverse=True)
            print(moves[0])
            mark = [False for _ in range(len(coals))]
            for move in moves:
                src = which_coal[move[0]]
                dest = move[1]
                if not mark[dest] and not mark[src]:
                    coals[src].remove(robot_dict[move[0]])
                    coals[dest].append(robot_dict[move[0]])
                    mark[src] = True
                    mark[dest] = True

            for i in range(len(coals) - 1, -1, -1):  # 倒序修改, 否则会漏掉
                if len(coals[i]) == 0:
                    coals.remove(coals[i])

            for i, coal in enumerate(coals):
                for robot in coal:
                    which_coal[robot.id] = i

            coal_value = [self.should_join_in(task, False, every_robot_value, coal, last_price) for coal in coals]

        if reserve:
            for r in self.robots:
                r.fn *= 2
                r.wn *= 2
                r.pn *= 2
                r.vn *= 2
            for coal_v in coal_value:
                coal_v[2] *= 2

        return coals, coal_value, step_values

    def cpt_best(self, coal_value, task: agents.Task):
        D = coal_value[0][2]
        idx = 0
        for i, val in enumerate(coal_value):
            if val[2] < D:
                D = val[2]
                idx = i
        K = task.data_size_K
        target_price = math.pow(self.price_iyta * self.price_p0 * D / K + math.pow(self.price_p0 * D, 2) / 4,
                                0.5) - self.price_p0 * D / 2
        target_price *= math.pow(D, 0.25)
        return coal_value[idx][1], coal_value[idx][2], target_price, idx


def plot_step_of_diff_algo(y_lable, step_values, algo_names, pre_msg):
    plt.figure()
    for i, single_algo_step_values in enumerate(step_values):
        sample_plot(vals=step_values[i], msg=algo_names[i])
        plt.legend()
    plt.xlabel("Time Slot")
    plt.ylabel(y_lable)
    plt.savefig('./results/' + y_lable + '.png')


def sample_plot(vals, msg):
    plt.plot([i for i in range(len(vals))],
             vals, label=msg)


def compare_step(R=default_R, T=default_T):
    y_labels = ["Energy", "Latency", "Price", "The Number of Coalitions", "The Size of Coalition", "Optimal Benefit"]
    algo_names = ['TRC-HG', 'IO', 'RR', 'WP']
    opm = Opm({'R': R, 'T': T})

    step_values_each_algo = []
    for i, algo_name in enumerate(algo_names):
        _, _, step_values = opm.run(opm.tasks[0], y_labels=y_labels, algo_type=algo_name)
        step_values_each_algo.append(step_values)

    for i, y_label in enumerate(y_labels):
        tmp = [step_values_each_algo[j][i] for j in range(len(algo_names))]
        # mat_data = {'x_label' :'Time Slot','y_label':y_label,'algo':tmp[0]}
        # sio.savemat('./results/' + y_label + ".mat", mat_data)

        plot_step_of_diff_algo(y_label, tmp, algo_names, pre_msg="step_value_R" + str(R) + "_T" + str(T) + "_")


def compare_value(val_list, xlabel, plt_save=True):  # 五种算法的时延 随节点数量（场景数量）的变化趋势
    plt.figure()
    if xlabel == 'R':
        xlabel_zh = "The Number of RNs"
    elif xlabel == 'T':
        xlabel_zh = "Crowdsourcing Task Size"
    else:
        raise ValueError
    y_labels = ["Energy", "Latency", "Price", "The Number of Coalitions", "The Size of Coalition", "Optimal Benefit"]
    algo_names = ['TRC-HG', 'IO', 'RR', 'WP']
    datas = [{algo_name: [] for algo_name in algo_names
              } for _ in range(len(y_labels))]
    for val in val_list:
        opm = Opm({xlabel: val})
        for i, algo_name in enumerate(algo_names):
            coals, coal_value, step_values = opm.run(opm.tasks[0], y_labels=y_labels, algo_type=algo_name)
            energy, latency, price, best_coal_idx = opm.cpt_best(coal_value, opm.tasks[0])
            datas[0][algo_names[i]].append(energy)
            datas[1][algo_names[i]].append(latency)
            datas[2][algo_names[i]].append(price)
            datas[3][algo_names[i]].append(len(coals))
            datas[4][algo_names[i]].append(len(coals[best_coal_idx]))
            datas[5][algo_names[i]].append(coal_value[best_coal_idx][0])

    for i, y_label in enumerate(y_labels):
        df = pd.DataFrame(datas[i], index=val_list)
        df.plot(kind='bar')
        plt.legend()
        # font = fm.FontProperties(fname=r'书法.ttf')
        plt.xlabel(xlabel_zh, fontproperties='simhei')
        plt.ylabel(ylabel=y_label, fontproperties='simhei')
        plt.xticks(rotation=360, fontproperties='simhei')
        if plt_save:
            plt.savefig('results/compare_' + xlabel + '_' + y_label + '.png')

    # 显示绘制结果
    if not plt_save:
        plt.show()


if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')

    # y_labels = ["Energy", "Latency", "Price", "The Number of Coalitions", "The Size of Coalition", "Optimal Benefit"]
    # opm = Opm({'R': 100, 'T': 10})
    # for task in opm.tasks:
    #     coal, coal_value, step_values = opm.run(task, y_labels)
    #     print(opm.cpt_best(coal_value, task))

    compare_value([10, 20, 50, 100], 'R')
    compare_value([1, 5, 10, 20], 'T')
    compare_step()
