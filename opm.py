import pickle

import agents
import random, math

GAOSS_SIGMA = 4


class Opm:
    def __init__(self, robot_num, tn_num, task_steps: [], generate=False):
        # 初始化robots
        if generate:
            f_list = self.get_random_gauss_list(9, 11, robot_num)
            w_list = self.get_random_gauss_list(4, 6, robot_num)
            p_list = [(f_list[i] + w_list[i]) for i in range(robot_num)]
            v_list = self.get_random_gauss_list(5, 10, robot_num)
            pos_list = self.get_random_pos(1, 10, 1, 10, robot_num)

            robots = [agents.Robot(pos_list[i], f_list[i], w_list[i], p_list[i], v_list[i], i) for i in
                           range(robot_num)]
            with open("./robots.txt", 'wb') as f:
                pickle.dump(robots, f)
                f.close()
        else:
            with open("./robots.txt", 'rb') as f:
                robots = pickle.load(f)
                f.close()
        self.robots = robots

        # 初始化TN
        t_pos_list = [[0, 0]]
        self.tns = [agents.TaskManager(t_pos_list[i]) for i in range(tn_num)]

        # 初始化task
        if generate:
            K_list = self.get_random_gauss_list(10, 100, len(task_steps))
            L_list = [self.get_random_pos(0, 10, 0, 10, step) for step in task_steps]
            H_list = self.get_random_gauss_list(10, 50, len(task_steps))
            O_list = self.get_random_gauss_list(1, 5, len(task_steps))
            tasks = [agents.Task(K_list[i], L_list[i], H_list[i], O_list[i], self.tns[0]) for i in
                          range(len(task_steps))]
            with open("./tasks.txt", 'wb') as f:
                pickle.dump(tasks, f)
                f.close()
        else:
            with open("./tasks.txt", 'rb') as f:
                tasks = pickle.load(f)
                f.close()
        self.tasks = tasks

        # 其他参数
        self.price_sigma = 1
        self.group_costs_belta = 10

    def get_random_gauss_list(self, bottom, top, size):
        mean = (bottom + top) // 2
        val_list = []
        while len(val_list) < size:
            val = random.gauss(mean, GAOSS_SIGMA)
            while val < bottom or val > top:
                val = random.gauss(mean, GAOSS_SIGMA)
            val_list.append(val)
        return val_list

    def get_random_pos(self, x_bottom, x_top, y_bottom, y_top, size):
        pos_list = []
        while len(pos_list) < size:
            x = random.randint(x_bottom, x_top)
            y = random.randint(y_bottom, y_top)
            pos_list.append([x, y])
        return pos_list

    def divide_task(self, task: agents.Task, coal: [agents.Robot]):
        f_list = [robot.fn for robot in coal]
        r_list = [robot.cpt_rn(task) for robot in coal]
        sum_f, sum_r = sum(f_list), sum(r_list)

        H_divided = [task.compute_res_H * f_val / sum_f for f_val in f_list]
        K_divided = [task.data_size_K * r_val / sum_r for r_val in r_list]
        O_divided = [task.output_size_O / len(coal) for i in range(len(coal))]
        L_divided = [[] for i in range(len(coal))]

        for pos in task.locations_L:
            best_dist, best_idx = 10000000, -1
            for i in range(len(coal)):
                d = get_dist(coal[i].pos, pos)
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

        v_together = last_price - self.price_sigma * (max(mul_T) + total_energy_cost) - self.group_costs_belta * len(new_coal)
        if v_respective < 0:
            v_respective = 0
        return v_together - v_respective

    def run(self, task:agents.Task, last_price, limit_group_2=False):
        coals = [[robot] for robot in self.robots]
        robot_dict = {robot.id: robot for robot in self.robots}
        which_coal = {self.robots[i].id: i for i in range(len(self.robots))}
        every_robot_value = {robot.id: robot.get_single_value(task, last_price, self.price_sigma) for robot in self.robots}
        coal_value = {i:0 for i in range(len(coals))}

        while True:
            moves = []
            for id in range(len(self.robots)):
                robot = robot_dict[id]
                best_coal = -1
                best_reward = 0

                for i, coal in enumerate(coals):
                    if which_coal[id] == i: continue
                    comp_val = self.should_join_in(task, robot, every_robot_value, coal, last_price)
                    if comp_val > best_reward:
                        best_coal = i
                        best_reward = comp_val
                if best_reward>0 and best_reward>coal_value[best_coal]\
                        and best_reward / (1 + len(coals[best_coal])) > coal_value[which_coal[id]] / len(coals[which_coal[id]]):
                    moves.append((id, best_coal, best_reward))
            if len(moves) == 0:
                break
            moves.sort(key=lambda x: x[2], reverse=True)
            print(moves[0])
            mark = [False for i in range(len(coals))]
            for move in moves:
                src = which_coal[move[0]]
                dest = move[1]
                if not mark[dest] and not mark[src]:
                    coals[src].remove(robot_dict[move[0]])
                    coals[dest].append(robot_dict[move[0]])
                    mark[src] = True
                    mark[dest] = True
            for i in range(len(coals)-1, -1, -1):
                if len(coals[i]) == 0:
                    coals.remove(coals[i])
            for i, coal in enumerate(coals):
                for robot in coal:
                    which_coal[robot.id] = i
            coal_value = [self.should_join_in(task, False, every_robot_value, coal, last_price) for coal in coals]

        pass



def get_dist(pos1, pos2):
    return math.pow((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]), 0.5)


if __name__ == "__main__":
    opm = Opm(50, 1, [5])
    for task in opm.tasks:
        opm.run(task, last_price=2500)