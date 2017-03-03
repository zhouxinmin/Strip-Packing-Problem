# coding:utf-8

# 打包核心算法
import copy


class Packing:
    def __init__(self, floor):
        self.floor = floor                              # 热压罐可用层数
        self.space_tree = []                            # 存放空间列表
        self.can_put = []                               # 能放进热压罐的零件列表

    def orders(self, autoclave_length, autoclave_wide, part_length, part_wide):
        """
        零件进罐的排序规则
        :param autoclave_length: 热压罐长
        :param autoclave_wide: 热压罐宽
        :param part_length: 零件长
        :param part_wide: 零件宽
        :return: order  排序的值
        """
        s = autoclave_length * autoclave_wide
        f = (autoclave_length - part_length) * (autoclave_wide - part_wide) +\
            (autoclave_length - part_wide) * (autoclave_wide - part_length)
        order = 1 * (s - f)
        return order

    def put_in(self, spaces, parts, e, part_length, part_wide):
        """

        :param spaces:         存放热压罐剩余空间的二叉树
        :param parts:         要进罐的零件数组
        :param e:             parts下标，指的是第几个零件
        :param part_length:     该阶段能进罐的热压罐长度
        :param part_wide:    该阶段能进罐的热压罐宽度
        :return:
        """
        space = spaces[-1]
        if spaces[-1]['length'] >= part_length and spaces[-1]['wide'] >= part_wide:
            spaces.pop()  # 移除已用空间
            spaces.append({'length': space['length'], 'wide': space['wide'] - part_wide})
            spaces.append({'length': space['length'] - part_length, 'wide': part_wide})
            parts[e]['angle'] = 0
            part = [parts[e]]
            del parts[e]
            return part, parts, spaces
        else:
            return [], parts, spaces

    def choose(self, parts_list, spaces):
        """
        按规则将工装放入热压罐
        :param spaces: 热压罐可用空间          spaces=[{'length':a , 'wide':b}]
        :param parts_list: 还没有进罐的零件  parts_list=[{'length':a , 'wide':b, 'height':h, 'is_fixed_direction':0/1}]
                                                                                    0表示没有方向要求 1表示有方向要求
        :return: 能放入热压罐的工装
        """
        space = spaces[-1]      # 本轮可用的空间
        od = []
        nona = self.orders(space['length'], space['wide'], 0, 0)
        # 筛选能放进布局空间的零件并计算出排序最靠前的布局块
        for part in parts_list:
            if part['is_fixed_direction'] == 0:        # 无方向要求
                if (space['length'] >= part['length'] and space['wide'] >= part['wide']) \
                        or (space['length'] >= part['wide'] and space['wide'] >= part['length']):
                    c = self.orders(space['length'], space['wide'], part['length'], part['wide'])
                    od.append(c)
                else:
                    od.append(nona)
            elif part['is_fixed_direction'] == 1:      # 有方向要求
                if space['length'] >= part['length'] and space['wide'] >= part['wide']:
                    c = self.orders(space['length'], space['wide'], float(part['length']), float(part['wide']))
                    od.append(c)
                else:
                    od.append(nona)

        # 判断，如果parts数组有能放进去的布局块就继续分割；
        # 如果没有，则表示parts数组的布局块都大于布局空间，更换布局空间
        if len(od) == 1 and od[0] == nona:
            return [], parts_list, []
        elif len(od) == 1:
            return parts_list, [], spaces
        elif max(od) == min(od) == nona:
            return [], parts_list, spaces[:-1]
        else:
            e = od.index(max(od))
            # 存放放入布局块的序号
            if parts_list[e]['is_fixed_direction'] == 1:                                   # 有方向要求
                spaces.pop()                                          # 移除已用空间
                spaces.append({'length': space['length'], 'wide': space['wide'] - parts_list[e]['wide']})
                spaces.append({'length': space['length']-parts_list[e]['length'], 'wide': parts_list[e]['wide']})
                parts_list[e]['angle'] = 0
                part = parts_list[e]
                del parts_list[e]
                return [part], [parts_list], [spaces]
            elif parts_list[e]['is_fixed_direction'] == 0:                                 # 无方向要求   @是个问题
                spaces1 = copy.deepcopy(spaces)
                parts_list1 = copy.deepcopy(parts_list)
                e1 = copy.deepcopy(e)
                spaces2 = copy.deepcopy(spaces)
                parts_list2 = copy.deepcopy(parts_list)
                e2 = copy.deepcopy(e)
                re_part = []
                re_parts_list = []
                re_spaces = []
                part1, parts_list1, spaces1 = self.put_in(spaces1, parts_list1, e1,
                                                          parts_list[e]['length'], parts_list[e]['wide'])

                part2, parts_list2, spaces2 = self.put_in(spaces2, parts_list2, e2,
                                                          parts_list[e]['wide'], parts_list[e]['length'])
                if len(part1) == len(part2) == 1:
                    part2[0]['angle'] = 90
                    re_part.append(part1[0])
                    re_part.append(part2[0])
                    re_parts_list.append(parts_list1)
                    re_parts_list.append(parts_list2)
                    re_spaces.append(spaces1)
                    re_spaces.append(spaces2)
                    return re_part, re_parts_list, re_spaces
                elif len(part1) == 1:
                    return part1, parts_list1, [spaces1]
                else:
                    return part2, parts_list2, [spaces2]

    def pack(self, part_list, total_area):
        """
        判断part_list里的零件能否全部进入total_area这个空间里
        :param part_list:[{'length':a , 'wide':b, 'height':h, 'is_fixed_direction':0/1}] 0表示没有方向要求 1表示有方向要求
        :param total_area:{'length':l, 'wide':w, 'height':h, 'can_multi_floor':f}
        :return: 放进去的零件和没有放进去的零件
        """
        parts = copy.deepcopy(part_list)
        space = [{'length': total_area['length'], 'wide': total_area['wide']}]
        plan_space = [space]
        plan_unsolved_put = [parts]
        plan_solved_put = [[]]
        e = 0
        unsolved_list = plan_unsolved_put[e]
        can_put_space = plan_space[e]
        while e < len(plan_space):
            part, unsolved_parts, can_space = self.choose(unsolved_list, can_put_space)
            # print "part", part
            # print "unsolved_parts", unsolved_parts
            # print "can_space", can_space
            if len(part) == 2:
                solved = copy.deepcopy(plan_solved_put[e])
                plan_solved_put[e].append(part[0])
                solved.append(part[1])
                plan_solved_put.insert(e+1, solved)
                plan_unsolved_put[e] = unsolved_parts[0]
                plan_unsolved_put.insert(e+1, unsolved_parts[1])
                plan_space[e] = can_space[0]
                plan_space.insert(e+1, can_space[1])
                unsolved_list = plan_unsolved_put[e]
                can_put_space = plan_space[e]
            elif len(part) == 1 and len(unsolved_parts) != 0:
                plan_solved_put[e].append(part[0])
                plan_unsolved_put[e] = unsolved_parts[0]
                plan_space[e] = can_space[0]
                unsolved_list = [plan_unsolved_put[e]]
                can_put_space = plan_space[e]
            elif len(part) == 1 and len(unsolved_parts) == 0:
                plan_solved_put[e].append(part[0])
                return plan_solved_put[e], plan_unsolved_put[e]
            elif len(part) == 0 and len(can_space) == 0:
                return plan_solved_put[e], plan_unsolved_put[e]
            elif len(part) == 0:
                try:
                    can_put_space = can_space
                    unsolved_list = plan_unsolved_put[e]
                except IndexError:
                    e += 1
                    unsolved_list = plan_unsolved_put[e]
                    can_put_space = plan_space[e]
        else:
            max_num = 0
            sub_index = 0
            for index, num in enumerate(plan_solved_put):
                num_spare = len(num)
                if num_spare > max_num:
                    max_num = num_spare
                    sub_index = index
            return plan_solved_put[sub_index], plan_unsolved_put[sub_index]

    def controller(self, part_list, total_area):
        """
        热压罐有N（N>=1）层, 判断part_list里的零件能否全部进入N层的total_area这个空间里
        :param part_list: [{'length':a , 'wide':b, 'height':h, 'is_fixed_direction':0/1}] 0表示没有方向要求 1表示有方向要求
        :param total_area: [{'length':l, 'wide':w, 'height':h, 'can_multi_floor':f}]
        :return: 能否进罐的指示 0代表不能 1代表可以进罐
        """
        parts = copy.deepcopy(part_list)
        count = self.floor
        while count and len(parts):
            unsolved, solved = self.pack(parts, total_area)
            parts = unsolved
            count -= 1
        else:
            if len(parts) == 0:
                return 1
            else:
                return 0


def demo():
    # total_area = {'length': 12, 'wide': 3}
    # part_list = [{'length': 2, 'wide': 1, 'is_fixed_direction': 0},
    #              {'length': 2, 'wide': 1, 'is_fixed_direction': 0}]
    a = Packing(1)
    # total_area = {'length': 15, 'wide': 3}
    # part_list = [{'length': 14, 'wide': 2, 'is_fixed_direction': 0},
    #              {'length': 14, 'wide': 2, 'is_fixed_direction': 0}]
    # a = Packing(1)
    # total_area = {'length': 4, 'wide': 5}
    # part_list = [{'length': 1.5, 'wide': 2, 'is_fixed_direction': 1},
    #              {'length': 2.5, 'wide': 3, 'is_fixed_direction': 1},
    #              {'length': 2.5, 'wide': 1, 'is_fixed_direction': 1},
    #              {'length': 1, 'wide': 1, 'is_fixed_direction': 1},
    #              {'length': 1.5, 'wide': 4, 'is_fixed_direction': 1}
    #              ]
    # a = Packing(3)
    t = a.controller(part_list, total_area)
    print t

if __name__ == '__main__':
    demo()