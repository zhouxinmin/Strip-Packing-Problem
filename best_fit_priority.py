# coding:utf-8

import copy
from operator import itemgetter


class BestFitPriority:
    def __init__(self, is_consider_floor):
        self.floor = is_consider_floor                  # 热压罐可用层数
        self.h = 0                                      # 最低左对齐最佳匹配算法所需要的高度
        self.put_in = []                                # 能放进热压罐的零件列表    output
        self.put_position = []                          # 装载过程中产生的轮廓线集合

    def full_fit_first(self, parts, position):
        """
        完全匹配优先
        在可装入的轮廓线中选取最低的水平线position.k,如果有多个线段,则优先选取最左边的一段
        从待装矩形中按照装入序列依次将矩形与position.k进行比较,如果存在宽度或者高度与该线段宽度position.k.w相等
        且装入后刚好左填平或者右填平的矩形则优先装入
        :param parts:要进罐的零件列表[{},{},{}]
        :param position:轮廓线集合列表  [{},{},{}]
        :return:
        """
        position = sorted(position, key=itemgetter("y", "x"))
        wide = position[0]['w']
        for part in parts:
            if part['wide'] == wide:
                for pos in range(0, len(position)):
                    if position[0]['x'] == position[pos]['x'] + position[pos]['w'] and\
                                            position[0]['y'] + part['length'] == position[pos]['y']:
                        self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                            'h': part['length'], 'angle': 0})
                        position[pos]['w'] += wide
                        position.remove(position[0])
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, part, position
                    elif position[0]['x'] + part['wide'] == position[pos]['x'] and\
                                            position[0]['y'] + part['length'] == position[pos]['y']:
                        self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                            'h': part['length'], 'angle': 0})
                        position[pos]['x'] = position[0]['x']
                        position[pos]['w'] += wide
                        position.remove(position[0])
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, part, position
            elif part['is_fixed_direction'] == 0 and part['length'] == wide:
                for pos in range(0, len(position)):
                    if position[0]['x'] == position[pos]['x'] + position[pos]['w'] and\
                                            position[0]['y'] + part['wide'] == position[pos]['y']:
                        self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                            'h': part['length'], 'angle': 90})
                        position[pos]['w'] += wide
                        position.remove(position[0])
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, part, position
                    elif position[0]['x'] + part['length'] == position[pos]['x'] and\
                                            position[0]['y'] + part['wide'] == position[pos]['y']:
                        self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                            'h': part['length'], 'angle': 90})
                        position[pos]['x'] = position[0]['x']
                        position[pos]['w'] += wide
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, part, position
        return 0, [], position

    def width_fit_first(self, parts, position):
        """
        宽度匹配优先
        优先装入宽度或者高度与最低水平线ek等宽的矩形,如果存在多个匹配矩形,则优先装入面积最大的
        :param parts:要进罐的零件列表[{w:c, h:d, is_fixed_direction:e},{},{}]
        :param position:轮廓线集合列表  [{x:a , y:b, w:c},{},{}}
        :return:
        indicate 能否进罐的指示 0代表不能 1代表可以进罐
        能进罐的零件part ={x:a, y:b, w:c, h:d, angle:e}
        """
        position = sorted(position, key=itemgetter("y", "x"))
        wide = position[0]['w']
        temp = []
        for part in parts:
            if part['wide'] == wide:
                temp.append(part)
        if len(temp) == 0:
            for part in parts:
                if part['is_fixed_direction'] == 0 and part['length'] == wide:
                    temp.append(part)
            if len(temp) == 0:
                return 0, [], position
            else:
                temp = sorted(temp, key=itemgetter('wide'), reverse=True)
                position[0]['y'] += temp[0]['wide']
                position = sorted(position, key=itemgetter("y"), reverse=True)
                self.h = position[0]['y']
                self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': temp[0]['wide'],
                                    'h': temp[0]['length'], 'angle': 90})
                return 1, temp[0], position
        else:
            temp = sorted(temp, key=itemgetter('length'), reverse=True)
            position[0]['y'] += temp[0]['length']
            position = sorted(position, key=itemgetter("y"), reverse=True)
            self.h = position[0]['y']
            self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': temp[0]['wide'],
                                'h': temp[0]['length'], 'angle': 0})
            return 1, temp[0], position

    def height_fit_first(self, parts, position):
        """
        高度匹配优先
        按照装入序列查询宽度或高度不大于最低水平线ek宽度且装入后能够实现左填平的矩形,若存在则装入查询到的首个矩形
        :param parts:要进罐的零件列表[{w:c, h:d, is_fixed_direction:e},{},{}]
        :param position:轮廓线集合列表  [{x:a , y:b, w:c},{},{}]
        :return:
        indicate 能否进罐的指示 0代表不能 1代表可以进罐
        能进罐的零件part ={x:a, y:b, w:c, h:d, angle:e}
        """
        position = sorted(position, key=itemgetter("y", "x"))
        wide = position[0]['w']
        temp = []
        for part in parts:
            if part['wide'] < wide:
                temp.append(part)
        if len(temp) == 0:
            for part in parts:
                if part['is_fixed_direction'] == 0 and part['length'] < wide:
                    temp.append(part)
            if len(temp) == 0:
                return 0, {}, position
            else:                                   # @
                for t in temp:
                    for pos in range(0, len(position)):
                        if position[pos]['x'] + position[pos]['w'] == position[0]['x'] and\
                                                position[0]['y'] + t['wide'] == position[pos]['y']:
                            self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': t['wide'],
                                                'h': t['length'], 'angle': 90})
                            position[pos]['w'] += t['length']
                            position[0]['x'] += t['length']
                            position[0]['w'] -= t['length']
                            position = sorted(position, key=itemgetter("y"), reverse=True)
                            self.h = position[0]['y']
                            return 1, t, position
                        elif position[0]['x'] + position[0]['w'] == position[pos]['x'] and\
                                                position[0]['y'] + t['wide'] == position[pos]['y']:
                            self.put_in.append({'x': position[pos]['x']-t['length'], 'y': position[0]['y'],
                                                'w': t['wide'], 'h': t['length'], 'angle': 90})
                            position[0]['w'] -= t['length']
                            position[pos]['x'] -= t['length']
                            position[pos]['w'] += t['length']
                            position = sorted(position, key=itemgetter("y"), reverse=True)
                            self.h = position[0]['y']
                            return 1, t, position
                return 0, {}, position

        else:
            for t in temp:
                for pos in range(0, len(position)):
                    if position[pos]['x'] + position[pos]['w'] == position[0]['x'] and\
                                            position[0]['y'] + t['length'] == position[pos]['y']:
                        self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': t['wide'],
                                            'h': t['length'], 'angle': 0})
                        position[pos]['w'] += t['wide']
                        position[0]['x'] += t['wide']
                        position[0]['w'] -= t['wide']
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, t, position
                    elif position[0]['x'] + position[0]['w'] == position[pos]['x'] and\
                                            position[0]['y'] + t['length'] == position[pos]['y']:
                        self.put_in.append({'x': position[pos]['x'] - t['wide'], 'y': position[0]['y'], 'w': t['wide'],
                                            'h': t['length'], 'angle': 0})
                        position[0]['w'] -= t['wide']
                        position[pos]['x'] -= t['wide']
                        position[pos]['w'] += t['wide']
                        position = sorted(position, key=itemgetter("y"), reverse=True)
                        self.h = position[0]['y']
                        return 1, t, position
            return 0, {}, position

    def joint_width_fit_first(self, parts, position):
        """
        组合宽度匹配优先
        按装入序列对两个矩形进行组合,如果组合后的宽度与最低水平线宽度ek相等,则优先装入组合序列中的首个矩形
        :param parts:要进罐的零件列表[{w:c, h:d, is_fixed_direction:e},{},{}]
        :param position:轮廓线集合列表  [{x:a , y:b, w:c},{},{}]
        :return:
        indicate 能否进罐的指示 0代表不能 1代表可以进罐
        能进罐的零件part ={x:a, y:b, w:c, h:d, angle:e}
        """
        position = sorted(position, key=itemgetter("y", "x"))
        wide = position[0]['w']
        for part in parts:
            for pt in parts:
                if part != pt and part['wide'] + pt['wide'] == wide:
                    self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                        'h': part['length'], 'angle': 0})
                    height = position[0]['y']
                    position[0]['y'] += part['length']
                    position[0]['w'] = part['wide']
                    position.append({'x': position[0]['x'] + position[0]['w'], 'y': height,
                                     'w': pt['wide']})
                    position = sorted(position, key=itemgetter("y"), reverse=True)
                    self.h = position[0]['y']
                    return 1, part, position
                elif part != pt and part['is_fixed_direction'] == 0 and part['length'] + pt['wide'] == wide:
                    self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                        'h': part['length'], 'angle': 90})
                    height = position[0]['y']
                    position[0]['y'] += part['wide']
                    position[0]['w'] = part['length']
                    position.append({'x': position[0]['x'] + position[0]['w'], 'y': height,
                                     'w': pt['wide']})
                    position = sorted(position, key=itemgetter("y"), reverse=True)
                    self.h = position[0]['y']
                    return 1, part, position
                elif part != pt and pt['is_fixed_direction'] == 0 and part['wide'] + pt['length'] == wide:
                    self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                        'h': part['length'], 'angle': 0})
                    height = position[0]['y']
                    position[0]['y'] += part['length']
                    position[0]['w'] = part['wide']
                    position.append({'x': position[0]['x'] + position[0]['w'], 'y': height,
                                     'w': pt['length']})
                    position = sorted(position, key=itemgetter("y"), reverse=True)
                    self.h = position[0]['y']
                    return 1, part, position
                elif part != pt and pt['is_fixed_direction'] == part['is_fixed_direction'] == 0 and part['length'] + pt['length'] == wide:
                    self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': part['wide'],
                                        'h': part['length'], 'angle': 90})
                    height = position[0]['y']
                    position[0]['y'] += part['wide']
                    position[0]['w'] = part['length']
                    position.append({'x': position[0]['x'] + position[0]['w'], 'y': height,
                                     'w': pt['length']})
                    position = sorted(position, key=itemgetter("y"), reverse=True)
                    self.h = position[0]['y']
                    return 1, part, position
        return 0, {}, position

    def place_first(self, parts, position):
        """
        可装入优先
        在一定范围内,从待装矩形件中按照装入序列依次查找宽度或高度不大于最低水平线ek宽度的矩形,若存在,则将其装入;
        若存在多个,则装入面积最大的矩形
        :param parts:要进罐的零件列表[{w:c, h:d, is_fixed_direction:e},{},{}]
        :param position:轮廓线集合列表  [{x:a , y:b, w:c},{},{}]
        :return:
        indicate 能否进罐的指示 0代表不能 1代表可以进罐
        能进罐的零件part ={x:a, y:b, w:c, h:d, angle:e}
        """
        position = sorted(position, key=itemgetter("y", "x"))
        wide = position[0]['w']
        temp = []
        for part in parts:
            if part['wide'] < wide:
                part['area'] = part['length'] * part['wide']
                temp.append(part)
        if len(temp) == 0:
            for part in parts:
                if part['is_fixed_direction'] == 0 and part['length'] < wide:
                    part['area'] = part['length'] * part['wide']
                    temp.append(part)
            if len(temp) == 0:
                return 0, {}, position
            else:
                temp = sorted(temp, key=itemgetter('area'), reverse=True)
                self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': temp[0]['wide'],
                                    'h': temp[0]['length'], 'angle': 90})
                position.append({'x': position[0]['x'] + temp[0]['length'], 'y': position[0]['y'],
                                 'w': position[0]['w'] - temp[0]['length']})
                position[0]['y'] += temp[0]['wide']
                position[0]['w'] = temp[0]['length']
                position = sorted(position, key=itemgetter("y"), reverse=True)
                self.h = position[0]['y']
                return 1, temp[0], position
        else:
            temp = sorted(temp, key=itemgetter('area'), reverse=True)
            self.put_in.append({'x': position[0]['x'], 'y': position[0]['y'], 'w': temp[0]['wide'],
                                'h': temp[0]['length'], 'angle': 0})
            position.append({'x': position[0]['x']+temp[0]['wide'], 'y': position[0]['y'],
                             'w': position[0]['w']-temp[0]['wide']})
            position[0]['y'] += temp[0]['length']
            position[0]['w'] = temp[0]['wide']
            position = sorted(position, key=itemgetter("y"), reverse=True)
            self.h = position[0]['y']
            return 1, temp[0], position

    def llabf(self, parts, position):
        """
        最低左对齐最佳匹配算法
        :param parts:  在parts中找出一个合适的零件放入到position
        :param position: 轮廓线集合
        :return: 返回合适的零件和新的轮廓线集合
        """
        indic, part, e = self.full_fit_first(parts, position)
        if indic == 1:
            # print type(parts), type(e)
            # print '完全匹配成功', part, e
            parts.remove(part)
            # print "还没有放进去的零件有：", parts, "\n"
            return [part], e
        else:
            indic, part, e = self.width_fit_first(parts, position)
            if indic == 1:
                # print type(parts), type(e)
                # print '宽度匹配成功', part, e
                parts.remove(part)
                # print "还没有放进去的零件有：", parts, "\n"
                return [part], e
            else:
                indic, part, e = self.height_fit_first(parts, position)
                if indic == 1:
                    # print type(parts), type(e)
                    # print '高度匹配成功', part, e
                    parts.remove(part)
                    # print "还没有放进去的零件有：", parts, "\n"
                    return [part], e
                else:
                    indic, part, e = self.joint_width_fit_first(parts, position)
                    if indic == 1:
                        # print type(parts), type(e)
                        # print '组合宽度匹配成功', part, e
                        parts.remove(part)
                        # print "还没有放进去的零件有：", parts, "\n"
                        return [part], e
                    else:
                        indic, part, e = self.place_first(parts, position)
                        if indic == 1:
                            # print type(parts), type(e)
                            # print '330，可装入优先成功', part, e
                            parts.remove(part)
                            # print "还没有放进去的零件有：", parts, "\n"
                            return [part], e
                        else:
                            return [], e

    def one_floor(self, part_list, total_area):
        """
        判断part_list里的零件能否全部进入total_area这个空间里
        :param part_list:[{'length':a , 'wide':b, 'height':h, 'is_fixed_direction':0/1}] 0表示没有方向要求 1表示有方向要求
        :param total_area:[{'length':l, 'wide':w, 'height':h, 'can_multi_floor':f}]
        :return: 放进去的零件和没有放进去的零件
        """
        parts = copy.deepcopy(part_list)
        position = [{'x': 0, 'y': 0, 'w': total_area['wide']}]
        self.h = 0
        solved = []
        unsolved = []
        while len(parts) and self.h <= total_area['length']:
            part, position = self.llabf(parts, position)
            if self.h <= total_area['length'] and len(part) != 0:
                solved.append(part[0])
                unsolved = parts
            else:
                return unsolved, solved
        else:
            return unsolved, solved

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
            unsolved, solved = self.one_floor(parts, total_area)
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
    # a = BestFitPriority(1)
    total_area = {'length': 15, 'wide': 3}
    part_list = [{'length': 14, 'wide': 2, 'is_fixed_direction': 0},
                 {'length': 14, 'wide': 2, 'is_fixed_direction': 0}]
    a = BestFitPriority(1)
    # total_area = {'length': 4, 'wide': 5}
    # part_list = [{'length': 1.5, 'wide': 2, 'is_fixed_direction': 1},
    #              {'length': 2.5, 'wide': 3, 'is_fixed_direction': 1},
    #              {'length': 2.5, 'wide': 1, 'is_fixed_direction': 1},
    #              {'length': 1, 'wide': 1, 'is_fixed_direction': 1},
    #              {'length': 1.5, 'wide': 4, 'is_fixed_direction': 1}
    #              ]
    # a = BestFitPriority(3)
    t = a.controller(part_list, total_area)
    print t

if __name__ == '__main__':
    demo()
