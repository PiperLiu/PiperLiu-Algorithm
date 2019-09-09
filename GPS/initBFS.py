# 2019年9月6日17:26:20

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import queue


# from collections import Counter


class Logger(object):

    def __init__(self):
        self.data = []
        # print("__init__")
        self.x_n = np.array([])
        self.y_n = np.array([])

    # 读数据
    def readData(self):
        file = open('./data.txt', 'r', 1)
        while True:
            lines = file.readline()
            if not lines:
                break
                pass
            x_tmp, y_tmp = [float(i) for i in lines.split('\t')]
            self.data.append([x_tmp, y_tmp])
        self.data = np.array(self.data)
        return self.data

    @staticmethod
    def pltRaw(data_tmp, color_='b'):
        try:
            plt.scatter(data_tmp[:, 0], data_tmp[:, 1], s=3, c=color_)
            # plt.show()
        except:
            # print("There is no point!")
            pass


class rectangle(object):

    def __init__(self, x_=0, y_=0, width_=0, height_=0):
        # 左下点为定位点
        self.x = x_
        self.y = y_
        self.width = width_
        self.height = height_
        self.left = x_
        self.bottom = y_
        self.top = y_ + height_
        self.right = x_ + width_
        # 新实例需要初始新points
        self.points = np.array([])

    def pointInRec(self, x_, y_):
        if self.left <= x_ <= self.right and self.bottom <= y_ <= self.top:
            return True
        else:
            return False

    def theyMyPoints(self, points_):
        try:
            rows, lines = points_.shape
            for row in range(rows):
                if self.pointInRec(points_[row, 0], points_[row, 1]):
                    # print(points_[row, :])
                    if self.points.shape[0] != 0:
                        self.points = np.vstack((self.points, points_[row, :]))
                    else:
                        self.points = points_[row, :]
        except:
            pass

    def addLine(self, y_):
        # 加一条横线
        rectangle_1 = rectangle(x_=self.x, y_=self.y, width_=self.width, height_=y_)
        rectangle_2 = rectangle(x_=self.x, y_=self.y + y_, width_=self.width, height_=self.height - y_)
        lst_tmp = [rectangle_1, rectangle_2]
        return lst_tmp

    def addColume(self, x_):
        # 加一条竖线
        rectangle_1 = rectangle(x_=self.x, y_=self.y, width_=x_, height_=self.height)
        rectangle_2 = rectangle(x_=self.x + x_, y_=self.y, width_=self.width - x_, height_=self.height)
        lst_tmp = [rectangle_1, rectangle_2]
        return lst_tmp

    @staticmethod
    def distPoints(point_1, point_2):
        return math.sqrt(pow((point_1[0] - point_2[0]), 2) + pow((point_1[1] - point_2[1]), 2))

    @staticmethod
    def separated(points_1, points_2):
        # 两个集合中的点是否可以分开？首先，两个集合中是否都有点？
        if np.sum(points_1) == 0 or np.sum(points_2) == 0:
            return False

        # 俩个集合之间的最小距离
        minDist = float("inf")
        rows_1, lines_1 = points_1.shape
        rows_2, lines_1 = points_2.shape
        for row_1 in range(rows_1):
            for row_2 in range(rows_2):
                tmpDist = rectangle.distPoints(points_1[row_1, :], points_2[rows_2, :])
                if tmpDist < minDist:
                    minDist = tmpDist
                    pass

        # points_1内point之间平均距离
        avgDis_1 = 0
        for row in range(rows_1):
            minDist_1 = float("inf")
            for row_ in range(rows_1):
                tmpDist = rectangle.distPoints(points_1[row, :], points_1[row_, :])
                if tmpDist < minDist_1:
                    minDist_1 = tmpDist
            avgDis_1 = avgDis_1 + minDist_1
        avgDis_1 = avgDis_1 / points_1.shape[0]

        # points_2内point之间平均距离
        avgDis_2 = 0
        for row in range(rows_2):
            minDist_2 = float("inf")
            for row_ in range(rows_2):
                tmpDist = rectangle.distPoints(points_2[row, :], points_2[row_, :])
                if tmpDist < minDist_2:
                    minDist_2 = tmpDist
            avgDis_2 = avgDis_2 + minDist_2
        avgDis_2 = avgDis_2 / points_2.shape[0]

        if minDist > avgDis_1 * 3 and minDist > avgDis_2 * 3:
            return True
        else:
            return False

    def yToAddLine(self):
        div = 0
        delta_div = self.height / 12
        while div < self.top:
            div = div + delta_div
            lst_rectangle = self.addLine(div)
            rectangle_1 = lst_rectangle[0]
            rectangle_2 = lst_rectangle[1]
            rectangle_1.theyMyPoints(self.points)
            rectangle_2.theyMyPoints(self.points)
            if rectangle.separated(rectangle_1.points, rectangle_2.points):
                return div
            pass
        return 0  # 0 means doing nothing

    def xToAddLine(self):
        div = 0
        delta_div = self.width / 12
        while div < self.right:
            div = div + delta_div
            lst_rectangle = self.addColume(div)
            rectangle_1 = lst_rectangle[0]
            rectangle_2 = lst_rectangle[1]
            rectangle_1.theyMyPoints(self.points)
            rectangle_2.theyMyPoints(self.points)
            if rectangle.separated(rectangle_1.points, rectangle_2.points):
                return div
            pass
        return 0  # 0 means doing nothing

    def drawMyLine(self):
        plt.plot([self.x, self.right], [self.y, self.y], 'k')
        plt.plot([self.right, self.right], [self.y, self.top], 'k')
        plt.plot([self.right, self.left], [self.top, self.top], 'k')
        plt.plot([self.left, self.left], [self.top, self.bottom], 'k')


class pixel(object):
    pixelWidth = 0.0012
    pixelHeight = 0.0012
    distNear = math.sqrt(pow(pixelWidth, 2) + pow(pixelHeight, 2))

    def __init__(self, x_=0.0, y_=0.0, width_=pixelWidth, height_=pixelHeight):
        # x，y为中心点
        self.x = x_
        self.y = y_
        self.width = width_
        self.height = height_
        self.left = self.x - self.width / 2
        self.right = self.x + self.width / 2
        self.top = self.y + self.height / 2
        self.bottom = self.y - self.height / 2

    def DoIExist(self, points_):
        if points_ != np.array([]):
            rows, lines = points_.shape
            for row in range(rows):
                if self.left <= points_[row, 0] <= self.right and self.bottom <= points_[row, 1] <= self.top:
                    return True
        else:
            return False
        return False

    def drawMePixel(self):
        ax_ = plt.gca()
        ax_.add_patch(
            patches.Rectangle(
                (self.left, self.bottom),
                self.width,
                self.height,
                color='yellow',
                fill=False
            )
        )


class rectangleWithPixel(rectangle):
    # BFS中用到的rectangle类

    def __init__(self, x_=0, y_=0, width_=0, height_=0):
        super().__init__(x_, y_, width_, height_)
        self.pixels = []
        self.pixelsRowNumber = math.ceil(self.width * 1.002 / pixel().pixelWidth)
        self.pixelsLineNumber = math.ceil(self.height * 1.002 / pixel().pixelHeight)
        # 从左上角的像素开始检查是否存在
        self.x_beginer = self.left + pixel().pixelWidth / 2 - self.width * 0.001
        self.y_beginer = self.top - pixel().pixelHeight / 2 + self.height * 0.001

    def theyMyPixels(self):
        # print(self.points)
        for row in range(self.pixelsRowNumber):
            x_tmp = self.x_beginer + row * pixel().pixelWidth
            for line in range(self.pixelsLineNumber):
                y_tmp = self.y_beginer - line * pixel().pixelHeight
                pixel_tmp = pixel(x_=x_tmp, y_=y_tmp)
                if pixel_tmp.DoIExist(self.points):
                    self.pixels.append(pixel_tmp)
                    pass

    def drawMyPixels(self):
        # print(self.pixels)
        if not self.pixels:
            return None
        for pixel_ in self.pixels:
            pixel_.drawMePixel()

    def fixMyPixels(self):
        # 没有找到检查具有特定属性的对象是否存在的函数，此处代码还可以优化
        # 查看(x, y+1)与(x, y-1)中夹着的元素是否存在，不存在则创建
        for pixel_up in self.pixels:  # 对于(x, y+1)
            for pixel_down in self.pixels:
                if pixel_down.y + 3 * pixel().pixelHeight > pixel_up.y > pixel_down.y + 1 * pixel().pixelHeight:  # 如果(x, y-1存在)
                    flag = True
                    for pixel_waiting in self.pixels:  # 则开始检验中间的像素是否存在
                        if pixel_up.y > pixel_waiting.y > pixel_down.y:
                            flag = False
                            break  # 夹在中间的像素已经存在，不需要添加，并跳出循环
                    if flag:  # 经历了for的遍历，夹在中间的像素不存在，flag依旧为True
                        # pixel_x = float(pixel_up.x)
                        # pixel_y = float(pixel_up)
                        pixel_ = pixel(x_=pixel_up.x, y_=pixel_up.y - pixel().pixelHeight)
                        self.pixels.append(pixel_)

        # 查看(x-1, y)与(x+1, y)中夹着的元素是否存在，不存在则创建
        for pixel_left in self.pixels:  # 对于(x-1, y)
            for pixel_right in self.pixels:
                if pixel_right.x - 3 * pixel().pixelWidth < pixel_left.x < pixel_right.x - 1 * pixel().pixelWidth:  # 如果(x+1, y存在)
                    flag = True
                    for pixel_waiting in self.pixels:  # 则开始检验中间的像素是否存在
                        if pixel_left.x < pixel_waiting.x < pixel_right.x:
                            flag = False
                            break  # 夹在中间的像素已经存在，不需要添加，并跳出循环
                    if flag:  # 经历了for的遍历，夹在中间的像素不存在，flag依旧为True
                        pixel_ = pixel(x_=pixel_left.x + pixel().pixelWidth, y_=pixel_left.y)
                        self.pixels.append(pixel_)


class zone(object):

    def __init__(self):
        self.pixels = []
        self.location = (0, 0)
        self.edgePixels = {'top': [],
                           'bottom': [],
                           'left': [],
                           'right': []
                           }

    @staticmethod
    def howManyZones(rec_: rectangleWithPixel):
        # 时间复杂度有点高，可能不是最好算法，日后可优化
        if not rec_.pixels:
            return None
        zones_ = []
        zone_new = zone()
        zone_new.pixels.append(rec_.pixels[0])
        zones_.append(zone_new)
        rec_.pixels.remove(rec_.pixels[0])  # 不清楚是不是传址调用rec_，如果是，这对rec_有伤害，之后不可使用rec_

        while rec_.pixels:
            isThereAnyPixelFitForCurrentZone = True
            while isThereAnyPixelFitForCurrentZone:
                isThereAnyPixelFitForCurrentZone = False
                zone_current = zones_[-1]
                for zone_pixel in zone_current.pixels:
                    for pixel_ in rec_.pixels:
                        dist = rectangle.distPoints(
                            point_1=(zone_pixel.x, zone_pixel.y),
                            point_2=(pixel_.x, pixel_.y)
                        )
                        # print(pixel().distNear)
                        # print(pixel().pixelWidth)
                        # print(pixel().pixelHeight)
                        if dist <= pixel().distNear:
                            # print(dist)
                            # print(math.sqrt(pixel().pixelWidth ** 2 + pixel().pixelHeight ** 2))
                            zone_current.pixels.append(pixel_)
                            isThereAnyPixelFitForCurrentZone = True
                            rec_.pixels.remove(pixel_)
            if rec_.pixels:  # rec_.pixels不为空
                # print('what')
                zone_new = zone()
                zone_new.pixels.append(rec_.pixels[0])
                zones_.append(zone_new)
                rec_.pixels.remove(rec_.pixels[0])

        return zones_

    def drawMyPixels(self):
        # print(self.pixels)
        if not self.pixels:
            return None
        for pixel_ in self.pixels:
            pixel_.drawMePixel()

    def generateStarter(self, rec_: rectangleWithPixel):
        # 生成边界起始点，这对于bfs算法很重要

        for key_ in self.edgePixels:
            for pixel_ in self.pixels:
                if key_ == 'top' or key_ == 'bottom':
                    point_2 = [pixel_.x, getattr(rec_, key_)]
                    threshold_distToEdge = pixel().pixelHeight
                else:
                    point_2 = [getattr(rec_, key_), pixel_.y]
                    threshold_distToEdge = pixel().pixelWidth
                distToEdge = rectangle.distPoints(
                    point_1=[pixel_.x, pixel_.y],
                    point_2=point_2
                )
                if distToEdge <= threshold_distToEdge * 0.85:
                    self.edgePixels[key_].append(pixel_)

        number_edgePixels = 0
        for key_ in self.edgePixels:
            number_edgePixels += len(self.edgePixels[key_])

        if number_edgePixels == 0:  # 没有边界点，选中离边界最近的点作为边界点
            min_dist = float('inf')
            for key_ in self.edgePixels:
                for pixel_ in self.pixels:
                    if key_ == 'top' or key_ == 'bottom':
                        point_2 = [pixel_.x, getattr(rec_, key_)]
                        threshold_distToEdge = pixel().pixelHeight
                    else:
                        point_2 = [getattr(rec_, key_), pixel_.y]
                        threshold_distToEdge = pixel().pixelWidth
                    distToEdge = rectangle.distPoints(
                        point_1=[pixel_.x, pixel_.y],
                        point_2=point_2
                    )
                    if distToEdge <= min_dist:
                        self.edgePixels[key_] = [pixel_]
                        min_dist = distToEdge

        if number_edgePixels == 1:
            # print(self.location, '上只有一个边界点！')
            max_dist = float(0)
            for key_ in self.edgePixels:
                if self.edgePixels[key_]:  # 即选中那唯一的边界点
                    thePixel_ = self.edgePixels[key_][0]
                    for pixel_ in self.pixels:
                        dist_ = rectangle.distPoints(
                            point_1=[thePixel_.x, thePixel_.y],
                            point_2=[pixel_.x, pixel_.y]
                        )
                        if dist_ > max_dist:
                            min_dist = dist_
                            self.edgePixels.update({'end': [pixel_]})
                            # print(self.edgePixels)
                    break


class bfsAlgorithm(object):
    # bfs: Breadth First Search

    @staticmethod
    def finished(pixel_, pixelCollection_: set):
        if pixel_ in pixelCollection_:
            return True
        else:
            return False

    @staticmethod
    def bfsP2P(zone_: zone):  # 用于点对点的连接
        pixelCollection_ = set()  # 不要忘记更新这两个集合
        connectionCollection_tmp = set()
        connectionCollection_ = set()

        # 在所有边界点中挑两个起始点
        pixel_starters = []
        for key_ in zone_.edgePixels:
            for pixel_ in zone_.edgePixels[key_]:
                if pixel_:  # pixel_存在点
                    pixel_starters.append(pixel_)
        pixel_starter = pixel_starters[0]
        pixel_ender = pixel_starters[-1]
        pixelCollection_.add(pixel_starter)
        pixelCollection_.add(pixel_ender)

        # 使用bfs连接两个点
        visited_ = set()
        visited_.add(pixel_starter)
        q_ = queue.Queue()
        q_.put(pixel_starter)
        while not q_.empty():
            pixel_current_ = q_.get()

            # 寻找可以与pixel_current_产生连接关系的pixel
            for pixel_ in zone_.pixels:  # 首先，应该是zone_.pixels中的点
                if pixel_ not in visited_:  # 其次，应该是没被遍历过的点
                    if rectangle.distPoints(
                            point_1=[pixel_current_.x, pixel_current_.y],
                            point_2=[pixel_.x, pixel_.y]
                    ) <= pixel().distNear:  # 最后，应是在pixel_current_周围的点
                        connectionCollection_tmp.add((pixel_current_, pixel_))
                        visited_.add(pixel_)
                        q_.put(pixel_)

                        if bfsAlgorithm.finished(pixel_, pixelCollection_):  # 触发终止条件
                            # 已经抵达终点，从末尾摸到开头，查看这条路径是什么
                            connect_tuple_1 = pixel_
                            flag_ = True
                            while flag_:
                                for connect_ in connectionCollection_tmp:
                                    if connect_[1] == connect_tuple_1:
                                        connectionCollection_.add(connect_)
                                        # connectionCollection_tmp.remove(connect_)
                                        connect_tuple_1 = connect_[0]
                                        if connect_tuple_1 == pixel_starter:
                                            flag_ = False
                                        else:
                                            pixelCollection_.add(connect_tuple_1)
                            q_ = queue.Queue()
                            break  # 已经找到最短路，跳出这两个循环

        return pixelCollection_, connectionCollection_

    @staticmethod
    def bfsP2Z(zone_: zone, pixelCollection_, connectionCollection_):
        # 用于三个点及以上的zone中，已经经过P2P的洗礼，连接点与片区

        # 在未使用边界点中挑一个起始点
        pixel_starters = []
        for key_ in zone_.edgePixels:
            for pixel_ in zone_.edgePixels[key_]:
                if pixel_:  # pixel_存在点
                    if pixel_ not in pixelCollection_:  # pixel_没在片区中
                        pixel_starters.append(pixel_)

        while pixel_starters:  # pixel_starters不为空
            pixel_starter = pixel_starters[-1]
            pixel_starters.remove(pixel_starter)
            pixelCollection_.add(pixel_starter)

            # 使用bfs将pixel_starter与pixelCollection_连接
            connectionCollection_tmp = set()
            visited_ = set()
            visited_.add(pixel_starter)
            q_ = queue.Queue()
            q_.put(pixel_starter)
            while not q_.empty():
                pixel_current_ = q_.get()

                # 寻找可以与pixel_current_产生连接关系的pixel
                for pixel_ in zone_.pixels:  # 首先，应该是zone_.pixels中的点
                    if pixel_ not in visited_:  # 其次，应该是没被遍历过的点
                        if rectangle.distPoints(
                                point_1=[pixel_current_.x, pixel_current_.y],
                                point_2=[pixel_.x, pixel_.y]
                        ) <= pixel().distNear:  # 最后，应是在pixel_current_周围的点
                            connectionCollection_tmp.add((pixel_current_, pixel_))
                            visited_.add(pixel_)
                            q_.put(pixel_)

                            if bfsAlgorithm.finished(pixel_, pixelCollection_):  # 触发终止条件
                                # 已经抵达终点，从末尾摸到开头，查看这条路径是什么
                                connect_tuple_1 = pixel_
                                flag_ = True
                                while flag_:
                                    for connect_ in connectionCollection_tmp:
                                        if connect_[1] == connect_tuple_1:
                                            connectionCollection_.add(connect_)
                                            # connectionCollection_tmp.remove(connect_)
                                            connect_tuple_1 = connect_[0]
                                            if connect_tuple_1 == pixel_starter:
                                                flag_ = False
                                            else:
                                                pixelCollection_.add(connect_tuple_1)
                                q_ = queue.Queue()
                                break  # 已经找到最短路，跳出这两个循环

        return pixelCollection_, connectionCollection_


def zoneProcessing(rec_dict_):
    zones_ = []
    for rec_key_ in rec_dict_:  # 对每个块进行切分zone操作
        howManyZones_result = zone.howManyZones(rec_dict_[rec_key_])
        if howManyZones_result:
            for zone_single in howManyZones_result:
                zone_single.location = rec_key_
                zones_.append(zone_single)

    for zone_ in zones_:  # 生成起始点
        zone_.generateStarter(rec_dict_[zone_.location])

    return zones_


def bfsProcessing(zones_):
    pixelCollection = set()
    connectionCollection = set()
    for zone_ in zones_:
        pixelCollection_thisZone, connectionCollection_thisZone = \
            bfsAlgorithm.bfsP2P(zone_)
        # print(connectionCollection_thisZone)
        pixelCollection_thisZone, connectionCollection_thisZone = \
            bfsAlgorithm.bfsP2Z(zone_,
                                pixelCollection_thisZone,
                                connectionCollection_thisZone)
        # print(connectionCollection_thisZone)
        pixelCollection = pixelCollection | pixelCollection_thisZone
        connectionCollection = connectionCollection | connectionCollection_thisZone
        # print(connectionCollection)

    return pixelCollection, connectionCollection


if __name__ == '__main__':
    data = Logger().readData()
    # Logger.pltRaw(data)

    # 划分rectangle
    left = np.min(data[:, 0])
    right = np.max(data[:, 0])
    top = np.max(data[:, 1])
    bottom = np.min(data[:, 1])
    width = right - left
    height = top - bottom
    width_rec = width * 1.02 / 18
    height_rec = height * 1.02 / 11

    rectangles = {}
    for horizon in range(18):
        x = 1.01 * left - 0.01 * right  # 因为边界向左右各扩展了1%，因此x_起始位置向左推进，省略数学推导步骤
        x = x + horizon * width_rec
        for vertical in range(11):
            y = 1.01 * bottom - 0.01 * top  # 因为边界向上下各扩展了1%，因此y_起始位置向下推进，省略数学推导步骤
            y = y + vertical * height_rec
            rectangles[(horizon, vertical)] = rectangleWithPixel(x, y, width_rec, height_rec)

    fig = plt.figure()
    ax_1 = fig.add_subplot(221)
    # counter = 0
    for rec_key in rectangles:
        # counter += 1
        rec = rectangles[rec_key]
        rec.theyMyPoints(data)
        # print(rec.points)
        # div = rec.yToAddLine()
        # if div:
        #     rectangles.remove(rec)
        #     rec_temps = rectangle.addLine(div)
        #     rectangles.append(rec_temps[0])
        #     rectangles.append(rec_temps[1])
        # div = rec.xToAddLine()
        # if div:
        #     rectangles.remove(rec)
        #     rec_temps = rectangle.addLine(div)
        #     rectangles.append(rec_temps[0])
        #     rectangles.append(rec_temps[1])
        # 不明原因，自动画线没有效果，并且计算量极大，运算5min
        rec.drawMyLine()
        rec.theyMyPixels()
        rec.fixMyPixels()
        # rec.fixMyPixels()
        rec.drawMyPixels()
        Logger.pltRaw(rec.points)

    ax_2 = fig.add_subplot(222)
    for rec_key in rectangles:
        rec = rectangles[rec_key]
        rec.drawMyLine()
        Logger.pltRaw(rec.points)
    zones = zoneProcessing(rectangles)
    # print(len(zones))
    zones[35].drawMyPixels()

    ax_3 = fig.add_subplot(223)
    for rec_key in rectangles:
        rec = rectangles[rec_key]
        rec.drawMyLine()
        Logger.pltRaw(rec.points)
    for zone in zones:
        zone.drawMyPixels()
        for key_thePixel in zone.edgePixels:
            if zone.edgePixels[key_thePixel]:  # 如果存在边界点
                for thePixel in zone.edgePixels[key_thePixel]:
                    plt.scatter(thePixel.x, thePixel.y, marker='s', c='r')

    ax_4 = fig.add_subplot(224)
    for rec_key in rectangles:
        rec = rectangles[rec_key]
        rec.drawMyLine()
    for zone in zones:
        zone.drawMyPixels()
    pixelCollect, connectionCollect = bfsProcessing(zones)
    for connection in connectionCollect:
        starter = connection[0]
        ender = connection[1]
        plt.plot([starter.x, ender.x], [starter.y, ender.y], c='b')

    plt.show()