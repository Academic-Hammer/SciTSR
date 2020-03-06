# Copyright (c) 2019-present, Zewen Chi
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from typing import List
from scitsr.table import Chunk


class Vertex(object):

  def __init__(self, vid: int, chunk: Chunk, tab_h, tab_w):
    """
    Args:
      vid: Vertex id
      chunk: the chunk to extract features
      tab_h: height of the table (y-axis)
      tab_w: width of the table (x-axis)
    """
    self.vid = vid
    self.tab_h = tab_h
    self.tab_w = tab_w
    self.chunk = chunk
    self.features = self.get_features()

  def get_features(self):
    return {
      "x1": self.chunk.x1,
      "x2": self.chunk.x2,
      "y1": self.chunk.y1,
      "y2": self.chunk.y2,
      "x center": (self.chunk.x1 + self.chunk.x2) / 2,
      "y center": (self.chunk.y1 + self.chunk.y2) / 2,
      "relative x1": self.chunk.x1 / self.tab_w,
      "relative x2": self.chunk.x2 / self.tab_w,
      "relative y1": self.chunk.y1 / self.tab_h,
      "relative y2": self.chunk.y2  / self.tab_h,
      "relative x center": (self.chunk.x1 + self.chunk.x2) / 2 / self.tab_w,
      "relative y center": (self.chunk.y2 + self.chunk.y2) / 2 / self.tab_h,
      "height of chunk": self.chunk.y2 - self.chunk.y1,
      "width of chunk": self.chunk.x2 - self.chunk.x1
    }


class Edge(object):

  def __init__(self, fr: Vertex, to: Vertex):
    self.fr = fr
    self.to = to
    self.features = self.get_features()

  def get_features(self):
    c1, c2 = self.fr.chunk, self.to.chunk
    tab_h = self.fr.tab_h
    tab_w = self.fr.tab_w

    # distance belong x/y
    x_dis, y_dis = 0, 0
    # coincide belong x/y
    x_cncd, y_cncd = 0, 0

    if c1.x2 <= c2.x1:
      x_dis = c2.x1 - c1.x2
    elif c2.x2 <= c1.x1:
      x_dis = c1.x1 - c2.x2
    elif c1.x2 <= c2.x2:
      if c1.x1 <= c2.x1:
        x_cncd = c1.x2 - c2.x1
      else:
        x_cncd = c1.x2 - c1.x1 
    elif c1.x2 > c2.x2:
      if c1.x1 <= c2.x1:
        x_cncd = c2.x2 - c2.x1
      else:
        x_cncd = c2.x2 - c1.x1

    if c1.y2 <= c2.y1:
      y_dis = c2.y1 - c1.y2
    elif c2.y2 <= c1.y1:
      y_dis = c1.y1 - c2.y2
    elif c1.y2 <= c2.y2:
      if c1.y1 <= c2.y1:
        y_cncd = c1.y2 - c2.y1
      else:
        y_cncd = c1.y2 - c1.y1
    elif c1.y2 > c2.y2:
      if c1.y1 <= c2.y1:
        y_cncd = c2.y2 - c2.y1
      else:
        y_cncd = c2.y2 - c1.y1

    c_h = (c1.y2 - c1.y1 + c2.y2 - c2.y1) / 2
    c_w = (c1.x2 - c1.x1 + c2.x2 - c2.x1) / 2 + 1e-7

    c1x = (c1.x1 + c1.x2) / 2
    c2x = (c2.x1 + c2.x2) / 2
    c1y = (c1.y1 + c1.y2) / 2
    c2y = (c2.y1 + c2.y2) / 2
    c_x_dis = abs(c2x - c1x)
    c_y_dis = abs(c2y - c1y)

    return {
      "x distance": x_dis,
      "y distance": y_dis,
      "relative (table) x distance": x_dis / tab_w,
      "relative (table) y distance": y_dis / tab_h,
      "relative (chunk) x distance": x_dis / c_w,
      "relative (chunk) y distance": y_dis / c_h,
      "x coincide": x_cncd,
      "y coincide": y_cncd,
      "relative (table) x coincide": x_cncd / tab_w,
      "relative (table) y coincide": y_cncd / tab_h,
      "relative (chunk) x coincide": x_cncd / c_w,
      "relative (chunk) y coincide": y_cncd / c_h,
      "Euler distance": math.sqrt(x_dis**2 + y_dis**2),
      "relative (table) Euler distance": math.sqrt((x_dis / tab_w)**2 + (y_dis / tab_h)**2),
      "relative (chunk) Euler distance": math.sqrt((x_dis / c_w)**2 + (y_dis / c_h)**2),
      "center x distance": c_x_dis,
      "relative (table) center x distance": c_x_dis / tab_w,
      "relative (chunk) center x distance": c_x_dis / c_w,
      "center y distance": c_y_dis,
      "relative (table) center y distance": c_y_dis / tab_h,
      "relative (chunk) center y distance": c_y_dis / c_h,
      "center Euler distance": math.sqrt(c_x_dis**2 + c_y_dis**2),
      "relative (table) center Euler distance": math.sqrt((c_x_dis / tab_w)**2 + (c_y_dis / tab_h)**2),
      "relative (chunk) center Euler distance": math.sqrt((c_x_dis / c_w)**2 + (c_y_dis / c_h)**2),
    }


class Graph(object):

  def __init__(self, E: List[Edge]=None, V: List[Vertex]=None, directed=False):
    self.E = E if E is not None else []
    self.V = V if V is not None else []
    self.directed = directed