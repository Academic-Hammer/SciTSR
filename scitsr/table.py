# Copyright (c) 2019-present, Zewen Chi
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from typing import Iterable, List, Tuple


def load_chunks(chunk_path):
  with open(chunk_path, 'r') as f:
    chunks = json.load(f)['chunks']
  # NOTE remove the chunk with 0 len
  ret = []
  for chunk in chunks:
    if chunk["pos"][1] < chunk["pos"][0]:
        chunk["pos"][0], chunk["pos"][1] = chunk["pos"][1], chunk["pos"][0]
        print("Warning load illegal chunk.")
    c = Chunk.load_from_dict(chunk)
    #if c.x2 == c.x1 or c.y2 == c.y1 or c.text == "": 
    #    continue
    ret.append(c)
  return ret


class Box(object):

  def __init__(self, pos):
    """pos: (x1, x2, y1, y2)"""
    self.set_pos(pos)
  
  def set_pos(self, pos):
    assert pos[0] <= pos[1]
    assert pos[2] <= pos[3]
    self.x1 = pos[0]
    self.x2 = pos[1]
    self.y1 = pos[2]
    self.y2 = pos[3]
    self.w = self.x2 - self.x1
    self.h = self.y2 - self.y1
    self.pos = pos
  
  def __lt__(self, other):
    return self.pos.__lt__(other.pos)
  
  def __contains__(self, other):
    if other.x1 >= self.x1 and other.x2 <= self.x2 and \
       other.y1 >= self.y1 and other.y2 <= self.y2:
       return True
    return False

  def __str__(self):
    return 'Box(%d, %d, %d, %d)' % self.pos
  
  def __hash__(self):
    return self.pos.__hash__()


class Chunk(Box):

  def __init__(self, text:str, pos:Tuple, size:float=0.0, cell_id=None):
    super(Chunk, self).__init__(pos)
    self.text = text
    self.size = size
    self.cell_id = cell_id

  def __str__(self):
    return 'Chunk(text="%s", pos=(%d, %d, %d, %d))' % (self.text, *self.pos)

  def __repr__(self):
    return self.__str__()
  
  def dump_as_json_obj(self):
    return {"text":self.text, "pos":self.pos, "cell_id":self.cell_id}
  
  @classmethod
  def load_from_dict(cls, d):
    assert type(d) == dict
    assert type(d["text"]) == str
    assert len(d["pos"]) == 4
    cell_id = d["cell_id"] if "cell_id" in d else None
    return cls(d["text"].strip(), d["pos"], cell_id=cell_id)
  

class Table(object):
  
  """
  The output of table segmentation.
  With the Table object, we can get the set of cells
  and their corresponding text.
  """
  def __init__(self, row_n, col_n, cells:Iterable[Chunk]=None, tid=""):
    # NOTE the Chunk object here represents the coordinate of
    # the cell in the table.
    # NOTE x in cell object represents the row id
    self.tid = tid
    self.row_n = row_n
    self.col_n = col_n
    self.coo2cell_id = [
      [ -1 for _ in range(col_n) ] for _ in range(row_n) ]
    self.cells:List[Chunk] = []
    for cell in cells:
      self.add_cell(cell)
  
  def reverse(self, is_col=True):
    cells = self.cells
    self.cells = []
    cell:Chunk = None
    for cell in cells:
      if is_col:
        _c = Chunk(cell.text, (
          self.row_n - cell.x2, self.row_n - cell.x1, cell.y1, cell.y2))
      else:
        _c = Chunk(cell.text, (
          cell.x1, cell.x2, self.col_n - cell.y1, self.col_n - cell.y2))
      self.add_cell(_c)

  def add_cell(self, cell:Chunk):
    # TODO Check conflicts of cells
    assert cell.y2 < self.col_n
    assert cell.x2 < self.row_n

    for x in range(cell.x1, cell.x2 + 1, 1):
      for y in range(cell.y1, cell.y2 + 1, 1):
        self.coo2cell_id[x][y] = len(self.cells)
    self.cells.append(cell)
  
  def __getitem__(self, id_tuple):
    row_id, col_id = id_tuple
    assert row_id < self.row_n and col_id < self.col_n
    return self.cells[self.coo2cell_id[row_id][col_id]]