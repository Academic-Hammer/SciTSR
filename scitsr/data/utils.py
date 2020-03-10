# Copyright (c) 2019-present, Zewen Chi, Heng-Da Xu
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
from typing import List

from tqdm import tqdm

from scitsr.table import Chunk, Table


def ds_iter(ds_dir, ds_ls):
  ds_dir_ls = [os.path.join(ds_dir, ds) for ds in ds_ls]
  ds_dir_ext = [os.path.splitext(os.listdir(d)[0])[1] for d in ds_dir_ls]
  for fn in os.listdir(ds_dir_ls[0]):
    fid, _ = os.path.splitext(fn)
    fid_fn = [os.path.join(
      ds_dir_ls[i], fid + ds_dir_ext[i]
    ) for i,ds in enumerate(ds_dir_ls)]
    ret = []
    try:
      for f in fid_fn:
        with open(f) as fp:
          ret.append(json.load(fp))
      if len(ret) != len(ds_ls):
        print("[W] 1 instance skipped")
      else:
        yield fid, ret
    except:
      continue


def json2Table(json_obj, tid="", splitted_content=False):
  """Construct a Table object from json object
  Args:
    json_obj: a json object
  Returns:
    a Table object
  """
  jo = json_obj["cells"]
  row_n, col_n = 0, 0
  cells = []
  for co in jo:
    content = co["content"]
    if content is None: continue
    if splitted_content:
      content = " ".join(content)
    else:
      content = content.strip()
    if content == "": continue
    start_row = co["start_row"]
    end_row = co["end_row"]
    start_col = co["start_col"]
    end_col = co["end_col"]
    row_n = max(row_n, end_row)
    col_n = max(col_n, end_col)
    cell = Chunk(content, (start_row, end_row, start_col, end_col))
    cells.append(cell)
  return Table(row_n + 1, col_n + 1, cells, tid)

def transform_coord(chunks):
    # Get table width and height
    coords_x, coords_y = [], []
    for chunk in chunks:
        coords_x.append(chunk.x1)
        coords_x.append(chunk.x2)
        coords_y.append(chunk.y1)
        coords_y.append(chunk.y2)
    # table_width = max(coords_x) - min(coords_x)
    # table_height = max(coords_y) - min(coords_y)

    # Coordinate transformation for chunks
    table_min_x, table_max_y = min(coords_x), max(coords_y)
    chunks_new = []
    for chunk in chunks:
        x1 = chunk.x1 - table_min_x
        x2 = chunk.x2 - table_min_x
        y1 = table_max_y - chunk.y2
        y2 = table_max_y - chunk.y1
        chunk_new = Chunk(
            text=chunk.text,
            pos=(x1, x2, y1, y2),
        )
        chunks_new.append(chunk_new)

    # return table_width, table_height
    return chunks_new


def _eul_dis(chunks, i, j):
  xi = (chunks[i].x1 + chunks[i].x2) / 2
  yi = (chunks[i].y1 + chunks[i].y2) / 2
  xj = (chunks[j].x1 + chunks[j].x2) / 2
  yj = (chunks[j].y1 + chunks[j].y2) / 2
  return (xj - xi)**2 + (yj-yi)**2


def construct_knn_edges(chunks, k=20):
  relations = []
  edges = set()
  for i in range(len(chunks)):
    _dis_ij = []
    for j in range(len(chunks)):
      if j == i: continue
      _dis_ij.append((_eul_dis(chunks, i, j), j))
    sorted_dis_ij = sorted(_dis_ij)
    for _, j in sorted_dis_ij[:k]:
      _i, _j = (i, j) if i < j else (j, i)
      if (_i, _j) not in edges:
        edges.add((_i, _j))
        relations.append((_i, _j, 0))
  return relations


def add_knn_edges(chunks, relations, k=20, debug=False):
  """Add edges according to knn of vertexes.
  """
  edges = set()
  rel_recall = {}
  for i, j, _ in relations:
    edges.add((i, j) if i < j else (j, i))
    rel_recall[(i, j) if i < j else (j, i)] = False
  for i in range(len(chunks)):
    _dis_ij = []
    for j in range(len(chunks)):
      if j == i: continue
      _dis_ij.append((_eul_dis(chunks, i, j), j))
    sorted_dis_ij = sorted(_dis_ij)
    for _, j in sorted_dis_ij[:k]:
      _i, _j = (i, j) if i < j else (j, i)
      if (_i, _j) in rel_recall: rel_recall[(_i, _j)] = True
      if (_i, _j) not in edges:
        edges.add((_i, _j))
        relations.append((_i, _j, 0))
  cnt = 0
  for _, val in rel_recall.items():
    if val: cnt += 1
  recall = 0 if len(rel_recall) == 0 else cnt / len(rel_recall)
  if debug:
    print("add knn edge. recall:%.3f" % recall)
  return relations, recall


def add_null_edges(chunks, relations):
  n_chunks = len(chunks)

  # Convert relations to adjcancy matrix
  adj = [[0] * n_chunks for _ in range(n_chunks)]
  for i, j, _ in relations:
    adj[i][j] = adj[j][i] = 1

  # Add null edges
  for i in range(n_chunks):
    x = (chunks[i].x1 + chunks[i].x2) / 2
    y = (chunks[i].y1 + chunks[i].y2) / 2
    for j in range(i + 1, n_chunks):
      if adj[i][j] == 1:
        continue
      xx = (chunks[j].x1 + chunks[j].x2) / 2
      yy = (chunks[j].y1 + chunks[j].y2) / 2
      if (xx - x)**2 + (yy - y)**2 > 30**2:
        continue
      adj[i][j] = adj[j][i] = 1
      relations.append((i, j, 0))
  
  return relations


def add_full_edges(chunks, relations):
  n_chunks = len(chunks)

  # Convert relations to adjcancy matrix
  adj = [[0] * n_chunks for _ in range(n_chunks)]
  for i, j, _ in relations:
    adj[i][j] = adj[j][i] = 1

  # Add null edges
  for i in range(n_chunks):
    x = (chunks[i].x1 + chunks[i].x2) / 2
    y = (chunks[i].y1 + chunks[i].y2) / 2
    for j in range(i + 1, n_chunks):
      if adj[i][j] == 1:
        continue
      adj[i][j] = adj[j][i] = 1
      relations.append((i, j, 0))
  
  return relations


def preprocessing(dataset, debug=True):
  # random.seed(0)
  dataset_new = []
  edge_recall_sum = 0
  cnt = 0
  if debug: recall_path = []
  for data in tqdm(dataset, desc='preprocessing'):
    data.chunks = transform_coord(data.chunks)
    #data.relations = add_null_edges(data.chunks, data.relations)
    data.relations, recall = add_knn_edges(data.chunks, data.relations)
    edge_recall_sum += recall
    cnt += 1
    if debug: recall_path.append((recall, data.path))
    # data.relations = add_full_edges(data.chunks, data.relations)
    # random.shuffle(relations)
  print("edge recall:%.3f" % (edge_recall_sum / cnt))
  return dataset