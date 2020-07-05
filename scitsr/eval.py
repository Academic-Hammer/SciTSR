# Copyright (c) 2019-present, Zewen Chi
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

from scitsr.relation import Relation
from scitsr.table import Table, Chunk


DIR_HORIZ = 1
DIR_VERT = 2
DIR_SAME_CELL = 3


def normalize(s:str, rule=0):
  if rule == 0:
    s = s.replace("\r", "")
    s = s.replace("\n", "")
    s = s.replace(" ", "")
    s = s.replace("\t", "")
    return s.upper()
  else:
    raise NotImplementedError


def eval_relations(gt:List[List], res:List[List], cmp_blank=True):
  """Evaluate results

  Args:
    gt: a list of list of Relation
    res: a list of list of Relation
  """

  #TODO to know how to calculate the total recall and prec

  assert len(gt) == len(res)
  tot_prec = 0
  tot_recall = 0
  total = 0
  # print("evaluating result...")

  # for _gt, _res in tqdm(zip(gt, res)):
  # for _gt, _res in tqdm(zip(gt, res), total=len(gt), desc='eval'):
  idx, t = 0, len(gt) 
  for _gt, _res in zip(gt, res):
    idx += 1
    print('Eval %d/%d (%d%%)' % (idx, t, idx / t * 100), ' ' * 45, end='\r')
    corr = compare_rel(_gt, _res, cmp_blank)
    precision = corr / len(_res) if len(_res) != 0 else 0
    recall = corr / len(_gt) if len(_gt) != 0 else 0
    tot_prec += precision
    tot_recall += recall
    total += 1
  # print()
  
  precision = tot_prec / total
  recall = tot_recall / total
  # print("Test on %d instances. Precision: %.2f, Recall: %.2f" % (
  #   total, precision, recall))
  return precision, recall

def compare_rel(gt_rel:List[Relation], res_rel:List[Relation], cmp_blank=True):
  count = 0

  #print("compare_rel =======================")
  #for gt in gt_rel:
  #  print("rel gt:", gt.from_text, gt.to_text, gt.direction)
  #for gt in res_rel:
  #  print("rel res:", gt.from_text, gt.to_text, gt.direction)
  #print("\n\n\n\n\n")

  dup_res_rel = [r for r in res_rel]
  for gt in gt_rel:
    to_rm = None
    for i, res in enumerate(dup_res_rel):
      if gt.equal(res, cmp_blank):
        to_rm = i
        count += 1
        break
    if to_rm is not None:
      dup_res_rel = dup_res_rel[:i] + dup_res_rel[i + 1:]
  
  return count

def Table2Relations(t:Table):
  """Convert a Table object to a List of Relation.
  """
  ret = []
  cl = t.coo2cell_id
  # remove duplicates with pair set
  used = set()

  # look right
  for r in range(t.row_n):
    for cFrom in range(t.col_n - 1):
      cTo = cFrom + 1
      loop = True
      while loop and cTo < t.col_n:
        fid, tid = cl[r][cFrom], cl[r][cTo]
        if fid != -1 and tid != -1 and fid != tid:
          if (fid, tid) not in used:
            ret.append(Relation(
              from_text=t.cells[fid].text,
              to_text=t.cells[tid].text,
              direction=DIR_HORIZ,
              from_id=fid,
              to_id=tid,
              no_blanks=cTo - cFrom - 1
            ))
            used.add((fid, tid))
          loop = False
        else:
          if fid != -1 and tid != -1 and fid == tid:
            cFrom = cTo
        cTo += 1
  
  # look down
  for c in range(t.col_n):
    for rFrom in range(t.row_n - 1):
      rTo = rFrom + 1
      loop = True
      while loop and rTo < t.row_n:
        fid, tid = cl[rFrom][c], cl[rTo][c]
        if fid != -1 and tid != -1 and fid != tid:
          if (fid, tid) not in used: 
            ret.append(Relation(
              from_text=t.cells[fid].text,
              to_text=t.cells[tid].text,
              direction=DIR_VERT,
              from_id=fid,
              to_id=tid,
              no_blanks=rTo - rFrom - 1
            ))
            used.add((fid, tid))
          loop = False
        else:
          if fid != -1 and tid != -1 and fid == tid:
            rFrom = rTo
        rTo += 1

  return ret

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

def json2Relations(json_obj, splitted_content):
  return Table2Relations(json2Table(json_obj, "", splitted_content))


