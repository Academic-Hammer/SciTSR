# Copyright (c) 2019-present, Zewen Chi, Heng-Da Xu
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

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