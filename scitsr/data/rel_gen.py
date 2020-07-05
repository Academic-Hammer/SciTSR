# Copyright (c) 2019-present, Zewen Chi
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

from collections import defaultdict
from tqdm import tqdm
from scitsr.table import Chunk
from scitsr.eval import Table2Relations, normalize
from scitsr.relation import Relation
from scitsr.data import utils


def dump_iters_as_tsv(filename, iterables, spliter="\t"):
  """
  Dump iters as tsv.
  item1\titem2\t... (from iterable1)
  item1\titem2\t... (from iterable2)
  """
  with open(filename, "w") as f:
    for iterable in iterables:
      iterable = [str(i) for i in iterable]
      f.write(spliter.join(iterable) + "\n")


def match(src:dict, trg:dict, src_chunks, trg_chunks, fid):
  """Match chunks to latex cells w.r.t. the contents."""
  sid2tid = {}
  print("--------%s---------------------------" % fid)
  for stxt, sids in src.items():
    if stxt in trg:
      tids = trg[stxt]
      if len(sids) == 1 and len(tids) == 1: sid2tid[sids[0]] = tids[0]
      elif len(sids) == len(tids):
        schunks = [(sid, src_chunks[sid]) for sid in sids]
        tchunks = [(tid, trg_chunks[tid]) for tid in tids]
        sorted_sc = sorted(schunks, key=lambda x: (-x[1].y1, x[1].x1))
        sorted_tc = sorted(tchunks, key=lambda x: (x[1].x1, x[1].y1))
        for (sid, _), (tid, _) in zip(sorted_sc, sorted_tc):
          sid2tid[sid] = tid
      else: 
        print("[W] length of sids and tids doesn't match")
    else: 
      print("[W] no match for text %s" % stxt)
  print("-----------------------------------------------------------")
  return sid2tid


def chunks2rel(ds_dir, rel_dir, chunk_ds="chunk", cell_ds="json"):
  if os.path.exists(rel_dir):
    print("%s exists." % rel_dir)
    return
  else: os.mkdir(rel_dir)

  skipped = 1

  for fid, (ch_json, cell_json) in tqdm(utils.ds_iter(ds_dir, [chunk_ds, cell_ds])):

    try:
      chunks = [Chunk.load_from_dict(cd) for cd in ch_json["chunks"]]
      table = utils.json2Table(cell_json, fid, splitted_content=True)
    except Exception as e:
      print(e)
      skipped += 1
      continue

    relations = Table2Relations(table)
    trg_chunks = table.cells
    rel_dict = {(r.from_id, r.to_id):r for r in relations}
    src_txt2id, trg_txt2id = defaultdict(list), defaultdict(list)
    for i, c in enumerate(chunks): src_txt2id[normalize(c.text)].append(i)
    for i, c in enumerate(trg_chunks): trg_txt2id[normalize(c.text)].append(i)

    sid2tid = match(src_txt2id, trg_txt2id, chunks, trg_chunks, fid)
    if sid2tid is None: continue
    tuples = []
    for i in range(len(chunks)):
      if i in sid2tid: ti = sid2tid[i]
      else: continue
      for j in range(i + 1, len(chunks), 1):
        if j in sid2tid: tj = sid2tid[j]
        else: continue
        if (ti, tj) in rel_dict: tuples.append((i, j, rel_dict[(ti, tj)]))
    
    dump_iters_as_tsv(os.path.join(rel_dir, fid + ".rel"), tuples)


if __name__ == "__main__":
  chunks2rel(
    ds_dir="/path/to/scitsr", 
    rel_dir="/path/to/scitsr/rel",
    chunk_ds="chunk",
    cell_ds="json"
  )
