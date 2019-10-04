# Copyright (c) 2019-present, Zewen Chi
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

def normalize(s:str, rule=0):

  if rule == 0:
    s = s.replace("\r", "")
    s = s.replace("\n", "")
    s = s.replace(" ", "")
    s = s.replace("\t", "")
    return s.upper()
  else:
    raise NotImplementedError


class Relation(object):

  def __init__(self, from_text, to_text, direction, from_id=0, to_id=0, no_blanks=0):
    self.from_text = from_text
    self.to_text = to_text
    self.direction = direction
    self.no_blanks = no_blanks
    self.from_id = from_id
    self.to_id = to_id
  
  def __eq__(self, rl):
    this_ft = normalize(self.from_text)
    this_tt = normalize(self.to_text)
    rl_ft = normalize(rl.from_text)
    rl_tt = normalize(rl.to_text)
    if len(this_ft) == 0 or len(this_tt) == 0 or \
       len(rl_ft) == 0 or len(rl_tt) == 0:
      print("Warning: Text comparison of 0-length strings after normalization",
        file=sys.stderr)
  
    return this_ft == rl_ft and this_tt == rl_tt and \
      self.direction == rl.direction and self.no_blanks == rl.no_blanks
  
  def equal(self, rl, cmp_blank=True):
    this_ft = normalize(self.from_text)
    this_tt = normalize(self.to_text)
    rl_ft = normalize(rl.from_text)
    rl_tt = normalize(rl.to_text)
    if len(this_ft) == 0 or len(this_tt) == 0 or \
       len(rl_ft) == 0 or len(rl_tt) == 0:
      print("Warning: Text comparison of 0-length strings after normalization",
        file=sys.stderr)

    return this_ft == rl_ft and this_tt == rl_tt and \
      self.direction == rl.direction and \
      (self.no_blanks == rl.no_blanks if cmp_blank else True)
  
  def __str__(self):
    return "%d:%d" % (self.direction, self.no_blanks)