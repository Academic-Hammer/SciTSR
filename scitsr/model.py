# Copyright (c) 2019-present, Heng-Da Xu
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch


class Attention(torch.nn.Module):
  """Attention unit"""

  def __init__(self, size):
    super(Attention, self).__init__()
    self.size = size
    self.linear_q = torch.nn.Linear(size, size, bias=False)
    self.linear_k = torch.nn.Linear(size, size, bias=False)
    self.linear_v = torch.nn.Linear(size, size, bias=False)
    self.layer_norm_1 = torch.nn.LayerNorm(size)
    self.feed_forward = torch.nn.Sequential(
      torch.nn.Linear(size, size, bias=False),
      torch.nn.ReLU(),
      torch.nn.Linear(size, size, bias=False),
    )
    self.layer_norm_2 = torch.nn.LayerNorm(size)
    self.d_rate = 0.4
    self.dropout = torch.nn.Dropout(self.d_rate)

  def masked_softmax(self, x, mask, dim):
    ex = torch.exp(x)
    masked_exp = ex * mask.float()
    masked_exp_sum = masked_exp.sum(dim=dim, keepdim=True)
    x = masked_exp / (masked_exp_sum + 1e-6)
    return x

  def forward(self, x, y, mask):
    """
    Shapes:
      mask: [nodes/edges, edges/nodes]
      q: [nodes/edges, h]
      k: [edges/nodes, h]
      v: [edges/nodes, h]
      score: [nodes/edges, edges/nodes]
      x_atten: [nodes/edges, h]
    """
    q = self.linear_q(x)
    k = self.linear_k(y)
    v = self.linear_v(y)
    score = torch.mm(q, k.t()) / math.sqrt(self.size)
    score = self.masked_softmax(score, mask, dim=1)
    x_atten = torch.mm(score, v)
    # dropout
    x_atten = self.dropout(x_atten)
    x = self.layer_norm_1(x + x_atten)
    x_linear = self.feed_forward(x)
    # dropout
    x_linear = self.dropout(x_linear)
    x = self.layer_norm_2(x + x_linear)
    return x


class AttentionBlock(torch.nn.Module):
  """Attention Block"""

  def __init__(self, size):
    super(AttentionBlock, self).__init__()
    self.atten_e2v = Attention(size)
    self.atten_v2e = Attention(size)

  def forward(self, nodes, edges, adjacency, incidence):
    new_nodes = self.atten_e2v(nodes, edges, incidence)
    new_edges = self.atten_v2e(edges, nodes, incidence.t())
    return new_nodes, new_edges


class GraphAttention(torch.nn.Module):
  """Graph Attention Model"""

  def __init__(self, n_node_features, n_edge_features, \
               hidden_size, output_size, n_blocks):
    super(GraphAttention, self).__init__()
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.hidden_size = hidden_size
    self.n_blocks = n_blocks
    self.node_transform = torch.nn.Sequential(
      torch.nn.Linear(n_node_features, hidden_size),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size, hidden_size),
    )
    self.edge_transform = torch.nn.Sequential(
      torch.nn.Linear(n_edge_features, hidden_size),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size, hidden_size),
    )
    self.attention_blocks = torch.nn.ModuleList()
    for _ in range(n_blocks):
      self.attention_blocks.append(AttentionBlock(hidden_size))
    self.output_linear = torch.nn.Linear(hidden_size, output_size)

  def forward(self, nodes, edges, adjacency, incidence):
    nodes = self.node_transform(nodes)
    edges = self.edge_transform(edges)
    for attention_block in self.attention_blocks:
      nodes, edges = attention_block(nodes, edges, adjacency, incidence)
    outputs = self.output_linear(edges)
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    return outputs
