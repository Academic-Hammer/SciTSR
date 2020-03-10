"""Load Real Data for Graph Attention Model
Author: Heng-Da Xu <dadamrxx@gmail.com>
Date Created: March 22, 2019
Modified by: Heng-Da Xu <dadamrxx@gmail.com>
Date Modified: March 22, 2019
"""
import json
import os
from typing import List
import random

from tqdm import tqdm
from pprint import pprint
import torch
from torch.utils.data import Dataset

from scitsr.data.utils import preprocessing, construct_knn_edges
from scitsr.graph import Edge, Vertex
from scitsr.table import Chunk
from scitsr.eval import Relation


class Data:

    def __init__(self, chunks, relations, cells=None,
                 path=None, nodes=None, edges=None, 
                 adj=None, incidence=None, labels=None):
        self.chunks = chunks
        self.relations = relations
        self.cells = cells
        self.path = path
        self.nodes = nodes
        self.edges = edges
        self.adj = adj
        self.incidence = incidence
        self.labels = labels


class TableDataset(Dataset):

    def __init__(self, dataset_dir, with_cells, trim=None,
                 node_norm=None, edge_norm=None, exts=None):
        if exts is None: exts = ['chunk12', 'rel12']
        raw_dataset = self.load_dataset(
            dataset_dir, with_cells, trim, exts=exts)
        raw_dataset = preprocessing(raw_dataset)

        dataset = []
        for data in tqdm(raw_dataset, desc='TableDataset'):
            if len(data.chunks) <= 2 or len(data.relations) <= 2:
                continue
            data.nodes, data.edges, data.adj, data.incidence, data.labels = \
                self.transform(data.chunks, data.relations)
            dataset.append(data)
        self.node_norm, self.edge_norm = self.feature_normalizaion(
            dataset, node_norm, edge_norm)
        self.dataset = dataset

        self.n_node_features = self.dataset[0].nodes.size(1)
        self.n_edge_features = self.dataset[0].edges.size(1)
        self.output_size = self.dataset[0].labels.max().item() + 1
    
    def shuffle(self):
        random.shuffle(self.dataset)
    
    def feature_normalizaion(self, dataset, node_param=None, edge_param=None):

        def _get_mean_std(features):
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            return mean, std
        
        def _norm(features, mean, std, eps=1e-6):
            return (features - mean) / (std + 1e-6)
        
        # normalize edge features
        if edge_param is None:
            all_edge_features = torch.cat([data.edges for data in dataset])
            edge_mean, edge_std = _get_mean_std(all_edge_features)
        else: edge_mean, edge_std = edge_param
        for data in dataset:
            data.edges = _norm(data.edges, edge_mean, edge_std)
        
        # normalize node features
        if node_param is None:
            all_node_features = torch.cat([data.nodes for data in dataset])
            node_mean, node_std = _get_mean_std(all_node_features)
        else: node_mean, node_std = node_param
        for data in dataset:
            data.nodes = _norm(data.nodes, node_mean, node_std)
        
        return (node_mean, node_std), (edge_mean, edge_std)


    def transform(self, chunks, relations):
        vertexes = self.get_vertexes(chunks)
        nodes = self.get_vertex_features(vertexes)
        adj, incidence = self.get_adjcancy(relations, len(chunks))
        edges = self.get_edges(relations, vertexes)
        labels = self.get_labels(relations)
        nodes, edges, adj, incidence, labels = \
            self.to_tensors(nodes, edges, adj, incidence, labels)
        #nodes, edges = self.normlize(nodes), self.normlize(edges)
        return nodes, edges, adj, incidence, labels

    def load_dataset(self, dataset_dir, with_cells, trim=None, debug=False, exts=None):
        dataset, cells = [], []
        if exts is None: exts = ['chunk', 'rel']
        if with_cells:
            exts.append('json')
        sub_paths = self.get_sub_paths(dataset_dir, exts, trim=trim)
        for i, paths in enumerate(sub_paths):
            if debug and i > 50:
                break
            chunk_path = paths[0]
            relation_path = paths[1]

            chunks = self.load_chunks(chunk_path)
            # TODO handle big tables
            #if len(chunks) > 100 or len(chunks) == 0: continue
            relations = self.load_relations(relation_path)
            #new_chunks, new_rels = self.clean_chunk_rel(chunks, relations)
            #chunks, relations = new_chunks, new_rels

            if with_cells:
                cell_path = paths[2]
                with open(cell_path) as f:
                    cell_json = json.load(f)
            else:
                cell_json = None

            dataset.append(Data(
                chunks=chunks,
                relations=relations,
                cells=cell_json,
                path=chunk_path,
            ))
        return dataset
    
    def clean_chunk_rel(self, chunks, relations):
        """Remove null chunks"""
        new_chunks = []
        oldid2newid = [-1 for i in range(len(chunks))]
        for i, c in enumerate(chunks):
            if c.x2 == c.x1 or c.y2 == c.y1 or c.text == "":
                continue
            oldid2newid[i] = len(new_chunks)
            new_chunks.append(c)
        new_rels = []
        for i, j, t in relations:
            ni = oldid2newid[i]
            nj = oldid2newid[j]
            if ni != -1 and nj != -1: new_rels.append((ni, nj, t))
        return new_chunks, new_rels

    def load_chunks(self, chunk_path):
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

    def load_relations(self, relation_path):
        with open(relation_path, 'r') as f:
            lines = f.readlines()
        relations = []
        for line in lines:
            i, j, t = line.split('\t')
            i, j, t = int(i), int(j), int(t.split(':')[0])
            relations.append((i, j, t))
        return relations

    def get_sub_paths(self, root_dir: str, sub_names: List[str], trim=None):
        # Check the existence of directories
        assert os.path.isdir(root_dir)
        # TODO: sub_dirs redundancy
        sub_dirs = []
        for sub_name in sub_names:
            sub_dir = os.path.join(root_dir, sub_name)
            assert os.path.isdir(sub_dir), '"%s" is not dir.' % sub_dir
            sub_dirs.append(sub_dir)

        paths = []
        d = os.listdir(sub_dirs[0])
        d = d[:trim] if trim else d
        for file_name in d:
            sub_paths = [os.path.join(sub_dirs[0], file_name)]
            name = os.path.splitext(file_name)[0]
            for ext in sub_names[1:]:
                sub_path = os.path.join(root_dir, ext, name + '.' + ext)
                assert os.path.exists(sub_path)
                sub_paths.append(sub_path)
            paths.append(sub_paths)
        
        return paths

    def get_vertexes(self, chunks):
        coords_x, coords_y = [], []
        for chunk in chunks:
            coords_x.append(chunk.x1)
            coords_x.append(chunk.x2)
            coords_y.append(chunk.y1)
            coords_y.append(chunk.y2)
        table_width = max(coords_x) - min(coords_x)
        table_height = max(coords_y) - min(coords_y)

        vertexes = []
        for index, chunk in enumerate(chunks):
            vertex = Vertex(index, chunk, table_width, table_height)
            vertexes.append(vertex)
        return vertexes

    def get_vertex_features(self, vertexes):
        vertex_features = []
        for vertex in vertexes:
            features = [v for v in vertex.get_features().values()]
            vertex_features.append(features)
        return vertex_features

    def get_adjcancy(self, relations, n_vertexes):
        n_edges = len(relations)
        adj = [[0] * n_vertexes for _ in range(n_vertexes)]
        incidence = [[0] * n_edges for _ in range(n_vertexes)]
        for idx, (i, j, _) in enumerate(relations):
            adj[i][j] = adj[j][i] = 1
            incidence[i][idx] = incidence[j][idx] = 1
        return adj, incidence

    def get_edges(self, relations, vertexes):
        edge_features = []
        for i, j, _ in relations:
            edge = Edge(vertexes[i], vertexes[j])
            features = [v for v in edge.get_features().values()]
            edge_features.append(features)
        return edge_features

    def get_labels(self, relations):
        labels = [label for id_a, id_b, label in relations]
        return labels

    def to_tensors(self, nodes, edges, adj, incidence, labels):
        nodes = torch.tensor(nodes, dtype=torch.float)
        edges = torch.tensor(edges, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.long)
        incidence = torch.tensor(incidence, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return nodes, edges, adj, incidence, labels

    #TODO normalize over dataset?
    def normlize(self, features):
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        features = (features - mean) / (std + 1e-6)
        return features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TableInferDataset(TableDataset):

    def __init__(self, dataset_dir, trim=None,
                 node_norm=None, edge_norm=None, exts=None):
        if exts is None: exts = ['chunk12', 'rel12']
        raw_dataset = self.load_dataset(
            dataset_dir, True, trim, exts=exts)
        raw_dataset = preprocessing(raw_dataset)

        dataset = []
        for data in tqdm(raw_dataset, desc='TableInferDataset'):
            if len(data.chunks) <= 2 or len(data.relations) <= 2:
                continue
            data.nodes, data.edges, data.adj, data.incidence, data.relations = \
                self.transform(data.chunks)
            dataset.append(data)
        self.node_norm, self.edge_norm = self.feature_normalizaion(
            dataset, node_norm, edge_norm)
        self.dataset = dataset

        self.n_node_features = self.dataset[0].nodes.size(1)
        self.n_edge_features = self.dataset[0].edges.size(1)
        self.output_size = 3

    def transform(self, chunks):
        vertexes = self.get_vertexes(chunks)
        nodes = self.get_vertex_features(vertexes)
        relations = construct_knn_edges(chunks)
        if len(relations) <= 2:
            return None
        adj, incidence = self.get_adjcancy(relations, len(chunks))
        edges = self.get_edges(relations, vertexes)
        nodes, edges, adj, incidence = \
            self.to_tensors(nodes, edges, adj, incidence)
        #nodes, edges = self.normlize(nodes), self.normlize(edges)
        return nodes, edges, adj, incidence, relations

    def to_tensors(self, nodes, edges, adj, incidence):
        nodes = torch.tensor(nodes, dtype=torch.float)
        edges = torch.tensor(edges, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.long)
        incidence = torch.tensor(incidence, dtype=torch.long)
        return nodes, edges, adj, incidence