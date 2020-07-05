# -*- coding: utf-8 -*-
"""Graph Attention Model Trainning
Author: Heng-Da Xu <dadamrxx@gmail.com>
Date Created: March 21, 2019
Modified by: Heng-Da Xu <dadamrxx@gmail.com>, Zewen
Date Modified: March 23, 2019
"""
import torch
from tqdm import tqdm

from scitsr.data.loader import TableDataset, TableInferDataset, Data
from scitsr.model import GraphAttention
from scitsr.table import Chunk


class Trainer:
    """Trainer"""

    def __init__(self, model, train_dataset, test_dataset, infer_dataset,
                 criterion, optimizer, n_epochs, device, weight_clipping):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.infer_dataset = infer_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.weight_clipping = weight_clipping
        self.device = device
        self.empty = 2000
        self.epoch_info_dict = {
            'loss': None,
            'acc': None,
            't_acc': None,
            'precision': None,
            'recall': None,
        }

    def _reset_epcho_info(self):
        for k in self.epoch_info_dict:
            self.epoch_info_dict[k] = None

    def _print_epoch_info(self, epoch, desc, **keywords):
        self.epoch_info_dict.update(keywords)
        print('[Epoch %2d] %s' % (epoch, desc), end='')
        n_none = 0
        for k, v in self.epoch_info_dict.items():
            if self.epoch_info_dict[k] is not None:
                print(' | %s: %.3f' % (k, v), end='')
            else:
                n_none += 1
        #print(end='\n' if n_none == 0 else '\r')
        print("")

    def train(self):

        print('Start training ...')
        for epoch in range(1, self.n_epochs + 1):
            self._reset_epcho_info()

            torch.cuda.empty_cache()
            loss = self.train_epoch(epoch, self.train_dataset)
            self._print_epoch_info(epoch, 'train', loss=loss)

            torch.cuda.empty_cache()
            #train_acc = self.test_epoch(epoch, self.train_dataset)
            #self._print_epoch_info(epoch, 'train', acc=train_acc)

            test_acc = self.test_epoch(epoch, self.test_dataset)
            self._print_epoch_info(epoch, 'test', t_acc=test_acc)

        print('Training finished.')
        return self.model

    def train_epoch(self, epoch, dataset, should_print=False):
        self.model.train()
        loss_list = []
        for index, data in tqdm(enumerate(dataset)):
            torch.cuda.empty_cache()
            self._to_device(data)
            # if index % 10 == 0:
            percent = index / len(dataset) * 100
            if should_print:
                print('[Epoch %d] Train | Data %d (%d%%): loss: | path: %s' % \
                (epoch, index, percent, data.path), ' ' * 20, end='\r')
            # try:
            outputs = self.model(data.nodes, data.edges, data.adj, data.incidence)
            # except Exception as e:
                # print(e, data.path)
            loss = self.criterion(outputs, data.labels)
            loss_list.append(loss.item())

            if should_print:
                print('[Epoch %d] Train | Data %d (%d%%): loss: %.3f | path: %s' % \
                (epoch, index, percent, loss.item(), data.path), ' ' * 20, end='\n')

            self.optimizer.zero_grad()
            loss.backward()
            if self.weight_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.weight_clipping
                )
            self.optimizer.step()
        loss = sum(loss_list) / len(loss_list)
        return loss

    def test_epoch(self, epoch, dataset, should_print=False, use_mask=True):
        """
        use_mask: mask the 0 label
        """
        self.model.eval()
        acc_list = []
        for index, data in tqdm(enumerate(dataset)):
            self._to_device(data)
            percent = index / len(dataset) * 100
            if should_print:
                print('[Epoch %d] Test | Data %d (%d%%): acc: | path: %s' % \
                (epoch, index, percent, data.path), ' ' * 30, end='\r')
            outputs = self.model(data.nodes, data.edges, data.adj, data.incidence)
            _lab_len = len(data.labels)
            if use_mask:
                for i in data.labels: 
                    if i == 0: _lab_len -= 1
                _labels = torch.LongTensor(
                    [(-1 if i == 0 else i) for i in data.labels]).to(self.device)
            else: _labels = data.labels
            acc = (outputs.max(dim=1)[1] == _labels).float().sum().item() / _lab_len
            acc_list.append(acc)
            # if index % 10 == 0:
            if should_print:
                print('[Epoch %d] Test | Data %d (%d%%): acc: %.3f | path: %s' % \
                (epoch, index, percent, acc, data.path), ' ' * 30, end='\n')
        acc = sum(acc_list) / len(acc_list)
        return acc

    def _to_device(self, data):
        data.nodes = data.nodes.to(self.device)
        data.edges = data.edges.to(self.device)
        data.adj = data.adj.to(self.device)
        data.incidence = data.incidence.to(self.device)
        if data.labels is not None:
            data.labels = data.labels.to(self.device)

            
def patch_chunks(dataset_folder):
	"""
	To patch the all chunk files of the train & test dataset that have the problem of duplicate last character
	of the last cell in all chunk files
	:param dataset_folder: train dataset path
	:return: 1
	"""
	import os
	import shutil
	from pathlib import Path

	shutil.move(os.path.join(dataset_folder, "chunk"), os.path.join(dataset_folder, "chunk-old"))
	dir_ = Path(os.path.join(dataset_folder, "chunk-old"))
	os.makedirs(os.path.join(dataset_folder, "chunk"), exist_ok=True)

	for chunk_path in dir_.iterdir():
		# print(chunk_path)
		with open(str(chunk_path), encoding="utf-8") as f:
			chunks = json.load(f)['chunks']
		chunks[-1]['text'] = chunks[-1]['text'][:-1]

		with open(str(chunk_path).replace("chunk-old", "chunk"), "w", encoding="utf-8") as ofile:
			json.dump({"chunks": chunks}, ofile)
	print("Input files patched, ready for the use")
	return 1


if __name__ == '__main__':

    train_path = "/path/to/train_folder"
    test_path = "/path/to/test_folder/"
    patch_chunks(train_path)
    patch_chunks(test_path)
    
    train_dataset = TableDataset(
        train_path, with_cells=False, exts=["chunk", "rel"])
    node_norm, edge_norm = train_dataset.node_norm, train_dataset.edge_norm
    infer_dataset = test_dataset = TableDataset(
        test_path, with_cells=True, node_norm=node_norm,
        edge_norm=edge_norm, exts=["chunk", "rel"])
    #device = 'cuda:1'
    device = "cpu"

    # Hyper-parameters 
    n_node_features = train_dataset.n_node_features
    n_edge_features = train_dataset.n_edge_features
    output_size = train_dataset.output_size

    #hidden_size = 64
    hidden_size = 4
    n_blocks = 3
    n_epochs = 15
    learning_rate = 0.0005
    weight_clipping = 1
    weight_decay = 1e-4
    random_seed = 0
    model_path = './gat-model.pt'

    # Random seed and device
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed(random_seed)
    device = torch.device(device)

    model = GraphAttention(
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        hidden_size=hidden_size,
        output_size=output_size,
        n_blocks=n_blocks,
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(
        torch.FloatTensor([0.2, 1, 1]).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        infer_dataset=infer_dataset,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=n_epochs,
        weight_clipping=weight_clipping,
        device=device,
    )
    model = trainer.train()

    # Save model
    torch.save(model.state_dict(), model_path)
    print('The model has been saved to "%s".' % model_path)

    # Save node_norm, edge_norm
    node_mean_path = "./tmp_gat_node_mean"
    node_std_path = "./tmp_gat_node_std"
    edge_mean_path = "./tmp_gat_edge_mean"
    edge_std_path = "./tmp_gat_edge_std"
    node_mean, node_std = node_norm
    edge_mean, edge_std = edge_norm
    
    torch.save(node_mean, node_mean_path)
    torch.save(node_std, node_std_path)
    torch.save(edge_mean, edge_mean_path)
    torch.save(edge_std, edge_std_path)
