import pandas as pd
import numpy as np
import os
import pickle

import torch
from torch.utils.data import random_split, Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, DataLoader

class Sample:
    def __init__(self, sample_id, group_attr, func_conn_path=None, stru_conn_path=None):
        self.sample_id = sample_id
        self.func_conn_path = func_conn_path
        self.stru_conn_path = stru_conn_path
        self.func_conn = None
        self.stru_conn = None
        self.cortical_thickness = pd.DataFrame()
        self.surface_area = pd.DataFrame()
        self.volume_size = pd.DataFrame()
        self.group_attr = group_attr

    def load_func_conn(self):
        if self.func_conn_path and os.path.exists(self.func_conn_path):
            self.func_conn = np.loadtxt(self.func_conn_path)
        else:
            self.func_conn = None
        return self.func_conn

    def load_stru_conn(self):
        if self.stru_conn_path and os.path.exists(self.stru_conn_path):
            self.stru_conn = np.loadtxt(self.stru_conn_path)
        else:
            self.stru_conn = None
        return self.stru_conn

    def get_func_conn(self):
        if self.func_conn is None:
            self.load_func_conn()
        return self.func_conn

    def get_stru_conn(self):
        if self.stru_conn is None:
            self.load_stru_conn()
        return self.stru_conn

    def set_cortical_thickness(self, thickness_df):
        self.cortical_thickness = thickness_df

    def set_surface_area(self, area_df):
        self.surface_area = area_df

    def set_volume_size(self, volume_df):
        self.volume_size = volume_df

    def get_node_features(self):
        return {
            'cortical_thickness': self.cortical_thickness,
            'surface_area': self.surface_area,
            'volume_size': self.volume_size
        }
    
def load_dataset(graph):
    print('loading data')
    num_graphs=len(graph)
    labels = [data.y.item() for data in graph]
    data_list = []
    for i in range(num_graphs):
        node_features = graph[i].x
        torch.FloatTensor(graph["graph_struct"][0][i][1])
        tepk = node_features.reshape(-1,1)
        tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(node_features.shape[0] * node_features.shape[0] * 0.2 - 1)]
        edge = torch.Tensor(np.where(node_features > mk, 1, 0))
        data_example = Data(x=node_features,edge_index=dense_to_sparse(edge)[0],y=label[i])
        data_list.append(data_example)

    return data_list

def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=None):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size = num_eval, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size = num_test, shuffle=False)
    return dataloader

def create_pyg_data(sample, all_seg_ids, indices_to_remove, device):
    # 获取特征矩阵，形状为 [num_nodes, num_features]（例如 110x3）
    #x = torch.tensor(align_features(sample, all_seg_ids), dtype=torch.float).to(device)
    #x = x[torch.tensor([i for i in range(x.shape[0]) if i not in indices_to_remove])]
    
    # 计算节点数
    num_nodes = len(all_seg_ids)
    # 获取功能连接和结构连接矩阵
    func_conn = sample.get_func_conn() if sample.get_func_conn() is not None else np.zeros((110, 110))
    stru_conn = sample.get_stru_conn() if sample.get_stru_conn() is not None else np.zeros((110, 110))
    
    # 移除不需要的节点
    func_conn = remove_nodes_from_matrix(func_conn, indices_to_remove)
    func_conn[func_conn == np.inf] = 0
    # 构建边特征矩阵
    # edge_attr = torch.stack([
    #     torch.tensor(func_conn.flatten(), dtype=torch.float),
    #     torch.tensor(stru_conn.flatten(), dtype=torch.float)
    # ], dim=1).to(device)
    x = torch.tensor(func_conn, dtype=torch.float)
    tepk = x.reshape(-1,1)
    tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
    mk = tepk[int(x.shape[0] * x.shape[0] * 0.2 - 1)]
    edge = torch.Tensor(np.where(x > mk, 1, 0))
    # 构建连接图的边索引
    #edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes)], dtype=torch.long).t().contiguous().to(device)
    
    # 构建标签
    label = 0 if sample.group_attr == 'ASD' \
        else 1 if sample.group_attr == 'FXS'\
            else 2
    y = torch.tensor([label], dtype=torch.long)
    
    # 返回 PyG Data 对象
    return Data(x=x, edge_index=dense_to_sparse(edge)[0], y=y)

def align_features(sample, all_seg_ids):
    features = np.zeros((len(all_seg_ids), 3), dtype=np.float32)
    
    if not sample.cortical_thickness.empty:
        for _, row in sample.cortical_thickness.iterrows():
            seg_id = row['SegId']
            if seg_id in all_seg_ids:
                idx = all_seg_ids.index(seg_id)
                features[idx, 0] = row['Mean_Thickness']
    
    if not sample.surface_area.empty:
        for _, row in sample.surface_area.iterrows():
            seg_id = row['SegId']
            if seg_id in all_seg_ids:
                idx = all_seg_ids.index(seg_id)
                features[idx, 1] = row['Mean_Area']
    
    if not sample.volume_size.empty:
        for _, row in sample.volume_size.iterrows():
            seg_id = row['SegId']
            if seg_id in all_seg_ids:
                idx = all_seg_ids.index(seg_id)
                features[idx, 2] = row['Volume']
    
    return features

def remove_nodes_from_matrix(matrix, indices_to_remove):
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[indices_to_remove] = False
    return matrix[mask][:, mask]

def load_seg_id_mapping(harvard_oxford_path):
    df = pd.read_excel(harvard_oxford_path, sheet_name='Sheet1', header=None, engine='openpyxl')
    seg_id_to_name = {}
    for _, row in df.iloc[1:].iterrows():
        seg_id = int(row[1])  # 第二列是SegId
        region = f"{row[3]}_{row[4]}"  # 合并4-5列为脑区名称
        seg_id_to_name[seg_id] = region
    return seg_id_to_name

if __name__ == "__main__":
    # 加载数据集
    with open('../data/dataset_SFvsc.pkl', 'rb') as f:
        dataset = pickle.load(f)

    data_path = '../data/HOA_atlas/HarvardOxford_Atlas_NewIndex_YCG.xlsx'
    # 提取所有SegId
    seg_id_to_name = load_seg_id_mapping(data_path)
    all_seg_ids = sorted(seg_id_to_name.keys())

    # 删除功能连接矩阵中的指定节点
    seg_ids_to_remove = [1016, 1116]
    indices_to_remove = [all_seg_ids.index(seg_id) for seg_id in seg_ids_to_remove]
    # 移除指定的seg_id并重置索引
    all_seg_ids = [seg_id for seg_id in all_seg_ids if seg_id not in seg_ids_to_remove]
    num_nodes = len(all_seg_ids)

    # 过滤掉缺失关键数据的样本
    valid_samples = []
    for sample in dataset.values():
        if not sample.cortical_thickness.empty and not sample.surface_area.empty and not sample.volume_size.empty and sample.get_func_conn() is not None and sample.get_stru_conn() is not None:
            valid_samples.append(sample)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pyg_dataset = [create_pyg_data(sample, all_seg_ids, indices_to_remove, device) for sample in valid_samples]
    pickle.dump(pyg_dataset, open('../data/pyg_dataset.pkl', 'wb'))