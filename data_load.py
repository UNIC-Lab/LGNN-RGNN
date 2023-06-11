'''
    创建数据集，分为用户点对，uav节点位置
'''
import torch
import numpy as np
import random
from configs import args
from sklearn.cluster import KMeans


def data_create(N, M):
    '''
    input:
         N:用户对数
         M:无人机个数
    
    Output：
         features：用户和无人机的位置，前2N个为用户坐标，后M个为无人机坐标
         edge_index:边的索引
    
    用户位置随机生成
    无人机位置为2N个用户的KMeans聚类
    '''
    # torch.manual_seed(50)
    # random.seed(50)
    x1 = torch.rand(N, 1)
    x2 = torch.rand(N, 1)
    y1 = torch.rand(N, 1)
    y2 = torch.rand(N, 1)

    idx = list(range(N))
    random.shuffle(idx)
    x1 = x1[idx]
    random.shuffle(idx)
    x2 = x2[idx]

    user_src = torch.column_stack([x1, y1])
    user_dst = torch.column_stack([x2, y2])
    users = torch.row_stack([user_src, user_dst])
    kmeans = KMeans(n_clusters=M, random_state=0).fit(users.numpy())
    uav = torch.FloatTensor(kmeans.cluster_centers_)
    # index_src = torch.repeat_interleave(torch.arange(len(users), len(users)+len(uav)).unsqueeze(1), repeats=len(users)+len(uav), dim=1)
    index_src = torch.repeat_interleave(torch.arange(len(users), len(users)+len(uav)).unsqueeze(1), repeats=len(users), dim=1)
    index_src = torch.reshape(index_src, (1, -1)).squeeze()
    
    # index_dst = torch.arange(0, len(users)+len(uav)).repeat(len(uav))
    index_dst = torch.arange(0, len(users)).repeat(len(uav))
    features = torch.row_stack([user_src, user_dst, uav]).to(args.device)
    edge_index = torch.row_stack([index_src, index_dst]).to(args.device)

    return features, edge_index

# features, edge_index = data_create(100, 10)
# print(edge_index)
# print(edge_index.shape)
# exit()
