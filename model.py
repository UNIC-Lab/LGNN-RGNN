
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch import Tensor
from configs import args
import numpy as np
import random
from tqdm import tqdm

'''
    UAV_conv: 一层LGNN
    PathSearch: 基于BF算法的链路搜索
    BFLoss: 基于BF算法的Loss计算
    RGNNLoss: 基于RGNN的Loss计算
    MLP: 基于MLP的位置优化
    UAV_Evolution: 基于遗传算法的位置优化
'''

class UAV_conv(MessagePassing):
    '''
    mlp进行消息传递和聚合
    lstm实现需要通信用户间的交互
    lstm实现uav间的交互
    '''
    def __init__(self, hidden_dim, alpha):
        super(UAV_conv, self).__init__(aggr='mean',flow='target_to_source')
        
        self.hidden_dim = hidden_dim
        # self.weight = nn.Parameter(torch.rand(size=(edge_num, 1)))  # 
        self.linear = nn.Linear(2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        

        # uav初始embedding
        self.uav_linear = nn.Linear(2, hidden_dim)

        # uav间通信过程
        self.uav_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True)
        # 需要通信的用户间的信息传递,编码
        self.users_lstm = nn.LSTM(input_size=2, hidden_size=int(hidden_dim/2), bidirectional=True)
        self.fc = nn.Sequential(
                    nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2),
                    nn.Sigmoid()
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # self.agg_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True)
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # 源节点和目标节点的索引
        self.source_index = list(range(edge_index[0][0], edge_index[0][-1]+1))
        self.users_num = int(edge_index[0][0]/2)

        # 用户间lstm实现编码
        user_pairs = torch.column_stack([x[:self.users_num, :].unsqueeze(1), x[self.users_num:2*self.users_num, :].unsqueeze(1)])
        user_pairs = user_pairs.transpose(0, 1)
        users, (_, _) = self.users_lstm(user_pairs)
        users = users.transpose(0, 1)
        users = torch.row_stack([users[:, 0, :], users[:, 1, :]])

        # uav间编码
        uavs = self.uav_linear(x[2*self.users_num:, :])

        x = torch.row_stack([users, uavs])

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j) -> torch.Tensor:
        # 消息传播机制

        # message_mlp计算用户到uav的信息传递
        
        self.outputs = self.message_mlp(torch.column_stack([x_i, x_j]))

        # 计算注意力
        self.att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(self.outputs)], dim=1))
        # self.att_weight = self.att(torch.cat([x_i, self.direction_f], dim=1))
        self.att_weight = self.att_weight.reshape(len(self.source_index), -1)
        self.att_weight = F.softmax(self.att_weight, dim=1).reshape(-1, 1)
        return self.outputs

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        # 消息聚合
        
        inputs = self.att_weight * inputs
        outputs = super().aggregate(inputs, index, ptr, dim_size)
        return outputs
    

    def update(self, aggr_out, x):

        x = self.update_mlp(torch.column_stack([aggr_out, x]))
        
        # 聚合之后uav间进行lstm信息传递
        uav_embeddings = x[2*self.users_num:, :].unsqueeze(1)
        uav_locations, (_ ,_) = self.uav_lstm(uav_embeddings)
        uav_locations = uav_locations.squeeze(1)
        # uav_locations = self.fc(uav_.squeeze(1))
        # x = F.sigmoid(self.linear2(x))
        # x[2*self.users_num:, :] = uav_locations
        x = torch.row_stack([x[:2*self.users_num, :], uav_locations])
        # print(x.shape)
        # exit()

        # 映射到01之间
        x = F.sigmoid(self.linear2(x))

        return x


class UAV(nn.Module):
    '''一层的卷积'''
    def __init__(self, hid_dim, alpha):
        super(UAV, self).__init__()
        self.uav_conv1 = UAV_conv(hid_dim, alpha)
        # self.uav_conv2 = UAV_conv(hid_dim, alpha)
    def forward(self, features, edge_index):
        # return self.uav_conv2(self.uav_conv1(features, edge_index), edge_index)
        return self.uav_conv1(features, edge_index)


class PathSearch:
    '''
        DFS搜索每对用户间的所有路径，返回snr
    '''
    def __init__(self, N, M) -> None:
        # 计算无人机和用户+无人机间的距离
        self.N = N
        self.M = M
    
    def search_maps(self, locations):
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]
        dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []
        for i in range(self.N):
            uav_map = dist_mat[:, 2*self.N:]
            uav_user = dist_mat[:, [i, i+self.N]]
            map_i = np.concatenate([uav_user, uav_map], axis=1)
            users_map = uav_user.T
            users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
            map_i = np.concatenate([users_map, map_i], axis=0)
            self.map_list.append(map_i)
        
        '''对所有图进行搜索'''
        best_path_snr = []     # 对于每对用户最佳路径所对应的snr的倒数
        for map_i in self.map_list:
            self.map_now = map_i
            self.set = np.zeros(self.M+2)
            self.top = -1
            self.stack = []
            self.path_max_edge = []     # 搜索到的路径的最长边的距离
            # self.path_list = []
            self.DFS(0)
            best_path_snr.append(min(self.path_max_edge))
            # best_path_list.append(self.path_list[np.argmin(self.path_max_edge)])    # 最佳路径列表
        return np.mean(best_path_snr)

    def search_maps_BF(self, locations):
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]
        dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []
        for i in range(self.N):
            uav_map = dist_mat[:, 2*self.N:]
            uav_user = dist_mat[:, [i, i+self.N]]
            map_i = np.concatenate([uav_user, uav_map], axis=1)
            users_map = uav_user.T
            users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
            map_i = np.concatenate([users_map, map_i], axis=0)
            self.map_list.append(map_i)
        
        # 构建边集
        edge_src = np.array([0 for _ in range(self.M)])
        edge_dst = np.arange(2, self.M+2)
        edge_src = np.concatenate([edge_src, np.arange(2, self.M+2), np.repeat(np.arange(2, self.M+2), repeats=self.M, axis=0)], axis=0)
        edge_dst = np.concatenate([edge_dst, np.full_like(edge_dst, 1)], axis=0)
        uav_dst = np.repeat(np.expand_dims(np.arange(2, self.M+2), axis=0), self.M, axis=0).reshape(-1, 1).squeeze(1)
        edge_dst = np.concatenate([edge_dst, uav_dst], axis=0)

        '''对所有图进行搜索'''
        best_path_list = []
        best_snr_list = []
        for map_i in self.map_list:
            # print(map_i)
            # 对每个节点创建max_edge表以及对应的链路
            # map_i为边权重
            max_edge = [np.inf for _ in range(self.M+2)]
            max_edge[0] = 0
            max_edge_path = [[] for _ in range(self.M+2)]
            max_edge_path[0] = [0]
            # print(max_edge_path)
            for _ in range(self.M+2):
                for i, j in zip(edge_src, edge_dst):
                    # 松弛每条边
                    if i != j and np.max([max_edge[i], map_i[i][j]]) < max_edge[j]:
                        max_edge[j] = np.max([max_edge[i], map_i[i][j]])
                        max_edge_path[j] = max_edge_path[i].copy()
                        max_edge_path[j].append(j)
                        
            best_snr_list.append(max_edge[1])
            best_path_list.append(np.array(max_edge_path[1]))
        return np.mean(best_snr_list), np.array(best_path_list)
    
    def search_maps_(self, locations):
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]
        self.dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []
        # for i in range(self.N):
        uav_map = self.dist_mat[:, 2*self.N:]
        uav_user = self.dist_mat[:, [0, self.N]]
        map_i = np.concatenate([uav_user, uav_map], axis=1)
        users_map = uav_user.T
        users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
        map_i = np.concatenate([users_map, map_i], axis=0)
        # self.map_list.append(map_i)
        
        '''对所有图进行搜索'''
        best_path_snr = []     # 对于每对用户最佳路径所对应的snr的倒数
        best_path_list = []
        # for idx, map_i in enumerate(self.map_list):
        for idx in range(self.N):
            self.map_now = map_i
            self.set = np.zeros(self.M+2)
            self.top = -1
            self.stack = []
            self.path_max_edge = []     # 搜索到的路径的最长边的距离
            # 第一张图搜索所有路径
            if idx == 0:
                self.edge_dist_mat = []     # N个列表，包含N条路径的边长
                self.path_list = [] # 所有的路径
                self.DFS(0)
            else:
                self.output_path_(idx)
            best_path_snr.append(min(self.path_max_edge))
            best_path_list.append(self.path_list[np.argmin(self.path_max_edge)])    # 最佳路径列表
            # print(best_path_list)
        return np.mean(best_path_snr), best_path_list

    def output_path(self):
        edge_dist = []
        for i in range(len(self.stack)-1):
            edge_dist.append(self.map_now[self.stack[i], self.stack[i+1]])
        self.path_list.append(np.array(self.stack.copy()))
        self.edge_dist_mat.append(edge_dist.copy())
        self.path_max_edge.append(max(edge_dist))


    def output_path_(self, idx):
        # print(self.edge_dist_mat)
        for k, p in enumerate(self.path_list):
            self.edge_dist_mat[k][0] = self.dist_mat[p[1]-2, idx]
            self.edge_dist_mat[k][-1] = self.dist_mat[p[-2]-2, idx + self.N]
            self.path_max_edge.append(max(self.edge_dist_mat[k]))

    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = np.expand_dims(np.sum(sq_a, axis=1), axis=1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = np.sum(sq_b, axis=1)[np.newaxis, :]  # n->[1, n]
        bt = b.T
        distance = np.sqrt(abs(sum_sq_a+sum_sq_b-2*np.matmul(a, bt)))
        return distance


class BFLoss(nn.Module):
    """
        使用Bellman-Ford计算最佳链路
        
    """
    def __init__(self, N, M):
        super(BFLoss, self).__init__()
        self.N = N
        self.M = M
        self.path_search = PathSearch(N, M)
    
    def forward(self, locations):
        locations_ = locations.cpu().data.numpy()
        _, path_list= self.path_search.search_maps_BF(locations_)
        
        best_snr = []
        for i in range(self.N):
            path_i = np.array(path_list[i])
            path_i[0] = i
            path_i[-1] = i+ self.N
            path_i[1:-1] += 2*(self.N-1)
            # path_i[1:-1] -= 2
            uav_path = path_i[1:-1]     # uav间的链路
            path_dist = [torch.norm(locations[path_i[0]]-locations[path_i[1]]), torch.norm(locations[path_i[-1]]-locations[path_i[-2]])]
            for j in range(len(uav_path)-1):
                path_dist.append(torch.norm(locations[uav_path[j]]-locations[uav_path[j+1]]))
            path_dist = torch.row_stack(path_dist)
            best_snr.append(torch.max(path_dist))
        best_snr = torch.row_stack(best_snr)
        return torch.mean(best_snr)

            

    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b, axis=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        distance = torch.sqrt(torch.abs(sum_sq_a+sum_sq_b-2*torch.mm(a, bt)))
        return distance



class RGNNLoss(nn.Module):
    def __init__(self, path_find_model, N):
        super(RGNNLoss, self).__init__()
        self.path_find_model = path_find_model
        self.N = N  # 用户对数
    
    def forward(self, outputs):
        users_src = outputs[:self.N].unsqueeze(1)
        users_dst = outputs[self.N:2*self.N].unsqueeze(1)
        uav_nodes = outputs[2*self.N:].repeat(self.N, 1, 1)
        
        uav_graph = torch.cat([users_src, uav_nodes, users_dst], dim=1)
        B = uav_graph.shape[0]
        size = uav_graph.shape[1]
        mask = torch.zeros(uav_graph.shape[0], uav_graph.shape[1]).to(args.device)
        mask[:, 0] = -np.inf
        x = uav_graph[:,0,:]
        max_dist = torch.zeros(uav_graph.shape[0]).to(args.device)
        h = None
        c = None
        for k in range(size):
            if k == 0:
                mask[[i for i in range(B)], -1] = -np.inf
                Y0 = x.clone()
            if k > 0:
                mask[[i for i in range(B)], -1] = 0
            
            output, h, c, _ = self.path_find_model(x=x, X_all=uav_graph, h=h, c=c, mask=mask)
            output = output.detach()
            
            idx = torch.argmax(output, dim=1)         # now the idx has B elements
            # idx_list.append(idx.clone().cpu().data.numpy()[0])
            Y1 = uav_graph[[i for i in range(B)], idx.data].clone()
            
            dist = torch.norm(Y1-Y0, dim=1)
            
            max_dist[dist > max_dist] = dist[dist > max_dist]
            
            Y0 = Y1.clone()
            x = uav_graph[[i for i in range(B)], idx.data].clone()
            
            mask[[i for i in range(B)], idx.data] += -np.inf 
            mask[idx.data==size] = -np.inf
        
        return max_dist.mean()



class MLP(nn.Module):
    '''利用mlp进行位置确定'''
    def __init__(self, input_dim, hidden_dim, output_num, users_num):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.users_num = users_num
        # 对节点进行编码
        self.linear = nn.Linear(input_dim, hidden_dim)
        # 两层的mlp
        self.location_mlp = nn.Sequential(
            nn.Linear(2*users_num*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_num*2),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, features):
        users = features[:2*self.users_num, :]
        users_cat = users.reshape(1, -1)
        uav_locations = self.location_mlp(users_cat)
        uav_locations = uav_locations.reshape(-1, 2)

        return uav_locations

class UAV_Evolution():
    def __init__(self, N, M, pop_num, total_epochs, features, Loss) -> None:
        self.flow_loss = Loss
        self.N = N
        self.M = M
        self.features = features
        self.pop_num = pop_num     # 初始种群规模
        self.retain_rate = 0.3      # 保存率
        self.mutate_rate = 0.2
        self.random_select_rate = 0.1
        # self.locations = locations
        self.total_epoch = total_epochs
        self.b = 2
    def evolution(self):
        populication = self.populication_create()
        max_distance = []
        for i in tqdm(range(self.total_epoch)):
            # print(populication.shape)
            parents, output_i = self.selection(populication)
            cs = self.cross_over(parents)
            cs = self.mutation(cs, i)
            populication = np.concatenate([parents, cs], axis=0)
            # output_i = self.adaptbility(populication)
            
            max_dist = np.min(output_i)       # 最好的一个个体对应的最大边
            
            max_distance.append(max_dist)
            # print('epoch == {}，max_distance of best_one == {}'.format(i, max_dist))
        
        return np.array(max_distance)
        

    def cross_over(self, parent):
        # 交叉, 单点交叉

        # 均匀交叉
        children = []
        get_child_num = self.pop_num-len(parent)
        while len(children) < get_child_num:
            i = random.randint(0, len(parent)-1)
            j = random.randint(0, len(parent)-1)
            male = parent[i]
            female = parent[j]
            select_p = np.random.rand(len(male))
            select_p[np.where(select_p < 0.5)] = 0
            select_p[np.where(select_p >= 0.5)] = 1
            child1 = select_p * male + (1-select_p) * female
            child2 = (1 - select_p) * male + select_p * female
            children.append(child1.reshape(1, len(child1)))
            children.append(child2.reshape(1, len(child2)))
            
        children = np.concatenate(children, axis=0)
        return children

    def populication_create(self):
        # 生成种群
        self.populication = np.random.rand(self.pop_num, self.M*2)
        self.users = torch.tensor(self.features[:2*self.N], device=args.device)
        
        return self.populication

    def mutation(self, cs, i):
        # 变异
        
        # 采用非一致性变异，每个位置都进行变异
        new_cs = cs.copy()
        for idx, c in enumerate(cs):
            if random.random() < self.mutate_rate:
                r = random.random()
                mut1 = (1-c)*np.random.rand(len(c))*(1-i/self.total_epoch)**self.b
                mut2 = c*np.random.rand(len(c))*(1-i/self.total_epoch)**self.b
                # print(mut1)
                if random.random() > 0.5:
                    c = c + mut1
                else:
                    c = c - mut2
                # print(c)
            new_cs[idx] = c
            # print(c)
        return new_cs
            

    def selection(self, populication):
        # 选择

        # 选择最佳的rate率的个体
        # 对种群从小到大进行排序
        adpt = self.adaptbility(populication)
        grabed = [[ad, one] for ad, one in zip(adpt, populication)]
        # print(grabed)
        # exit()
        sorted_grabed = sorted(grabed, key=lambda x: x[0])
        grabed = np.array([x[1] for x in sorted_grabed])
        index = int(len(populication)*self.retain_rate)

        live = grabed[:index]

        # 选择幸运个体
        for i in grabed[index:]:
            if random.random() < self.random_select_rate:
                live = np.concatenate([live, i.reshape(1, len(i))], axis=0)
        
        return live, adpt
    
    def adaptbility(self, populication):
        max_dist = []
        for p in torch.FloatTensor(populication).to(args.device):
            p = p.reshape(int(len(p)/2), 2)
            # p = torch.FloatTensor(p).to(device)
            users_uav = torch.cat([self.users, p], dim=0)
            max_dist.append(self.flow_loss(users_uav).cpu().data.numpy())
        return max_dist

