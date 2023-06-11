
import torch
from data_load import data_create
from model import RGNNLoss, UAV
import numpy as np
import matplotlib.pyplot as plt
from configs import args
import time
import pandas as pd


def model_train(model, epochs, lr, batch_size, train_num):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_func = RGNNLoss(path_find_model, N)

    features = []
    edge_index = []
    for _ in range(train_num):
        features_i, edge_index_i = data_create(N, M)    # 创建数据集
        features.append(features_i.unsqueeze(0))
        edge_index.append(edge_index_i.unsqueeze(0))
    print('data_create finished')
    features = torch.row_stack(features)
    edge_index = torch.row_stack(edge_index)

    loss_list = []
    for epoch in range(epochs):
        
        start = time.time()
        batch_loss_list = []
        # 推理、train过程
        for idx in range(train_num//batch_size):
            batch_features = features[idx*batch_size:(idx+1)*batch_size]
            batch_edge = edge_index[idx*batch_size:(idx+1)*batch_size]
            batch_loss = []
            for one_features, one_edge in zip(batch_features, batch_edge):
            
                outputs = model(one_features, one_edge)
                outputs = torch.cat([one_features[:2*N], outputs[2*N:]], dim=0)
                loss = loss_func(outputs)
                batch_loss.append(loss)
            batch_loss = torch.row_stack(batch_loss)
        #
            batch_loss = batch_loss.mean()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss_list.append(batch_loss)
            # loss_list.append(batch_loss.cpu().data.numpy())
        batch_loss_mean = sum(batch_loss_list)/len(batch_loss_list)
        print('epoch == {}, loss == {}'.format(epoch, batch_loss_mean))
        batch_loss_mean = batch_loss_mean.cpu().data.numpy()
        loss_list.append(batch_loss_mean)
        end = time.time()
        print('time used {} seconds'.format(end-start))
        if epoch % 100 == 99:
            torch.save(model, './model/location_model_{}_{}_{}_lstm.pt'.format(N, M, args.train_num))
    loss_list = np.array(loss_list)
    pd.DataFrame(loss_list).to_excel('./data/train_loss_{}_{}_{}_lstm.xlsx'.format(N, M, train_num))
    
    return model

if __name__=='__main__':
    
    # 用户对数和无人机个数
    N = 20
    M = 5
    features, edge_index = data_create(N, M)
    path_find_model = torch.load(args.path_model, map_location=args.device)     # 加载预训练LGNN模型
    uav_model = UAV(args.hidden_dim, args.alpha).to(args.device)
    location_model = model_train(uav_model, args.epochs, args.learning_rate, args.batch_size, args.train_num)
    torch.save(location_model, './model/lgnn_model_{}_{}_{}.pt'.format(N, M, args.train_num))
