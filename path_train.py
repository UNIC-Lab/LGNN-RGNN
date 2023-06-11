import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from rgnn import RGNN

'''
    训练RGNN模型
'''

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default=10, help="number of nodes")
    parser.add_argument('--epoch', default=100, help="number of epochs")
    parser.add_argument('--batch_size', default=512, help='')
    parser.add_argument('--train_size', default=2500, help='')
    parser.add_argument('--val_size', default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--device', default='cuda')
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']    # learning rate
    B = int(args['batch_size'])    # batch_size
    B_val = int(args['val_size'])    # validation size
    steps = int(args['train_size'])    # training steps
    n_epoch = int(args['epoch'])    # epochs
    device = args['device']
    save_root ='./model/rgnn_'+str(size)+'.pt'
    
    print('=========================')
    print('prepare to train')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')
    
    
    model = RGNN(n_feature=2, n_hidden=128).to(device)
    # load model
    # model = torch.load(save_root).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 2500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                         lr_decay_step), gamma=lr_decay_rate)
    
    # validation data
    X_val = np.random.rand(B_val, size, 2)

    C = 0     # baseline
    R = 0     # reward

    # R_mean = []
    # R_std = []
    for epoch in range(n_epoch):
        for i in tqdm(range(steps)):
            optimizer.zero_grad()
        
            X = np.random.rand(B, size, 2)        
        
            X = torch.Tensor(X).cuda()
            
            mask = torch.zeros(B,size).cuda()
            mask[:, 0] = -np.inf
            max_dist = torch.zeros(B).cuda()
        
            R = 0
            logprobs = 0
            reward = 0
            
            Y = X.view(B,size,2)
            x = Y[:,0,:]
            h = None
            c = None
        
            for k in range(size):
                
                if k == 0:
                    mask[:, -1] = -np.inf
                    Y0 = x.clone()
                if k > 0:
                    mask[:, -1] = 0
                
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
        
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                
                dist = torch.norm(Y1-Y0, dim=1)
                # if dist > max_dist:
                #     max_dist
                max_dist[dist > max_dist] = dist[dist > max_dist]
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
                
                # R += reward
                    
                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY) 
                
                mask[[i for i in range(B)], idx.data] += -np.inf 
                mask[idx.data==size] = -np.inf
                
            # R += torch.norm(Y1-Y_ini, dim=1)
            
            
            # self-critic base line
            mask = torch.zeros(B,size).cuda()
            mask[:, 0] = -np.inf
            max_dist = torch.zeros(B).cuda()
            
            baseline = torch.zeros(B).cuda()
            
            Y = X.view(B,size,2)
            x = Y[:,0,:]
            h = None
            c = None
            
            for k in range(size):
                
                if k == 0:
                    Y0 = x.clone()
                    mask[:, -1] = -np.inf
                if k > 0:
                    mask[:, -1] = 0
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            
                idx = torch.argmax(output, dim=1)    # greedy baseline
            
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                
                dist = torch.norm(Y1-Y0, dim=1)
                # if dist > max_dist:
                #     max_dist
                baseline[dist > baseline] = dist[dist > baseline]
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
            
                # C += baseline
                
                mask[[i for i in range(B)], idx.data] += -np.inf
                mask[idx.data==size] = -np.inf
        
            # C += torch.norm(Y1-Y_ini, dim=1)
        
            gap = (max_dist-baseline).mean()
            loss = ((max_dist-baseline-gap)*logprobs).mean()
        
            loss.backward()
            # print(loss)
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            if i % 50 == 0:
                print("epoch:{}, batch:{}/{}, reward:{}"
                    .format(epoch, i, steps, max_dist.mean().item()))
                
                # greedy validation
                
                tour_len = 0

                X = X_val
                X = torch.Tensor(X).cuda()
                
                mask = torch.zeros(B_val,size).cuda()
                # mask = torch.zeros(B,size).cuda()
                mask[:, 0] = -np.inf
                max_dist = torch.zeros(B_val).cuda()

                logprobs = 0
                Idx = []
                reward = 0
                
                Y = X.view(B_val, size, 2)    # to the same batch size
                x = Y[:,0,:]
                h = None
                c = None
                
                for k in range(size):
                
                    if k == 0:
                        Y0 = x.clone()
                        mask[:, -1] = -np.inf
                    if k > 0:
                        mask[:, -1] = 0
                    output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                

                    idx = torch.argmax(output, dim=1)    # 
                
                    Y1 = Y[[i for i in range(B_val)], idx.data].clone()
                    
                    dist = torch.norm(Y1-Y0, dim=1)
                    # if dist > max_dist:
                    #     max_dist
                    max_dist[dist > max_dist] = dist[dist > max_dist]
                    Y0 = Y1.clone()
                    x = Y[[i for i in range(B_val)], idx.data].clone()
                
                    # C += baseline
                    
                    mask[[i for i in range(B_val)], idx.data] += -np.inf
                    mask[idx.data==size] = -np.inf
            
                # R += torch.norm(Y1-Y_ini, dim=1)
                tour_len += max_dist.mean().item()
                print('validation tour length:', tour_len)

        print('save model to: ', save_root)
        torch.save(model, save_root)
