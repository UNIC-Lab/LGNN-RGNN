import  argparse

args = argparse.ArgumentParser()
args.add_argument('--batch_size', default=512)
args.add_argument('--train_num', default=4096)
args.add_argument('--device', default='cuda:0')
args.add_argument('--hidden_dim', default=128)
args.add_argument('--alpha', default=0.2)
args.add_argument('--learning_rate', default=1e-4)
args.add_argument('--epochs', default=500)
args.add_argument('--path_model', default='./model/rgnn_10.pt')

args = args.parse_args()
print(args)




