import torch
from torch import nn, optim
import argparse

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets
import os
import numpy as np 
import math
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from itertools import chain

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda:0'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size,args) :
        super().__init__()
        self.input_size = input_size # input 特征的维度
        self.hidden_size = hidden_size # 隐藏层节点个数。
        self.num_layers = num_layers # 层数，默认为1
        self.output_size = output_size # 
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_seq):
        input_seq = input_seq.to(args.device)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, input_seq.size(0), self.hidden_size).to(args.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, input_seq.size(0), self.hidden_size).to(args.device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # 
        # pred = self.linear_2(output)  
        pred = self.linear(output)  
        pred = pred[:, -1, :] 
        return pred

# 创建数据集
def create_dataset(dataset, time_step, pre_len):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - pre_len - 1):
        # print(f'dataset:{dataset}\n')
        x = dataset[i, 0:time_step]
        dataX.append(x)
        y = dataset[i+pre_len:i+pre_len+1, 0]
        dataY.append(y)
        # print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def get_data(path):
    data_csv = pd.read_csv(path) # 
    # print(f"data_csv:{data_csv}")
    # exit(0)
    #数据预处理
    data_csv = data_csv.dropna() #去掉na数据
    dataset = data_csv.values      #字典(Dictionary) values()：返回字典中的所有值。
    dataset = dataset.astype('float64')   #astype(type):实现变量类型转换  
    return dataset

# 数据归一化预处理
def normalization(data,label):

    mm_x = MinMaxScaler() # 导入sklearn的预处理容器
    mm_y = MinMaxScaler()
    # data=data.values    # 将pd的系列格式转换为np的数组格式
    # label=label.values
    data = mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label = mm_y.fit_transform(label)
    return data, label, mm_y

# 数据分离
def split_data(x, y, split_ratio):

    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size
    
    train_X = x[:train_size]
    train_Y = y[:train_size]
    
    test_X = x[train_size:]
    test_Y = y[train_size:]
    # print(train_X.shape)
    # print(train_Y.shape)
    # exit()
    train_X = train_X.reshape(train_X.shape[0],1, train_X.shape[1]) 
    # train_Y = train_Y.reshape(train_Y.shape[0],1,1)
    test_X = test_X.reshape(test_X.shape[0],1,test_X.shape[1])
    # test_Y = test_Y.reshape(test_Y.shape[0],1,1)
    
    x_train=Variable(torch.Tensor(train_X))
    y_train=Variable(torch.Tensor(train_Y))
    y_test=Variable(torch.Tensor(test_Y))
    x_test=Variable(torch.Tensor(test_X))

    x_data=(torch.Tensor(np.array(x)))
    y_data=(torch.Tensor(np.array(y)))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x.shape,y.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test

# 数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_test,y_test)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=False) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

    return train_loader,test_loader,num_epochs

def train(model,train_loader,test_loader,num_epochs,loss_function,optimizer,args):
    # train
    # 初始化训练和测试loss的列表
    train_losses = []
    test_losses = []
    model.train()
    iter=0
    for epochs in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        for i,(x_train, y_train) in enumerate (train_loader):
            x_train, y_train= x_train.to(args.device),y_train.to(args.device)
            y_pred = model(x_train)
            loss = loss_function(y_pred ,y_train) # 计算损失
            optimizer.zero_grad()   # 将每次传播时的梯度累积清除
            loss.backward() # 反向传播
            optimizer.step()
            iter+=1
            # if iter % 100 == 0:
            #     print("iter: %d, loss: %1.5f" % (iter, loss.item()))
            train_loss += loss.item() * x_train.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 测试模式
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate (test_loader):
                x_test, y_test= x_test.to(args.device),y_test.to(args.device)
                outputs = model(x_test)
                loss = loss_function(outputs, y_test)
                test_loss += loss.item() * x_test.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epochs+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


    # 绘制图像
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./train_test_loss.png')
    plt.show()
    if not os.path.isdir(args.model_save_path):
        os.mkdir(args.model_save_path)
    torch.save(model.state_dict(),os.path.join(args.model_save_path, "best.pt"))

def test(model,test_loader,mm_y,args):
    y_pred = []
    y_true = []
    
    print('predicting...')
    for i,(x_test, y_test) in enumerate (test_loader):
        x_test, y_test= x_test.to(args.device),y_test.to(args.device)
        model.eval()
        # y_test = list(chain.from_iterable(y_test.data.tolist()))
        y_true.extend(y_test.cpu().numpy())
        with torch.no_grad():
            pred  = model(x_test)
            # pred = list(chain.from_iterable(pred.data.tolist()))
            y_pred.extend(pred.cpu().numpy())
            
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true = np.array(y_true)
    # print(y_true)
    # print(y_pred.shape)
    y_true = mm_y.inverse_transform(y_true)
    y_pred = mm_y.inverse_transform(y_pred)

    # evaluation
    print(f'y_true shape:{y_true.shape}, {y_true}\n')
    print(f'y_pred shape:{y_pred.shape}, {y_pred}\n')
    evaluation(y_true, y_pred)

    # plot 
    plot(y_true, y_pred)

    # # save data
    # save_data(y_true, y_pred)

# MAPE和SMAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def evaluation(y_test, y_predict):
    print("evaluation P")
    MAE_P = mean_absolute_error(y_test[:,0], y_predict[:,0])
    RMSE_P = math.sqrt(mean_squared_error(y_test[:,0], y_predict[:,0]))
    MAPE_P = mape(y_test[:,0], y_predict[:,0])
    SMAPE_P = smape(y_test[:,0], y_predict[:,0])
    
    print('RMSE_P: %.4f ' % (RMSE_P))
    print('MAE_P: %.4f ' % MAE_P)
    print('MAPE_P: %.4f ' % MAPE_P)
    print('SMAPE_P: %.4f ' % SMAPE_P)
    print("R2_P: %.4f" % r2_score(y_test[:,0], y_predict[:,0]))

    # print("\nevaluation D ")
    # MAE_D = mean_absolute_error(y_test[:,1], y_predict[:,1])
    # RMSE_D = math.sqrt(mean_squared_error(y_test[:,1], y_predict[:,1]))
    # MAPE_D = mape(y_test[:,1], y_predict[:,1])
    # SMAPE_D = smape(y_test[:,1], y_predict[:,1])

    # print('MAE_D: %.4f ' % MAE_D)
    # print('RMSE_D: %.4f ' % (RMSE_D))
    # print('MAPE_D: %.4f ' % MAPE_D)
    # print('SMAPE_D: %.4f ' % SMAPE_D)
    # print("R2_D: %.4f" % r2_score(y_test[:,1], y_predict[:,1]))

def plot(y_test, y_predict):
    # 指定默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号)

    plt.figure(figsize=(10, 10))
    plt.plot(y_test[:,0])
    plt.plot(y_predict[:,0])
    plt.title('P real vs pred ')
    plt.legend(['pred', 'real'], loc='lower right')
    plt.savefig('./pred_real_p.png')
    plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.plot(y_test[:,1])
    # plt.plot(y_predict[:,1])
    # plt.title('D real vs pred ')
    # # plt.ylabel('流量')
    # # plt.xlabel('时间')
    # plt.legend(['pred', 'real'], loc='lower right')
    # plt.savefig('./pred_real_d.png')
    # plt.show()

def save_data(y_test, y_predict):
    # 转换为Pandas DataFrame
    df = pd.DataFrame(y_predict, columns=['P', 'D'])
    # 保存到CSV文件，如果不需要索引可以设置index=False
    df.to_csv("data_predict.csv", index=False)

    df = pd.DataFrame(y_test, columns=['P', 'D'])
    # 保存到CSV文件，如果不需要索引可以设置index=False
    df.to_csv("data_test.csv", index=False)

import random
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LSTM model")
    parser.add_argument('--seed', type=int, default=402)
    parser.add_argument('--test_set', action='store_true', default=False)
    # data
    parser.add_argument("--data_file_path", type=str, default="./m2csv_result.csv")
    parser.add_argument("--model_save_path", type=str, default="./model_saved")
    parser.add_argument("--num_n_iters", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--time_step", type=int, default=3)
    parser.add_argument("--pre_len", type=int, default=2)
    parser.add_argument("--split_ratio", type=float, default=0.80)
    # model
    parser.add_argument("--input_size", type=int, default=101)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
     # training
    parser.add_argument("--lr", type=float, default=0.0001)
    
    return parser.parse_args()

def run(args):
    # create data
    dataset = get_data(args.data_file_path)

    # data,label = create_dataset(dataset,args.time_step,args.pre_len)
    data,label = dataset[:,:-2], dataset[:,-2:-1]
   
    data,label,mm_y = normalization(data,label)
    x_data,y_data,x_train,y_train,x_test,y_test = split_data(data,label,args.split_ratio)
    train_loader,test_loader,num_epochs = data_generator(x_train,y_train,x_test,y_test,args.num_n_iters,args.batch_size)
    
    model=LSTM(args.input_size,args.hidden_size,args.num_layers,args.output_size,args.batch_size,args).to(args.device)
    loss_function = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
    # optimizer = torch.optim.RMSprop(model.parameters(),lr=args.lr)
    print(model)

    
    train(model,train_loader,test_loader,num_epochs,loss_function,optimizer,args)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, "best.pt")))

    # test_all(model,x_data,y_data,mm_y,args)
    print("test\n")
    test(model,test_loader,mm_y,args)

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'args.n_gpu:{args.n_gpu}\n')
    set_seed(args)
    run(args)