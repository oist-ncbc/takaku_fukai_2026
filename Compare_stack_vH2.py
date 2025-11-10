#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:18:13 2023

@author: munenori
vH2...Big batch model
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:48:53 2022

@author: munenori
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt
import glob


def mkDataSet(data_size, data_length=50, freq=60., noise=0.02):
    """
    params
      data_size : データセットサイズ
      data_length : 各データの時系列長
      freq : 周波数
      noise : ノイズの振幅
    returns
      train_x : トレーニングデータ（t=1,2,...,size-1の値)
      train_t : トレーニングデータのラベル（t=sizeの値）
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.cos(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise), math.sin(4 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.cos(2 * math.pi * (offset + data_length) / freq),math.sin(4 * math.pi * (offset + data_length) / freq)])
    print(len(train_x[0]))
    return train_x, train_t

def correct_traj(traj):
    Rightflag = False
    Leftflag = False
    Pointflag = False
    correct_number = 0
    for i in range(traj.shape[0]):
        if traj[i,0] < 0.3 and Pointflag == False:
            Leftflag = True
            Pointflag = True
            if Rightflag == True:
                correct_number += 1
                Rightflag = False
        if traj[i,0] > 0.3 and traj[i,0] < 0.5 and Pointflag == True:
            Pointflag = False
            
        if traj[i,0] > 0.7 and Pointflag == False:
            Rightflag = True
            Pointflag = True
            if Leftflag == True:
                correct_number += 1
                Leftflag = False
        if traj[i,0] < 0.7 and traj[i,0] > 0.5 and Pointflag == True:
            Pointflag = False

    return correct_number

def mkOwnDataSet(data_size, data_length=100, freq=60., noise=0.005):
    
    x = np.loadtxt("primal_long131test_r.csv",delimiter=',')
    y = np.loadtxt("primal_long131test_l.csv",delimiter=',')
    train_x = []
    train_y = []
    
    for offset in range(data_size):
        train_x.append([[x[i][0] + np.random.normal(loc=0.0, scale=noise),x[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(0,x.shape[0],round(x.shape[0]/data_length))])
        train_y.append([[y[i][0] + np.random.normal(loc=0.0, scale=noise),y[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(0,y.shape[0],round(y.shape[0]/data_length))])
    return train_x, train_y

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す
    """
    batch_x = []
    batch_t = []
    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - batch_size)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    return torch.tensor(batch_x).transpose(0,1), torch.tensor(batch_t)

def mkOwnRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す
    """
    batch_x = []
    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - batch_size)
        batch_x.append(train_x[idx])
    return torch.tensor(batch_x).transpose(0,1)

def make_Ttraj(direction):
    traj_x = np.array([])
    traj_y = np.array([])
    straight_x = np.ones(5)*0.5
    if direction == 1:
        branch1_x = np.linspace(0.5, 0.75, 5)
        branch2_x = np.linspace(0.75, 0.5, 5)
    elif direction == 2:
        branch1_x = np.linspace(0.5, 0.25, 5)
        branch2_x = np.linspace(0.25, 0.5, 5)        
    backstraight_x = np.ones(5)*0.5

    straight_y = np.linspace(0, 0.5, 5)
    branch1_y = np.linspace(0.5, 0.5, 5)
    branch2_y = np.linspace(0.5, 0.5, 5)
    backstraight_y = np.linspace(0.5 , 0, 5)
    
    traj_x = np.append(traj_x, [straight_x,branch1_x,branch2_x,backstraight_x])
    traj_y = np.append(traj_y, [straight_y,branch1_y,branch2_y,backstraight_y])
    traj = np.concatenate([[traj_x],[traj_y]],axis=0).T
    return traj


def make_Ytraj(direction):
    traj_x = np.array([])
    traj_y = np.array([])
    straight_x = np.ones(5)*0.5
    if direction == 1:
        branch1_x = np.linspace(0.5, 0.75, 5)
        branch2_x = np.linspace(0.75, 0.5, 5)
    elif direction == 2:
        branch1_x = np.linspace(0.5, 0.25, 5)
        branch2_x = np.linspace(0.25, 0.5, 5)        
    backstraight_x = np.ones(5)*0.5

    straight_y = np.linspace(0, 0.5, 5)
    branch1_y = np.linspace(0.5, 0.75, 5)
    branch2_y = np.linspace(0.75, 0.5, 5)
    backstraight_y = np.linspace(0.5 , 0, 5)
    
    traj_x = np.append(traj_x, [straight_x,branch1_x,branch2_x,backstraight_x])
    traj_y = np.append(traj_y, [straight_y,branch1_y,branch2_y,backstraight_y])
    traj = np.concatenate([[traj_x],[traj_y]],axis=0).T
    return traj

def make_Htraj(direction1, direction2):
    traj_x = np.array([])
    traj_y = np.array([])
    straight_x = np.ones(10)*0.5
    if direction1 == 1:
        branch1_x = np.linspace(0.5, 0.75, 5)
        branch2_x = np.linspace(0.75, 0.75, 5)
        branch3_x = np.linspace(0.75, 0.75, 5)
        branch4_x = np.linspace(0.75, 0.5, 5)
    elif direction1 == 2:
        branch1_x = np.linspace(0.5, 0.25, 5)
        branch2_x = np.linspace(0.25, 0.25, 5)   
        branch3_x = np.linspace(0.25, 0.25, 5)  
        branch4_x = np.linspace(0.25, 0.5, 5)       
    backstraight_x = np.ones(10)*0.5

    straight_y = np.linspace(0, 0.5, 10)
    branch1_y = np.linspace(0.5, 0.5, 5)
    if direction2 == 1:
        branch2_y = np.linspace(0.5, 0.25, 5)
        branch3_y = np.linspace(0.25, 0.5, 5)
    elif direction2 == 2:
        branch2_y = np.linspace(0.5, 0.75, 5)
        branch3_y = np.linspace(0.75, 0.5, 5)
    branch4_y = np.linspace(0.5, 0.5, 5)
    backstraight_y = np.linspace(0.5 , 0, 10)
    
    traj_x = np.append(traj_x, straight_x)
    traj_x = np.append(traj_x, [branch1_x,branch2_x,branch3_x,branch4_x])
    traj_x = np.append(traj_x, backstraight_x)
    traj_y = np.append(traj_y, straight_y)
    traj_y = np.append(traj_y, [branch1_y,branch2_y,branch3_y,branch4_y])
    traj_y = np.append(traj_y, backstraight_y)
    traj = np.concatenate([[traj_x],[traj_y]],axis=0).T
    return traj


def make_traj(delay_length):
    freq=60
    noise=0.005
    data_length = 40
    
    # x = np.loadtxt("right1.csv",delimiter=',')
    x_1 = make_Htraj(1,1)
    x_2 = make_Htraj(1,2)
    # y = np.loadtxt("left1.csv",delimiter=',')
    y_1 = make_Htraj(2,1)
    y_2 = make_Htraj(2,2)
    z = np.loadtxt("stay1.csv",delimiter=',')
    data_list = []
    ###  ver1  ###
    orders = []
    order  = np.array([1])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [4])
    orders.append(order) 
    order  = np.array([2])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [3])
    orders.append(order) 
    order  = np.array([3])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [1])
    orders.append(order) 
    order  = np.array([4])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [2])
    orders.append(order) 
    
    # ###  ver2  ###
    # orders = []
    # order  = np.array([1])
    # order  = np.append(order,np.zeros(delay_length))
    # order = np.append(order, [2])
    # orders.append(order) 
    # order  = np.array([2])
    # order  = np.append(order,np.zeros(delay_length))
    # order = np.append(order, [3])
    # orders.append(order) 
    # order  = np.array([3])
    # order  = np.append(order,np.zeros(delay_length))
    # order = np.append(order, [4])
    # orders.append(order) 
    # order  = np.array([4])
    # order  = np.append(order,np.zeros(delay_length))
    # order = np.append(order, [1])
    # orders.append(order) 
    
    # order = [1,0,0,0,2]
    # order = [2,0,0,1,0,0,0,2,0,0,1]
    for order in orders:
        data = np.array([[0,0]])
        for idx in order:
            if idx == 1:
                target = x_1
            elif idx == 2:
                target = x_2
            elif idx == 3:
                target = y_1
            elif idx == 4:
                target = y_2
            elif idx == 0:
                target = np.array([[z[i][0],z[i][1]] for i in np.random.choice(z.shape[0], data_length)])
            if idx != 0:
                target += np.random.normal(loc=0.0, scale=noise, size=target.shape)
    
            data = np.append(data,target,axis=0)
        data_list.append(data[1:])
    return data_list[0],data_list[1],data_list[2],data_list[3]

def mkOwnDataSet_auto(data_size, delay_length, freq=60., noise=0.01):
   
    x1,x2,y1,y2 = make_traj(delay_length)
    train_x1 = []
    train_x2 = []
    train_y1 = []
    train_y2 = []
    
    for offset in range(data_size):
        train_x1.append([[x1[i][0] + np.random.normal(loc=0.0, scale=noise),x1[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(x1.shape[0])])
        train_x2.append([[x2[i][0] + np.random.normal(loc=0.0, scale=noise),x2[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(x2.shape[0])])

        train_y1.append([[y1[i][0] + np.random.normal(loc=0.0, scale=noise),y1[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(y1.shape[0])])
        train_y2.append([[y2[i][0] + np.random.normal(loc=0.0, scale=noise),y2[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(y2.shape[0])])

    return train_x1, train_x2, train_y1, train_y2


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.LSTM1 = nn.LSTMCell(input_size, hidden_size)
        self.LSTM2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, input, hiddens):
        hidden1 = self.LSTM1(input, hiddens[0])
        hidden2 = self.LSTM2(hidden1[0], hiddens[1])
        output = self.linear(hidden2[0])
        return output, [hidden1,hidden2]

    def initHidden(self):
        hidden = [torch.zeros(self.batch_size, self.hidden_size), torch.zeros(self.batch_size, self.hidden_size)]
        return [hidden,hidden]
    
    def initHidden_rand(self):
        hidden = [torch.rand(self.batch_size, self.hidden_size)*0.01, torch.rand(self.batch_size, self.hidden_size)*0.01]
        return [hidden,hidden]    

class MyLSTM_single(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MyLSTM_single, self).__init__()

        self.hidden_size = hidden_size
        self.LSTM1 = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, input, hiddens):
        hidden1 = self.LSTM1(input, hiddens[0])
        output = self.linear(hidden1[0])
        return output, [hidden1]

    def initHidden(self):
        hidden = [torch.zeros(self.batch_size, self.hidden_size), torch.zeros(self.batch_size, self.hidden_size)]
        return [hidden]
    
    def initHidden_rand(self):
        hidden = [torch.rand(self.batch_size, self.hidden_size)*0.01, torch.rand(self.batch_size, self.hidden_size)*0.01]
        return [hidden]    


def choose_maxmodel(path,sparse,indexnum):
    model_list = glob.glob(path+'*s'+str(sparse)+'_100_'+str(indexnum)+'_*epoch*.pth')
    model_list = sorted(model_list)
    model_list = sorted(model_list,key=len,reverse=False)
    n = model_list[-1].split("_epoch")[-1].split(".pth")[0]
    return model_list[-1], int(n)

def main():
    training_size = 8000
    test_size = 1000
    epochs_num = 100
    hidden_size = 30
    batch_size = 20
    data_length = 100
    inputsize = 2
    outputsize = 2
    sparse = sys.argv[1]
    # sparse = 1
    delay_length = 2
    indexnum = 10
    past_epochnum = 0

    # train_x,train_y = mkOwnDataSet(training_size,data_length)
    train_x1,train_x2,train_y1,train_y2 = mkOwnDataSet_auto(training_size,delay_length)


    rnn = MyLSTM(inputsize, hidden_size, outputsize, batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.0002, weight_decay=1e-8)
    
    # model_path = 'model/compare_H/Compare_1'+str(delay_length)+'1H_s'+str(sparse)+'_100_'+str(indexnum)+'_epoch100.pth'
    # model_path = 'model/compare_cont/Compare_151_s1_100_4_epoch395.pth'
    # model_path = 'model/compare30_151/compare30_151_s6_100_2_epoch195.pth'
    # rnn.load_state_dict(torch.load(model_path))
    
    model_path,past_epochnum = choose_maxmodel("model/compare30_H_bigbatch/",sparse,indexnum)
    rnn.load_state_dict(torch.load(model_path))

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            ### Toy model ###
            # data, label = mkRandomBatch(train_x, train_t, batch_size)
            

            # hidden = rnn.initHidden()
            # for k in range(data.shape[0]):
            #     #print(data.shape[0])
            #     output,hidden = rnn(data[k],hidden)
            # loss = criterion(output, label)
            # loss.backward()
            # optimizer.step()
            
            
            
            
            ### Learning Part ###
            pattern = np.random.randint(1, 5)
            if  pattern == 1:
                data = mkOwnRandomBatch(train_x1, train_x1, batch_size)
            elif pattern == 2:
                data = mkOwnRandomBatch(train_x2, train_x2, batch_size)
            elif pattern == 3:
                data = mkOwnRandomBatch(train_y1, train_y1, batch_size)                
            else:
                data = mkOwnRandomBatch(train_y2, train_y2, batch_size)
            # Define learning time
            ids = np.sort(np.random.randint(1,data.shape[0]-2,(50)))
            #ids = np.arange(0,data.shape[0]-1,70)
            
            
            hidden = rnn.initHidden_rand()
            k=0
            while k < data.shape[0]-1:
                #print(data.shape[0])
                output,hidden = rnn(data[k],hidden)
                label = data[k+1]
                if np.any(ids==k):
                    #print("training:"+str(k))
                    optimizer.zero_grad()
                    loss = criterion(output, label)
                    # loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                    
                    k=0
                    ids = ids[1:]
                    # hidden = [(hidden[0][0].detach(),hidden[0][1].detach()),(hidden[1][0].detach(),hidden[1][1].detach())]
                    hidden = rnn.initHidden_rand()
                    continue
                k+=1

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.05)
            
            if i % 100 == 0:
                print("Now training... step %d ..." % (i))
                print(output,label)

        # #test
        # test_accuracy = 0.0
        # for i in range(int(test_size / batch_size)):
        #     offset = i * batch_size
        #     data, label = torch.tensor(test_x[offset:offset+batch_size]).transpose(0,1), torch.tensor(test_t[offset:offset+batch_size])
        #     hidden = rnn.initHidden()
        #     for k in range(data_length):
        #         #print(data[k].shape)
        #         output,hidden = rnn(data[k],hidden)

        #     test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.02)
        
        # training_accuracy /= (training_size * inputsize)
        # test_accuracy /= (test_size * inputsize)

        # print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
        #     epoch + 1, running_loss, training_accuracy, test_accuracy))
        
        
        ########    MY TEST!!!!!!!!!!!!      ########
        if epoch % 5 == 0:
            traj_test = []
            PFCstate = []
            HPCstate = []
            Restate = []
            hidden_test = rnn.initHidden_rand()
            data_limit = 1
            for k in range(data_limit):
                    #print(data[k].shape)
                    output_test,hidden_test = rnn(data[k],hidden_test)
                    traj_test.append(output_test.tolist())
            for k in range(data.shape[0]*5):
                    output_test,hidden_test = rnn(output_test,hidden_test) 
                    traj_test.append(output_test.tolist())
            traj_test = torch.tensor(traj_test)
            pltdata = torch.squeeze(data).numpy()
            traj_test = torch.squeeze(traj_test).numpy()
            correct_num = correct_traj(traj_test[:,0])
            if correct_num > -1 or epoch == 50:
                # model_path = 'model/compare_transfer/Compare_1'+str(delay_length)+'1H_s'+str(sparse)+'_100_'+str(indexnum)+'add_epoch'+str(epoch+100)+'.pth'
                # model_path = 'model/compare30_transfer3B/Compare_1'+str(delay_length)+'1H_transfers62_s'+str(sparse)+'_100_'+str(indexnum)+'_epoch'+str(epoch+0)+'.pth'
                model_path = 'model/compare30_H_bigbatch/Compare30_1'+str(delay_length)+'1H_s'+str(sparse)+'_100_'+str(indexnum)+'_epoch'+str(epoch+past_epochnum)+'.pth'
                torch.save(rnn.state_dict(), model_path)
                s = str(epoch+past_epochnum) + "," + str(correct_num) + "," + model_path +"\n"
                with open("model/compare30_H_bigbatch/correct_list.txt", mode="a") as f:
                    f.write(s)
                
                # if correct_num > 2:
                #     break
                
    # traj = []
    # hidden = rnn.initHidden()
    # for k in range(100):
    #         #print(data[k].shape)
    #         output,hidden = rnn(data[k],hidden)
    #         traj.append(output.tolist())
    # for k in range(data.shape[0]*3):
    #         #print(data[k].shape)
    #         #output = torch.cat([output,torch.zeros(batch_size,1)],dim=1)
    #         output,hidden = rnn(output,hidden) 
    #         traj.append(output.tolist())
    # traj = torch.tensor(traj)
    # pltdata = torch.squeeze(data).numpy()
    # traj = torch.squeeze(traj).numpy()
    # fig = plt.figure()
    # print(pltdata.shape)
    # plt.plot(pltdata[:,0,0],pltdata[:,0,1],"--")
    # plt.plot(traj[:,0,0],traj[:,0,1])
    # plt.show()
    
    # model_path = 'compstack_model_long131test_10.pth'
    # model_path = 'model/compare_transfer/compstack_model_1'+str(delay_length)+'1H_s'+str(sparse)+'_100_'+str(indexnum)+'.pth'
    # model_path = 'model/compare_H/compstack_model_1'+str(delay_length)+'1Y_s10_100_10_'+str(sparse)+'.pth'
    # torch.save(rnn.state_dict(), model_path)
    
if __name__ == '__main__':
    main()
