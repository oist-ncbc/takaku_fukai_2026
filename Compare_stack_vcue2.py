#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:57:33 2023

@author: munenori
v2_bigbatch
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:41:41 2022

@author: munenori
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:58:28 2021
for cue ver
@author: munenori
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:00:21 2021

@author: munenori
"""

#v4 without Re
#v4_1 revised forward function

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys


def reset_grad(hidden):
    HPC_hidden = [hidden[0][0].detach(), hidden[0][1].detach()]
    PFC_hidden = [hidden[1][0].detach(), hidden[1][1].detach()]
    Re_hidden = hidden[2].detach()
    return [PFC_hidden,HPC_hidden,Re_hidden]

def mkOwnDataSet(data_size, data_length=100, freq=60., noise=0.01):
   
    x = np.loadtxt("primal_long131test_r.csv",delimiter=',')
    y = np.loadtxt("primal_long131test_l.csv",delimiter=',')
    train_x = []
    train_y = []
    
    for offset in range(data_size):
        train_x.append([[x[i][0] + np.random.normal(loc=0.0, scale=noise),x[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(0,x.shape[0],round(x.shape[0]/data_length))])
        train_y.append([[y[i][0] + np.random.normal(loc=0.0, scale=noise),y[i][1]+ np.random.normal(loc=0.0, scale=noise)] for i in np.arange(0,y.shape[0],round(y.shape[0]/data_length))])
    return train_x, train_y

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

def make_traj(delay_length):
    freq=60
    noise=0.005
    data_length = 20
    
    # x = np.loadtxt("right1.csv",delimiter=',')
    x = make_Ttraj(1)
    # y = np.loadtxt("left1.csv",delimiter=',')
    y = make_Ttraj(2)
    z = np.loadtxt("stay1.csv",delimiter=',')
    data_x = np.array([[0,0,0]])
    data_y = np.array([[0,0,0]])
    cue = np.zeros((data_length,1))
    cue[0,0] += 1
    order  = np.array([1])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [2])
    # order = [1,0,0,0,2]
    # order = [2,0,0,1,0,0,0,2,0,0,1]
    for idx in order:
        if idx == 1:
            target_x = x
            target_x += np.random.normal(loc=0.0, scale=noise, size=target_x.shape)
            target_x = np.append(target_x,cue,axis=1)
            target_y = y
            target_y += np.random.normal(loc=0.0, scale=noise, size=target_y.shape)
            target_y = np.append(target_y,cue,axis=1)
        elif idx == 0:
            target = np.array([[z[i][0],z[i][1], 0] for i in np.random.choice(z.shape[0], 20)])
            target_x = target
            target_y = target
        else:
            target_x = y
            target_x += np.random.normal(loc=0.0, scale=noise, size=target_x.shape)
            target_x = np.append(target_x,cue,axis=1)
            target_y = x
            target_y += np.random.normal(loc=0.0, scale=noise, size=target_y.shape)
            target_y = np.append(target_y,cue,axis=1)

        data_x = np.append(data_x,target_x,axis=0)
        data_y = np.append(data_y,target_y,axis=0)
        
    return data_x[1:],data_y[1:]

def make_traj_randamize(delay_length):
    freq=60
    noise=0.005
    data_length = 20
    
    # x = np.loadtxt("right1.csv",delimiter=',')
    x = make_Ttraj(1)
    # y = np.loadtxt("left1.csv",delimiter=',')
    y = make_Ttraj(2)
    z = np.loadtxt("stay1.csv",delimiter=',')
    data_x = np.array([[0,0,0]])
    data_y = np.array([[0,0,0]])
    cue = np.zeros((data_length,1))
    cue[0,0] += 1
    order  = np.array([1])
    order  = np.append(order,np.zeros(delay_length))
    order = np.append(order, [2])
    # order = [1,0,0,0,2]
    # order = [2,0,0,1,0,0,0,2,0,0,1]
    n = 0
    for idx in order:
        n+=1
        if idx == 1:
            target_x = x
            target_x += np.random.normal(loc=0.0, scale=noise, size=target_x.shape)
            target_x = np.append(target_x,cue,axis=1)
            target_y = y
            target_y += np.random.normal(loc=0.0, scale=noise, size=target_y.shape)
            target_y = np.append(target_y,cue,axis=1)
        elif idx == 0:
            target = np.array([[z[i][0],z[i][1], 0] for i in np.random.choice(z.shape[0], 20)])
            target_x = target
            target_y = target
        else:
            target_x = y
            target_x += np.random.normal(loc=0.0, scale=noise, size=target_x.shape)
            target_x = np.append(target_x,cue,axis=1)
            target_y = x
            target_y += np.random.normal(loc=0.0, scale=noise, size=target_y.shape)
            target_y = np.append(target_y,cue,axis=1)

        if n == int(order.size):
            data_x = data_x[:-np.random.randint(1,(n-3)*data_length)]
            data_y = data_y[:-np.random.randint(1,(n-3)*data_length)]
        data_x = np.append(data_x,target_x,axis=0)
        data_y = np.append(data_y,target_y,axis=0)
        
    return data_x[1:],data_y[1:]


def mkOwnDataSet_auto(data_size, delay_length, freq=60., noise=0.01):
   
    x,y = make_traj_randamize(delay_length)
    train_x = []
    train_y = []
    
    for offset in range(data_size):
        train_x.append([[x[i][0] + np.random.normal(loc=0.0, scale=noise),x[i][1]+ np.random.normal(loc=0.0, scale=noise),x[i][2]] for i in np.arange(x.shape[0])])
        train_y.append([[y[i][0] + np.random.normal(loc=0.0, scale=noise),y[i][1]+ np.random.normal(loc=0.0, scale=noise),y[i][2]] for i in np.arange(y.shape[0])])
    return train_x, train_y


def mkOwnRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す
    """
    batch_x = []
    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - batch_size)
        batch_x.append(train_x[idx])
    return torch.tensor(batch_x).transpose(0,1)



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
    
class MyLSTM_cue2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MyLSTM_cue2, self).__init__()

        self.hidden_size = hidden_size
        self.LSTM1 = nn.LSTMCell(input_size-1, hidden_size)
        self.LSTM2 = nn.LSTMCell(hidden_size+1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, input, hiddens):
        input = input.float()
        hidden1 = self.LSTM1(input[:,:2], hiddens[0])
        # print(hidden1[0].shape,torch.reshape(input[:,2],(10,1)))
        hidden2_input = torch.cat((hidden1[0],torch.reshape(input[:,2],(10,1))),dim=1)
        hidden2 = self.LSTM2(hidden2_input, hiddens[1])
        output = self.linear(hidden2[0])
        return output, [hidden1,hidden2]

    def initHidden(self):
        hidden = [torch.zeros(self.batch_size, self.hidden_size), torch.zeros(self.batch_size, self.hidden_size)]
        return [hidden,hidden]
    
    def initHidden_rand(self):
        hidden = [torch.rand(self.batch_size, self.hidden_size)*0.2, torch.rand(self.batch_size, self.hidden_size)*0.2]
        # hidden = [torch.tensor(torch.rand(self.batch_size, self.hidden_size)*0.2,retain_graph=True), torch.tensor(torch.rand(self.batch_size, self.hidden_size)*0.2,retain_graph=True)]
        return [hidden,hidden]    
    
    
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




def calc_KL_exp2(data1):
    x = np.linspace(0,2,200)
    params1 = scipy.stats.expon.fit(np.abs(data1[data1>=0]))
    params2 = scipy.stats.expon.fit(np.abs(data1[data1<=0]))
    pdf1 = scipy.stats.expon.pdf(x,params1[0],params1[1])
    pdf2 = scipy.stats.expon.pdf(x,params2[0],params2[1])
    return scipy.stats.entropy(pdf1,pdf2)

    
def myloss(input, target, Re_weight, Rein_weight):
    loss = F.mse_loss(input, target)
    Re_weight_loss = calc_KL_exp2(Re_weight)
    Rein_weight_loss = calc_KL_exp2(Rein_weight)
    return loss + Re_weight_loss + Rein_weight_loss

def main():
    training_size = 8000
    test_size = 1000
    epochs_num = 100
    hidden_size = 30
    batch_size = 10
    data_length = 100
    inputsize = 3
    outputsize = 2
    sparse = sys.argv[1]
    # sparse = 1
    print(sparse)
    index_num = 10

    rnn = MyLSTM(inputsize, hidden_size, outputsize, batch_size)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(rnn.parameters(), lr=0.0005)
    optimizer = optim.Adam(rnn.parameters(), lr=0.0005, weight_decay=1e-8)
    # optimizer = optim.Adam(rnn.LSTM1.parameters(), lr=0.0001)
    # optimizer = optim.Adam(rnn.LSTM2.parameters(), lr=0.0001)
    # optimizer = optim.Adam(rnn.linear.parameters(), lr=0.0001)

    
    # model_path = 'model/ReModel_interRNNrand_AddRe_OUT5_long_s9_200_2_1.pth'
    # model_path = 'model/ReModel_L2_interRNNrand_AddRe_OUT5_cue18_s'+str(sparse)+'_100_1.pth'
    # model_path = 'model/compare30_cue_train/Compare30_cue_131_s'+str(sparse)+'_100_'+str(index_num)+'_epoch195.pth'
    model_path = 'model/compare30_cue_1until3_smallbatch5_decay/Compare30_cue_131_s'+str(sparse)+'_100_'+str(index_num)+'_epoch195.pth'
    rnn.load_state_dict(torch.load(model_path))
    
    # for n, p in rnn.named_parameters():
    #     if n == "LSTM2.weight_ih":
    #           p.requires_grad=False
    #     if n == "LSTM2.weight_hh":
    #           p.requires_grad=False



    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        # delay_length = np.random.randint(1, 3)
        delay_length = 3
        train_x,train_y = mkOwnDataSet_auto(training_size,delay_length)
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()
                       
            ### Learning Part ###       
            if np.random.randint(1, 3) == 1:
                data = mkOwnRandomBatch(train_x, train_x, batch_size).float()
            else:
                data = mkOwnRandomBatch(train_y, train_y, batch_size).float()
            
            # Define learning time
            ids = np.sort(np.random.randint(1,data.shape[0]-2,(30)))
            #ids = np.arange(0,data.shape[0]-1,70)
            print(np.sort(ids))
            
            #ids_PFC = ids[1::2]
            ids_PFC = ids
            
            
            hidden = rnn.initHidden_rand()
            outputs = []
            labels = []
            learn_count = 0
            k=0
            while k < data.shape[0]-1:
                #print(data.shape[0])
                output,hidden = rnn(data[k],hidden)
                label = data[k+1,:,:2]
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

            
            # # erase the weight between Re and other gate 
            # for n, p in rnn.named_parameters():
            #     if n == "PFC.weight_ih":
            #         p.data[hidden_size:,hidden_size:].sub_(p.data[hidden_size:,hidden_size:])
            #     if n == "HPC.weight_ih":
            #         p.data[hidden_size:,inputsize:].sub_(p.data[hidden_size:,inputsize:])
 
            # # erase the weight between Re and internal/other gate 
            # for n, p in rnn.named_parameters():
            #     if n == "PFC.weight_ih":
            #         p.data[hidden_size*2:,hidden_size:].sub_(p.data[hidden_size*2:,hidden_size:])
            #     if n == "HPC.weight_ih":
            #         p.data[hidden_size*2:,inputsize:].sub_(p.data[hidden_size*2:,inputsize:])
                    
            # # mix!!! 
            # for n, p in rnn.named_parameters():
            #     if n == "PFC.weight_ih":
            #         p.data[hidden_size:hidden_size*2,hidden_size:].sub_(p.data[hidden_size:hidden_size*2,hidden_size:])
            #         p.data[hidden_size*3:hidden_size*4,hidden_size:].sub_(p.data[hidden_size*3:hidden_size*4,hidden_size:])
            #     if n == "HPC.weight_ih":
            #         p.data[hidden_size:,inputsize:].sub_(p.data[hidden_size:,inputsize:])
    

            # running_loss += loss.data
            # training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.05)

            if i % 100 == 0:
                print("Now training... step %d ..." % (i))
                print(output,label)


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
                    output_test = torch.cat([output_test,torch.zeros(batch_size,1)],axis=1)
            for k in range(data.shape[0]*5):
                    output_test,hidden_test = rnn(output_test,hidden_test) 
                    traj_test.append(output_test.tolist())
                    if k == (delay_length+1)*20 or k == (delay_length*2+2)*20 or k == (delay_length*3+3)*20:
                        output_test = torch.cat([output_test,torch.ones(batch_size,1)],axis=1)
                    else:
                        output_test = torch.cat([output_test,torch.zeros(batch_size,1)],axis=1)
            traj_test = torch.tensor(traj_test)
            pltdata = torch.squeeze(data).numpy()
            traj_test = torch.squeeze(traj_test).numpy()
            correct_num = correct_traj(traj_test[:,0])
            if correct_num > -1 or epoch == 50:
                # model_path = 'model/compare30_cue_transfer_1'+str(delay_length)+'1/Compare30_cue_transfersame_1'+str(delay_length)+'1_s'+str(sparse)+'_100_'+str(index_num)+'_epoch'+str(0+epoch)+'.pth'
                model_path = 'model/compare30_cue_1until3_smallbatch5_decay/Compare30_cue_131_s'+str(sparse)+'_100_'+str(index_num)+'_epoch'+str(195+epoch)+'.pth'
                # model_path = 'model/R20_feed/FeedModel_L2_interRNNrand_OUT1_1'+str(delay_length)+'1_s'+str(sparse)+'_100_2_epoch'+str(epoch)+'.pth'
                # model_path = 'model/R20_uniPFCHPC/ReModel_L2_interRNNrand_OUT1_uniPFCHPC_1'+str(delay_length)+'1_s'+str(sparse)+'_100_'+str(index_num)+'_epoch'+str(epoch)+'.pth'
                torch.save(rnn.state_dict(), model_path)
                s = str(0+epoch) + "," + str(correct_num) + "," + model_path +"\n"
                with open("model/compare30_cue_1until3_smallbatch5_decay/correct_list.txt", mode="a") as f:
                    f.write(s)
        

    
    # model_path = 'model/ReModel_interRNNrand_AddRe_OUT5_long131test_s'+str(sparse)+'_100_1.pth'
    model_path = 'model/compare30_cue_1until3_smallbatch5_decay/Compare30_cue7_s'+str(sparse)+'_100_'+str(index_num)+'.pth'
    # model_path = 'model/ReModeltest_30reset3addlong_est_L2_interRNNrand_Reinh_AddRe_OUT5_1'+str(delay_length)+'1_s'+str(sparse)+'_100_2.pth'
    # model_path = 'model/ReModel_interRNNrand_AddRe_OUT5_long131test_s8_100_2_pluslong_'+str(sparse)+'.pth'
    # model_path = 'model/ReModel_interRNNrand_AddRe_OUT5_long_s9_200_2_1_'+str(sparse)+'.pth'
    # model_path = 'model/ReModel_interRNN_invYl_s5_200_'+str(sparse)+'_4000.pth'
    torch.save(rnn.state_dict(), model_path)
    print(model_path+" saving")
    
if __name__ == '__main__':
    main()
