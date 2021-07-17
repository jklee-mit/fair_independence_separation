"""Soft-HGR for separation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from knnie import revised_mi, revised_multi_mi
import scipy.io as sio

from facl.independence.density_estimation.pytorch_kde import kde
from facl.independence.hgr import chi_2_cond, hgr_cond

import pandas as pd


def chi_squared_l1_kde(X, Y, Z):
    return torch.mean(chi_2_cond(X, Y, Z, kde))

def evaluate(model, x, y, z):
    Y = y
    Z = z
    X = x

    prediction = model(X).detach().flatten()
    loss = nn.MSELoss()(prediction, Y)
    hgr_infty = np.max(hgr_cond(prediction, Z, Y, kde))
    return loss.item(), hgr_infty

def max_corr_obj_cont(f,g,data_x,data_y):
    """Computes soft-HGR objective"""
    num_samps = data_x.shape[0]
    
    outputs_f = f(data_x)
    outputs_f -= outputs_f.mean(dim=0)
    cov_f = torch.mm(torch.t(outputs_f),outputs_f)/(num_samps-1)

    outputs_g = g(data_y)
    outputs_g -= outputs_g.mean(dim=0)
    cov_g = torch.mm(torch.t(outputs_g),outputs_g)/(num_samps-1)
    
    loss = torch.trace(torch.mm(torch.t(outputs_f),outputs_g)/(num_samps-1)) - 0.5*torch.trace(torch.mm(cov_f,cov_g))
    
    return loss

def get_std_devs(net,inputs):
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    stds = torch.sqrt(torch.diag(torch.mm(outputs.permute(1,0),outputs)))
    stds[stds==0] = 1

    return stds

def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):

    # create names
    names = []
    with open('datasets/communities/communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv('datasets/communities/communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]

def eval_model_mse(net, x_test, y_test):
    return np.sum((net(x_test).detach().numpy().flatten() - y_test.detach().numpy())**2)/x_test.detach().numpy().shape[0]

import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(121, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class SimpleNetComb(nn.Module):
    def __init__(self):
        super(SimpleNetComb, self).__init__()
        self.fc1 = nn.Linear(122, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class SimpleNetComb1(nn.Module):
    def __init__(self):
        super(SimpleNetComb1, self).__init__()
        self.fc1 = nn.Linear(122, 50)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

class SimpleNetComb2(nn.Module):
    def __init__(self):
        super(SimpleNetComb2, self).__init__()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return x

class ScalarNet(nn.Module):
    def __init__(self):
        super(ScalarNet, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ScalarNet2(nn.Module):
    def __init__(self):
        super(ScalarNet2, self).__init__()
        self.fc1 = nn.Linear(50, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class DoubleNet(nn.Module):
    def __init__(self):
        super(DoubleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

data1, data2, data3, data1_test, data2_test, data3_test = read_crimes()

data1 = torch.Tensor(data1)
data2 = torch.Tensor(data2)
data3 = torch.Tensor(data3)

data1_test = torch.Tensor(data1_test)
data2_test = torch.Tensor(data2_test)
data3_test = torch.Tensor(data3_test)

criteria = lambda w,x,y,z : -1 * max_corr_obj_cont(w,x,y,z)
criteria_chi = chi_squared_l1_kde
criteria_mse = nn.MSELoss()

print("Training networks with chi-squared penalty...")

num_lambdas = 6
lambdas = torch.linspace(0,1,num_lambdas)

all_res_single_reg_y = []
all_res_single_reg_d = []
all_res_single_reg_d_chi = []
all_res_single_reg_d_mi = []

all_res_single_reg_y_var = []
all_res_single_reg_d_var = []
all_res_single_reg_d_chi_var = []
all_res_single_reg_d_mi_var = []

data_comb_test = torch.cat((data2_test.view(-1,1),data3_test.view(-1,1)),1)
data1_test_comb = torch.cat((data1_test,data3_test.view(-1,1)),1)

dataset = data_utils.TensorDataset(data1, data2, data3)
dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

for j in range(num_lambdas):
    print(j)
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    for kk in range(30):
        net1 = SimpleNet()
        net2 = DoubleNet()
        net3 = ScalarNet()
        net4 = ScalarNet()
        optimizer1 = optim.SGD(net1.parameters(), lr=5e-4, momentum=0.9)
        optimizer2 = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9)
        optimizer3 = optim.SGD(net3.parameters(), lr=0.1, momentum=0.9)
        optimizer4 = optim.SGD(net4.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(60):  # loop over the dataset multiple times
            for iii, (data1,data2,data3) in enumerate(dataset_loader):
                data_comb = torch.cat((data2.view(-1,1),data3.view(-1,1)),1)
                
                # zero the parameter gradients
                optimizer1.zero_grad()
            
                # forward + backward + optimize
                loss = criteria_mse(net1(data1).flatten(), data2) + lambdas[j]*criteria_chi(net1(data1).flatten(), data3, data2)
                loss.backward()
                optimizer1.step()
        
                for sub_epoch in range(5):
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()
                    optimizer4.zero_grad()
                    loss = criteria(nn.Sequential(net1,net3), net2, data1, data_comb) - criteria(nn.Sequential(net1,net3), net4, data1, data3.view(-1,1))
                    loss.backward()
                    optimizer2.step() 
                    optimizer3.step() 
                    optimizer4.step() 
        
        try:
            mse_tmp, hgr_val = evaluate(net1, data1_test, data2_test, data3_test)
            assert(not np.isnan(hgr_val))
            assert(not np.isnan(criteria(net1,net2,data1_test,data_comb_test).detach().numpy()))
            assert(not np.isnan(revised_multi_mi(net1(data1_test).detach().numpy(),data2_test.detach().numpy().reshape(-1,1),data3_test.detach().numpy().reshape(-1,1)) - 
                         revised_mi(net1(data1_test).detach().numpy(),data3_test.detach().numpy().reshape(-1,1))))
            tmp1.append(eval_model_mse(net1,data1_test,data2_test))
            tmp2.append(criteria(net1,net2,data1_test,data_comb_test).detach().numpy())
            tmp3.append(hgr_val)
            tmp4.append(revised_multi_mi(net1(data1_test).detach().numpy(),data2_test.detach().numpy().reshape(-1,1),data3_test.detach().numpy().reshape(-1,1)) - 
                         revised_mi(net1(data1_test).detach().numpy(),data3_test.detach().numpy().reshape(-1,1)))
        except:
            if len(tmp1):
                tmp1.append(np.max(np.array(tmp1)))
                tmp2.append(np.max(np.array(tmp2)))
                tmp3.append(np.max(np.array(tmp3)))
                tmp4.append(np.max(np.array(tmp4)))
            continue
    with torch.no_grad():
        all_res_single_reg_y.append(np.mean(np.array(tmp1)))
        all_res_single_reg_d.append(np.mean(np.array(tmp2)))
        all_res_single_reg_d_chi.append(np.mean(np.array(tmp3)))
        all_res_single_reg_d_mi.append(np.mean(np.array(tmp4)))
        all_res_single_reg_y_var.append(np.var(np.array(tmp1)))
        all_res_single_reg_d_var.append(np.var(np.array(tmp2)))
        all_res_single_reg_d_chi_var.append(np.var(np.array(tmp3)))
        all_res_single_reg_d_mi_var.append(np.var(np.array(tmp4)))
        
print("Training networks with soft-HGR objective...")
all_res_double_reg_y = []
all_res_double_reg_d = []
all_res_double_reg_d_chi = []
all_res_double_reg_d_mi = []

all_res_double_reg_y_var = []
all_res_double_reg_d_var = []
all_res_double_reg_d_chi_var = []
all_res_double_reg_d_mi_var = []
for j in range(num_lambdas):
    print(j)
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    for kk in range(30):
        net1 = nn.Sequential(SimpleNetComb1(),SimpleNetComb2())
        net2 = DoubleNet()
        net3 = ScalarNet2()
        net4 = ScalarNet()
        optimizer1 = optim.SGD(net1.parameters(), lr=5e-4, momentum=0.9)
        optimizer2 = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9)
        optimizer3 = optim.SGD(net3.parameters(), lr=0.1, momentum=0.9)
        optimizer4 = optim.SGD(net4.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(60):  # loop over the dataset multiple times
            for iii, (data1,data2,data3) in enumerate(dataset_loader):
                data_comb = torch.cat((data2.view(-1,1),data3.view(-1,1)),1)
                data1 = torch.cat((data1,data3.view(-1,1)),1)
                
                # zero the parameter gradients
                optimizer1.zero_grad()
            
                # forward + backward + optimize
                loss = criteria_mse(net1(data1).flatten(), data2) - lambdas[j]*(criteria(nn.Sequential(net1[0],net3), net2, data1, data_comb)**2 - criteria(nn.Sequential(net1[0],net3), net4, data1, data3.view(-1,1))**2)
                loss.backward()
                optimizer1.step()
        
                for sub_epoch in range(5):
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()
                    optimizer4.zero_grad()
                    loss = criteria(nn.Sequential(net1[0],net3), net2, data1, data_comb)**2 - criteria(nn.Sequential(net1[0],net3), net4, data1, data3.view(-1,1))**2
                    loss.backward()
                    optimizer2.step() 
                    optimizer3.step() 
                    optimizer4.step() 
        
        try:
            mse_tmp, hgr_val = evaluate(net1, data1_test_comb, data2_test, data3_test)
            assert(not np.isnan(hgr_val))
            assert(not np.isnan(criteria(net1,net2,data1_test_comb,data_comb_test).detach().numpy()))
            assert(not np.isnan(revised_multi_mi(net1(data1_test_comb).detach().numpy(),data2_test.detach().numpy().reshape(-1,1),data3_test.detach().numpy().reshape(-1,1)) - 
                         revised_mi(net1(data1_test_comb).detach().numpy(),data3_test.detach().numpy().reshape(-1,1))))
            tmp1.append(eval_model_mse(net1,data1_test_comb,data2_test))
            tmp2.append(criteria(net1,net2,data1_test_comb,data_comb_test).detach().numpy())
            tmp3.append(hgr_val)
            tmp4.append(revised_multi_mi(net1(data1_test_comb).detach().numpy(),data2_test.detach().numpy().reshape(-1,1),data3_test.detach().numpy().reshape(-1,1)) - 
                         revised_mi(net1(data1_test_comb).detach().numpy(),data3_test.detach().numpy().reshape(-1,1)))
        except:
            if len(tmp1):
                tmp1.append(np.max(np.array(tmp1)))
                tmp2.append(np.max(np.array(tmp2)))
                tmp3.append(np.max(np.array(tmp3)))
                tmp4.append(np.max(np.array(tmp4)))
            continue
    with torch.no_grad():
        all_res_double_reg_y.append(np.mean(np.array(tmp1)))
        all_res_double_reg_d.append(np.mean(np.array(tmp2)))
        all_res_double_reg_d_chi.append(np.mean(np.array(tmp3)))
        all_res_double_reg_d_mi.append(np.mean(np.array(tmp4)))
        all_res_double_reg_y_var.append(np.var(np.array(tmp1)))
        all_res_double_reg_d_var.append(np.var(np.array(tmp2)))
        all_res_double_reg_d_chi_var.append(np.var(np.array(tmp3)))
        all_res_double_reg_d_mi_var.append(np.var(np.array(tmp4)))


plt.scatter(np.abs(all_res_double_reg_d_chi),all_res_double_reg_y)
plt.scatter(np.abs(all_res_single_reg_d_chi),all_res_single_reg_y)
plt.xlabel("discrmination (discrete-HGR)")
plt.ylabel("MSE")
plt.legend("Ours","Chi-Squared")
plt.show()

plt.scatter(np.abs(all_res_double_reg_d_mi),all_res_double_reg_y)
plt.scatter(np.abs(all_res_single_reg_d_mi),all_res_single_reg_y)
plt.xlabel("discrmination (Mutual Info)")
plt.ylabel("MSE")
plt.legend("Ours","Chi-Squared")
plt.show()


#sio.savemat("output.mat", {"all_res_double_reg_y":all_res_double_reg_y,
#                           "all_res_double_reg_d_chi":all_res_double_reg_d_chi,
#                           "all_res_double_reg_d_mi":all_res_double_reg_d_mi,
##                           "all_res_single_reg_d":all_res_single_reg_d,
#                           "all_res_single_reg_y":all_res_single_reg_y,
#                           "all_res_single_reg_d_chi":all_res_single_reg_d_chi,
#                           "all_res_single_reg_d_mi":all_res_single_reg_d_mi,
##                           "all_res_double_reg_d_var":all_res_double_reg_d_var,
#                           "all_res_double_reg_y_var":all_res_double_reg_y_var,
#                           "all_res_double_reg_d_chi_var":all_res_double_reg_d_chi_var,
#                           "all_res_double_reg_d_mi_var":all_res_double_reg_d_mi_var,
##                           "all_res_single_reg_d_var":all_res_single_reg_d_var,
#                           "all_res_single_reg_y_var":all_res_single_reg_y_var,
#                           "all_res_single_reg_d_chi_var":all_res_single_reg_d_chi_var,
#                           "all_res_single_reg_d_mi_var":all_res_single_reg_d_mi_var})
