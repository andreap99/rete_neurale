import torch
import torch.nn as nn
from torch.nn import MSELoss, Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm




# ----- DATASET CLASS -----
class VectorialDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        v = self.X[idx]
        l = self.y[idx]
        if self.transform:
            v = self.transform(v)
        if self.target_transform:
            l = self.target_transform(l)
        return v,l



# ----- MODEL DEFINITION -----
class Net(Module):

    def __init__(self, D_in, H1=20, H2=None, H3=None):
        super().__init__()
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.fc1 = nn.Linear(D_in, H1)
        if H2 is None:
            self.fc2 = nn.Linear(H1, 1)
        else:
            self.fc2 = nn.Linear(H1, H2)
            if H3 is None:
                self.fc3 = nn.Linear(H2, 1)
            else:
                self.fc3 = nn.Linear(H2, H3)
                self.fc4 = nn.Linear(H3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):   # x is a minibatch of dim [dim. batch * dim. vector]
        #x = self.dropout(x)
        x = self.fc1(x)     # applies the linear combinations to each vector of the minibatch
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        if self.H2 is not None:
            print
            x = self.relu(x)
            x = self.fc3(x)
        if self.H3 is not None:
            x = self.relu(x)
            x = self.fc4(x)

        return x.squeeze()



# ----------------  TEST FUNCTION  -------------------
def test(net, X_test, y_test, verbose=True, criterion=MSELoss()):
    # from numpy array/pandas dataframe to torch dataset
    testset = VectorialDataset(X_test, y_test, transform=torch.Tensor)

    # create dataloader
    testloader = DataLoader(testset, batch_size=512, shuffle=False)

    net.train(False)    # set model to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use gpu if available

    outputs = np.array([])
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(testloader), disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs_batch = net(inputs.float())
            loss = criterion(outputs_batch, labels.float())

            outputs = np.concatenate([outputs, outputs_batch.detach().cpu().numpy()])
            running_loss += loss.item()
    avg_test_loss = running_loss / (i + 1)

    return outputs, avg_test_loss



#  ---------------  TRAINING FUNCTION  ---------------
def train(net, X_train, y_train, n_epochs=300, batch_size=256, verbose=False, X_test=None, y_test=None, lr=0.0001, lr_final=1e-7, patience=5):
    """Trains the model.
    Args:
        net: pytorch model to train
        X_train (numpy array or pandas dataframe): dataset used for training.
        y_train (numpy array): ground truth values.
        n_epochs (int): number of epochs to train.
        verbose (boolean):
        X_test:
    """

    # from numpy array/pandas dataframe to torch dataset
    trainset = VectorialDataset(X_train, y_train, transform=torch.Tensor)

    # create dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    n_minibatches_per_epoch = len(X_train) // batch_size    # length of the trainloader
    
    # define loss function
    criterion = nn.MSELoss()

    # choose optimizer algorithm
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # if optimizer is SGD, define lr scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, threshold_mode='rel')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use gpu if available
    net.to(device)  # move model to gpu if available

    # train model
    avg_train_loss_minibatch = np.full([n_epochs,n_minibatches_per_epoch],-1.0)    # avg training loss per minibatch/iteration (as a 2D array)
    avg_train_loss_epoch = np.full(n_epochs,-1.0)  # avg training loss per epoch (as a 1D array)
    if verbose:
        avg_test_loss_epoch = np.full(n_epochs,-1.0)  # avg test loss per epoch (as a 1D array)
    
    for epoch in range(n_epochs):

        if verbose:
            '''
            current_lr = []
            for param_group in optimizer.param_groups:
                current_lr.append(param_group['lr'])
            '''
            current_lr = optimizer.param_groups[0]['lr']
            print(f'--- Starting epoch {epoch+1}/{n_epochs}, LR = {current_lr} ---')
        
        net.train(True) # set model to training mode

        running_loss = 0.0  # sum of MSEs on the minibatches of the current epoch; needs to be divided by the n. of minibatches in an epoch in the end
        for iter, (inputs, labels) in tqdm(enumerate(trainloader), disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.float())   # from double to float; this is to speed up computations, at the expense of precision
            loss = criterion(outputs, labels.float())

            # backward + update weight
            loss.backward()
            optimizer.step()

            # save loss (for reporting purposes)
            running_loss += loss.item()
            avg_train_loss_minibatch[epoch,iter] = loss.item()

        avg_train_loss_current_epoch = avg_train_loss_minibatch[epoch].mean()

        # eventually, reduce lr if a plateau is reached
        scheduler.step(avg_train_loss_current_epoch)

        # eventually, halt the training process if lr goes below a certain threshold, e.g. 1e-7
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr <= lr_final:
            print(f'######## TRAINING STOPPED AT EPOCH {epoch+1} DUE TO LR BELOW {lr_final} ########')
            break

        if verbose:
            _, avg_test_loss_current_epoch = test(net, X_test, y_test, verbose=True, criterion=criterion)
            avg_test_loss_epoch[epoch] = avg_test_loss_current_epoch
            print(f"--- Epoch {epoch+1} ended. ---\nAvg. MSE for training set: {avg_train_loss_current_epoch}.\nMSE for test set: {avg_test_loss_current_epoch}")
    
    #avg_train_loss_epoch = avg_train_loss_minibatch.mean(axis=1)

    if verbose:
        return avg_train_loss_minibatch, avg_test_loss_epoch

    

def plot_loss_curve(train_loss, test_loss):
    '''
    Plot loss curve
    '''
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Loss (MSE) evolution over epochs')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    # plot train loss
    if len(train_loss.shape) == 2:
        n_epochs, n_minibatches_per_epoch = train_loss.shape
        n_iterations = n_epochs * n_minibatches_per_epoch
        # flatten the train_loss matrix
        train_loss = train_loss.flatten()
        ticks = np.linspace(n_epochs, 0, n_iterations, endpoint=False)[::-1]
        ax.plot(ticks, train_loss, color='blue', linewidth=2)
    else:
        ax.plot(train_loss, color='blue', linewidth=2)

    # plot test loss
    ax.plot(test_loss, color='red', marker='o', linewidth=2, markersize=12)
    fig.show()