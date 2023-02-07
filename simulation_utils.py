import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm


class Net(nn.Module):
    """
    Simple fully connected neural network
    """
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.inp = nn.Linear(input_size, 100)
        self.act1 = nn.ReLU()
        self.hid = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.act1(h)
        h = self.hid(h)
        h = self.act2(h)
        y = self.out(h)
        return y
    

class EarlyStopper:
    """
    Implementation of early stopping scheme for torch models
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
def make_batches(x, y, batch_size):
    """
    Generates random batches of data
    """
    shuffle_idx = torch.randperm(x.shape[0])
    x_shuffle = x[shuffle_idx]
    y_shuffle = y[shuffle_idx]
    x_split = torch.split(x_shuffle, batch_size)
    y_split = torch.split(y_shuffle, batch_size)
    for i in range(len(x_split)):
        yield x_split[i], y_split[i]
        
        
def bradley_terry_loss(pred, targets):
    """
    Implements a torch-compatible version of Bradley-Terry loss
    """
    pred = pred[None, :]
    targets = targets[None, :]
    pred_diff = pred - pred.transpose(1, 0)
    contrastive_preds = F.logsigmoid(pred_diff)  
    inverse_preds = F.logsigmoid(-1*pred_diff)
    contrast_labels = torch.sign(targets-targets.transpose(1,0))*0.5 + 0.5
    
    losses = -contrast_labels*contrastive_preds - (1-contrast_labels)*inverse_preds
    self_mask = 1-torch.eye(losses.shape[0], device=losses.device)
    losses = losses*self_mask

    contrastive_pred_loss = torch.sum(losses*self_mask)/torch.sum(self_mask)
    return contrastive_pred_loss


def fit_net(L, f, phi, 
            train_idx, 
            val_idx,
            test_idx, 
            batch_size=32, 
            loss='mse', 
            lr=1e-3,
            max_epochs=1000, 
            patience=10,
            verbose=False,
            device='cpu'
            ):
    """
    Fit neural network with Adam and early stopping. Returns results
    in a dictionary containing test set Spearman, Pearson and MSE on
    test set.
    """
    model = Net(L).to(device)
    phi = torch.Tensor(phi)
    X = phi[:, 1:L+1].to(device)
    y = torch.Tensor(f).to(device)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx].to('cpu')
    X_test = X[test_idx]
    y_test = y[test_idx].to('cpu')
    
    if loss == 'mse':
        loss_func = nn.MSELoss()
    elif loss== 'bradley_terry':
        loss_func = bradley_terry_loss
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if verbose:
        prog = tqdm(range(max_epochs))
    else:
        prog = range(max_epochs)
    for t in prog:
        epoch_loss = 0 
        num_batches = 0
        for X_batch, y_batch in make_batches(X_train, y_train, batch_size=batch_size):
            model.zero_grad()
            output = model(X_batch).flatten()
            l = loss_func(output, y_batch)
                
            l.backward()
            opt.step()
            epoch_loss += l.detach().to('cpu').numpy()
            num_batches += 1
            
        with torch.no_grad(): 
            val_out = model(X_val).detach().flatten().to('cpu').numpy()
            if loss == 'mse':
                val_loss = -pearsonr(val_out, y_val)[0]
            elif loss == 'bradley_terry':
                val_loss = -spearmanr(val_out, y_val)[0]
            if early_stopper.early_stop(val_loss):             
                break
        des = "loss = %.3f, val_loss=%.3f" % (epoch_loss / num_batches, val_loss)
        if verbose:
            prog.set_description(des)
    with torch.no_grad():
        test_out = model(X_test).flatten().detach().to('cpu').numpy()
        results = {}
        results['test_spearman'] = spearmanr(test_out, y_test)[0]
        results['test_pearson'] = pearsonr(test_out, y_test)[0]
        results['test_mse'] = np.mean((test_out - y_test.numpy())**2)
    return model, results


def calc_entropy(x):
    """
    Calculates entropy of given vector
    """
    p = x**2 / np.linalg.norm(x, 2)**2
    H = -np.nansum(p*np.log(p))
    return H


### Nonlinearities:

def arcsinh(x, alpha=1):
    return np.arcsinh(alpha*x)

def sigmoid(x, alpha=1):
    return 1 / (1+np.exp(-alpha*x))

def left_censored(x, alpha=0):
    return np.maximum(0, x-alpha)

def exponential(x, alpha=1):
    return np.exp(alpha*x)

def cubic(x, alpha):
    return x**3 + alpha*x


