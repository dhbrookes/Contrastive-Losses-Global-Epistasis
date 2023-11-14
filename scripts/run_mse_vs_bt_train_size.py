import sys
sys.path.append('../')
import sparsity_utils
import simulation_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Script that runs the simulations shown in Fig. 2c
"""

K = 1
VERSION = "K=%i" % K

# sample fitness function:
L = 8
q = 2
# K = 2
V = 'random'
N = q**L
f, beta, phi, seqs = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V) 

# normalize to 0 mean:
f-=np.mean(f)
beta[0] = 0 

# apply nonlinearity and renormalize:
g = simulation_utils.sigmoid
alpha = 20
sig = 0
f_ = g(f, alpha=alpha) 
beta_ = phi.T@f_
fnorm_ = (f_ - np.mean(f_)) / np.std(f_) + sig*np.random.normal(size=N)
betanorm_ = phi.T@fnorm_


ns = np.arange(25, 1025, 25)
replicates = 200

results = {
    'train_size': [],
    'loss': [],
    'test_spearman': [],
    'test_pearson': [],
    'test_mse': []
}


def add_to_results(train_size, loss, sub_results):
    results['loss'].append(loss)
    results['train_size'].append(train_size)
    results['test_spearman'].append("%.3f" % sub_results['test_spearman'])
    results['test_pearson'].append("%.3f" %sub_results['test_pearson'])
    results['test_mse'].append("%.3f" % sub_results['test_mse'])


mse_lr = 1e-3
bt_lr = 1e-3
patience = 20
batch_size = 64
max_epochs = 2500
verbose = True
sig = 0
DEVICE = 'cpu'

np.save(f"../results/train_size_results/train_size_f_{VERSION}.npy", f)
np.save(f"../results/train_size_results/train_size_fprime_{VERSION}.npy", fnorm_)

for i, n in enumerate(ns):
    num_train = int(n*0.8)
    num_val = int(n*0.2)
    for j in range(replicates):
        train_idx, _ = train_test_split(list(range(N)), train_size=num_train+num_val) 
        train_idx, val_idx = train_test_split(train_idx, train_size=num_train)
        test_idx = list(range(N)) # Test on everything 
        _, mse_results = simulation_utils.fit_net(L, 
                                                  fnorm_, 
                                                  phi, 
                                                  train_idx, 
                                                  val_idx, 
                                                  test_idx, 
                                                  batch_size=batch_size,
                                                  loss='mse', 
                                                  max_epochs=max_epochs,
                                                  patience=patience, 
                                                  lr=mse_lr, 
                                                  verbose=verbose,
                                                  device=DEVICE)
        add_to_results(n, 'mse', mse_results)
        
        _, bt_results = simulation_utils.fit_net(L, 
                                                 fnorm_,
                                                 phi,
                                                 train_idx, 
                                                 val_idx, 
                                                 test_idx, 
                                                 batch_size=batch_size, 
                                                 loss='bradley_terry', 
                                                 max_epochs=max_epochs,
                                                 patience=patience, 
                                                 lr=bt_lr, 
                                                 verbose=verbose,
                                                 device=DEVICE)
        add_to_results(n, 'bradley_terry', bt_results)
        df = pd.DataFrame(results)
        df.to_csv("../results/train_size_results/train_size_results_%s.csv" % VERSION)


