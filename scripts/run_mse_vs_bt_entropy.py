import sys
sys.path.append("../")
import numpy as np
import seaborn as sns
import pandas as pd
import sparsity_utils
import simulation_utils
from sklearn.model_selection import train_test_split


"""
Script that runs the simulations shown in Fig. 2b
"""

K = 3
eps = 0.1
VERSION = f"K={K}_eps={eps}"
# parameters of simulation:
L = 10
q = 2
# K = 1
V = 'random'
N = q**L
num_train = 256
num_val = int((num_train / 0.8) * 0.2)
# replicates = 50
replicates=10

# nonlinearities
gs = {
    'exponential':simulation_utils.exponential,
    'sigmoid': simulation_utils.sigmoid,
    'cubic': simulation_utils.cubic
}

# settings of parameter in nonlinearity
gs_alphas = {
    'exponential':np.logspace(-1, np.log10(50), 20),
    'sigmoid': np.logspace(np.log10(0.5), np.log10(25), 20),
    'cubic': np.logspace(1, -3, 20)
}

results = {
    'nonlinearity': [],
    'alpha': [],
    'loss': [],
    'entropy': [],
    'test_spearman': [],
    'test_pearson': [],
    'test_mse': []
}


def add_to_results(nonlinearity, alpha, entropy, loss, sub_results):
    results['loss'].append(loss)
    results['nonlinearity'].append(nonlinearity)
    results['alpha'].append(alpha)
    results['entropy'].append(entropy)
    results['test_spearman'].append("%.3f" % sub_results['test_spearman'])
    results['test_pearson'].append("%.3f" %sub_results['test_pearson'])
    results['test_mse'].append("%.3f" % sub_results['test_mse'])


mse_lr = 1e-3
bt_lr = 1e-3
patience = 100
batch_size = 64
max_epochs = 2500
verbose = True
sig = 0

for g_name in gs.keys():
    g = gs[g_name]
    alphas = gs_alphas[g_name] 
    for alpha in alphas:
        for i in range(replicates):
            # sample and transform
            f, beta, phi, seqs = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V) 
            f_ = g(f, alpha=alpha)
            fnorm_ = (f_ - np.mean(f_)) / np.std(f_)
            betanorm_ = phi.T@fnorm_
            
            entropy = simulation_utils.calc_entropy(betanorm_)
            
            fnorm_ += np.random.randn(N)*eps
            
            train_idx, test_idx = train_test_split(list(range(N)), train_size=num_train+num_val) 
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
                                                      device='cuda')
            
            add_to_results(g_name, alpha, entropy, 'mse', mse_results)

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
                                                     verbose=verbose)
            add_to_results(g_name, alpha, entropy, 'bradley_terry', bt_results)
            df = pd.DataFrame(results)
            df.to_csv("../results/entropy_results/entropy_results_%s.csv" % VERSION)


