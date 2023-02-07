import sys
sys.path.append('../')
import torch
import numpy as np
import pandas as pd
import sparsity_utils
import simulation_utils

"""
Runs simulation shown in Fig 1
"""


LOAD_F = False
VERSION = 'TEST'

L = 8
q = 2
K = 2
V = 'random'
N = q**L

if LOAD_F:  # Load existing fitness function
    _, _, phi, _ = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V)
    f = np.load(f"../results/complete_recovery_results/main_text/f_{VERSION}.npy")
    f_ = np.load(f"../results/complete_recovery_results/main_text/fprime_{VERSION}.npy")
    beta = phi.T@f
    beta_ = phi.T@f_
else:  # sample new fitness function
    f, beta, phi, seqs = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V)
    f-=np.mean(f)
    beta[0] = 0
    g = simulation_utils.exponential
    alpha = 10
    f_ = g(f, alpha=alpha) 
    beta_ = phi.T@f_

fnorm_ = (f_ - np.mean(f_)) / np.std(f_)
betanorm_ = phi.T@fnorm_


# fit network to all of the data
train_idx = list(range(N))
val_idx = list(range(N))
test_idx = list(range(N))
max_epochs = 4000
patience = 100
loss = 'bradley_terry'
bt_model, results = simulation_utils.fit_net(L, f_, phi, train_idx, val_idx, 
                                             test_idx, batch_size=256, loss=loss,
                                             max_epochs=max_epochs, patience=patience,
                                             lr=1e-3, verbose=True)

# Calculate fhat by making all predictions
X = torch.Tensor(phi[:, 1:L+1])
fhat = bt_model(X).detach().numpy().flatten()

np.save(f"../results/complete_recovery_results/main_text/f_{VERSION}.npy", f)
np.save(f"../results/complete_recovery_results/main_text/fprime_{VERSION}.npy", fnorm_)
np.save(f"../results/complete_recovery_results/main_text/fhat_{VERSION}.npy", fhat)