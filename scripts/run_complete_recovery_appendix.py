import sys
sys.path.append('../')
import torch
import numpy as np
import pandas as pd
import sparsity_utils
import simulation_utils

"""
Runs simulation shown in Appendix B
"""

alphas = {'sigmoid': 10,
          'exponential': 10,
          'cubic': 0.1,
          'arcsinh': 20
         }

L = 8
q = 2
V = 'random'
N = q**L
train_idx = list(range(N))
val_idx = list(range(N))
test_idx = list(range(N))
max_epochs = 4000
patience = 100
loss = 'bradley_terry'

gs = [simulation_utils.exponential,
      simulation_utils.sigmoid, 
      simulation_utils.cubic,
      simulation_utils.arcsinh]

for K in [1, 2, 3]:
    for g in gs:
        f, beta, phi, seqs = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V) 
        f-=np.mean(f)
        beta[0] = 0

        alpha = alphas[g.__name__]
        f_ = g(f, alpha=alpha) 
        fnorm_ = (f_ - np.mean(f_)) / np.std(f_)
        
        
        bt_model, bt_spear = simulation_utils.fit_net(L, f_, phi, train_idx, val_idx, 
                                                      test_idx, batch_size=256, loss=loss,
                                                      max_epochs=max_epochs,
                                                      patience=patience,
                                                      lr=1e-3, verbose=True)
        
        X = torch.Tensor(phi[:, 1:L+1])
        bt_f = bt_model(X).detach().numpy().flatten()
        
        results_dict = {"f": f, "fprime": fnorm_, "fhat": bt_f}
        # np.save(f"../results/appendix/complete_recovery_results/appendix/f_{K}_{g.__name__}_results.npy", results_dict)