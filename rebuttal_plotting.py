# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: dyno
#     language: python
#     name: dyno
# ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sparsity_utils
import simulation_utils
from sklearn.preprocessing import QuantileTransformer
plt.rcParams['figure.dpi']=300
plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# # Additional Fig 2 sims

# ## Entropy

# +
Ks = [1, 3]
L = 10
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, K in enumerate(Ks):
    VERSION = f"K={K}"
    ax = axes[i]
    df = pd.read_csv("results/entropy_results/entropy_results_%s.csv" % VERSION, index_col=0)
    bt_color = colors[0]
    mse_color = colors[2]
    s = 10
    alpha=0.75
    markers = ['o', '^', 'X']
    for i, g in enumerate(df['nonlinearity'].unique()):
        lbl1=None
        lbl2 = None
        bt_data = df.query(f"nonlinearity=='{g}' and loss=='bradley_terry'")
        mse_data = df.query(f"nonlinearity=='{g}' and loss=='mse'")

        ax.scatter(mse_data['entropy'], mse_data['test_spearman'], 
                   c=mse_color, s=s, alpha=alpha, marker=markers[i],
                  edgecolor='none', label=lbl1)
        ax.scatter(bt_data['entropy'], bt_data['test_spearman'], 
                   c=bt_color, s=s, alpha=alpha, marker=markers[i],
                  edgecolor='none', label=lbl2)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Spearman correlation")
    ax.set_xlabel("$H(\\beta_y)$")
    ax.set_xlim([1.1, L*np.log(2)*1.01])
    ax.set_ylim([0, 1.01])
    # ax.legend(frameon=False)

    legend_elements = [Patch(facecolor=bt_color, label='Bradley-Terry'),
                       Patch(facecolor=mse_color, label='MSE'),
                       Line2D([0], [0], marker=markers[0], color='w', label='Exponential',
                              markerfacecolor='k', markersize=7),
                       Line2D([0], [0], marker=markers[1], color='w', label='Sigmoid',
                              markerfacecolor='k', markersize=7),
                       Line2D([0], [0], marker=markers[2], color='w', label='Cubic',
                              markerfacecolor='k', markersize=7),

                      ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8)

axes[0].text(-0.25, 1, 'a',
     horizontalalignment='center',
     verticalalignment='center',
     transform = axes[0].transAxes,
     fontsize=17)
axes[1].text(-0.25, 1, 'b',
     horizontalalignment='center',
     verticalalignment='center',
     transform = axes[1].transAxes,
     fontsize=17)


plt.tight_layout()
plt.show()
# -

# # Train size

# +
Ks = [1, 2, 3]
alphas = [10, 20]

fig, axes = plt.subplots(2, 3, figsize=(8.5, 6.5))

bt_color = colors[0]
mse_color = colors[2]

pearson_marker = '^'
pearson_ls = ':'

spearman_marker = 'o'
spearman_ls = '-'
lw=0.75
ms = 3

for i, K in enumerate(Ks):
    for j, alpha in enumerate(alphas):
        ax = axes[j, i]
        if K == 2 and alpha == 10:
            ax.set_axis_off()
            continue
        VERSION = f"K={K}_alpha={alpha}"
        ax.set_title(f"$K={K}, a={alpha}$")
        
        df = pd.read_csv("results/train_size_results/train_size_results_%s.csv" % VERSION, index_col=0)

        bt_results = df.query("loss=='bradley_terry'").groupby("train_size").agg({"test_pearson":['mean', 'std'],
                                                                         "test_spearman": ['mean', 'std']})
        mse_results = df.query("loss=='mse'").groupby("train_size").agg({"test_pearson":['mean', 'std'],
                                                                         "test_spearman": ['mean', 'std']})

        mse_pearsons = mse_results['test_pearson']
        mse_spearmans = mse_results['test_spearman']
        bt_pearsons = bt_results['test_pearson']
        bt_spearmans = bt_results['test_spearman']


        # spearmans
        ax.plot(mse_spearmans['mean'].index, 
                              mse_spearmans['mean'], 
                              marker=spearman_marker,
                              ls=spearman_ls,
                              linewidth=lw, 
                              c=mse_color, 
                              markersize=ms,
                              label='MSE')

        ax.plot(bt_spearmans['mean'].index, 
                              bt_spearmans['mean'], 
                              marker=spearman_marker,
                              ls=spearman_ls,
                              linewidth=lw, 
                              c=bt_color, 
                              markersize=ms,
                              label='Bradley-Terry')

        # pearsons
        ax.plot(mse_pearsons['mean'].index, 
                              mse_pearsons['mean'], 
                              marker=pearson_marker,
                              ls=pearson_ls,
                              linewidth=lw, 
                              c=mse_color, 
                              markersize=ms,
                              label='MSE')

        ax.plot(bt_pearsons['mean'].index, 
                              bt_pearsons['mean'], 
                              marker=pearson_marker,
                              ls=pearson_ls,
                              linewidth=lw, 
                              c=bt_color, 
                              markersize=ms,
                              label='Bradley-Terry')

        ax.set_xlabel("Training set size")
        ax.set_ylabel("Correlation")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0 and j == 0:
            legend_elements = [Patch(facecolor=bt_color, label='Bradley-Terry'),
                               Patch(facecolor=mse_color, label='MSE'),
                               Line2D([0], [0], marker=spearman_marker, color='k', label='Spearman', ls=spearman_ls,
                                      markerfacecolor='k', markersize=5),
                               Line2D([0], [0], marker=pearson_marker, color='k', label='Pearson', ls=pearson_ls,
                                      markerfacecolor='k', markersize=5),

                              ]
            ax.legend(handles=legend_elements, frameon=False, fontsize=8)
        ax.set_xticks([0, 250, 500, 750, 1000])
    
plt.tight_layout()

# -

# # Fig. 1 baseline

# +
LOAD_F = True
VERSION = 'SUBMIT'

L = 8
q = 2
K = 2
V = 'random'
N = q**L

if LOAD_F:  # Load existing fitness function
    _, _, phi, _ = sparsity_utils.sample_gnk_fitness_function(L, q, K=K, V=V)
    
    f = np.load(f"/home/jupyter/contrastive_loss_paper/results/complete_recovery_results/main_text/f_{VERSION}.npy")
    f_ = np.load(f"/home/jupyter/contrastive_loss_paper/results/complete_recovery_results/main_text/fprime_{VERSION}.npy")
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

qt = QuantileTransformer(output_distribution='uniform', n_quantiles=10)
fhat_uni = qt.fit_transform(f_.reshape(-1, 1)).flatten()
betahat_uni = phi.T@fhat_uni

qt = QuantileTransformer(output_distribution='normal', n_quantiles=10)
fhat_normal = qt.fit_transform(f_.reshape(-1, 1)).flatten()
betahat_normal = phi.T@fhat_normal


# +
fig = plt.figure(constrained_layout=True, figsize=(6, 3))
gs = fig.add_gridspec(2, 8)

ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1, :3])
ax4 = fig.add_subplot(gs[1, 3:])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax1.scatter(f, fhat_uni, s=3,  c='k', alpha=0.5)
r2 = pearsonr(f, fhat_uni)[0]**2
ax1.text(0.3, 0.9, f'$R^2={r2:0.3f}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax1.transAxes,
    fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel("$f$")
ax1.set_ylabel("$f_q$", rotation=0)
ax1.set_box_aspect(4/5)
ax1.set_xticks([])

sparsity_utils.make_barplot(L, [beta, betahat_uni], labels=["$\\beta$", "$\\beta_q$"], 
                            up_to=3, ax=ax2, colors=[colors[1], colors[3]])
ax2.set_ylabel("Squared magnitude",fontsize=8)
ax2.set_xlabel(None)
ax2.set_ylim([0, 0.35])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(0.03, 0.9, 'I',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax2.transAxes,
     fontname='Helvetica',
     family='serif'
     )
ax2.text(0.14, 0.9, 'II',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax2.transAxes,
     fontname='Helvetica',
     family='serif'
     )
ax2.text(0.45, 0.9, 'III',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax2.transAxes,
     fontname='Helvetica',
     family='serif'
     )

ax2.set_xticks([])

#########

ax3.scatter(f, fhat_normal, s=3, c='k', alpha=0.5)
r2 = pearsonr(f, fhat_normal)[0]**2
ax3.text(0.3, 0.9, f'$R^2={r2:0.3f}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax3.transAxes,
        fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel("$f_q$", rotation=0)
ax3.set_xlabel("$f$")
ax3.set_box_aspect(4/5)

sparsity_utils.make_barplot(L, [beta, betahat_normal], labels=["$\\beta$", "$\\beta_q$"], up_to=3, ax=ax4, 
             colors=[colors[1], colors[3]])
ax4.set_ylim([0, 0.35])
ax4.set_ylabel("Squared magnitude",fontsize=8)
ax4.set_xlabel("Coefficient index", fontsize=8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax4.text(0.03, 0.9, 'I',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax4.transAxes,
     fontname='Helvetica',
     family='serif'
     )
ax4.text(0.14, 0.9, 'II',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax4.transAxes,
     fontname='Helvetica',
     family='serif'
     )
ax4.text(0.45, 0.9, 'III',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax4.transAxes,
     fontname='Helvetica',
     family='serif'
     )

ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax4.tick_params(axis='both', which='major', labelsize=8)
# ax5.axis('off')
# ax5.text(-0.35, 1, 'a',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax5.transAxes,
#      fontsize=16)
ax1.text(-0.35, 1, 'a',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax1.transAxes,
     fontsize=16)
ax3.text(-0.35, 1, 'b',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax3.transAxes,
     fontsize=16)

plt.subplots_adjust(wspace=1.25, hspace=0.15)
# plt.tight_layout()
plt.show()

# +
f = np.load(f"/home/jupyter/contrastive_loss_paper/results/train_size_results/train_size_f_K={K}_alpha={alpha}_eps={eps}.npy")
# y_ = np.load(f"/home/jupyter/contrastive_loss_paper/results/train_size_results/train_size_fprime_K={K}_alpha={alpha}_eps={eps}.npy")
f -= np.mean(f)
f_ = np.exp(10*f)

fnorm_ = (f_ - np.mean(f_)) / np.std(f_)
sig=0.1
spears = []
pearsons = []
for i in range(10000):
    y = fnorm_ + sig*np.random.randn(len(fnorm_))
    spear = spearmanr(fnorm_, y)[0]
    spears.append(spear)

        
print(np.mean(spears), np.max(spears))
# print(np.mean(pearsons), np.max(pearsons))

max_spear = np.max(spears)
# -

# # Noisy sim

# +
K = 2
alpha = 10
eps_s = [0.025, 0.1]
maxes = [0.587, 0.412]

fig, axes = plt.subplots(2, 2, figsize=(6,6))

bt_color = colors[0]
mse_color = colors[2]

pearson_marker = '^'
pearson_ls = ':'

spearman_marker = 'o'
spearman_ls = '-'
lw=0.75
ms = 3

for i, eps in enumerate(eps_s):
    # Load functions
    ax = axes[i, 0]
    f = np.load(f"/home/jupyter/contrastive_loss_paper/results/train_size_results/train_size_f_K={K}_alpha={alpha}_eps={eps}.npy")
    y = np.load(f"/home/jupyter/contrastive_loss_paper/results/train_size_results/train_size_fprime_K={K}_alpha={alpha}_eps={eps}.npy")
    rank_f = np.argsort(f).argsort()
    rank_y = np.argsort(y).argsort()

    ax.scatter(rank_f, rank_y, s=3,  c='k', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Rank in $f$")
    ax.set_ylabel("Rank in $y$")


    # load train size

    ax = axes[i, 1]
    VERSION = f"K={K}_alpha={alpha}_eps={eps}"

    df = pd.read_csv("results/train_size_results/train_size_results_%s.csv" % VERSION, index_col=0)

    bt_results = df.query("loss=='bradley_terry'").groupby("train_size").agg({"test_pearson":['mean', 'std'],
                                                                     "test_spearman": ['mean', 'std']})
    print(bt_results)
    mse_results = df.query("loss=='mse'").groupby("train_size").agg({"test_pearson":['mean', 'std'],
                                                                     "test_spearman": ['mean', 'std']})
    print(mse_results)

    mse_pearsons = mse_results['test_pearson']
    mse_spearmans = mse_results['test_spearman']
    bt_pearsons = bt_results['test_pearson']
    bt_spearmans = bt_results['test_spearman']
    ax.axhline(maxes[i], c='k', ls='--', label='Max Spearman')

    # spearmans
    ax.plot(mse_spearmans['mean'].index, 
                          mse_spearmans['mean'], 
                          marker=spearman_marker,
                          ls=spearman_ls,
                          linewidth=lw, 
                          c=mse_color, 
                          markersize=ms,
                          label='MSE')

    ax.plot(bt_spearmans['mean'].index, 
                          bt_spearmans['mean'], 
                          marker=spearman_marker,
                          ls=spearman_ls,
                          linewidth=lw, 
                          c=bt_color, 
                          markersize=ms,
                          label='Bradley-Terry')

    # pearsons
    ax.plot(mse_pearsons['mean'].index, 
                          mse_pearsons['mean'], 
                          marker=pearson_marker,
                          ls=pearson_ls,
                          linewidth=lw, 
                          c=mse_color, 
                          markersize=ms,
                          label='MSE')

    ax.plot(bt_pearsons['mean'].index, 
                          bt_pearsons['mean'], 
                          marker=pearson_marker,
                          ls=pearson_ls,
                          linewidth=lw, 
                          c=bt_color, 
                          markersize=ms,
                          label='Bradley-Terry')

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Correlation")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [Patch(facecolor=bt_color, label='Bradley-Terry'),
                       Patch(facecolor=mse_color, label='MSE'),
                       Line2D([0], [0], marker=spearman_marker, color='k', label='Spearman', ls=spearman_ls,
                              markerfacecolor='k', markersize=5),
                       Line2D([0], [0], marker=pearson_marker, color='k', label='Pearson', ls=pearson_ls,
                              markerfacecolor='k', markersize=5),
                       Line2D([0], [0], color='k', label='Max Spearman', ls='--'),
                      ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=6)
    ax.set_xticks([0, 250, 500, 750, 1000])

axes[0, 0].text(-0.25, 1, 'a',
     horizontalalignment='center',
     verticalalignment='center',
     transform = axes[0, 0].transAxes,
     fontsize=16)
axes[1, 0].text(-0.25, 1, 'b',
     horizontalalignment='center',
     verticalalignment='center',
     transform = axes[1, 0].transAxes,
     fontsize=16)

plt.tight_layout()

# -




