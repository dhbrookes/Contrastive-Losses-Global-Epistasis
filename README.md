# Code for NeurIPS 2024 paper "Contrastive losses as generalized models of global epistasis"


## Overview

`Contrastive-Losses-Global-Epistasis` contains Python sourcecode for running the simulations used in the main text and appendix of "Contrastive losses as generalized models of global epistasis" by David Brookes, Jakub Otwinowski and Sam Sinai.

## System Requirements

`Contrastive-Losses-Global-Epistasis` can be run on a standard computr. The code has been tested on a Linux (Debian GNU/Linus 11) system with Python 3.10.9. It has been tested with the following versions of Python dependencies:
* torch (2.0.1)
* matplotlib (3.7.1)
* numpy (1.23.5)
* pandas (1.5.3)
* scipy (1.14.0)
* tqdm (4.65.0)
* sklearn (1.2.2)

## Installation

To install from GitHub:
```
git clone https://github.com/dhbrookes/Contrastive-Losses-Global-Epistasis.git
cd Contrastive-Losses-Global-Epistasis

```

## Contents

* `scripts/` contains self-contained Python scripts that run the simulations used in the paper
    * `scripts/run_complete_recovery_main_text.py` runs the simulations whose results are shown in Figure 1.
    * `scripts/run_mse_vs_bt_entropy.py` runs the simulations whose results are shown in Figure 2b.
    * `scripts/run_mse_vs_bt_train_size.py` runs the simulations whose results are shown in Figure 2c.
    *     
