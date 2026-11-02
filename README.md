<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.23-3776AB?logo=python&logoColor=white" alt="Python 3.9.23">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6.0">
  <img src="https://img.shields.io/badge/dimod-0.12.20-6B7280" alt="dimod 0.12.20">
</p>

<h1 align="center">Factorization Machine for Binary Quadratic Models</h1>

<p align="center">
  A PyTorch implementation of a second-order Factorization Machine for learning
  and reconstructing Binary Quadratic Models (BQMs).
</p>

---

## Overview

This repository implements a second-order **Factorization Machine (FM)** in PyTorch and demonstrates how it can be trained as an energy surrogate for a **Binary Quadratic Model (BQM)**.

The main experiment generates a fully connected binary BQM, samples binary configurations, and evaluates their exact energies. A neural FM is then trained to approximate the energy function. After training, its parameters are converted back into the corresponding BQM coefficients for evaluation and comparison.

The implementation is intended for experiments involving quadratic energy modeling and the connection between factorized interaction models and BQMs.

## Model Formulation

For a binary input vector \(x \in \{0,1\}^{n}\), the second-order Factorization Machine is defined as

\[
\hat{E}(x)
=
b
+
\sum_{i=1}^{n} h_i x_i
+
\sum_{i<j}
\langle v_i,v_j\rangle x_i x_j ,
\]

where:

- \(b\) is the scalar offset.
- \(h_i\) is the linear coefficient associated with variable \(i\).
- \(v_i \in \mathbb{R}^{k}\) is the latent factor vector for variable \(i\).
- \(k\) is the factorization rank.

The pairwise interaction coefficients are represented through the factorization

\[
Q_{ij} = \langle v_i,v_j\rangle,\qquad i<j.
\]

Instead of learning every pairwise coefficient independently, the model learns a low-rank factor matrix \(V\). Its interaction term is evaluated efficiently using

\[
\frac{1}{2}\sum_{f=1}^{k}
\left[
\left(\sum_{i=1}^{n}v_{if}x_i\right)^2
-
\sum_{i=1}^{n}v_{if}^{\,2}x_i^2
\right].
\]

This reduces the interaction evaluation cost from \(O(n^2)\) to \(O(nk)\).

For binary variables, the trained parameters can be mapped to a BQM:

\[
E(x)=b+\sum_i h_i x_i+\sum_{i<j}Q_{ij}x_ix_j.
\]

The implementation extracts the upper-triangular interaction coefficients from \(V^\mathsf{T}V\), excluding the diagonal self-interactions.

## Workflow

The main notebook follows this pipeline:

~~~mermaid
flowchart TD
    A["Generate a random BQM"] --> B["Sample binary configurations"]
    B --> C["Evaluate reference BQM energies"]
    C --> D["Train the PyTorch Factorization Machine"]
    D --> E["Extract offset, linear, and quadratic coefficients"]
    E --> F["Reconstruct a BQM with dimod"]
    F --> G["Compare predicted and reference energies"]
    E --> H["Visualize the learned interaction matrix"]
~~~

## Repository Structure

| File / Directory | Description |
| :--- | :--- |
| [factorization_machine.py](factorization_machine.py) | PyTorch implementation of the Factorization Machine and parameter extraction utilities. |
| [BQM Factorization Machine.ipynb](BQM%20Factorization%20Machine.ipynb) | Main experiment: BQM generation, dataset construction, model training, coefficient extraction, and evaluation. |
| [test_demo_1.ipynb](test_demo_1.ipynb) | Earlier demonstration and experimental notebook. |
| `data/` | Generated training/test datasets and initial model parameters. Created when the main notebook is executed. |
| `results/` | Generated Q-matrix visualization and prediction data. Created during the experiment. |

## Environment

The experiment was developed with the following package versions.

| Package | Version | Installation source |
| :--- | :--- | :--- |
| Python | `3.9.23` | Conda |
| PyTorch | `2.6.0+cu124` | PyTorch CUDA 12.4 wheel |
| NumPy | `2.0.2` | pip |
| Matplotlib | `3.9.4` | pip |
| JupyterLab | `4.4.4` | pip |
| dimod | `0.12.20` | pip |

## Installation

Create and activate a dedicated Conda environment:

~~~bash
conda create -n fm-bqm python=3.9.23 -y
conda activate fm-bqm
~~~

Install PyTorch with CUDA 12.4 support:

~~~bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
~~~

Install the remaining dependencies:

~~~bash
pip install numpy==2.0.2 matplotlib==3.9.4 jupyterlab==4.4.4 dimod==0.12.20
~~~

The training notebook automatically selects CUDA when available and otherwise falls back to CPU. The CUDA-enabled PyTorch installation requires a compatible NVIDIA driver and GPU environment.

## Running the Experiment

Launch JupyterLab from the repository directory:

~~~bash
jupyter lab
~~~

Open **BQM Factorization Machine.ipynb** and execute its cells in order.

The notebook performs the following steps:

1. Generates a random, fully connected BQM with binary variables.
2. Samples configurations and evaluates their reference energies using dimod.
3. Saves the training/test datasets and initial FM parameters.
4. Trains the PyTorch model using mean squared error and the Adam optimizer.
5. Extracts the learned offset, linear coefficients, and quadratic interactions.
6. Reconstructs a BQM and compares its predicted energies with the reference energies.

## Main Experiment Configuration

The default configuration in the main notebook is:

| Parameter | Value |
| :--- | :---: |
| Number of binary variables | 32 |
| Factorization rank | 32 |
| Training samples | 10,000 |
| Test samples | 50,000 |
| Training epochs | 1,000 |
| Batch size | 512 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Loss function | Mean Squared Error (MSE) |

These values are experiment defaults and can be changed in the notebook.

## Using the Model

The model can also be imported independently:

~~~python
import torch
from factorization_machine import FactorizationMachine

model = FactorizationMachine(
    input_size=32,
    factorization_size=8,
    act="identity"
)

# A batch of binary configurations: (batch_size, input_size)
x = torch.randint(0, 2, (16, 32)).float()

# Predicted energies: (batch_size, 1)
predicted_energy = model(x)

# Extract the equivalent BQM parameters
offset, linear, Q_upper = model.get_bhQ()
~~~

The parameter extraction method returns:

- `offset`: scalar BQM offset.
- `linear`: NumPy array containing the linear coefficients.
- `Q_upper`: upper-triangular NumPy matrix containing pairwise interaction coefficients.

For BQM reconstruction, the notebook uses `dimod.BinaryQuadraticModel` with the extracted offset, linear terms, and nonzero upper-triangular quadratic terms.

## Generated Outputs

After running the main notebook, the following files are generated:

| Output | Description |
| :--- | :--- |
| `data/xs_train.npy` | Binary training configurations. |
| `data/ys_train.npy` | Reference energies for the training configurations. |
| `data/xs_test.npy` | Binary test configurations. |
| `data/ys_test.npy` | Reference energies for the test configurations. |
| `data/initial_bias.npy` | Initial model offset. |
| `data/initial_h.npy` | Initial linear parameters. |
| `data/initial_V.npy` | Initial factorization matrix. |
| `results/Q_matrix_pytorch.png` | Visualization of the learned symmetric interaction matrix. |
| `results/ys_pred_pytorch.npy` | Predicted energies on the test set. |
| `py_energy_comparison.png` | Comparison between reference and predicted test energies. |

## Scope and Reproducibility

This repository focuses on **training a Factorization Machine as a BQM energy surrogate and recovering its quadratic coefficients**. The current notebooks do not submit the reconstructed BQM to a quantum annealer or execute an annealing sampler.

The main notebook declares `RANDOM_SEED = 42`, but does not currently apply it to the NumPy or PyTorch random number generators. Therefore, independently generated datasets and model initializations are not guaranteed to be identical across runs.

---

<p align="center">
  <sub>PyTorch implementation for Factorization Machine-based binary quadratic energy modeling.</sub>
</p>
