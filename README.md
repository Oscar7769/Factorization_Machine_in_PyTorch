<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.23-3776AB?logo=python&logoColor=white" alt="Python 3.9.23">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6.0">
  <img src="https://img.shields.io/badge/dimod-0.12.20-6B7280" alt="dimod 0.12.20">
</p>

<h1 align="center">Factorization Machine in PyTorch<br>以二階 FM 學習 Binary Quadratic Model 的能量函數</h1>

<p align="center">
  使用 PyTorch 訓練二階 Factorization Machine，從隨機 BQM 的能量資料中學習二次交互作用，並由模型參數還原等效的 BQM 係數。
</p>

---

## 摘要

Binary Quadratic Model（BQM）是組合最佳化與量子退火中常用的能量模型。本專案以 **PyTorch** 實作二階 **Factorization Machine（FM）**，並將其作為 BQM 的**能量代理模型（energy surrogate）**：先由隨機 BQM 產生「二元組態－能量」資料，再以監督式學習訓練 FM 逼近該能量函數，最後由訓練後的參數還原偏置、線性項與二次交互作用係數，重建等效 BQM 並與參考能量比較。

本專案展示以下流程：

1. 以 PyTorch 建立並訓練二階 FM。
2. 以低秩因子矩陣表示變數間的二次交互作用。
3. 將 FM 參數轉換為 BQM 係數。
4. 以 `dimod` 重建 BQM，並於未參與訓練的測試組態上評估能量預測。

## 目錄

- [摘要](#摘要)
- [理論背景](#理論背景)
- [實驗流程](#實驗流程)
- [專案結構](#專案結構)
- [環境與安裝](#環境與安裝)
- [快速開始](#快速開始)
- [實驗設定](#實驗設定)
- [API 使用範例](#api-使用範例)
- [實驗輸出](#實驗輸出)
- [可重現性與限制](#可重現性與限制)
- [參考文獻](#參考文獻)

---

## 理論背景

### 1. Binary Quadratic Model

設有 $n$ 個二元變數，組態為 $\mathbf{x} \in \{0,1\}^{n}$。BQM 的能量函數定義為：

$$
E(\mathbf{x}) = b + \sum_{i=1}^{n} h_i x_i + \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} Q_{ij}\, x_i x_j
$$

其中：

- $b$：偏置項（offset）。
- $h_i$：變數 $x_i$ 的線性係數。
- $Q_{ij}$：變數 $x_i$ 與 $x_j$ 的二次交互作用係數（$i<j$）。

### 2. 二階 Factorization Machine

二階 FM（Rendle, 2010）以相同形式預測能量，但不為每一對變數獨立配置參數，而是令每個變數對應一個潛在因子向量 $\mathbf{v}_i \in \mathbb{R}^{k}$，並以內積表示交互作用：

$$
\hat{E}(\mathbf{x}) = b + \sum_{i=1}^{n} h_i x_i + \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j
$$

其中 $k$ 為因子分解維度（factorization rank）。與 BQM 比較可知，FM 以下式參數化二次係數：

$$
Q_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{if}\, v_{jf}, \qquad i < j
$$

令因子矩陣 $V = [\mathbf{v}_1, \ldots, \mathbf{v}_n] \in \mathbb{R}^{k \times n}$，則完整的交互作用矩陣為 $V^{\top}V \in \mathbb{R}^{n \times n}$。當 $k < n$ 時，$V^{\top}V$ 的秩至多為 $k$，因此 FM 等同對 $Q$ 施加低秩限制；本專案預設 $k = n = 32$，模型表達能力足以涵蓋一般的 $Q$。

### 3. 前向傳播的高效率計算

直接計算所有變數配對的交互作用需要 $O(n^2 k)$ 的運算。利用平方和恆等式，二次項可改寫為：

$$
\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j
= \frac{1}{2} \sum_{f=1}^{k} \left[ \left( \sum_{i=1}^{n} v_{if}\, x_i \right)^{2} - \sum_{i=1}^{n} v_{if}^{2}\, x_i^{2} \right]
$$

因此前向傳播的複雜度由 $O(n^2 k)$ 降為 $O(nk)$。

### 4. 由 FM 參數還原 BQM

訓練完成後，FM 參數與 BQM 係數的對應關係如下：

| BQM 係數 | 來源 |
| :--- | :--- |
| 偏置項 $b$ | FM 的偏置參數 |
| 線性係數 $h_i$ | FM 的線性權重 |
| 二次係數 $Q_{ij}$ | $V^{\top}V$ 的嚴格上三角元素（$i<j$） |

對角線元素 $\langle \mathbf{v}_i, \mathbf{v}_i \rangle$ 對應變數自身的平方項，不屬於模型中的交互作用，因此還原時予以排除。程式中的 `get_bhQ()` 會回傳上述三項參數。

---

## 實驗流程

~~~mermaid
flowchart TD
    A["產生隨機 BQM"] --> B["取樣二元變數組態"]
    B --> C["以 dimod 計算 BQM 參考能量"]
    C --> D["以 PyTorch 訓練 Factorization Machine"]
    D --> E["擷取偏置、線性與二次係數"]
    E --> F["以 dimod 重建 BQM"]
    F --> G["比較測試集的預測能量與參考能量"]
    E --> H["繪製二次交互作用矩陣"]
~~~

**實驗目的**：檢驗 FM 能否僅憑「隨機組態－能量」的資料，學得隨機 BQM 的二次能量關係，並對未參與訓練的測試組態做出準確預測。使用隨機產生的 BQM，而非人工設計的特定結構，可避免結論僅適用於特殊問題。

---

## 專案結構

| 檔案／資料夾 | 說明 |
| :--- | :--- |
| [`factorization_machine.py`](factorization_machine.py) | FM 模型定義、前向傳播與參數擷取函式 `get_bhQ()`。 |
| [`BQM Factorization Machine.ipynb`](BQM%20Factorization%20Machine.ipynb) | 主要實驗：產生 BQM、建立資料集、訓練、擷取係數與評估。 |
| [`test_demo_1.ipynb`](test_demo_1.ipynb) | 初期模型測試與 BQM 轉換示範。 |
| `data/` | 執行主要 Notebook 後產生的資料集與模型初始參數。 |
| `results/` | 儲存交互作用矩陣圖與預測能量資料。 |

---

## 環境與安裝

### 套件版本

| 套件 | 版本 | 安裝來源 | 用途 |
| :--- | :--- | :--- | :--- |
| Python | `3.9.23` | Conda | 執行環境 |
| PyTorch | `2.6.0+cu124` | pip | 模型訓練（CUDA 12.4） |
| NumPy | `2.0.2` | pip | 數值運算 |
| Matplotlib | `3.9.4` | pip | 結果視覺化 |
| JupyterLab | `4.4.4` | pip | 執行 Notebook |
| dimod | `0.12.20` | pip | BQM 建立、能量計算與重建 |

### 安裝步驟

**1. 建立並啟用 Conda 環境**

~~~bash
conda create -n fm-bqm python=3.9.23 -y
conda activate fm-bqm
~~~

**2. 安裝 PyTorch（CUDA 12.4）**

~~~bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
~~~

**3. 安裝其餘套件**

~~~bash
pip install numpy==2.0.2 matplotlib==3.9.4 jupyterlab==4.4.4 dimod==0.12.20
~~~

Notebook 會自動偵測 CUDA：可用時以 GPU 訓練，否則改用 CPU。使用 GPU 需具備相容的 NVIDIA GPU 與驅動程式。

---

## 快速開始

在專案根目錄啟動 JupyterLab：

~~~bash
jupyter lab
~~~

開啟 `BQM Factorization Machine.ipynb`，依序執行所有儲存格。各階段的作業如下：

| 步驟 | 內容 |
| :---: | :--- |
| 1 | 產生具有 32 個二元變數的隨機 BQM。 |
| 2 | 隨機取樣訓練與測試組態，並以 `dimod` 計算對應能量。 |
| 3 | 儲存資料集與 FM 初始參數至 `data/`。 |
| 4 | 以 MSE 為損失函數、Adam 為優化器訓練模型。 |
| 5 | 以 `get_bhQ()` 擷取偏置、線性係數與二次交互作用係數。 |
| 6 | 以擷取的係數重建 BQM，比較測試集預測能量與參考能量。 |

---

## 實驗設定

| 參數 | 預設值 |
| :--- | :---: |
| 二元變數數量（`VAR_NUM`） | 32 |
| 因子分解維度（`K`） | 32 |
| 訓練資料筆數（`DATASET_SIZE`） | 10,000 |
| 測試資料筆數（`TEST_NUM`） | 50,000 |
| 訓練 Epoch（`NUM_EPOCH`） | 1,000 |
| Batch Size | 512 |
| Optimizer | Adam |
| Learning Rate | `1e-3` |
| 損失函數 | Mean Squared Error（MSE） |

訓練目標為最小化預測能量與參考能量的均方誤差：

$$
\mathcal{L} = \frac{1}{N} \sum_{m=1}^{N} \left( \hat{E}(\mathbf{x}^{(m)}) - E(\mathbf{x}^{(m)}) \right)^{2}
$$

以上為 Notebook 的預設值，可依需求調整。

---

## API 使用範例

### 獨立使用 FM 模型

~~~python
import torch
from factorization_machine import FactorizationMachine

model = FactorizationMachine(
    input_size=32,
    factorization_size=8,
    act="identity"
)

# 一批二元輸入組態，形狀為 (batch_size, input_size)
x = torch.randint(0, 2, (16, 32)).float()

# 預測能量，形狀為 (batch_size, 1)
predicted_energy = model(x)

# 擷取等效 BQM 參數
offset, linear, Q_upper = model.get_bhQ()
~~~

`get_bhQ()` 的回傳值：

| 回傳值 | 說明 |
| :--- | :--- |
| `offset` | BQM 偏置項（純量）。 |
| `linear` | 各變數線性係數的 NumPy 陣列。 |
| `Q_upper` | 二次交互作用係數的上三角 NumPy 矩陣（不含對角線）。 |

### 以 `dimod` 重建 BQM

下列為示意寫法，實際寫法請以主要 Notebook 為準：

~~~python
import dimod

n = len(linear)
bqm = dimod.BinaryQuadraticModel(
    {i: float(linear[i]) for i in range(n)},
    {(i, j): float(Q_upper[i, j]) for i in range(n) for j in range(i + 1, n)},
    float(offset),
    dimod.BINARY,
)
~~~

---

## 實驗輸出

| 輸出檔案 | 說明 |
| :--- | :--- |
| `data/xs_train.npy` | 訓練用二元變數組態。 |
| `data/ys_train.npy` | 訓練組態的 BQM 參考能量。 |
| `data/xs_test.npy` | 測試用二元變數組態。 |
| `data/ys_test.npy` | 測試組態的 BQM 參考能量。 |
| `data/initial_bias.npy` | FM 初始偏置參數。 |
| `data/initial_h.npy` | FM 初始線性參數。 |
| `data/initial_V.npy` | FM 初始因子矩陣。 |
| `results/Q_matrix_pytorch.png` | 訓練後的對稱二次交互作用矩陣視覺化。 |
| `results/ys_pred_pytorch.npy` | 模型對測試集的預測能量。 |
| `py_energy_comparison.png` | 測試集參考能量與預測能量的比較圖。 |

---

## 可重現性與限制

**隨機種子**：Notebook 宣告 `RANDOM_SEED = 42`，但目前尚未將其傳入 NumPy 與 PyTorch 的隨機數產生器，因此僅宣告並不會固定 BQM 係數、資料集或模型初始權重。若需完全重現，請在產生 BQM、資料集與模型參數之前，分別設定：

~~~python
import numpy as np
import torch

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
~~~

**研究範圍**：本專案聚焦於「以 FM 學習 BQM 能量函數，並由模型參數還原二次交互作用係數」。Notebook 並未將重建後的 BQM 提交至 D-Wave 量子退火器，亦未執行量子退火取樣。

---

## 參考文獻

1. S. Rendle, "Factorization Machines," *2010 IEEE International Conference on Data Mining (ICDM)*, pp. 995–1000, 2010.
2. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.
3. D-Wave Systems, [`dimod` documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/index.html).

---

<p align="center">
  <sub>以 PyTorch 實作 Factorization Machine，進行二元二次能量模型的學習與重建。</sub>
</p>
