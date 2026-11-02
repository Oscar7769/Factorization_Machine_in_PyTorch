<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.23-3776AB?logo=python&logoColor=white" alt="Python 3.9.23">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6.0">
  <img src="https://img.shields.io/badge/dimod-0.12.20-6B7280" alt="dimod 0.12.20">
</p>

<h1 align="center">以 PyTorch 實作的 Factorization Machine：BQM 能量模型學習</h1>

<p align="center">
  使用二階 Factorization Machine 學習 Binary Quadratic Model，並由訓練後的模型參數還原二次交互作用係數。
</p>

---

## 專案簡介

本專案使用 **PyTorch** 實作二階 **Factorization Machine（FM）**，並展示如何將 FM 作為 **Binary Quadratic Model（BQM）** 的能量近似模型（energy surrogate）。

實驗首先產生一個具有二元變數的隨機 BQM，再隨機取樣多組二元組態，並透過 `dimod` 計算各組態的參考能量。接著使用 PyTorch 訓練 FM，使模型預測值逼近 BQM 的能量。完成訓練後，再從 FM 參數取回偏置、線性項與二次交互作用係數，重建等效的 BQM，並比較預測能量與參考能量。

本專案主要展示以下流程：

- 使用 PyTorch 建立及訓練二階 Factorization Machine。
- 以 FM 的因子分解矩陣表示變數之間的二次交互作用。
- 將訓練後的 FM 參數轉換成 BQM 係數。
- 使用 `dimod` 重建 BQM，並評估能量預測結果。

## 數學模型

### Factorization Machine

對於輸入二元向量 $\mathbf{x}\in\{0,1\}^{n}$，二階 Factorization Machine 的預測能量可表示為：

專案使用的 FM 預測能量函數如下：

```math
\hat{E}(\mathbf{x})
=
b
+
\sum_{i=1}^{n} h_i x_i
+
\sum_{i<j}
\langle \mathbf{v}_i,\mathbf{v}_j\rangle x_i x_j
```

其中：

- $b$：偏置項（offset）。
- $h_i$：第 $i$ 個變數的線性係數。
- $\mathbf{v}_i\in\mathbb{R}^{k}$：第 $i$ 個變數對應的潛在因子向量。
- $k$：因子分解維度（factorization rank）。
- $\langle \mathbf{v}_i,\mathbf{v}_j\rangle$：變數 $i$ 與 $j$ 之間的二次交互作用係數。

FM 不需要直接為每一組變數交互作用獨立配置參數，而是透過因子矩陣 $V$ 表示交互作用：

其中，兩個不同變數之間的二次交互作用係數為：

```math
Q_{ij}=\langle \mathbf{v}_i,\mathbf{v}_j\rangle,
\qquad i<j
```

### 二次交互作用的高效率計算

若直接計算所有變數配對，二次交互作用的計算量會隨變數數量呈平方成長。FM 利用平方和恆等式，將二次交互作用改寫為：

其二次交互作用項可利用下式高效率計算：

```math
\frac{1}{2}\sum_{f=1}^{k}
\left[
\left(\sum_{i=1}^{n}v_{if}x_i\right)^2
-
\sum_{i=1}^{n}v_{if}^{\,2}x_i^2
\right]
```

因此，前向傳播的計算複雜度可由 $O(n^2)$ 降低為 $O(nk)$，其中 $n$ 為輸入變數數量，$k$ 為因子分解維度。

### 從 FM 參數還原 BQM

對於二元變數 $x_i\in\{0,1\}$，BQM 的能量函數可寫成：

BQM 的能量函數表示如下：

```math
E(\mathbf{x})
=
b
+
\sum_i h_i x_i
+
\sum_{i<j}Q_{ij}x_ix_j
```

訓練完成後，FM 的參數可轉換為對應的 BQM 係數：

- 偏置項：$b$。
- 線性係數：$h$。
- 二次係數：由 $V^\mathsf{T}V$ 取得，並僅保留上三角矩陣中不含對角線的元素。

程式中的 `get_bhQ()` 會回傳上述參數。由於對角線代表變數自身的平方項，而本實作只保留不同變數之間的交互作用，因此會排除矩陣對角線。

## 實驗流程

主要實驗 Notebook 的執行流程如下：

~~~mermaid
flowchart TD
    A["產生隨機 BQM"] --> B["取樣二元變數組態"]
    B --> C["計算 BQM 參考能量"]
    C --> D["使用 PyTorch 訓練 Factorization Machine"]
    D --> E["擷取偏置、線性與二次係數"]
    E --> F["使用 dimod 重建 BQM"]
    F --> G["比較預測能量與參考能量"]
    E --> H["繪製二次交互作用矩陣"]
~~~

## 專案結構

| 檔案／資料夾 | 說明 |
| :--- | :--- |
| [factorization_machine.py](factorization_machine.py) | Factorization Machine 模型、前向傳播，以及參數擷取函式。 |
| [BQM Factorization Machine.ipynb](BQM%20Factorization%20Machine.ipynb) | 主要實驗：產生 BQM、建立資料集、訓練模型、擷取係數及評估預測結果。 |
| [test_demo_1.ipynb](test_demo_1.ipynb) | 初期模型測試與 BQM 轉換示範。 |
| `data/` | 執行主要 Notebook 後產生的訓練／測試資料集與模型初始參數。 |
| `results/` | 儲存訓練後的交互作用矩陣圖與能量預測資料。 |

## 環境需求

本專案使用以下套件版本：

| 套件 | 版本 | 安裝來源 | 用途 |
| :--- | :--- | :--- | :--- |
| Python | `3.9.23` | Conda | Python 執行環境 |
| PyTorch | `2.6.0+cu124` | pip | 深度學習與模型訓練（CUDA 12.4） |
| NumPy | `2.0.2` | pip | 數值運算 |
| Matplotlib | `3.9.4` | pip | 結果視覺化 |
| JupyterLab | `4.4.4` | pip | 執行 Notebook |
| dimod | `0.12.20` | pip | BQM 建立、能量計算與模型重建 |

## 安裝方式

### 1. 建立並啟用 Conda 環境

~~~bash
conda create -n fm-bqm python=3.9.23 -y
conda activate fm-bqm
~~~

### 2. 安裝 PyTorch（CUDA 12.4）

~~~bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
~~~

### 3. 安裝其餘套件

~~~bash
pip install numpy==2.0.2 matplotlib==3.9.4 jupyterlab==4.4.4 dimod==0.12.20
~~~

Notebook 會自動檢查 CUDA 是否可用：若環境支援 CUDA，便使用 GPU 訓練；否則會自動改用 CPU。若要使用 GPU，仍須具備相容的 NVIDIA GPU 與驅動程式。

## 執行實驗

在專案根目錄啟動 JupyterLab：

~~~bash
jupyter lab
~~~

開啟 **BQM Factorization Machine.ipynb**，並依照 Notebook 中的順序執行各個儲存格。主要步驟如下：

1. 產生具有 32 個二元變數的隨機 BQM。
2. 隨機產生訓練與測試組態，並使用 `dimod` 計算對應能量。
3. 儲存資料集與 FM 模型初始參數。
4. 使用 Mean Squared Error（MSE）作為損失函數，並透過 Adam optimizer 訓練模型。
5. 從訓練後的模型擷取偏置、線性係數與二次交互作用係數。
6. 使用擷取出的係數重建 BQM，並比較測試資料的預測能量與參考能量。

## 主要實驗參數

主要 Notebook 預設使用以下設定：

| 參數 | 數值 |
| :--- | :---: |
| 二元變數數量（VAR_NUM） | 32 |
| 因子分解維度（K） | 32 |
| 訓練資料筆數（DATASET_SIZE） | 10,000 |
| 測試資料筆數（TEST_NUM） | 50,000 |
| 訓練 Epoch（NUM_EPOCH） | 1,000 |
| Batch Size | 512 |
| Optimizer | Adam |
| Learning Rate | `1e-3` |
| Loss Function | Mean Squared Error（MSE） |

上述數值為 Notebook 的預設實驗參數，可依實際需求修改。

## 模型使用範例

除了執行完整 Notebook，也可以在 Python 程式中獨立呼叫模型：

~~~python
import torch
from factorization_machine import FactorizationMachine

model = FactorizationMachine(
    input_size=32,
    factorization_size=8,
    act="identity"
)

# 建立一批二元輸入組態，尺寸為 (batch_size, input_size)
x = torch.randint(0, 2, (16, 32)).float()

# 預測能量，輸出尺寸為 (batch_size, 1)
predicted_energy = model(x)

# 擷取等效 BQM 參數
offset, linear, Q_upper = model.get_bhQ()
~~~

`get_bhQ()` 回傳三項資料：

| 回傳值 | 說明 |
| :--- | :--- |
| `offset` | BQM 的偏置項（純量）。 |
| `linear` | 儲存各變數線性係數的 NumPy 陣列。 |
| `Q_upper` | 儲存二次交互作用係數的上三角 NumPy 矩陣，不包含對角線。 |

主要 Notebook 會將這些參數傳入 `dimod.BinaryQuadraticModel`，建立重建後的 BQM。

## 實驗輸出

執行主要 Notebook 後，會產生以下檔案：

| 輸出檔案 | 說明 |
| :--- | :--- |
| `data/xs_train.npy` | 訓練用二元變數組態。 |
| `data/ys_train.npy` | 訓練組態對應的 BQM 參考能量。 |
| `data/xs_test.npy` | 測試用二元變數組態。 |
| `data/ys_test.npy` | 測試組態對應的 BQM 參考能量。 |
| `data/initial_bias.npy` | FM 初始偏置參數。 |
| `data/initial_h.npy` | FM 初始線性參數。 |
| `data/initial_V.npy` | FM 初始因子分解矩陣。 |
| `results/Q_matrix_pytorch.png` | 訓練後的對稱二次交互作用矩陣視覺化結果。 |
| `results/ys_pred_pytorch.npy` | 模型對測試資料的預測能量。 |
| `py_energy_comparison.png` | 測試資料的參考能量與預測能量比較圖。 |

## 隨機 BQM 實驗與可重現性

本專案的實驗目的，是確認 Factorization Machine 是否能從隨機產生的 BQM 與隨機取樣的二元組態中，學習其能量函數，並對未參與訓練的測試組態做出正確的能量預測。這樣可以檢驗模型是否確實學到 BQM 的二次能量關係，而不是只對某個特定、人工設計的結構有效。

Notebook 中設定 `RANDOM_SEED = 42`，是為了對照原始實驗所使用的隨機設定。實驗會隨機產生 BQM 的係數，並隨機取樣訓練與測試用的二元變數組態，再以 BQM 計算出的能量作為監督學習標籤，訓練 FM 並評估其預測能力。

**可重現性注意事項：**目前 Notebook 僅宣告 `RANDOM_SEED = 42`，尚未將它實際傳入 NumPy 或 PyTorch 的隨機數產生器。因此，這個數值目前表達的是實驗預定採用的 seed 設定，並不會單憑宣告就固定 BQM 係數、資料集或模型初始權重。若要讓每次執行都使用相同的隨機序列，仍需在產生 BQM、資料集及模型參數之前，分別設定 NumPy 與 PyTorch 的 random seed。

本專案目前聚焦於 **使用 Factorization Machine 學習 BQM 能量函數，並由模型參數還原二次交互作用係數**；Notebook 並未將重建後的 BQM 提交至 D-Wave 量子退火器，也沒有執行量子退火取樣。

---

<p align="center">
  <sub>以 PyTorch 實作 Factorization Machine，進行二元二次能量模型的學習與重建。</sub>
</p>
