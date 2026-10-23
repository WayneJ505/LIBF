# LIBF: Local Implicit Bathymetry Function

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

LIBF introduces implicit neural representations into the task of bathymetric super-resolution.
It learns a continuous implicit representation`f(x, y) → depth` that supports both arbitrary-scale 
grid super-resolution and sparse ship-borne sonar point queries.

### Key Features

- **Arbitrary-scale super-resolution**: predict at any resolution beyond training scales
- **Continuous point queries**: directly query depth at arbitrary coordinates
- **Dual-source training**: leverage both GEBCO grids and sparse ship soundings via mixed sampling
- **RDN encoder + MLP decoder**: local ensemble and cell decoding for smooth continuous output

### Project Structure

```
bathy_liif/
├── libf/                    # Python package
│   ├── __init__.py
│   ├── utils.py             # Metrics, coordinate helpers, inference utilities
│   ├── train.py             # Training entry point
│   ├── eval.py              # Batch evaluation (grid SR + ship point metrics, with cache)
│   ├── demo.py              # Single-file super-resolution demo
│   ├── models/
│   │   ├── models.py        # Model registry
│   │   ├── liif.py          # LIIF core (encoder + imnet)
│   │   ├── rdn.py           # RDN encoder
│   │   ├── edsr.py          # EDSR encoder
│   │   ├── rcan.py          # RCAN encoder
│   │   └── mlp.py           # MLP decoder
│   └── datasets/
│       ├── bathy_tile.py    # Tile-based dataset loader
│       └── bathy_wrapper.py # Mixed grid/ship sampling wrapper
│
├── configs/
│   └── train-bathy.yaml     # Training configuration
├── index/
│   ├── bathy_train.txt      # Training tile index
│   └── bathy_val.txt        # Validation tile index
├── run_all.sh               # Train + eval pipeline
├── split_data.py            # Data split preparation
└── requirements.txt         # Dependencies
```

### Installation

```bash
git clone <repo-url>
cd bathy_liif
pip install -r requirements.txt
```

### Data Preparation

Each tile directory should contain:

```
tile_SXX_WXXX/
├── GEBCO2026_1m.nc       # Low-resolution input (GEBCO 1 arc-minute)
├── GEBCO2026_15s.nc      # High-resolution ground truth (GEBCO 15 arc-second)
└── cleaned_all.txt       # Ship soundings (tab-separated: longitude, latitude, depth)
```

Create train/val splits:

```bash
# Edit index files with paths to tile directories
# index/bathy_train.txt  (e.g., 800 tiles)
# index/bathy_val.txt    (e.g., 200 tiles)
```

### Training

```bash
# Pure grid supervision
python -m libf.train --config configs/train-bathy.yaml --hr_ratio 1.0 --name bathy-grid

# Pure ship point supervision
python -m libf.train --config configs/train-bathy.yaml --hr_ratio 0.0 --name bathy-ship
```

### Evaluation

```bash
python -m libf.eval --val_index index/bathy_val.txt --max_tiles 20
```

Results include PSNR/RMSE at 4 scales (2x–8x) and ship point RMSE/MAE/R².

### Demo

Super-resolve a single NetCDF file:

```bash
python -m libf.demo \
    --input /path/to/bathymetry.nc \
    --checkpoint output/bathy-grid/LIBF-G.pth \
    --scale 4 \
    --output sr_x4.nc
```

The depth variable is auto-detected. Optional ground truth comparison:

```bash
python -m libf.demo \
    --input tile/GEBCO2026_1m.nc \
    --checkpoint output/bathy-grid/LIBF-S.pth \
    --scale 4 \
    --gt tile/GEBCO2026_15s.nc
```

### Configuration

See `configs/train-bathy.yaml`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hr_ratio` | Grid pixel fraction in training (0=pure ship, 1=pure grid) | 1.0 |
| `inp_size` | LR patch size | 48 |
| `scale_min/max` | Random downsampling factor range | 2–4 |
| `sample_q` | Query points per patch | 2304 |
| `epoch_max` | Maximum training epochs | 500 |
| `loss` | Loss function (`l1` or `huber`) | l1 |


### License

MIT License.

---

<a name="chinese"></a>
## 中文

### 简介

LIBF将隐式神经表示引入海底地形超分辨率任务。
模型学习一个连续隐式表示 `f(x, y) → depth`，同时支持任意尺度网格超分和船载声纳测点的连续查询。

### 主要特性

- **任意尺度超分**：可在训练尺度之外的任意分辨率进行预测
- **连续坐标查询**：直接在任意坐标处查询深度值
- **双数据源训练**：通过混合采样同时利用 GEBCO 网格和稀疏船测数据
- **RDN 编码器 + MLP 解码器**：局部集成和单元解码实现平滑连续输出

### 项目结构

```
bathy_liif/
├── libf/                    # Python 包
│   ├── __init__.py
│   ├── utils.py             # 指标、坐标工具、推理函数
│   ├── train.py             # 训练入口
│   ├── eval.py              # 批量评估（网格超分 + 船测点指标，支持缓存）
│   ├── demo.py              # 单文件超分演示
│   ├── models/              # 模型（RDN、EDSR、RCAN、MLP、LIIF）
│   └── datasets/            # 数据加载与混合采样
├── configs/                 # 训练配置
├── index/                   # 数据划分索引
├── run_all.sh               # 一键训练+评估
├── split_data.py            # 数据预处理
└── requirements.txt         # 依赖
```

### 安装

```bash
git clone <repo-url>
cd bathy_liif
pip install -r requirements.txt
```

### 数据准备

每个瓦片目录需包含：

```
tile_SXX_WXXX/
├── GEBCO2026_1m.nc       # 低分辨率输入（GEBCO 1 弧分）
├── GEBCO2026_15s.nc      # 高分辨率真值（GEBCO 15 弧秒）
└── cleaned_all.txt       # 船测数据（制表符分隔：经度、纬度、深度）
```

创建训练/验证划分：

```bash
# 编辑 index 文件，填入瓦片目录路径
# index/bathy_train.txt  （例如 800 个瓦片）
# index/bathy_val.txt    （例如 200 个瓦片）
```

### 训练

```bash
# 纯网格监督
python -m libf.train --config configs/train-bathy.yaml --hr_ratio 1.0 --name bathy-grid

# 纯船测点监督
python -m libf.train --config configs/train-bathy.yaml --hr_ratio 0.0 --name bathy-ship
```

### 评估

```bash
python -m libf.eval --val_index index/bathy_val.txt --max_tiles 20
```

评估结果包含 4 个尺度（2x–8x）的 PSNR/RMSE 及船测点 RMSE/MAE/R²。

### 演示

对单个 NetCDF 文件超分：

```bash
python -m libf.demo \
    --input /path/to/bathymetry.nc \
    --checkpoint output/bathy-grid/LIBF-G.pth \
    --scale 4 \
    --output sr_x4.nc
```

深度变量自动识别。可选与真值比较：

```bash
python -m libf.demo \
    --input tile/GEBCO2026_1m.nc \
    --checkpoint output/bathy-grid/LIBF-S.pth \
    --scale 4 \
    --gt tile/GEBCO2026_15s.nc
```

### 配置

详见 `configs/train-bathy.yaml`。关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `hr_ratio` | 训练中网格像素占比（0=纯船测点，1=纯网格） | 1.0 |
| `inp_size` | 低分辨率裁剪尺寸 | 48 |
| `scale_min/max` | 随机下采样倍率范围 | 2–4 |
| `sample_q` | 每批次查询点数 | 2304 |
| `epoch_max` | 最大训练轮数 | 500 |
| `loss` | 损失函数（`l1` 或 `huber`） | l1 |


### 许可证

MIT License.
