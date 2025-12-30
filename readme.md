# 论文复现仓库

本仓库用于复现论文中的结构和代码。在这里，我将重新实现论文中提出的算法和模型，以验证论文的实验结果。

## 仓库说明
- 该仓库包含了论文复现所需的所有代码文件。
- 旨在提供可复现的实验环境和步骤。

# 仓库结构

## 基础机器学习算法 (basic_ml)
- **linear_regression/**: 线性回归算法实现（Python、PyTorch、增强版）
- **logistic_regression/**: 逻辑回归算法实现（二分类、多分类）
- **nlp/text_classification/**: 文本分类教程（Bag-of-Words、TF-IDF、CNN、RNN）
- **cv/image_classification/**: 图像分类教程（基础CNN、ResNet、VGG）
- **fine_tuning/**: 模型微调教程（BERT微调、迁移学习）

## 论文复现 (fan)
- paper: 论文文件
- note: 笔记
- code: 代码实现
- data: 数据文件

## 推荐系统 (recall)
- Content_recall: 内容召回
- ItemCF: 物品协同过滤
- UserCF_Swing: 用户协同过滤
- TwoTowerModel: 双塔模型

# 环境配置
- pytorch 3+
- pytorch 2.8

# 环境配置

## Python库依赖

### 基础库
- **numpy**: 数值计算
- **scikit-learn**: 模型评估、数据预处理
- **matplotlib**: 数据可视化
- **seaborn**: 高级数据可视化

### 深度学习
- **PyTorch**: 深度学习框架
- **torchvision**: 计算机视觉工具
- **transformers**: Hugging Face transformers库（用于BERT等）

### 安装命令
```bash
# 基础库
pip install numpy scikit-learn matplotlib seaborn

# PyTorch (根据你的CUDA版本选择)
pip install torch torchvision

# Transformers (用于BERT微调)
pip install transformers
```

## 快速开始

### 1. 线性回归
```bash
cd basic_ml/linear_regression
python data_generator.py
python linear_regression_python.py
```

### 2. 逻辑回归
```bash
cd basic_ml/logistic_regression
python data_generator.py
python logistic_regression_python.py
```

### 3. 文本分类（NLP）
```bash
cd basic_ml/nlp/text_classification
python text_classification_basic.py
```

### 4. 图像分类（CV）
```bash
cd basic_ml/cv/image_classification
python cnn_basic.py
```

### 5. BERT微调
```bash
cd basic_ml/fine_tuning
python bert_finetuning.py
```

### 6. 迁移学习（CV）
```bash
cd basic_ml/fine_tuning
python transfer_learning_cv.py
```

<div align = "center">
  <img src="basic_ml_algorithm.png" width="60%">
</div>