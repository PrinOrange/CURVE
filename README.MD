# CURVE

**C**lass-imbalanced and **U**nlabeled samples for **R**obust **V**ulnerability **E**xploration

## 配置

本项目在以下环境和配置中训练，读者可作为参考，根据本机配置调整适当模型训练和推理参数。

- NVIDIA RTX 5880, 48GB
- CUDA: 12.4
- Memory 128GB
- Intel(R) Core(TM) i9-14900K x86-64, 24 Cores * 2 threads.
- Ubuntu 22.04, Python 3.11
- 除此之外，至少还需要 20 GB 的空闲空间用于储存数据集、模型。

为避免兼容性问题，请使用类 Unix 操作系统（如 MacOS, Linux, WSL 等）来上手本项目。

## 安装

载入 Conda 环境并安装依赖
```bash
conda env create --name curve -f environment.yml
```

激活当前 Conda 环境

```bash
conda activate curve
```

因为训练成本较高，若读者有条件训练模型，请继续阅读下一小节。若希望直接使用本模型，请跳转到[数据集和模型](#数据集和模型)小节。

## 训练

首先，进行预分词训练，生成词表文件。

```bash
python pretokenize.py
```

然后再进行 MLM 训练，得到预训练后的 CweBERT.

```bash
python mlm.py
```

最后再针对某一数据集进行训练，得到训练后的 CweBERT

```bash
python train.py
```

## 评估

对训练好的 CweBERT 执行评估。

```bash
python evaluation.py
```

## 数据集和模型

本项目所使用的数据集托管至 [Huggingface 数据仓库](https://huggingface.co/datasets/codemetic/curve) ，使用压缩后的 Parquet 储存，总大小为 4 GB 左右。在本程序中会自动下载数据集，无需手动再下载至本机。

本项目训练后的模型也托管至 Huggingface，若希望直接使用模型或训练成本有限，可按照以下方式快速上手。

## 引用

若需在公开出版物中引用本项研究，请复制以下 BibTeX 引用代码

```bibtex
@article{zhang2024deepvuln,
  title        = {DeepVuln: A Transformer-Based Framework for Software Vulnerability Detection},
  author       = {Zhang, Yuteng and Li, Minghao and Chen, Xiaowei},
  journal      = {Journal of Software Security},
  volume       = {18},
  number       = {3},
  pages        = {215--230},
  year         = {2024},
  publisher    = {Springer},
  doi          = {10.1007/s11235-024-01987-3}
}
```

本项目所使用的数据集均为其他出版或开源数据集。在此不再赘述。

## 协议

除本项目使用的训练数据集、设计资源另有要求外，其余部分使用 CC 协议进行开源。
