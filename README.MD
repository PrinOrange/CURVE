# CURVE

**C**lass-imbalanced and **U**nlabeled samples for **R**obust **V**ulnerability **E**xploration

## 安装和配置

为避免兼容性问题，请使用类 Unix 操作系统（如 MacOS, Linux, WSL 等）来上手本项目。

载入 Conda 环境并安装依赖
```bash
conda env create --name curve -f environment.yml
```

本项目在以下环境和配置中训练，读者可作为参考，根据本机配置调整适当模型训练和推理参数。

- NVIDIA RTX 5880, 48GB
- CUDA: 12.4
- Memory 128GB
- Intel(R) Core(TM) i9-14900K x86-64, 24 Cores * 2 threads.
- Ubuntu 22.04, Python 3.10
- 除此之外，至少还需要 10 GB 的空闲空间用于储存数据集、模型。

## 训练

进行预分词训练，生成词表文件。

```bash
python pretrain-tokenizer.py
```

再进行 MLM 训练，得到预训练后的 CweBERT.

```bash
python pretrain-mlm.py
```

## 评估

对训练好的 CweBERT 执行评估。

```bash
python evaluation.py
```

## 引用

若需在公开出版物中引用本项目，请复制以下 BibTeX 引用代码

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
