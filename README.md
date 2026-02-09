# Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning

This repository is the official implementation of [Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](https://arxiv.org/abs/2509.23049).

> Federated learning (FL) is increasingly adopted in domains like healthcare, where data privacy is paramount. A fundamental challenge in these systems is statistical heterogeneity—the fact that data distributions vary significantly across clients (e.g., different hospitals may treat distinct patient demographics). While current FL algorithms focus on aggregating model updates from these heterogeneous clients, the potential of the central server remains under-explored.
>
> This paper is motivated by a healthcare scenario: could a central server not only coordinate model training but also guide a new patient to the hospital best equipped for their specific condition? We generalize this idea to propose a novel paradigm for FL systems where the server actively guides the allocation of new tasks or queries to the most appropriate client. To enable this, we introduce a density ratio model and empirical likelihood-based framework that simultaneously addresses two goals: (1) learning effective local models on each client, and (2) finding the best matching client for a new query. Empirical results demonstrate the framework's effectiveness on benchmark datasets, showing improvements in both model accuracy and the precision of client guidance compared to standard FL approaches. This work opens a new direction for building more intelligent and resource-efficient FL systems that leverage heterogeneity as a feature, not just a bug.

---

## Installation

We provide step-by-step installation commands as follows:

```bash
conda create -n FedDRM python=3.12
conda activate FedDRM

pip install -r requirements.txt

git clone https://github.com/zijianwang0510/FedDRM.git
cd FedDRM
```

## Datasets

This project is evaluated on CIFAR-10, CIFAR-100, and RETINA.

### CIFAR-10 / CIFAR-100

These datasets are automatically downloaded at runtime. No manual action is required.

```text
# Expected structure (automatically created)
./FedDRM/datasets/CIFAR10/
./FedDRM/datasets/CIFAR100/
```

### RETINA

This dataset requires manual download.

- Download the dataset from [here](https://drive.google.com/file/d/1MMK8bourqVFyJ-UbuMgB40n-xTYHlHl2/view).
- Place the extracted data into the `datasets` directory within the project root.

```text
# Expected structure
./FedDRM/datasets/retina_balanced/
```

---

## Usage

### Training

To train the model, use the `exp/run.py` script. You must specify the configuration file using the `--config` argument (without the `.yaml` extension).

#### Example: Running FedDRM

```bash
python exp/run.py --config feddrm
```

#### Example: Running Baselines

```bash
python exp/run.py --config fedas
```

### Configuration

All hyperparameters are defined in YAML files located in the `configs/` directory. You can modify these files to change experimental settings.

### Outputs

By default, training artifacts are organized as follows:

- `logs/` for logs
- `save/` for checkpoints and other experiment outputs

---

## Citation

If this repository is useful for your research, please consider citing:

```bibtex
@inproceedings{wang2026beyond,
  title     = {Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning},
  author    = {Wang, Zijian and Zhang, Xiaofei and Zhang, Xin and Liu, Yukun and Zhang, Qiong},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
