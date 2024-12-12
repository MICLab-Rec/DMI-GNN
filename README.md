## DMI-GNN

This is the source code for AAAI 2025 Paper: *Dynamic Multi Interest Graph Neural Network for Session-Based Recommendation.*

### Requirements
- Python 3
- PyTorch = 1.11.0 + cu113
- tqdm

### Usage
Data preprocessing:

Train and evaluate the model:

```
python build_graph.py --dataset Tmall
python main.py --dataset Tmall
```