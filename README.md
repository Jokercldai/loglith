<p align="center" width="100%">
<img src="assets/network_v2.pdf" width="80%">
</p>

<div align="center">
    <strong>
    LogGPT: Chunk-Based Autoregressive Self-Supervised Pretraining  
    for Lithology Identification From Well Logs
    </strong>
    </br>
    <a href='https://github.com/yourname' target='_blank'>Your Name<sup>1</sup></a>&emsp;
    <a href='#'>Co-author<sup>1</sup></a>&emsp;
</div>

<div align="center">
    <sup>1</sup> Your Institution &emsp;
</div>

---

# 🌟 LogGPT

Lithology identification is a fundamental task in hydrocarbon exploration and reservoir characterization. However, it remains challenging due to:

- Complex nonlinear responses  
- Limited labeled data with class imbalance  
- Point-wise modeling limitations  
- Widespread missing data  

To address these challenges, we propose **LogGPT**, a chunk-based autoregressive self-supervised framework for lithology identification.

---

## 🔥 Key Ideas

- **Chunk-based modeling**  
  Instead of point-wise prediction, logs are partitioned into chunks to capture stratigraphic context.

- **Autoregressive pretraining (GPT-style)**  
  Enables learning deeper geological patterns rather than local interpolation.

- **Self-supervised learning**  
  Leverages large-scale unlabeled well logs.

- **Missing-aware masking**  
  Handles incomplete logs naturally.

- **Focal Tversky loss**  
  Mitigates severe class imbalance.

---

## 🧠 Method Overview

<p align="center">
<img src="assets/framework.png" width="80%">
</p>

LogGPT is built upon a GPT-2 architecture:

1. Continuous well logs → chunk partitioning  
2. Chunk-level autoregressive pretraining  
3. Fine-tuning for lithology classification  

---

## 📊 Results

<p align="center">
<img src="assets/results.png" width="80%">
</p>

Experiments on the Yuanba dataset show that LogGPT:

- Outperforms BiLSTM, ResNet, Transformer  
- Achieves higher **Recall** and **F1-score**  
- Provides more stable performance  

---

## ⚙️ Installation

```bash
conda create -n loggpt python=3.10
conda activate loggpt

pip install -r requirements.txt


## 🚀 Usage
Training
python train.py
Evaluation
python test.py
