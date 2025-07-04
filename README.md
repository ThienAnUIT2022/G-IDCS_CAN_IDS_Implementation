# G-IDCS for CAN Protocol: Reimplementation and Evaluation

This repository contains our reimplementation of the **G-IDCS (Graph-Based Intrusion Detection and Classification System)** for the CAN protocol, as proposed in the research paper:

> _G-IDCS: Graph-Based Intrusion Detection and Classification System for CAN Protocol_, published in IEEE Access.

## 📘 Project Context

This is a final project for the course **NT204.P22.ANTT**, where we study about IDS/IPS, SIEM, SOC and reproduce the methodology and results of the G-IDCS system, which includes:

- Building graphs from CAN message windows
- Extracting temporal-structural features from those graphs
- Applying:
  - **TH_classifier**: threshold-based detection
  - **ML_classifier**: machine-learning based classification (Random Forest)

## 🛠️ Features

- ✅ Parsing CAN messages from CSV dataset (Car Hacking Dataset)
- ✅ Graph construction per 200-message window (non-overlapping)
- ✅ Feature extraction: elapsed time, max degree, number of edges
- ✅ Threshold-based detection (G-IDCS TH)
- ✅ RandomForest-based classification (G-IDCS ML)
- ✅ Metrics evaluation & visualization
- ✅ Comparison with paper results

## 📊 Dataset

This project uses the Car-Hacking Dataset originally created by Prof. Huy Kang Kim's research group at Korea University.

> **Acknowledgment:**
> The dataset used in this study was not publicly available at the time of implementation. We received permission and access by directly contacting **Prof. Huy Kang Kim** (Korea University). We sincerely thank the authors for their support.

## 📚 Related References

Please cite the following works when referring to the dataset or related research:

- Song, Hyun Min, Jiyoung Woo, and Huy Kang Kim.
  _“In-vehicle network intrusion detection using deep convolutional neural network.”_
  _Vehicular Communications_ 21 (2020): 100198. [https://doi.org/10.1016/j.vehcom.2019.100198](https://doi.org/10.1016/j.vehcom.2019.100198)

- Seo, Eunbi, Hyun Min Song, and Huy Kang Kim.
  _“GIDS: GAN based Intrusion Detection System for In-Vehicle Network.”_
  _2018 16th Annual Conference on Privacy, Security and Trust (PST). IEEE, 2018._ [https://doi.org/10.1109/PST.2018.8576040](https://doi.org/10.1109/PST.2018.8576040)

## 🔍 Results Summary

| Classifier    | Accuracy | Precision | Recall | F1-score |
| ------------- | -------- | --------- | ------ | -------- |
| TH_classifier | 99.45%   | 99.53%    | 99.31% | 99.42%   |
| ML_classifier | 96.15%   | 94.41%    | 95.17% | 94.57%   |

Our implementation closely follows the original paper and achieves comparable results, confirming the method's reproducibility and robustness.

## 📁 Structure

```bash
├─── Illustration         # Graph figures and video demo
├─── metrics              # Our experimental results
├─── Test_ML              # Processed dataset for testing ML_classifier
├─── Test_TH              # Processed dataset for testing TH_classifier
├─── Train_ML             # Processed dataset for training ML_classifier
├─── Train_TH             # Processed dataset for training TH_classifier
├─── windows              # Dumped JSON files for post investigation
├─── processed.py         # Standardize data
├─── train.py             # Training TH and ML classifiers
├─── test.py              # Evaluation scripts
├─── report.py            # Export graphs and visual charts
├─── ml_classifier.joblib # Saved ML_classifier models
├─── th_classifier.json   # Threshold values of TH_classifier
```

## 🧑‍💻 Contributors

- 22520028 - Pham Truong Thien An (Me)
- 22520132 - Nguyen Huu Binh
- 22520180 - Bui Phuong Dai
- 22520199 - Le Cong Danh
