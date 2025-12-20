# 3D Brain Tumor Segmentation using U-Net (BraTS 2020)

This repository provides a professional implementation of a 3D medical image segmentation pipeline using the **U-Net** architecture. The project leverages the **MONAI** framework and **PyTorch** to perform automated multi-modal segmentation of brain tumors into clinically relevant sub-regions.

---

## 🏗 Overall Workflow

<img width="1532" height="442" alt="Image" src="https://github.com/user-attachments/assets/4950d917-e475-40b9-8971-b3e29b255ba1" />

---

## 📊 Dataset Specifications

The model is developed and validated using the **BraTS 2020 Dataset (Training + Validation)**. This dataset consists of multi-institutional MRI scans, providing a robust benchmark for glioma segmentation.

- **Download Source:** [BraTS 2020 on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

### 1. General Dataset Statistics

| Characteristic         | Detail                                      |
| :--------------------- | :------------------------------------------ |
| **Total Samples**      | 494 volumes                                |
| **Training Set Size**  | 369 volumes                                |
| **Validation Set Size**| 125 volumes                                |
| **Input Modalities**   | 4 sequences per patient (FLAIR, T1, T1ce, T2) |
| **Voxel Shape (Input)**| 240 × 240 × 155                            |
| **Voxel Size**         | 1.0 × 1.0 × 1.0 mm³                        |
| **Data Type (Raw)**    | uint8 (Standard 96³ patches used for training) |
| **Data Type (Processed)** | float64 (Standardized for intensity normalization) |

### 2. Label Mapping and Segmentation Targets

The original BraTS labels are re-organized into three nested sub-regions for clinical evaluation:

| BraTS Original Label | Region Description       | Target Segmentation Channel (Output) |
| :------------------- | :----------------------- | :----------------------------------- |
| **Label 0**          | Background / Healthy Tissue | Not part of any target channel      |
| **Label 1**          | Necrotic Core (NCR)     | Part of Tumor Core (TC) and Whole Tumor (WT) |
| **Label 2**          | Peritumoral Edema (ED)  | Part of Whole Tumor (WT) only       |
| **Label 4**          | Enhancing Tumor (ET)    | Part of TC, WT, and ET              |

**Target Channels:**
- **Target TC (Tumor Core):** Necrotic Core + Enhancing Tumor (Channel 1)
- **Target WT (Whole Tumor):** NCR + ED + ET (Channel 2)
- **Target ET (Enhancing Tumor):** Enhancing Tumor only (Channel 3)

---

## 🧠 Model Architecture: 3D U-Net

The architecture utilizes a 3D U-Net with batch normalization and residual units to capture high-resolution features and spatial context.

<img width="1623" height="672" alt="Image" src="https://github.com/user-attachments/assets/149d7cbe-b743-482f-9868-11bba81520c6" />

---

## 🧪 Comparative Analysis: Impact of Input Patch Size

A core component of this research was evaluating how the **Input Patch Dimension** affects the Dice Similarity Coefficient (DSC) and computational efficiency. We compared two standard voxel dimensions: **128³** and **96³**.

### Experimental Results

| Metric                  | Input 128 × 128 × 128 Voxel | Input 96 × 96 × 96 Voxel |
| :---------------------- | :-------------------------: | :-----------------------: |
| **Val Mean Dice**       | 0.7926                     | **0.8171**               |
| **Val Loss**            | 0.2225                     | **0.1986**               |
| **Train Loss**          | **0.1595**                 | 0.2078                   |
| **Dice: Tumor Core (TC)** | 0.7610                   | **0.8069**               |
| **Dice: Whole Tumor (WT)** | 0.8654                  | **0.8759**               |
| **Dice: Enhancing Tumor (ET)** | 0.7515              | **0.7687**               |
| **Time (sec) per picture** | 1.43                    | **1.35**                 |

### 🏁 Conclusion

- **Superior Accuracy:** The **96 × 96 × 96** patch size achieved a significantly higher Mean Dice Score (0.8171), proving to be the optimal hyperparameter for this task.
- **Class Imbalance Mitigation:** Increasing the patch size to 128³ diluted the density of the tumor Region of Interest (ROI) relative to the background, worsening the class imbalance and hindering the model's ability to extract specific features.
- **Efficiency:** The 96³ configuration provided faster inference times and superior model generalization.

---

## 📂 Project Components

- `format.py`: Data integrity verification and deterministic dataset splitting.
- `train.py`: Implementation of the 3D training loop, augmentations, and W&B logging.
- `test.py`: Quantitative evaluation script for final metrics and mask reconstruction.
- `Brain_Tumor_Report.pdf`: Comprehensive research report and statistical analysis.
- `result_1.png` & `result_2.png`: Visual segmentation results for the 96×96×96 U-Net model.

---

## 🚀 Installation & Usage

### 1. Environment Setup

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Dataset Formatting

Configure the source paths in `format.py` and run the script to organize data into Train, Validation, and Test sets:

```bash
python format.py
```

### 3. Model Training

Execute the training script (ensure your Weights & Biases entity is configured in the code):

```bash
python train.py
```

### 4. Evaluation & Testing

To evaluate the best-performing model and generate NIfTI predicted masks:

```bash
python test.py
```
