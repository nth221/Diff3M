# Diff3M: Harnessing EHRs for Diffusion-based Anomaly Detection on Chest X-rays

This repository contains our implementation of **"Harnessing EHRs for Diffusion-based Anomaly Detection on Chest X-rays"** (MICCAI 2025).


## Paper Overview

Unsupervised anomaly detection (UAD) in medical imaging is crucial for identifying pathological abnormalities without requiring extensive labeled data. However, existing diffusion-based UAD models rely solely on imaging features, limiting their ability to distinguish between normal anatomical variations and pathological anomalies. To address this, we propose Diff3M, a multi-modal diffusion-based framework that integrates chest X-rays and structured Electronic Health Records (EHRs) for enhanced anomaly detection. Specifically, we introduce a novel image-EHR cross-attention module to incorporate structured clinical context into the image generation process, improving the model’s ability to differentiate normal from abnormal features. Additionally, we develop a static masking strategy to enhance the reconstruction of normal-like images from anomalies. Extensive evaluations on CheXpert and MIMIC-CXR/IV demonstrate that Diff3M achieves state-of-the-art performance, outperforming existing UAD methods in medical imaging.


The Diffusion code used in Diff3M references the implementation provided by [Wolleb's repository](https://gitlab.com/cian.unibas.ch/diffusion-anomaly). 


## Data Preprocessing

We define normal samples as cases with *No Finding*, while all other cases are considered anomalies. For demographic features, we use sex, age, and AP/PA view. In MIMIC-CXR, we further incorporate BMI, blood pressure, height, and weight from MIMIC-IV as additional EHR features, selecting records within three months of the X-ray imaging date.

Due to policy restrictions, we cannot share the preprocessed MIMIC dataset. Instead, we are sharing the preprocessing code used in this study.
Run five codes located in the `all_preprocessing` sequentially:

`preprocess_MIMIC > connect_mimic_plus > join_label_ehr > file_check_and_update_MIMIC > analysis_mimic`



### Citation

Kim, H., Wang, Y., Ahn, M., Choi, H., Zhou, Y., Hong, C. (2026). Harnessing EHRs for Diffusion-Based Anomaly Detection on Chest X-Rays. In: Gee, J.C., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. MICCAI 2025. Lecture Notes in Computer Science, vol 15962. Springer, Cham. https://doi.org/10.1007/978-3-032-04947-6_23
