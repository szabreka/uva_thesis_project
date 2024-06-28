# A Lightweight CLIP-Based Ensemble Model for State-of-the-art Industrial Smoke Detection from Video Clips

This work was submitted as a thesis project to the Master Information Studies, Data Science track at the University of Amsterdam.

**Supervisor of the project:** Dr. Yen-Chia Hsu (y.c.hsu@uva.nl)

**Date of submission:** 30.06.2024

**Abstarct of the thesis report:** Advanced computer vision techniques have led to the development
of various vision-based approaches to tackle the complex challenge
of smoke detection. A lightweight AI tool that can recognize poten-
tially harmful industrial emissions would aid local communities in
their fight for clean air. This thesis addresses the research gap in
the potential benefits of utilizing OpenAI’s Contrastive Language-
Image Pre-training (CLIP) model for industrial smoke detection.
The proposed Vision-Language model exploits the advantages of
comparing text and image semantics. Prior research has primar-
ily utilized Convolutional Neural Networks, Vision Transformers,
and segmentation methods. This study evaluates CLIP in indus-
trial smoke detection in terms of performance and computational
efficiency. Experiments show that fully-supervised CLIP training,
particularly when utilizing linear probe representation learning,
performs better than zero-shot and few-shot setups for smoke de-
tection. Additionally, an ensemble approach, combining a fully-
supervised CLIP model with a small MobileNetV3 using a GRU
or LSTM layer outperforms the baseline Two-Stream Inflated 3D
ConvNet model with a single Timeception layer in terms of light-
weight design, as the model can be run using only CPU cores. The
ensemble model achieved an average F1 score of 0.72, indicating a
promising approach for smoke detection.

### Used datasets
- IJmond-video-dataset-2024-01-22: https://github.com/MultiX-Amsterdam/ijmond-camera-monitor/tree/main/dataset/2024-01-22#ijmond-video-dataset-2024-01-22
- RISE-Dataset-2020-02-24: https://github.com/CMU-CREATE-Lab/deep-smoke-machine/tree/master/back-end/data/dataset/2020-02-24

### Used splits on the combined dataset

| View | S<sub>0</sub> | S<sub>1</sub> | S<sub>2</sub> | S<sub>4</sub> | S<sub>5</sub> |
| --- | --- | --- | --- | --- | --- |
| 0-0 | Train | Train | Test | Train | Train |
| 0-1 | Test | Train | Train | Train | Train |
| 0-2 | Train | Test | Train | Train | Train |
| 0-3 | Train | Train | Validate | Train | Test |
| 0-4 | Validate | Train | Train | Test | Validate |
| 0-5 | Train | Validate | Train | Train | Test |
| 0-6 | Train | Train | Test | Train | Validate |
| 0-7 | Test | Train | Train | Validate | Train |
| 0-8 | Train | Train | Validate | Test | Train |
| 0-9 | Train | Test | Train | Validate | Train |
| 0-10 | Validate | Train | Train | Test | Train |
| 0-11 | Train | Validate | Train | Train | Test |
| 0-12 | Train | Train | Test | Train | Train |
| 0-13 | Test | Train | Train | Train | Train |
| 0-14 | Train | Test | Train | Train | Train |
| 1-0 | Test | Test | Test | Test | Test |
| 2-0 | Test | Test | Test | Test | Test |
| 2-1 | Test | Test | Test | Test | Test |
| 2-2 | Test | Test | Test | Test | Test |
| 3-0 | Test | Test | Test | Test | Test |
| 3-1 | Test | Test | Test | Test | Test |
| 3-2 | Validate | Train | Test | Train | Train |
| 3-3 | Train | Train | Test | Train | Validate |
| 3-4 | Train | Train | Validate | Test | Train |
| 4-0 | Test | Train | Train | Train | Validate |
| 4-1 | Test | Train | Validate | Train | Train |
| 4-2 | Validate | Train | Test | Train | Train |
| 4-3 | Train | Train | Train | Test | Train |
| 5-0 | Train | Validate | Train | Train | Test |
| 5-1 | Train | Test | Train | Validate | Train |