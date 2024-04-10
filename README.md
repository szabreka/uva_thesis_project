# A Lightweight CLIP-Based Ensemble Model for State-of-the-art Industrial Smoke Detection from Video Clips

This work is a further improvement of the work of Hsu et al. (2021). The original paper is noted below:

Yen-Chia Hsu, Ting-Hao (Kenneth) Huang, Ting-Yao Hu, Paul Dille, Sean Prendi, Ryan Hoffman, Anastasia Tsuhlares, Jessica Pachuta, Randy Sargent, and Illah Nourbakhsh. 2021. Project RISE: Recognizing Industrial Smoke Emissions. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). https://ojs.aaai.org/index.php/AAAI/article/view/17739

**Supervisor of the project:** Dr. Yen-Chia Hsu (y.c.hsu@uva.nl)

### Used datasets
- IJmond-video-dataset-2024-01-22: https://github.com/MultiX-Amsterdam/ijmond-camera-monitor/tree/main/dataset/2024-01-22#ijmond-video-dataset-2024-01-22
- RISE-Dataset-2020-02-24: https://github.com/CMU-CREATE-Lab/deep-smoke-machine/tree/master/back-end/data/dataset/2020-02-24


### Used splits on the rise dataset

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

### Used splits on the ijmond dataset

| View | S<sub>0</sub> | S<sub>1</sub> | S<sub>2</sub> | S<sub>4</sub> | S<sub>5</sub> |
| --- | --- | --- | --- | --- | --- |
| 0-0 | Test | Test | Test | Test | Test |
| 0-1 | Test | Test | Test | Test | Test |
| 0-2 | Validate | Train | Test | Train | Train |
| 0-3 | Train | Train | Test | Train | Validate |
| 0-4 | Train | Train | Validate | Test | Train |
| 1-0 | Test | Train | Train | Train | Validate |
| 1-1 | Test | Train | Validate | Train | Train |
| 1-2 | Validate | Train | Test | Train | Train |
| 1-3 | Train | Train | Train | Test | Train |
| 2-0 | Train | Validate | Train | Train | Test |
| 2-1 | Train | Test | Train | Validate | Train |