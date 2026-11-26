# Sky Observer — Datasets & Labeling Guide

## 1. Verified Public Sky/Cloud Datasets
Sky Observer includes metadata and format adapters for the following verified scientific datasets (download manually in accordance with each dataset's license before importing into `/data/datasets`):

| Dataset | Citation | Samples | Annotations | Suitability | License *(verify at source)* |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SWIMSEG** | Dev et al., IEEE TGRS 2016 | 1,013 | Binary sky/cloud masks | Segmentation | Academic / Research — verify exact terms at NTU source |
| **SWINSEG** | Dev et al., IEEE TGRS 2017 | 115 | Nighttime binary sky/cloud masks | Night Segmentation | Academic / Research — verify exact terms at NTU source |
| **SWINySEG** | Dev et al., IGARSS 2019 | 6,768 | Day & night binary masks | Day/Night Segmentation | Academic / Research — verify exact terms at NTU source |
| **WSISEG** | Xie et al., AMT 2020 | 400 | 3-class whole-sky masks | Segmentation | Research / CC-BY — verify at source |
| **CCSN** | Zhang et al., JAOT 2018 | 2,543 | 11 cloud/sky class folders (no pixel masks) | Classification | Academic / Research — verify at source |
| **SWIMCAT** | Dev et al., ICIP 2015 | 784 | 5 sky/cloud patch categories | Classification | Academic / Research — verify at NTU source |
| **MGCD / GCD** | Liu et al., IEEE TGRS 2020 | 8,000+ | Ground-based cloud genus images | Classification | Academic / Research — verify at source |
| **LenghuSky-8** | Rui et al., CVPR Findings 2025 | 429,000+ | 8-year day/night all-sky frames + 1,111 labeled benchmark + nowcasting split | Day/Night Segmentation & Nowcasting Benchmark | Code: Apache-2.0; DINOv3 backbone weights subject to separate **DINOv3 License** — verify at source |
| **SKIPP'D** | Nie et al., Solar Energy 2023 | 349,000+ | Sky images + PV generation time series | Nowcasting & Solar/PV | Academic / Research — verify at Stanford source |
| **CloudSEN12** | Aybar et al., Sci. Data 2022 | 49,400 | Satellite nadir multi-class masks | Reference only (nadir vs ground view) | CC0-1.0 (verify at source) |

### Dataset & Pretrained-Weight License Notice
- **Sky Observer's source code is MIT-licensed**, but **public scientific datasets and external pretrained backbones carry their own independent licenses** (many datasets such as SWIMSEG, SWINSEG, SWINySEG, and CCSN restrict usage to academic or non-commercial research).
- **Model weights fine-tuned on third-party datasets or built atop third-party backbones (e.g., DINOv3) may inherit upstream license restrictions.** Do not redistribute trained weights without verifying compatibility with the upstream dataset and backbone license terms directly at each source repository.
- For an overview of 72 open-source sky image datasets, see Nie et al. (*arXiv:2211.14709*). Note that most public benchmarks use professional all-sky fisheye cameras; always evaluate final accuracy on frames captured with your target low-cost camera (smartphone, USB webcam, ESP32-CAM).

## 2. Supported Local Directory Formats (`POST /api/dataset/import`)
1. **Paired Images + Masks**:
   ```text
   my_dataset/
   ├── images/   (*.jpg, *.png)
   └── masks/    (or GTmaps/, matching filename stems)
   ```
2. **Class Subfolders (CCSN-style)**:
   ```text
   my_clouds/
   ├── cumulus/
   ├── cirrus/
   └── stratus/
   ```

## 3. Leakage-Free Splitting
All imported and bootstrapped datasets split by **group/sequence/day (`group_key`)** (`70% train / 15% val / 15% test`) rather than randomly shuffling near-identical consecutive camera frames.

## 4. Local Camera Adaptation (`Dataset Labeling` Page)
- Browse frames captured by your Android camera.
- Paint `Dense Cloud`, `Thin Cloud`, or `Clear Sky` pixel masks directly over the captured frame.
- Assign cloud formation, event, and quality labels.
- Labeled frames are stored in `user_camera_labels` and marked `is_training_sample = True` so storage retention cleanup never deletes them.
