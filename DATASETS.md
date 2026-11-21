# Sky Observer — Datasets & Labeling Guide

## 1. Verified Public Sky/Cloud Datasets
Sky Observer includes metadata and format adapters for the following verified scientific datasets (download manually in accordance with each dataset's license before importing into `/data/datasets`):

| Dataset | Citation | Samples | Annotations | Suitability | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SWIMSEG** | Dev et al., IEEE TGRS 2016 | 1,013 | Binary sky/cloud masks | Segmentation | Academic / Non-Commercial (NTU) |
| **SWINySEG** | Dev et al., IGARSS 2019 | 6,768 | Day & night binary masks | Segmentation | Academic / Non-Commercial (NTU) |
| **WSISEG** | Xie et al., AMT 2020 | 400 | 3-class whole-sky masks | Segmentation | Research / CC-BY |
| **CCSN** | Zhang et al., JAOT 2018 | 2,543 | 11 cloud genus folders (no pixel masks) | Classification | Academic / Research |
| **CloudSEN12** | Aybar et al., Sci. Data 2022 | 49,400 | Satellite nadir multi-class masks | Reference only (nadir vs ground view) | CC0-1.0 |

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
