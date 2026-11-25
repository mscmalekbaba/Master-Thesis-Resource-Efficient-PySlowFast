# Resource-efficient Deep Learning for Real-time Recognition of Worker Activities in Industrial Assembly

Code for my M.Sc. thesis (University of Stuttgart, Institute for Artificial Intelligence, 2025; first examiner Jun.-Prof. Dr.-Ing. Alina Roitberg). The repository extends [PySlowFast](https://github.com/facebookresearch/SlowFast) with structured pruning, knowledge distillation (KD) and model-efficiency measurements, and evaluates them on the egocentric [MECCANO](https://github.com/fpv-iplab/MECCANO) dataset (61 fine-grained action classes).

The core idea is published in a shorter form as **PDNet** (VISAPP 2026): [github.com/mscmalekbaba/PDNet](https://github.com/mscmalekbaba/PDNet).

## Main result

Isomorphic structured pruning (L1 criterion, 25% of channels) shrinks X3D-M from 3.09M to 1.80M parameters. Distillation from SlowFast then recovers most of the accuracy lost to pruning.

| Model (MECCANO test set) | Top-1 (%) | Top-5 (%) | Size (MB) | GFLOPs |
| --- | --- | --- | --- | --- |
| SlowFast R50 (teacher, dataset authors' model) | 45.16 | 73.75 | 128.88 | 66.06 |
| X3D-M (baseline) | 35.40 | 61.23 | 11.82 | 6.2 |
| X3D-M, pruned only | 25.54 | 58.64 | 6.90 | 3.78 |
| **X3D-M, pruned + KD** | **39.46** | **68.72** | **6.90** | **3.78** |

*Size = number of parameters x 4 bytes (fp32 weights): 3.09M parameters for X3D-M, 1.80M after pruning. Checkpoint files on disk can be larger because they store more than the weights.*

**CPU inference** on a 2-minute test video (Intel i5-1035G1, no GPU): SlowFast 75 s, X3D-M 44.8 s, pruned + KD 36.2 s.

**Limitations.** Each setting was trained once (no seeds or confidence intervals). MECCANO is long-tailed: mean per-class accuracy is 12.01% for X3D-M and 10.95% for pruned + KD, so most of the Top-1 accuracy comes from a few frequent classes.

## Method

```
train baseline  ->  structured pruning  ->  fine-tune with KD  ->  measure params / FLOPs / CPU time
 (X3D-M)           (Torch-Pruning,          (SlowFast teacher,
                    isomorphic, L1, 25%)     alpha = 0.5, T = 4)
```

| Stage | Entry point | Config |
| --- | --- | --- |
| Train / test a model | `slowfast/tools/run_net.py` | `slowfast/configs/meccano/X3D_M.yaml` (also `SLOWFAST_8x8_R50`, `I3D_8x8_R50`, `X3D_L`, `3DMobileNet`) |
| Prune | `slowfast/tools/prune_net.py` | `slowfast/configs/meccano/pruning/X3D_M_Pruned.yaml` |
| Distill | `slowfast/tools/distill_net.py` | `slowfast/configs/meccano/distillation/SlowFast_to_X3D_M_v4.yaml` (final pruned + KD setup: alpha 0.5, T 4, batch 12, 180 epochs) |
| Params / FLOPs / CPU time | `slowfast/tools/calculate.matrices.py` | any of the above |

The other `SlowFast_to_X3D_M*.yaml` files are the alpha sweep (alpha = 1.0, 0.9, 0.5). `slowfast/tools/pipeline.py` is an unfinished stub for chaining the stages; for now, run them one after another.

## Installation

Requirements: Python 3.12, GCC >= 4.9, and the packages in `requirements.txt`. Detectron2 and PyTorchVideo must be built from source.

```bash
pip install -r requirements.txt

git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

git clone https://github.com/facebookresearch/pytorchvideo.git pyTorchVideo_repo
pip install -e pyTorchVideo_repo
```

## Usage

The MECCANO dataset is not included; request access from the [dataset authors](https://github.com/fpv-iplab/MECCANO). The config files still contain absolute paths from my machine (`DATA.PATH_TO_DATA_DIR`, checkpoint and output directories). Override them on the command line or edit them before running.

```bash
cd slowfast
export PYTHONPATH=$(pwd):$PYTHONPATH

# 1. Train the baseline
python tools/run_net.py --cfg configs/meccano/X3D_M.yaml \
  DATA.PATH_TO_DATA_DIR /path/to/meccano_rgb_frames OUTPUT_DIR /path/to/out/x3d_m

# 2. Prune (set PRUNING.CHECKPOINT_FILE_PATH and OUTPUT_DIR in the config first)
python tools/prune_net.py --cfg configs/meccano/pruning/X3D_M_Pruned.yaml

# 3. Fine-tune the pruned model with KD
#    (set DISTILLATION.TEACHER_CFG_FILE and DISTILLATION.STUDENT_CHECKPOINT in the config first;
#     STUDENT_CHECKPOINT is the L1 isomorphic pruned model produced in step 2. The default path in
#     the config comes from my local runs.)
python tools/distill_net.py --cfg configs/meccano/distillation/SlowFast_to_X3D_M_v4.yaml
```

## Configs and checkpoints

Final configs and trained checkpoints used for the thesis results: *[link to be added]*

| Model | Config | Checkpoint |
| --- | --- | --- |
| SlowFast R50 (teacher) | `SLOWFAST_8x8_R50_MECCANO.yaml` | `SLOWFAST_8x8_R50_RGB_MECCANO.pyth` |
| X3D-M baseline | `X3D_M.yaml` | `x3d_baseline.pyth` |
| X3D-M pruned + KD | `X3D_M_Pruned_KD.yaml` | `x3d_pruning_KD.pyth` |

## Roadmap

- Dockerfile and a pinned environment
- A working end-to-end pipeline script and smoke tests
- Architecture diagram
- Class-balanced training for the long-tailed classes

## Citation

If you use this code, please cite the PDNet paper (see [PDNet](https://github.com/mscmalekbaba/PDNet)).

## Acknowledgements

[facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast), [fpv-iplab/MECCANO](https://github.com/fpv-iplab/MECCANO), [Torch-Pruning](https://github.com/VainF/Torch-Pruning).

## License

MIT
