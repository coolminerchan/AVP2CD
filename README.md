# AVP2CD

AVP2CD (Anatomical Visual Perception to Clinical Diagnosis) is a hierarchical post-training framework for drafting lumbar MRI reports from radiologist-selected sagittal T1-weighted and T2-weighted slices. It adapts Qwen2.5-VL-3B-Instruct through anatomical grounding, diagnostic visual question answering (VQA), and report-generation refinement.

**Paper.** [From anatomy to diagnosis: A hierarchical post-training framework for lumbar MRI report generation with limited multimodal data](https://doi.org/10.1016/j.array.2026.101139), *Array* 31 (2026), 101139.

## Code

| Entry point or module | Purpose |
| --- | --- |
| `sft.py` | Supervised fine-tuning (SFT) for Stage 2 diagnostic VQA and Stage 3 report generation |
| `dapo.py`, `rl_training.py` | DAPO-style reinforcement learning from the SFT checkpoints |
| `reward_fn.py` | Clinical entity and BERTScore components of the reinforcement-learning reward |
| `SFT_dataloader.py` | Case loading and construction of image–text prompts |
| `model_evaluation.py` | Local checkpoint evaluation and aggregate metric calculation |

The training workflow starts from an existing Stage 1 anatomy-VQA checkpoint. Stage 1 training and preprocessing code and the study checkpoints are outside the scope of this release. The hospital MRI images, reports and case-level outputs are confidential. The paper describes the public [SPIDER dataset](https://doi.org/10.5281/zenodo.8009679) used for anatomical supervision.

## Setup

Use Python 3.11 and install PyTorch and torchvision builds that match your CUDA version. The model training and BERTScore reward require a CUDA GPU. Run the commands below from the repository root in a dedicated Python environment.

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet
```

## Inputs

Prepare independent `train`, `val`, and `test` folders for each stage and keep each case in the same split across stages. Each split contains one folder per case. Each case folder holds `report.json` and its selected MRI slices.

| Stage | Images | `report.json` fields |
| --- | --- | --- |
| 2, diagnostic VQA | `T1_slice.png`, `T2_slice.png`, or both | `Age`, `Discs`, `Vertbras`, `DegenerativeChanges`, `OtherSystems`, `GeneralFindings` |
| 3, two-section report | Both `T1_slice.png` and `T2_slice.png` | `Age`, `Description`, `Impression` |

The data loader defines the annotation fields and prompt construction. Preserve the `Vertbras` field spelling expected by the loader.

## Run

Replace the example checkpoint and dataset paths with your local paths.

```bash
python sft.py --pipeline stage2_stage3 --model-dir checkpoints/stage1 --stage2-data-root data/stage2 --stage3-data-root data/stage3

python dapo.py --pipeline stage2_stage3 --stage2-model-folder SFT_stage2_vqa --stage3-model-folder SFT_stage3_report --stage2-data-root data/stage2 --stage3-data-root data/stage3

python model_evaluation.py --model-dir dapo_outputs_stage3 --data-root data/stage3 --mode full --split test --output-json eval/stage3.json
```

Use `--mode partial` and the Stage 2 dataset for diagnostic VQA evaluation. Evaluation writes a case-level JSON with predictions and reference text, plus a separate aggregate `.summary.json`. Keep these files with your local study data.

## Citation

```bibtex
@article{zuo2026avp2cd,
  author  = {Zuo, Qunyang and Wang, Zifan and Li, Jiayi and Guan, Yifan and Chen, Jianjun and Yang, Chuanhong and Fu, Zhihui},
  title   = {From anatomy to diagnosis: A hierarchical post-training framework for lumbar MRI report generation with limited multimodal data},
  journal = {Array},
  year    = {2026},
  volume  = {31},
  pages   = {101139},
  doi     = {10.1016/j.array.2026.101139},
  url     = {https://doi.org/10.1016/j.array.2026.101139}
}
```
