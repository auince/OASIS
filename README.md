# OASIS

**Open-dimension Assessment System for Image Scoring**

OASIS predicts continuous image-quality scores from natural-language dimension definitions. This repository provides the core two-stage training, inference, and evaluation code, built on Qwen3-VL-8B-Instruct.

## Downloads

| Resource | Baidu Netdisk | Extraction code |
| --- | --- | --- |
| MD61 annotations and image index | [MD61.zip](https://pan.baidu.com/s/1LPc9knROOfbwUPfmeLftdA?pwd=xxqu) | `xxqu` |
| OASIS model weights | [oasis-score.zip](https://pan.baidu.com/s/1BTW59k8MZ-SIm5qBEeHPgg?pwd=i28u) | `i28u` |

MD61 includes annotations, image indices, and dimension definitions. Due to redistribution restrictions on the source datasets, we provide only indices for the images. Image files are obtained separately from their source datasets.

## Installation

Python 3.10 or later is required. The dependencies use PyTorch 2.10.0 with CUDA 12.8.

```bash
git clone https://github.com/auince/OASIS.git
cd OASIS
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Inference

Place the downloaded `full_train_state.pt` and associated model, tokenizer, and processor files in `weights/oasis-score/`.

Create `requests.jsonl` with one JSON object per line:

```json
{"id":"example","image":"example.jpg","definition":{"name":"Sharpness","description":"Assess visible image sharpness.","positive_evidence":"Clear edges and fine detail.","negative_evidence":"Blurred edges and lost detail.","score_guidance":"Higher scores mean greater sharpness."}}
```

```bash
oasis-score \
  --model-dir weights/oasis-score \
  --input requests.jsonl \
  --output predictions.jsonl \
  --codebook configs/score_tokens.json
```

Scores are returned on a 0–100 scale. Image paths are relative to the request file; use `--image-root` to specify a separate image directory.

## Training and Evaluation

| Task | Commands |
| --- | --- |
| Two-stage training and model export | `oasis-train`, `oasis-export` |
| IQA training and evaluation | `oasis-iqa-train`, `oasis-iqa` |
| MD61-Open evaluation | `oasis-mt61-test` |
| Driving, depth, medical, and remote-sensing evaluation | `oasis-downstream` |
| Dataset format conversion | `oasis-data` |

Run any command with `--help` for its arguments. Downstream evaluation uses externally supplied task-model predictions and ground-truth annotations.

## Acknowledgements

OASIS builds on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). See [NOTICE](NOTICE) and the [upstream license](licenses/Qwen3-VL-LICENSE) for attribution and terms.
