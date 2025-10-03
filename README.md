# <p align="center">Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation</p>
## Introduction
This repository contains the code for the paper [Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation](https://arxiv.org/pdf/2412.13375)
## Abstract
Large language models (LLMs) have made great progress in classification and text generation tasks. However, they are mainly trained on English data and often struggle with low-resource languages. In this study, we explore adding a new language, i.e., Persian, to Llama (a model with a limited understanding of Persian) using parameter-efficient fine-tuning. We employ a multi-stage approach involving pretraining on monolingual Persian data, aligning representations through bilingual pretraining and instruction datasets, and instruction-tuning with task-specific datasets. We evaluate the model's performance at each stage on generation and classification tasks. Our findings suggest that incorporating the Persian language, through bilingual data alignment, can enhance classification accuracy for Persian tasks, with no adverse impact and sometimes even improvements on English tasks. Additionally, the results highlight the model's initial strength as a critical factor when working with limited training data, with cross-lingual alignment offering minimal benefits for the low-resource language. Knowledge transfer from English to Persian has a marginal effect, primarily benefiting simple classification tasks.

## Pipeline Overview

### 1. Train a SentencePiece Model and Extract Extra Vocabulary
Train a SentencePiece model and extract additional vocabulary based on your training file.

Run the following file: ```tokenizer/extract_vocab.py```

### 2. Merge the Extracted Vocabulary with the Original Model Tokenizer
Merge the newly extracted vocabulary with the original tokenizer.

Run the following file: ```tokenizer/merge.py```

### 3. Pretrain the Model
Start model pretraining with the option to:

1. Train the full model.
2. Train using LoRA (Low-Rank Adaptation).
3. Freeze all parameters except model heads and embeddings.


Run the following file: ```pretrain/run_clm_pt_with_peft.py```

**Options:**
- `--use_lora`: Enable LoRA training.
- `--freeze_transformer`: Freeze all model parameters except heads and embeddings.
- **For training LLaMA models**, do not forget to include the `--llama` argument.

### 4. Instruct-Tune the Model
Fine-tune the model using supervised fine-tuning (SFT) for instruction-based tasks.

Run the following file: ```sft/run_clm_sft_with_peft.py```

**Note:** For training LLaMA models, include the `--llama` argument.

### 5. Test the Model Performance
Evaluate the model's performance using inference scripts.

Run the following file: ```inference/inference.py```

## Slurm Bash Scripts
To simplify execution, slurm bash scripts for running the codes are also provided. These scripts include example commands and parameter configurations for each stage of the pipeline.

Example:
```
sbatch pretrain/run_pt.sh
sbatch sft/run_sft.sh
sbatch inference/inference.sh
```
## Citation
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{mahdizadeh-sani-etal-2025-extending,
    title = "Extending {LLM}s to New Languages: A Case Study of Llama and {P}ersian Adaptation",
    author = "Mahdizadeh Sani, Samin  and
      Sadeghi, Pouya  and
      Vu, Thuy-Trang  and
      Yaghoobzadeh, Yadollah  and
      Haffari, Gholamreza",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.594/",
    pages = "8868--8884"
}
```
