# Retention Score: Quantifying Jailbreak Risks for Vision Language Models

This is the official implementation of the paper "Retention Score: Quantifying Jailbreak Risks for Vision Language Models", accepted at AAAI 2025.

## Table of Contents
- [Code Explanation](#code-explanation)
- [Detailed Implementation for Models and Dataset](#detailed-implementation-for-models-and-dataset)
- [Evaluation Settings](#evaluation-settings)
- [Reference](#reference)

## Code Explanation

- `generation_code/*`: Contains utilities and model descriptions for generating adversarial examples.
- `evaluation_code/*`: Contains the evaluation code for assessing the robustness of the generated examples.
- `minigpt_adversarial_generation.py`: Script to create adversarial images for any images in the specified directory.
- `minigpt_real_our.py`: Generates responses for each prompt and image using the specified datasets.
- `get_metric.py`: Utilizes the Perspective API to generate evaluation metrics.
- `cal_score_acc.py`: Computes the retention score and ASR (Adversarial Success Rate).
- `gemini_evaluation.py` and `gpt4v_evaluation.py`: Scripts for API evaluation.

## Detailed Implementation for Models and Dataset

1. Create two directories named `samples` and `models` to store generated samples and robust models.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   This will install all necessary dependencies for the project.
3. To generate samples, follow these steps:
   - Use the diffusion generator to generate images and save them into the specified directory.
   - Refer to the provided examples for generating adversarial images.
   - Use the specified datasets for generating responses and save the results to a JSONL file.

## Evaluation Settings

1. **Image Evaluation**:
   - Use the `minigpt_adversarial_generation.py` to create adversarial images.
   - Use `get_metric.py` with the Perspective API to generate evaluation metrics.
   - Use `cal_score_acc.py` to compute the retention score and ASR.

2. **Text Evaluation**:
   - Use the paraphrasing model to get paraphrased prompts for adversarial behavior.
   - Generate responses using the specified scripts and evaluate the scores for each response.

You can modify the sample size in each sub-setting to change the number of samples for evaluation.

## Reference

```bibtex
@misc{li2024retentionscorequantifyingjailbreak,
      title={Retention Score: Quantifying Jailbreak Risks for Vision Language Models}, 
      author={Zaitang Li and Pin-Yu Chen and Tsung-Yi Ho},
      year={2024},
      eprint={2412.17544},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.17544}, 
}
