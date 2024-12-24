# Here we would like to give the example code in evaluation_code directory and generation_code directory.
# In generation_code directory, we provide the utils and model descriptions.
# In evaluation_code directory, we provide the evaluation code 



# To reimplement the experiments, we divide into 2 parts. we construct our code based on https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
## Image: We here use minigpt4 model as example to show, for other two models, please replace the name.
### 1: Use diffution generator to generate the images and save into directory. 
### 2: Use minigpt_adversarial_generation.py to create adversarial images for any images in the directory.
### 3. Use minigpt_real_our.py to generate the responses for each prompt and images. Here we refers to use https://allenai.org/data/real-toxicity-prompts as our prompts datasets，hence save the results to jsonl file
### 4. Use get_metric.py with perspective api to generate evaluation
### 5. Using cal_score_acc.py to get the retention score and ASR.
### 6: For API evaluation, use gemini_evaluation.py and gpt4v_evaluation.py



## Text:
### 1: Use paraphrasing model to get paraphrased prompts for adv_behavior in AdvBench https://github.com/llm-attacks/llm-attacks , generate a plain noise image
### 2: Use minigpt_plain_noise.py to generate responses for the step 1 image and prompts. 
### 3: use AutoDAN https://github.com/SheltonLiu-N/AutoDAN to generate adv prompts in step 1
### 4: use minigpt_noise_attack.py to generate adv responses.
### 5. Use adv_bench_evaluation_llama_70B.py to evaluate the score for each responses.
### 6. Use adv_bench_keyword_evaluation.py to get ASR
### 7. use extract_score to calculate the Retention Score.
### 8. For pure LLM, please refers to test_transformer.py and replace the model and tokennizer to vicuna-v0, v1.1, llama-13B, and use similar step for the ASR and score.