# **A Decoding Algorithm Based on Directed Acyclic Transformers for Length-Control Summarization**

> This repo contails code, model outputs, training/evaluation scripts, and additional results of our paper. 

<!--
## Evaluaiton with ChatGPT

In addition to the results reported in the paper, we also compare our Seq-MAP with three baseline models using [ChatGPT:gpt-3.5-turbo-0301](https://chat.openai.com/). Specifically, we compare our Seq-MAP with each of the baseline model. For each comparison, we give the ChatGPT the source text, the hypotesis of each of the models, and ask which one is better, or cannot decide. To reduce the impacts of the promts, our results are based on two prompts (listed the files under `prompts` folder). The results are shown in the following table.
-->


# Replication
## Environment setup
Our code based on the [FairSeq](https://github.com/facebookresearch/fairseq) library. Our code is written in PyTorch, and uses CUDA for GPU acceleration.
To install the required packages, run the following command:
```bash
# Under `code` dir
bash env_setup.sh
```

## Data
Our data is directly obtained from [NAUS](https://github.com/MANGA-UOFA/NAUS), here we only need the `gigaref` dataset. To download the data, run the following command:
```bash
gdown  https://drive.google.com/uc\?id\=1H0csbbv-1L5In9ebJjhk1GamrN8ysnbZ
```
After downloading the data, unzip it and put it under the `datasets/data` folder. Where you will have a `datasets/data/gigaword_ref` folder.

To preprocess the data, run the following command:
```bash
bash datasets/preprocess.sh
```
Where a binarized version of the data will be saved under the `code/data-bin/gigaword_ref` folder.

## Training
The training is conduced under the `code` folder, where the `trainer_with_config.sh` file will convert a config file under `code/config_summary` to a FairSeq training command. 

### Training DAT model
```bash
# Under `code` dir
bash trainer_with_config.sh config_summary/train/dat.yaml
```

### Training Seq-MAP model
**NOTE**: The Seq-MAP method is mainly implemented [here](https://github.com/emnlp2023anonymous/DAT-LC/blob/65340fac16ce9e122112a69c6f89ef955cbf6989/code/fs_plugins/models/glat_decomposed_with_link.py#L1061).
```bash
# Under `code` dir
bash trainer_with_config.sh config_summary/reranker/dat_reranker_decoder_rouge_20_5.ymal
```

Note that you need change the `finetune_from_model` in the configure file according to the path of a trained DAT model.

*Unfortunately, we cannot provide the trained DAT model without comprising annonymity. This is because of the size limit of Github.*

Under `code/config_summary`, we also provide the configure files for training the baseline models, different settings of the Seq-MAP, and ablation studies.


## Evaluation
```bash
# Under `code` dir
checkpoint_path=path_to_the_trained_model
result_path=path_to_save_the_results
ratio=0.25
decoding_algorithm=seq_map 
K_m=20
K_v=5

bash eval_rouge.sh $checkpoint_path $result_path $ratio $decoding_algorithm $K_m $K_v 
```

The supported decoding algorithms are: `seq_map`, `seq_map_no_rerank`, `joint_map`, `ctc`, `at`.

`ratio` is the length ratio of the summary to the source text. /