#!/bin/bash

ckpt_path=${1:-'checkpoint_gigaref.pt'}
result_path=${2:-'tmp_eval_rouge'}
ratio=${3:-0.25}
decoding_algorithm=${4:-'seq_map'}
bs_k=${5:-10}
bs_kv=${6:-5}


bin_path=data-bin/gigaword_ref
rerank_method=final_bert_v3
rerank_lambda=0.0
subset=test

# grep ^H $tmp_path/dat_giga10.txt | cut -f 3 > $tmp_path/dat_giga10.H.tmp_
# grep ^T $tmp_path/dat_giga10.txt | cut -f 2 > $tmp_path/dat_giga10.T.tmp_
# grep ^S $tmp_path/dat_giga10.txt | cut -f 2 > $tmp_path/dat_giga10.S.tmp_

# rouge -f $tmp_path/dat_giga10.H.tmp_ $tmp_path/dat_giga10.T.tmp_ --avg

tmp_path=$result_path
mkdir -p $tmp_path
# rm $tmp_path/*
output_file_name=tmp_gen.txt


data_dir=$bin_path
average_checkpoint_path=$ckpt_path # ckpt_summary/checkpoints_giga10_ctc_by_rouge/checkpoint_best.pt

batch_size=2048


if [ $decoding_algorithm == "seq_map_no_rerank" ]; then
    python fairseq_cli/generate.py ${data_dir} \
        --gen-subset $subset --user-dir fs_plugins --task translation_ctc_rouge \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
        --remove-bpe --seed 0 \
        --model-overrides "{\"decode_strategy\": \"length_control_bs\", \"specified_length_ratio\": ${ratio}, \"not_force_length\": False, \"minimal_target_length\": 2, \"decode_max_batchsize\": 200, \"length_beam_K\": ${bs_k}, \"length_beam_KV\": ${bs_kv}}" \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens $batch_size \
        --path ${average_checkpoint_path} > $tmp_path/$output_file_name
elif [ $decoding_algorithm == "joint_map" ]; then
    python fairseq_cli/generate.py ${data_dir} \
        --gen-subset test --user-dir fs_plugins --task translation_ctc_rouge \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
        --remove-bpe --seed 0 \
        --model-overrides "{\"decode_strategy\": \"length_control\", \"specified_length_ratio\": ${ratio}, \"not_force_length\": False, \"minimal_target_length\": 2, \"decode_max_batchsize\": 256}" \
        --max-tokens $batch_size \
        --skip-invalid-size-inputs-valid-test \
        --path ${average_checkpoint_path} > $tmp_path/$output_file_name
elif [ $decoding_algorithm == "ctc" ]; then
    python -m debugpy --listen 5678 fairseq_cli/generate.py ${data_dir} \
        --gen-subset test --user-dir fs_plugins --task translation_ctc_rouge \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
        --remove-bpe --seed 0 \
        --model-overrides "{\"lc_beam_size\": 6, \"margin_criteria\": \"max\", \"specified_length_ratio\": True, \"specified_length_fixed\": ${ratio} }" \
        --max-tokens $batch_size \
        --skip-invalid-size-inputs-valid-test \
        --path ${average_checkpoint_path} > $tmp_path/$output_file_name

elif [ $decoding_algorithm == "seq_map" ]; then
    python  fairseq_cli/generate_simplified.py ${data_dir} \
        --gen-subset test --user-dir fs_plugins --task translation_ctc_rouge \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
        --remove-bpe --seed 0 \
        --model-overrides "{\"decode_strategy\": \"length_control_bs\", \"specified_length_ratio\": ${ratio}, \"not_force_length\": False, \"minimal_target_length\": 2, \
            \"decode_max_batchsize\": 256, \"length_beam_K\": ${bs_k}, \"length_beam_KV\": ${bs_kv}, \"iter_decode_with_external_reranker\": True, \
            \"rerank_method\": \"${rerank_method}\", \"rerank_lambda\": ${rerank_lambda}}" \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens $batch_size \
        --path ${average_checkpoint_path} > $tmp_path/$output_file_name
elif [ $decoding_algorithm == "at" ]; then
    python -m debugpy --listen 5678 fairseq_cli/generate_simplified.py ${data_dir} \
        --gen-subset test --user-dir fs_plugins --task translation_rouge \
        --beam 4 \
        --remove-bpe --seed 0 \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens $batch_size  \
        --path ${average_checkpoint_path} > $tmp_path/$output_file_name
fi

# make if condition for decoding_algorithm is "at" or "nat"
if [ $decoding_algorithm == "at" ]; then
    echo "Truncate the output for AT models"
    python my_scripts/eval_naus_rouge/extract_output_truncate.py $tmp_path/$output_file_name ${ratio}
else
    python my_scripts/extract_output.py $tmp_path/$output_file_name
fi


python my_scripts/eval_naus_rouge/eval_rouge_naus.py $tmp_path/${output_file_name}.H $tmp_path/${output_file_name}.T  2>&1 | tee $tmp_path/final_score.txt

# example bash eval_rouge.sh data-bin/gigaword_10 checkpoints/ckpt_summary/checkpoints_giga10_ref_valid_dat_by_len10_rouge_noglat_drop0.3.pt 
# 

