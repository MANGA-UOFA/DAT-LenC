#!/bin/bash


bin_path=code/data-bin/gigaword_ref
data_path=datasets/data/gigaword_ref/

rm -rf $bin_path
fairseq-preprocess \
    --source-lang article --target-lang summary \
    --trainpref $data_path/ train\
    --validpref $data_path/valid_ref \
    --testpref $data_path/test \
    --destdir $bin_path \
    --workers 128 --joined-dictionary \
    --srcdict datasets/data/gigaref.dict


