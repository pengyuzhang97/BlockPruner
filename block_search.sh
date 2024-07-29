export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

model_name=llama2-7b
nsamples=256
dataset=alpaca
block_num=20
block_type=mix

python block_search.py \
        --model-path ../llm_model/${model_name} \
        --block-type ${block_type} \
        --cal-nsamples ${nsamples} \
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-path ppls \
        --ppl-eval-batch-size 2 \
        --device cuda 