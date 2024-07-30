export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

model_name=llama2-7b
nsamples=256
dataset=alpaca


python fp_search.py \
        --model-path ../llm_model/${model_name} \
        --cal-nsamples ${nsamples} \
        --cal-dataset ${dataset} \
        --ppl-search-path fp_ppls \
        --ppl-eval-batch-size 2 \
        --device cuda 