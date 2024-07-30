export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com


ori_model_in_ppls=Llama-2-7b
local_model_name=llama2-7b
block_num=17
dataset=alpaca
ppl_search_file=ppls/${local_model_name}_mix_alpaca_ns_256_del_order_list.json


python eval.py \
        --do-eval \
        --model-path ../llm_model/${local_model_name}\
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-file ${ppl_search_file}\
        --ppl-eval-batch-size 10 \
        --device cuda \
        --compute-dtype bf16 