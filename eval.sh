export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com


ori_model_in_ppls=Llama-2-7b
local_model_name=llama2-7b
block_num=17
dataset=alpaca
ppl_search_file=ppls/${local_model_name}_mix_alpaca_ns_256_del_order_list.json              # baseline 17 for 21.99% 61.64 avg
#ppl_search_file=fp_ppls/${local_model_name}_mix_alpaca_ns_256_del_order_list.json
#ppl_search_file=fp_ppls/${local_model_name}_mix_alpaca-sft_ns_2560_del_order_list.json
#ppl_search_file=fp_ppls/llama2-7b_mix_alpaca-sft_ns_256_lwth_0_upth_31_elem_False_mat_True_del_order_list.json    # 3588
#ppl_search_file=fp_ppls/llama2-7b_mix_alpaca-sft_ns_256_lwth_3_upth_29_elem_False_mat_True_del_order_list.json   # 4993   24
#ppl_search_file=fp_ppls/llama2-7b_mix_alpaca-sft_ns_256_lwth_5_upth_25_elem_False_mat_True_del_order_list.json   # 4938   18-25.01
#ppl_search_file=fp_ppls/llama2-7b_mix_alpaca-sft_ns_256_lwth_5_upth_25_elem_False_mat_True_del_order_list.json   # 5152   16-22
#ppl_search_file=fp_ppls/llama2-7b_mix_alpaca_ns_256_del_order_list.json

python eval.py \
        --do-eval\
        --model-path ../llm_model/${local_model_name}\
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-file ${ppl_search_file}\
        --ppl-eval-batch-size 10 \
        --device cuda \
        --compute-dtype bf16