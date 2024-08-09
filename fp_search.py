from typing import Optional, Tuple
import argparse
import json
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed;

set_seed(42)
import utils
import random

from utils import zo_step_for_acc_grad

from peft import prepare_model_for_kbit_training

import copy

from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bf16",
        help="Data type for computation ('bf16', 'fp32', 'fp64').",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default='../llm_model/llama2-7b',
        help="Path to load the model and tokenizer",
    )
    parser.add_argument(
        "--ppl-search-path",
        type=str,
        help="Path to save the perplexity search results.",
        default="fp_ppls",
    )
    parser.add_argument(
        "--del-block-num",
        type=int,
        help="Number of blocks to delete.",
        default=0,
    )
    parser.add_argument(
        "--block-type",
        type=str,
        help="Block type for searching ('mha', 'mlp', 'mix').",
        choices=["mha", "mlp", "mix"],
        default="mix",
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset for calibration.",
        choices=["wikitext2", "alpaca"],
        default="alpaca",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples for calibration.",
        default=256,
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    return parser.parse_args()


def get_model_config(load_in_8bit = False, load_in_4bit = True):
    if load_in_8bit and load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit
        )
        # Copy the model to each device
        # device_map = {"": Accelerator().local_process_index}
        device_map = 'auto'
        # device_map = 'sequential'
        torch_dtype = torch.bfloat16
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        # device_map = {"": Accelerator().local_process_index}
        device_map = 'auto'
        # device_map = 'sequential'
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = torch.bfloat16
    return device_map, quantization_config, torch_dtype



class MaskedLlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mask_block = ""

    def setting_layer(self, layer):
        if "mha" not in self.mask_block:
            self.input_layernorm = layer.input_layernorm
            self.self_attn = layer.self_attn
        else:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" not in self.mask_block:
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.mlp = layer.mlp
        else:
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if "mha" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None
            present_key_value = None

        if "mlp" not in self.mask_block:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def get_model_params(model):
    return sum(int(p.nelement()) for p in model.parameters())


@torch.no_grad
def block_search_by_ppl(args, model, test_loader=None, model_size=None):
    # Initialize best results dictionary
    best_results = {}

    # Split blocks into MHA and MLP lists
    mha_block_ids = list(range(
        model.config.num_hidden_layers)) if args.block_type != "mlp" else []  # You can use BI to reduce the search space if needed
    mlp_block_ids = list(range(model.config.num_hidden_layers)) if args.block_type != "mha" else []

    logging.info(f"mha_block_ids: {mha_block_ids}")
    logging.info(f"mlp_block_ids: {mlp_block_ids}")

    # iterate search process
    current_sequence = set()
    current_ppl = float('inf')

    pbar = tqdm(range(1, args.del_block_num + 1), desc=f"searching block del order based on {args.cal_dataset} ppl")
    for del_num in pbar:
        best_candidate = None
        best_candidate_ppl = float('inf')

        candidate_blocks = [("mha", mha_id) for mha_id in mha_block_ids if ("mha", mha_id) not in current_sequence] \
                           + [("mlp", mlp_id) for mlp_id in mlp_block_ids if ("mlp", mlp_id) not in current_sequence]

        for block_type, block_id in candidate_blocks:
            candidate_sequence = frozenset(current_sequence) | {(block_type, block_id)}
            del_layer_dict = apply_block_masks(model, candidate_sequence)
            candidate_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
            revert_block_masks(model, del_layer_dict)

            if candidate_ppl < best_candidate_ppl:
                best_candidate_ppl = candidate_ppl
                best_candidate = candidate_sequence

        if best_candidate is not None:
            current_sequence = best_candidate
            current_ppl = best_candidate_ppl

        del_order_list = list(current_sequence)
        best_results[str(del_num)] = sorted(del_order_list, key=lambda x: x[1], reverse=False)

        print(f"best_ppl: {current_ppl}")
        print(f"best_seq ({del_num}): {sorted(del_order_list, key=lambda x: x[1], reverse=False)}")

    file_name = f"{args.ppl_search_path}/{args.model_path.split('/')[-1]}_{args.block_type}_{args.cal_dataset}_ns_{args.cal_nsamples}_del_order_list.json"
    with open(file_name, "w") as f:
        json.dump(best_results, f)
    logging.info(f"del_order_list path: {file_name}")



def fp_search_by_is( args, model, test_loader=None, model_size=None, load_in_8bit=False, load_in_4bit=True,
                     elem_imp=True, matrix_imp=False, lower_threshold = 5, upper_threshold = 29):
    if load_in_8bit or load_in_4bit:  # this line will set parameters to False
        #         1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        #         head to fp32
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    import bitsandbytes.functional as bnbF

    global_step = 0
    module_name_list = ['v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']
    named_grads_to_store = {}
    named_parameters_to_optim = {}

    for name, param in model.named_parameters():
        if any(ta_name in name for ta_name in module_name_list):
            ori_weight_bf16 = bnbF.dequantize_nf4(param.data, param.quant_state)
            # weight_fp = torch.empty((4096, 11008), dtype=torch.bfloat16, device= m.weight.data.device)
            # bnbF.dequantize_nf4(m.weight.data, m.weight.quant_stat, out= weight_fp)
            # named_grads_to_store.append((name, torch.empty(shape, dtype=torch.bfloat16, device=m.weight.data.device)))
            named_grads_to_store[name] = torch.empty(ori_weight_bf16.size(), dtype=ori_weight_bf16.dtype, device=ori_weight_bf16.device)
            del ori_weight_bf16

        if 'self_attn' in name or 'mlp' in name:
            named_parameters_to_optim[name] = param


    for step, inputs in (enumerate(tqdm(test_loader))):

        # Note that loss is averaged on the batch size
        loss, named_grads_to_store = zo_step_for_acc_grad(model, inputs, global_step, module_name_list, named_parameters_to_optim , named_grads_to_store, len(test_loader))


        # loss = model_instance.training_step(model, inputs)  # backward is conducted in this line

        global_step += 1


    # compute score
    target_dtype = torch.bfloat16
    elem_score_dict_sum = {}
    score_dict_mul = {}
    mat_score_dict_sum = {}


    score_method = 'sum'  # TODO: try normalized group
    for name, param in model.named_parameters():
        if any(ta_name in name for ta_name in module_name_list):
            if elem_imp:
                if score_method == 'sum':
                    elem_score_dict_sum[name] = torch.prod( torch.abs( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype) * named_grads_to_store[name].to(target_dtype) ))
                    del named_grads_to_store[name]
                    torch.cuda.empty_cache()
                elif score_method == 'all':
                    elem_score_dict_sum[name] = torch.sum( torch.abs( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype) * named_grads_to_store[name].to(target_dtype) ))
                    score_dict_mul[name] = torch.prod(  torch.abs( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype) * named_grads_to_store[name].to(target_dtype) ) )
                    del named_grads_to_store[name]
                    torch.cuda.empty_cache()
            elif matrix_imp:
                mat_score_dict_sum[name] = torch.abs( torch.trace( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype).T @ named_grads_to_store[name].to(target_dtype)) )
                del named_grads_to_store[name]
                torch.cuda.empty_cache()


            # ori_weight_bf16 = bnbF.dequantize_nf4(param.data, param.quant_state)

    # keys filter
    if elem_imp:
        keys = list(elem_score_dict_sum.keys())
        values = list(elem_score_dict_sum.values())
    elif matrix_imp:
        keys = list(mat_score_dict_sum.keys())
        values = list(mat_score_dict_sum.values())

    import re
    pattern = re.compile(r'model\.layers\.(\d+)\..*\.(.*)\.weight')

    def simplify_key(key):
        match = pattern.match(key)
        if match:
            layer_index = match.group(1)
            module_name = match.group(2)
            return f'{layer_index}.{module_name}'
        return key

    simplified_keys = [simplify_key(key) for key in keys]
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=False)
    sorted_simplified_keys = [simplified_keys[i] for i in sorted_indices]

    modified_keys_list = []
    for item in sorted_simplified_keys:
        if 'v_proj' in item or 'o_proj' in item:
            modified_keys_list.append(item.replace('v_proj', 'mha').replace('o_proj', 'mha'))
        else:
            modified_keys_list.append(item.replace('up_proj', 'mlp').replace('gate_proj', 'mlp').replace('down_proj', 'mlp'))

    filtered_list = []
    seen = set()

    # lower_threshold = 5
    # upper_threshold = 29

    for item in modified_keys_list:
        layer_index = int(item.split('.')[0])
        if lower_threshold <= layer_index <= upper_threshold and item not in seen:
            filtered_list.append(item)
            seen.add(item)


    final_dict = {}
    temp_list = []
    num = 1
    for item in filtered_list:
        layer_index = int(item.split('.')[0])
        layer_type = item.split('.')[1]
        module = [layer_type, layer_index]
        temp_list.append(module)
        final_dict[str(num)] = copy.deepcopy(temp_list)
        num += 1

    for key in final_dict:
        final_dict[key] = sorted(final_dict[key], key=lambda x: x[1], reverse=False)


    import json
    import os

    file_name =  f"{args.ppl_search_path}/{args.model_path.split('/')[-1]}_{args.block_type}_{args.cal_dataset}_ns_{args.cal_nsamples}_del_order_list.json"

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(final_dict, f)
    # logging.info(f"del_order_list path: {file_name}")

    # print(final_dict)


def apply_block_masks(model, seq):
    del_layer_dict = {}
    for block_type, block_id in seq:
        chosen_layer = model.model.layers[block_id]
        if isinstance(chosen_layer, MaskedLlamaDecoderLayer):
            chosen_layer.mask_block += block_type
            chosen_layer.setting_layer(del_layer_dict[str(block_id)])
        else:
            new_layer = MaskedLlamaDecoderLayer()
            new_layer.mask_block += block_type
            new_layer.setting_layer(chosen_layer)
            del_layer_dict[str(block_id)] = chosen_layer
            model.model.layers[block_id] = new_layer
    return del_layer_dict


def revert_block_masks(model, del_layer_dict):
    for k, v in del_layer_dict.items():
        layer_id = int(k)
        model.model.layers[layer_id] = v


def main() -> None:
    args = parse_args()
    logging.info(args)
    logging.info(f"PyTorch device: {args.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    if args.compute_dtype == "bf16":
        compute_dtype = torch.bfloat16
    elif args.compute_dtype == "fp32":
        compute_dtype = torch.float32
    elif args.compute_dtype == "fp64":
        compute_dtype = torch.float64
    else:
        raise NotImplementedError("Unsupported compute type.")

    device_map, quantization_config, torch_dtype = get_model_config()

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 quantization_config=quantization_config,
                                                 torch_dtype=compute_dtype, trust_remote_code=True,
                                                 device_map="auto", use_cache=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model_size = get_model_params(model)
    logging.info(f"original model size: {model_size / 1e9:.3f}B")

    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    sampled_test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), args.cal_nsamples))
    # print(len(sampled_test_dataset))
    test_loader = utils.prepare_test_dataloader(
        dataset=sampled_test_dataset,
        tokenizer=tokenizer,
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size
    )
    # print(test_loader)

    fp_search_by_is(args, model, test_loader=test_loader, model_size=None, load_in_8bit=False, load_in_4bit=True,
                     elem_imp=True, matrix_imp=False, lower_threshold = 5, upper_threshold = 29)

    # block_search_by_ppl(args, model, test_loader, model_size)


if __name__ == "__main__":
    main()
