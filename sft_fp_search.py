from typing import Optional, Tuple
import argparse
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

from dataclasses import dataclass, field


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


@dataclass
class ScriptArguments:
    # candidate_data = ['TIGER-Lab/MathInstruct', "vicgalle/alpaca-gpt4", 'lucasmccabe-lmi/CodeAlpaca-20k']
    # canddate_model = ['../llm_model/llama2-7b']
    # model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    model_name_or_path: Optional[str] = field(default="../llm_model/llama2-7b", metadata={"help": "the model name"})

    dataset_name: Optional[str] = field(
        default="alpaca-gpt4", metadata={"help": "the dataset name"}
    )
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})

    # lucasmccabe-lmi/CodeAlpaca-20k, "vicgalle/alpaca-gpt4", 'TIGER-Lab/MathInstruct',
    # 'FinGPT/fingpt-sentiment-train', 'medalpaca/medical_meadow_medical_flashcards'


    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})

    optimizer: Optional[str] = field(default='adamw_hf', metadata={"help": "optimizer, sgd, adamw_hf"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate, default 2e-5"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Max Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    single_output_dir: Optional[str] = field(default="single_output", metadata={"help": "the output directory"})

    peft_lora_r: Optional[int] = field(default=4, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=22, metadata={"help": "the alpha parameter of the adapters"})   # original alpha 16

    # target_modules: List = field(default_factory=lambda: ["q_proj","k_proj", "v_proj"])

    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2024, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.0, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})


    single_training_epoch : Optional[int] = field(default=20, metadata={"help": "the number of training epochs"})
    eval_epoch: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


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

def get_single_training_args(script_args, new_lr, w_decay):
    training_args = TrainingArguments(
        output_dir=script_args.single_output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        per_device_eval_batch_size= script_args.batch_size,
        optim=script_args.optimizer,
        learning_rate=new_lr,
        weight_decay=w_decay,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        # max_steps=script_args.max_steps,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        bf16= True
    )
    return training_args



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


# ===== layer selection =====

def compute_score(args, model, local_data, zo_eps, data_collator, tokenizer, formatting_prompts_func, lower_threshold = 5 , upper_threshold = 29):
    import bitsandbytes.functional as bnbF
    from fp_utils.modified_trainer import get_sft_trainer_fp

    training_args = get_single_training_args(script_args, script_args.learning_rate, 0)

    trainer = get_sft_trainer_fp(script_args, model, tokenizer, training_args, local_data,
                              None, formatting_prompts_func, data_collator, zo_eps)
    train_dataloader = trainer.get_train_dataloader()
    # pruner = IterSNIP(generator.masked_parameters(model))


    global_step = 0
    module_name_list = ['v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']
    # module_name_list = [ 'o_proj', 'down_proj']


    # named_grads_to_store = []

    named_grads_to_store = {}

    # for name, param in model.named_parameters():
    #     # if param.requires_grad and any(ta_name in name for ta_name in module_name_list):
    #     #     named_grads_to_store.append((name, torch.zeros_like(param, dtype=param.data.dtype)))
    #     if any(ta_name in name for ta_name in module_name_list):
    #         bnbF.dequantize_nf4(param.data, param.quant_state)
    #         named_grads_to_store.append((name, torch.zeros_like(param, dtype=torch.bfloat16)))

    # for name, m in model.named_modules():   this is for dequantization
        # bnbF.dequantize_nf4(m.weight.data, m.weight.quant_stat, out= weight_fp)
        # named_grads_to_store.append((name, torch.empty(shape, dtype=torch.bfloat16, device=m.weight.data.device)))

    for name, param in model.named_parameters():
        if any(ta_name in name for ta_name in module_name_list):
            ori_weight_bf16 = bnbF.dequantize_nf4(param.data, param.quant_state)
            # weight_fp = torch.empty((4096, 11008), dtype=torch.bfloat16, device= m.weight.data.device)
            # bnbF.dequantize_nf4(m.weight.data, m.weight.quant_stat, out= weight_fp)
            # named_grads_to_store.append((name, torch.empty(shape, dtype=torch.bfloat16, device=m.weight.data.device)))
            named_grads_to_store[name] = torch.empty(ori_weight_bf16.size(), dtype=ori_weight_bf16.dtype, device=ori_weight_bf16.device)
            del ori_weight_bf16

    for step, inputs in (enumerate(tqdm(train_dataloader))):

        # Note that loss is averaged on the batch size
        loss, named_grads_to_store = trainer.zo_step_for_acc_grad(model, inputs, global_step, module_name_list, named_grads_to_store, len(train_dataloader))
        # loss = model_instance.training_step(model, inputs)  # backward is conducted in this line

        global_step += 1


    # compute score
    target_dtype = torch.bfloat16
    elem_score_dict_sum = {}
    score_dict_mul = {}
    mat_score_dict_sum = {}
    elem_imp = False
    matrix_imp = True

    score_method = 'sum'
    for name, param in model.named_parameters():
        if any(ta_name in name for ta_name in module_name_list):
            if elem_imp:
                if score_method == 'sum':
                    elem_score_dict_sum[name] = torch.sum( torch.abs( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype) * named_grads_to_store[name].to(target_dtype) ))
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


    print('done')



    # # for minibatch and uncertainty computation
    # for step, inputs in enumerate(train_dataloader):
    #
    #     loss = trainer.zo_step_for_grad(model, inputs,
    #                                     ipt = {} ,exp_avg_ipt = {}, exp_avg_unc = {},
    #                                     global_step = global_step, beta1=0.85, beta2=0.95)
    #     # loss = model_instance.training_step(model, inputs)  # backward is conducted in this line
    #
    #     global_step += 1


    # compute score
    # # uncertainty
    # is_dict = {}
    # for n, p in model.named_parameters():
    #     if beta2 > 0 and beta2 < 1:
    #         is_dict[n] = exp_avg_ipt[n] * exp_avg_unc[n]
    #     elif beta2 == 1.:
    #         is_dict[n] = exp_avg_ipt[n]
    #     elif beta2 == 2.:
    #         is_dict[n] = exp_avg_ipt[n] * exp_avg_unc.sqrt()
    #     else:
    #         # Handling the uncepted beta2 to default setting
    #         is_dict[n] = exp_avg_ipt[n] * (ipt[n] - exp_avg_ipt[n]).abs()


    # structured sorting






def fp_search_by_is( args, model, test_loader=None, model_size=None, load_in_8bit=False, load_in_4bit=True):


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
    elem_imp = False
    matrix_imp = True

    score_method = 'sum'
    for name, param in model.named_parameters():
        if any(ta_name in name for ta_name in module_name_list):
            if elem_imp:
                if score_method == 'sum':
                    elem_score_dict_sum[name] = torch.sum( torch.abs( bnbF.dequantize_nf4(param.data, param.quant_state).to(target_dtype) * named_grads_to_store[name].to(target_dtype) ))
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

    lower_threshold = 5
    upper_threshold = 29

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

    if script_args.load_in_8bit or script_args.load_in_4bit:  # this line will set parameters to False
        #         1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        #         head to fp32
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ===== Define the tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # following vicuna



    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    sampled_test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), args.cal_nsamples))

    from fp_utils.utils_sft import get_formatting_prompts_func
    from trl import DataCollatorForCompletionOnlyLM

    # ===== Define the formatting function (cater to TRL SFTTrainer)=====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
                            2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


    model_size = get_model_params(model)
    logging.info(f"original model size: {model_size / 1e9:.3f}B")

    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    sampled_test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), args.cal_nsamples))

    zo_eps = 1e-3
    compute_score(args, model, sampled_test_dataset, zo_eps, data_collator, tokenizer, formatting_prompts_func, lower_threshold = 5 , upper_threshold = 29)

    # # print(len(sampled_test_dataset))
    # test_loader = utils.prepare_test_dataloader(
    #     dataset=sampled_test_dataset,
    #     tokenizer=tokenizer,
    #     seqlen=args.ppl_eval_seqlen,
    #     batch_size=args.ppl_eval_batch_size
    # )
    # # print(test_loader)
    #
    # fp_search_by_is(args, model, test_loader=test_loader, load_in_8bit=False, load_in_4bit=True)

    # block_search_by_ppl(args, model, test_loader, model_size)


if __name__ == "__main__":
    main()
