


















"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""


alpaca_template_no_Instruct = """{}. ### Response: {}{}"""



vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'alpaca_no_instruct': (alpaca_template_no_Instruct, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),

}


def get_formatting_prompts_func(template_name, eos_token):

    overall_temp, response_temp = TEMPLATE_DICT[template_name]

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)
            output_texts.append(text)
        return output_texts

    def formatting_prompts_func_4_alpaca_no_instruct(example):
        output_texts = []
        for i in range(len(example['text'])):
            text = overall_temp.format(example['text'][i], example['output'][i], eos_token)
            output_texts.append(text)
        return output_texts

    if template_name == 'alpaca_no_instruct':
        return formatting_prompts_func_4_alpaca_no_instruct, response_temp
    else:
        return formatting_prompts_func, response_temp


