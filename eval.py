import sys
import json
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, 
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--no_lora', type=bool, default=False)
parser.add_argument('--load_8bit', type=bool, default=False)
parser.add_argument('--base_model', type=str)
parser.add_argument('--lora_weights', type=bool)
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
args = parser.parse_args()

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

print(args)

NO_LORA = args.no_lora
LOAD_8BIT = args.load_8bit
BASE_MODEL = args.base_model
LORA_WEIGHTS = args.lora_weights

if args.model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
elif args.model_type == "bloom":
    tokenizer = BloomTokenizerFast.from_pretrained(BASE_MODEL)
elif args.model_type == "chatglm":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    if args.model_type == "llama":
        device="cuda:1"
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="cuda:1",
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                torch_dtype=torch.float16,
            )
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                torch_dtype=torch.float16,
            )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                torch_dtype=torch.float16,
            )
elif device == "mps":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
else:
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
            )

    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
            )   
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,trust_remote_code=True,
            device_map={"": device}, low_cpu_mem_usage=True
        )
        if not NO_LORA:
            model = PeftModel.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
            )
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=1.0,
    top_p=0.9,
    top_k=40,
    num_beams=4,
    max_new_tokens=512,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.8,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

"""
gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="alpaca4",
    description="Alpaca4",
).launch()

# Old testing code follows.

"""
if __name__ == "__main__":
    with open("./test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    for data in test_data["en"]:
        response = evaluate(data["instruction"])
        if response[-4:] == "</s>":
            response = response[:-4]
        data["output_gpt"] = response
        print({"question":data["instruction"], "answer": response})
    for data in test_data["zh"]:
        response = evaluate(data["instruction"])
        if response[-4:] == "</s>":
            response = response[:-4]
        data["output_gpt"] = response
        print({"question":data["instruction"], "answer": response})
    open("./eval_output.json", "w+").write(json.dumps(test_data, ensure_ascii=False))
