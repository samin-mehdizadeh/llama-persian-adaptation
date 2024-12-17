import argparse
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import  PeftModel
import logging
import sys
from os import listdir
from os.path import isfile, join


DEFAULT_PAD_TOKEN = "[PAD]"
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_dir',default=None, type=str,help="A directory that contains json instruction test files")
parser.add_argument('--with_prompt',action='store_true',help="wrap the input with the prompt automatically")
parser.add_argument('--interactive',action='store_true',help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_dir', default='./predictions.json', type=str)
parser.add_argument('--cache_dir', default='./cache', type=str)

args = parser.parse_args()
print("start running the codes")

logger = logging.getLogger(__name__)


generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
    )

prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)


if __name__ == '__main__':

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)],)
    
    load_type = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,cache_dir=args.cache_dir)
    
    if tokenizer.pad_token is None:
        logger.info(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    base_model =AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir
        ).to(device)

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    logger.info(f"Vocab of the base model: {model_vocab_size}")
    logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        logger.info("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        logger.info("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map={"": device})
    else:
        model = base_model

    if device=='cpu':
        model.float()

    model.eval()

    files = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]
    for file in files:
        f = open(f'{args.data_dir}/{file}')
        data = json.load(f)
        results = []
        logger.info(f"processing {file}")
        logger.info("start inference")
        for i,x in enumerate(data):
            try:
                PROMPT = x['instruction'].strip()+"\n"+x['input'].strip()
                if(args.with_prompt):
                    instruction = prompt_input.format_map({'instruction': PROMPT})
                else:
                    instruction = PROMPT
                
                with torch.no_grad():
                    input_tokens = tokenizer(instruction,return_tensors="pt").to(device) 
                    generate_ids = model.generate(**input_tokens, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
                    model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    response = model_output[len(instruction):]

                results.append({"input":PROMPT,"output":response,"answer":x['output']})
                logger.info(f"process {i}")

            except Exception as e:
                logger.info(f"{i} encounter err: {e}")

        dirname = os.path.dirname(args.predictions_dir)
        os.makedirs(dirname,exist_ok=True)
        os.makedirs(args.predictions_dir,exist_ok=True)
        with open(f"{args.predictions_dir}/{file}",'w',encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        logger.info(f"finish processing {file}, stored in {args.predictions_dir}/{file}")