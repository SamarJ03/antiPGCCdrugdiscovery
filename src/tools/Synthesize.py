import sys
import argparse
import os
import json
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

class Synthesizer:
    def get_hf_tokenizer_pipeline(self, model, is_8bit=False):
        model = model.lower()
        if model == 'falcon-7b':
            hf_model = "tiiuae/falcon-7b-instruct"
        elif model == 'falcon-40b':
            hf_model = "tiiuae/falcon-40b-instruct"
            is_8bit = True
        elif model == "galactica-6.7b":
            hf_model = "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
        elif model == "galactica-30b":
            hf_model = "GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k"
            is_8bit = True
        elif model == "chemllm-7b":
            hf_model = "AI4Chem/ChemLLM-7B-Chat"
        elif model == "chemdfm":
            hf_model = "X-LANCE/ChemDFM-13B-v1.0"
        else:
            raise NotImplementedError(f"Cannot find Hugging Face tokenizer for model {model}.")
        model_kwargs = {}
        quantization_config = None
        if is_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0,)
        if model == "chemllm-7b":
            pipeline = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16,
                                                            device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_model,trust_remote_code=True)
        elif model == "chemdfm":
            cache = os.path.join(os.getcwd(),'tools', 'cache')
            os.makedirs(cache, exist_ok=True)
            offload_dir = os.path.join(cache, 'offload')
            os.makedirs(offload_dir, exist_ok=True)
            tokenizer = LlamaTokenizer.from_pretrained(hf_model)
            pipeline = LlamaForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16, device_map="auto", offload_folder=offload_dir)
        else:
            model_kwargs['quantization_config'] = quantization_config
            tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False, padding_side="left", trust_remote_code=True)
            offload_dir = os.path.join(os.getcwd(), "offload")
            os.makedirs(offload_dir, exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(hf_model, 
                                                torch_dtype=torch.float16, 
                                                device_map="auto", 
                                                trust_remote_code=True,
                                                offload_folder=offload_dir,
                                                # quantization_config=quantization_config,
                                                **model_kwargs)
            pipeline = model
        return tokenizer, pipeline

    def get_synthesize_prompt(self):
        """
        Read prompt json file to load prompt for the task, return a task name list and a prompt list
        """
        args = self.args
        prompt_file = os.path.join(args['input_folder'], args['input_file'])
        pk_prompt_list = []
        task_list = []
        with open(prompt_file, 'r') as f:
            prompt_dict = json.load(f)
        if args['model'].lower() in ["falcon-7b", "galactica-6.7b", "chemllm-7b", "chemdfm"]:
            dataset_key = args['dataset'] + "_small"
        else:
            dataset_key = args['dataset'] + "_big"
        print(f'Extracting {dataset_key} dataset prior knowledge prompt ....')
        if not args['subtask']:
            task_list.append(args['dataset'])
            pk_prompt_list.append(prompt_dict[dataset_key])
        elif args['subtask']:  # for tox21 and sider
            print(f"Extracting {args['subtask']} task prior knowledge prompt ....")
            task_list.append(args['dataset'] + "_" + args['subtask'] )
            pk_prompt_list.append(prompt_dict[dataset_key][args['subtask']])
        else:
            raise NotImplementedError(f"""No prior knowledge prompt for task {args['dataset']}.""")
        return task_list, pk_prompt_list

    def get_pk_model_response(self, model, tokenizer, pipeline, pk_prompt_list):
        model = model.lower()
        if model in ["galactica-6.7b", "galactica-30b", "chemllm-7b"]:
            system_prompt = ("Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\n{instruction}\n\n### Response:\n")
        elif model in ['falcon-7b', 'falcon-40b']:
            system_prompt = "{instruction}\n"
        elif model in ["chemdfm"]:
            system_prompt = "Human: {instruction}\nAssistant:"
        else:
            system_prompt = "{instruction}\n"
        response_list = []
        if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b", "chemllm-7b", "chemdfm"]:
            for pk_prompt in pk_prompt_list:
                input_text = system_prompt.format_map({'instruction': pk_prompt.strip()})
                len_input_text = len(tokenizer.tokenize(input_text))
                print(input_text)
                max_new_token = self.get_token_limit(model, for_response=True) - len_input_text
                if model in ['chemllm-7b']:
                    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
                    generation_config = GenerationConfig(
                                        do_sample=True,
                                        top_k=1,
                                        temperature=float(0.5),
                                        max_new_tokens=max_new_token,
                                        repetition_penalty=float(1.2),
                                        pad_token_id=tokenizer.eos_token_id
                                        )
                    outputs = pipeline.generate(**inputs, generation_config=generation_config)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elif model in ["chemdfm"]:
                    
                    # inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = tokenizer(input_text, return_tensors="pt").to(device)

                    generation_config = GenerationConfig(
                                        do_sample=True,
                                        top_k=20,
                                        top_p=0.9,
                                        temperature=0.9,
                                        max_new_tokens=max_new_token,
                                        repetition_penalty=1.05,
                                        pad_token_id=tokenizer.eos_token_id
                                        )
                    outputs = pipeline.generate(**inputs, generation_config=generation_config)
                    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
                else:
                    text_generator = pipeline(
                        input_text,
                        min_new_tokens=0,
                        max_new_tokens=max_new_token,
                        do_sample=False,
                        num_beams=3,
                        temperature=float(0.5),
                        repetition_penalty=float(1.2),
                        renormalize_logits=True
                    )
                    generated_text = text_generator[0]['generated_text']
                if model in ["galactica-6.7b", "galactica-30b", 'chemllm-7b']:
                    generated_text = generated_text.split('### Response:\n')[1]
                elif model in ['falcon-7b', 'falcon-40b', 'chemdfm']:
                    pass
                print(generated_text)
                response_list.append(generated_text)
        else:
            raise NotImplementedError(f"""get_model_response() is not implemented for model {model}.""")
        return response_list
    
    @staticmethod
    def get_token_limit(model, for_response=False):
        """Returns the token limitation of provided model"""
        model = model.lower()
        if for_response:  # For get response
            if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b", "chemdfm"]:
                num_tokens_limit = 2048
            elif model in ['chemllm-7b']:
                num_tokens_limit = 4096
        else:  # For split input list
            if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b", "chemdfm"]:
                num_tokens_limit = round(2048*3/4)  # 1/4 part for the response, 512 tokens
            elif model in ['chemllm-7b']:
                num_tokens_limit = round(4096*3/4)
            else:
                raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
        return num_tokens_limit

    def run(self):
        start = time.time()
        args = self.args
        tokenizer, pipeline = self.get_hf_tokenizer_pipeline(args['model'])
        task_list, pk_prompt_list = self.get_synthesize_prompt()
        response_list = self.get_pk_model_response(args['model'], tokenizer, pipeline, pk_prompt_list)
        output_file_folder = os.path.join(args['output_folder'], args['model'], args['dataset'])
        if args['subtask'] != '':
            subtask_name = "_" + args['subtask']
        else:
            subtask_name = ''
        output_file = os.path.join(output_file_folder, f'{args["model"]}{subtask_name}_pk_response.txt')
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
        with open(output_file, 'w') as f:
            for i in range(len(task_list)):
                f.write(f'task name: {task_list[i]}\n')
                f.write('Response from model: \n')
                f.write(response_list[i])
                f.write("\n\n================================\n\n")
        end = time.time()
        print(f"Synthesize/Time elapsed: {end-start} seconds")

    def __init__(self, args, dataset=None, subtask=None, model=None):
        if args is None: 
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset', type=str, default='metaFingerprints', help='dataset/task name')
            parser.add_argument('--subtask', type=str, default='', help='subtask of rdkit dataset')
            parser.add_argument('--model', type=str, default='chemdfm', help='LLM model name')
            parser.add_argument('--input_folder', type=str, default='prompt_file', help='Synthesize prompt file folder')
            parser.add_argument('--input_file', type=str, default='synthesize_prompt.json', help='synthesize prompt json file')
            parser.add_argument('--output_folder', type=str, default='synthesize_model_response', help='Synthesize output folder')
            args = parser.parse_args()
        if isinstance(args, argparse.Namespace): args = vars(args)
        if isinstance(args, dict): 
            if model is None: args['model'] = 'chemdfm'
            elif model not in ('chemdfm', 'chemllm-7b', 'falcon-7b', 'falcon-40b', 'galactica-6.7b', 'galactica-30b'):
                raise Exception('Invalid model..')
            else: args['model'] = model
            
            if dataset not in ('ecfp4', 'maccs', 'metaFingerprints', 'rdkit'): 
                raise Exception('Invalid dataset..')
            else: args['dataset'] = dataset
            
            if dataset=='rdkit' and subtask is None: subtask = 'all'
            elif dataset=='rdkit' and subtask not in (
                'all', 'E-State', 'fingerprintBased', 'functionalGroup', 'molecularTopology',
                'physiochemical', 'structural', 'surfaceArea'
            ): raise Exception('Invalid rdkit subtask..')
            elif dataset!='rdkit' and subtask is not None: 
                print('!! subtask only valid for rdkit dataset..')
                subtask = None
            args['subtask'] = subtask

        self.args = args

        # print(args)