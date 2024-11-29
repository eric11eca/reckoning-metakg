import os
import torch
import logging

from typing import Optional
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)
from vllm import LLM, SamplingParams

from .utils.py_io import *

util_logger = logging.getLogger("metakg.inference")
util_logger.setLevel(logging.INFO)

model_repo_registry = {
    "t5": "t5-small",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-neo": "EleutherAI/gpt-neo-2.7B",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "phi-2": "susnato/phi-2",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "tiny-llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
}

class LLM_Generator:
    def __init__(
        self,
        model_repo_id: str,
        model: AutoModelForCausalLM,
        tokenizer: Optional[AutoTokenizer],
        device: str = "cuda:0"
    ) -> None:
        self.model_repo_id = model_repo_id
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def vllm_generate(self, queries, **kwargs):
        prompts = [query["prompt"] for query in queries]
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0,
            max_tokens=kwargs["max_new_tokens"])
        generator = LLM(
            model="gpt2",
            tokenizer="gpt2",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16")
        outputs = generator.generate(prompts, sampling_params)
        for query, output in zip(queries, outputs):
            prompt = output.prompt
            assert prompt == query["prompt"]
            generated_text = output.outputs[0].text
            query["output"] = generated_text

    @torch.no_grad()
    def generate(self, queries, **kwargs): 
        for query in queries:
            max_length = 1024 - kwargs["max_new_tokens"] - 2
            query["orig_prompt"] = query["prompt"]
            truncated = self.tokenizer(
                query["prompt"], 
                truncation=True,
                add_special_tokens=True,
                max_length=max_length)
            query["prompt"] = self.tokenizer.decode(
                truncated["input_ids"], skip_special_tokens=True)
        
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device)
        
        for query in queries:
            response = generator(
                query["prompt"],
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=kwargs["pad_token_id"],
                do_sample=kwargs["do_sample"],
                max_new_tokens=kwargs["max_new_tokens"],
            )
            
            query["prompt"] = query.pop("orig_prompt")
            query["output"] = response[0]["generated_text"]

if __name__ == "__main__":
    import uuid
    from datetime import datetime

    repo_id = model_repo_registry["phi-2"]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True)
    generator = LLM_Generator(model, tokenizer)

    generations = []
    generations.append({
        "print_data": {
            "guid": str(uuid.uuid4()),
            "prompt": "Question: Which of these branches of the trigeminal nerve contain somatic motor processes?\nOptions: A. The supraorbital nerve\nB. The infraorbital nerve\nC. The mandibular nerve\nD. None of the above\n\nAnswer:",
            "answer": "C",
            "metadata": {
                "task": "mcat",
                "language": "en"
            }
        },
        "loss": 0.1
    })

    generations.append({
        "print_data": {
            "guid": str(uuid.uuid4()),
            "prompt": "Question: If the effects of IAA treatment in nerve cells are the same as those observed in myocytes, which feature of an action potential would be most affected by IAA treatment?\nOptions: A. Initiation of depolarization\nB. Rising phase of depolarization\nC. Falling phase to undershoot\nD. Return to resting potential\n\nAnswer:",
            "answer": "D",
            "metadata": {
                "task": "mcat",
                "language": "en"
            }
        },
        "loss": 0.5
    })

    generation_kwargs = {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id
    }
    results = generator.generate(generations, **generation_kwargs)
    for result in results:
        print(result["print_data"]["prompt"])
        print(result["response"])

    run_id = int(datetime.timestamp(datetime.now()))
    run_dir = f"./runs/icr-phi-2-mcat-{run_id}/outputs"
    os.makedirs(run_dir, exist_ok=True)
    write_generations(results, run_dir, f"epoch-{1}-val.jsonl")
    util_logger.info(f"Saved {len(generations)} generations to {run_dir}")