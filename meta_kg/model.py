import torch
import logging
import numpy as np
import torch.nn as nn

from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    GenerationConfig,
)

from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModelForCausalLM
)


model_class_registry = {
    "t5": T5ForConditionalGeneration,
    "gpt2": GPT2LMHeadModel,
    "gptj": AutoModelForCausalLM,
    "llama": AutoModelForCausalLM,
    "mllama": AutoModelForCausalLM,
    "mistral": AutoModelForCausalLM,
    "gemma": AutoModelForCausalLM,
    "qwen": AutoModelForCausalLM,
}

util_logger = logging.getLogger("meta_knowledge.model")

class CausalLM(nn.Module):
    """
    Generic transformer-based autoregressive language model (e.g., GPT-2, GPT-J, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation.
    """

    def __init__(self, model, tokenizer, model_config, global_config):
        super().__init__()
        self.lm_model = model
        self.tokenizer = tokenizer
        self.config = model_config
        self.global_config = global_config

    @staticmethod
    def freeze_params(model, range: List[int]):
        for name, param in model.named_parameters():
            if np.any([x in name for x in range]):
                print(f"Freezing {name}")
                param.requires_grad = False

    @classmethod
    def from_config(cls, config):
        """
        Loads a pretrained decoder-only causal LLM from configuration

        :param config: the global configuration
        :rtype CausalLM
        """
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        model_class = model_class_registry[config.model_type]
        model = model_class.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
            cache_dir="/mnt/u14157_ic_nlp_001_files_nfs/scartch/home/zechen/.cache/huggingface/hub",
        )

        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        if config.model_type == "llama":
            tokenizer.pad_token_id = 128004
            model.config.pad_token_id = 128004

        if config.freeze_partial:
            cls.freeze_params(model, config.freeze_range)

        if config.peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8, lora_alpha=32,
                lora_dropout=0.1)
            model = get_peft_model(model, peft_config)

            trainable_params, all_param = model.get_nb_trainable_parameters()
            util_logger.info("PEFT model loaded")
            util_logger.info(f"trainable parameters: {trainable_params:,d} || all params: {all_param:,d}")
            util_logger.info(f"trainable%: {100 * trainable_params / all_param:.4f}")

        return cls(
            model,
            tokenizer,
            model_config,
            config,
        )

    def forward(self, features):
        """A modified version of forward method for the underlying transformer model.
        :param features: the target inputs
        :param print_out: data to print out during evaluation
        """
        outputs = self.lm_model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            labels=features["labels"],
            return_dict=True,
        )
        return outputs

class MetaReasonSeq2Seq(CausalLM):
    def forward(self, features, print_out, is_inner=False):
        """A modified version of forward method for the underlying transformer model.
        :param features: the target inputs
        :param print_out: data to print out during evaluation
        """
        main_out = {"print_out": print_out}
        labels = (
            features["input_ids"]
            if "gpt" in self.global_config.model_type
            else features["labels"]
        )

        outputs = self.lm_model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        main_out["loss"] = outputs["loss"]

        if "evaluate" in features and features["evaluate"]:
            if "question" in print_out:
                main_out["print_out"]["gen_out"] = self.generate(print_out)
            else:
                main_out["print_out"]["gen_out"] = self.generate(
                    main_out["print_out"]["prompt"]
                )
        return main_out

    def preprocess_generation(self, print_out):
        output_length = []
        for answer in print_out["answer"]:
            out_ids = self.tokenizer(answer, return_tensors="pt").input_ids
            output_length.append(out_ids.size(1))
        max_out_length = max(output_length)

        input_ids_batch = []
        for question in print_out["question"]:
            input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(
                self.lm_model.device
            )
            input_ids_batch.append(input_ids)

        return input_ids_batch, max_out_length

    def generate_step(self, input_ids, max_out_length):
        generation_config = GenerationConfig.from_pretrained(
            "t5-small",
            num_beams=5,
            max_new_tokens=max_out_length,
            early_stopping=True,
            top_p=None,
            do_sample=False,
            num_return_sequences=1,
        )

        greedy_output = self.lm_model.generate(
            input_ids=input_ids, generation_config=generation_config
        )
        out = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True).strip()

        return out

    def generate(self, print_out):
        input_ids_batch, max_out_length = self.preprocess_generation(print_out)

        outputs = [
            self.generate_step(input_ids, max_out_length)
            for input_ids in input_ids_batch
        ]

        return outputs