import numpy as np

from peft import get_peft_model, PrefixTuningConfig, LoraConfig, TaskType

from torch.optim import AdamW

from meta_kg.module import MetaReasonLMModule, CausalLMModule


class KGMAMLPrefixModule(MetaReasonLMModule):
    def __init__(self, config):
        super().__init__(config)

        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=config.prefix_dim
        )
        self.model.model = get_peft_model(self.model.model, peft_config)

        self.prefix_params = {}
        self.model_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prefix_params[name] = param
            else:
                self.model_params[name] = param

        self.lm_head = self.model.model.base_model.lm_head
        for name, param in self.lm_head.named_parameters():
            param.requires_grad = True
            self.prefix_params[name] = param

        self.inner_lr_schedular_config(config.n_inner_iter, config.inner_lr)

        self.num_prefix_params = len(self.prefix_params)
        self.num_model_params = len(self.model_params)

        # util_logger.info(
        #     f"Number of prefix parameters: {self.num_prefix_params}")
        # util_logger.info(
        #     f"Number of model parameters: {self.num_model_params}")

    def set_model_params_grad(self, grad: bool = True):
        for _, param in self.model_params.items():
            param.requires_grad = grad

    def set_prefix_params_grad(self, grad: bool = True):
        for _, param in self.prefix_params.items():
            param.requires_grad = grad

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :param model_params: the model parameters
        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        for _, param in self.prefix_params.items():
            model_params.append({"params": param, "lr": self.hparams.inner_lr})

        inner_opt = AdamW(model_params, amsgrad=False)
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """

        parameters_prefix = [p for _, p in self.prefix_params.items()]

        optimizer_grouped_parameters = [
            {"params": parameters_prefix, "weight_decay": self.hparams.weight_decay}
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


class KGMAMLLoraModule(MetaReasonLMModule):
    def __init__(self, config):
        super().__init__(config)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model.model = get_peft_model(self.model.model, peft_config)

        self.trainable_params = {}
        self.frozen_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.trainable_params[name] = param
            else:
                self.frozen_params[name] = param

        for name, param in self.model.named_parameters():
            if np.any([x in name for x in ["lm_head", "wpe", "wte"]]):
                param.requires_grad = True
                self.trainable_params[name] = param

        for name, param in self.model.named_parameters():
            if np.any([x in name for x in [".47."]]):
                print(f"Enable training: {name}")
                param.requires_grad = True
                self.trainable_params[name] = param

        self.num_prefix_params = len(self.trainable_params)

        self.inner_lr_schedular_config(config.n_inner_iter, config.inner_lr)

    def set_model_params_grad(self, grad: bool = True):
        for _, param in self.model_params.items():
            param.requires_grad = grad

    def set_prefix_params_grad(self, grad: bool = True):
        for _, param in self.lora_params.items():
            param.requires_grad = grad

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :param model_params: the model parameters
        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        for _, param in self.trainable_params.items():
            model_params.append({"params": param, "lr": self.hparams.inner_lr})

        inner_opt = AdamW(model_params, amsgrad=False)
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p
            for n, p in self.trainable_params.items()
            if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p
            for n, p in self.trainable_params.items()
            if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {"params": parameters_first, "weight_decay": self.hparams.weight_decay},
            {"params": parameters_sec, "weight_decay": 0.0},
            {"params": self.inner_schedular.parameters(), "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


class CausalLoraModule(CausalLMModule):
    def __init__(self, config):
        super().__init__(config)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
        )

        self.model.model = get_peft_model(self.model.model, peft_config)
        self.model.model.print_trainable_parameters()
