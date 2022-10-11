import os
import torch
import higher
import numpy as np

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from meta_kg.dataset import MetaKnowledgeDataset

import torch

from torch.utils.data import DataLoader


def run(args, logger):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    train_data = MetaKnowledgeDataset(
        logger, args, args.train_dir,
        data_type="train", is_training=True)
    dev_data = MetaKnowledgeDataset(
        logger, args, args.train_dir,
        data_type="dev", is_training=False)

    print(len(dev_data))

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key): value for key, value in state_dict.items()}
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path, state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        parameters_first = [
            p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_first,
                "weight_decay": 0.0
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=1000
        )

        train(args, logger, model, tokenizer,
              train_data, dev_data, optimizer, scheduler)


def train(args, logger, model, tokenizer, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_batch = 0
    global_step = 0
    train_losses = []
    dev_losses = []
    best_accuracy = -1.0
    stop_training = False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):
            batch = batch[0]
            global_batch += 1

            # if torch.cuda.is_available():
            #     batch = batch.to(torch.device("cuda"))

            train_input_ids = batch["train_input_ids"].to(
                torch.device("cuda"))
            train_attention_mask = batch["train_attention_mask"].to(
                torch.device("cuda"))
            train_labels = batch["train_labels"].to(
                torch.device("cuda"))

            dev_input_ids = batch["input_ids"].to(torch.device("cuda"))
            dev_attention_mask = batch["attention_mask"].to(
                torch.device("cuda"))
            dev_labels = batch["labels"].to(torch.device("cuda"))

            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # Train
                for _ in range(args.n_inner_iter):
                    train_out = fmodel(
                        input_ids=train_input_ids,
                        attention_mask=train_attention_mask.to(
                            torch.device("cuda")),
                        labels=train_labels.to(torch.device("cuda"))
                    )
                    train_loss = train_out.loss
                    train_losses.append(train_loss.item())
                    diffopt.step(train_loss)

                # Dev
                dev_out = fmodel(
                    dev_input_ids,
                    dev_attention_mask,
                    labels=dev_labels
                )
                dev_loss = dev_out.loss
                dev_losses.append(dev_loss.detach().cpu())
                dev_loss.backward()

                if global_batch % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    if global_step % args.eval_period == 0:
                        logger.info("evaluating...")
                        model.eval()
                        curr_em = inference(
                            model, tokenizer, dev_data.dataloader)
                        logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em,
                            epoch))
                        train_losses = []
                        dev_losses = []

                    logger.info("train loss: {}; dev loss: {}".format(
                        np.mean(train_losses), np.mean(dev_losses)))

                    #     if best_accuracy < curr_em:
                    #         model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    #         torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    #         logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                    #                 (dev_data.metric, best_accuracy, curr_em, epoch, global_step))
                    #         best_accuracy = curr_em
                    #         wait_step = 0
                    #         stop_training = False
                    #     else:
                    #         wait_step += 1
                    #         if wait_step >= args.wait_step:
                    #             stop_training = True
                    #             break
                    #     model.train()

                if global_step >= args.total_steps:
                    stop_training = True
                    break

            if stop_training:
                break


def generate(
    model,
    input_ids,
    attention_mask,
    max_length=4,
    no_repeat_ngram_size=None,
    num_beams=10,
    do_sample=None,
    top_p=None,
    min_length=1,
    top_k=None,
    num_return_sequences=None
):
    if do_sample and top_p:
        top_k = 0
    elif do_sample and top_k:
        top_p = None

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        use_cache=True
    )
    return outs


def inference(model, tokenizer, eval_dataloader):
    model.eval()
    all_out = []
    all_label = []
    for batch in tqdm(eval_dataloader, desc="Inference"):
        batch = batch[0]
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(model.parameters(), lr=1e-1)

        with higher.innerloop_ctx(model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(n_inner_iter):
                train_out = fmodel(
                    input_ids=batch["train_input_ids"].to(
                        torch.device("cuda")),
                    attention_mask=batch["train_attention_mask"].to(
                        torch.device("cuda")),
                    labels=batch["train_labels"].to(torch.device("cuda"))
                )
                train_loss = train_out.loss
                diffopt.step(train_loss)

            with torch.no_grad():
                output = generate(
                    fmodel,
                    batch["input_ids"].to(torch.device("cuda")),
                    batch["attention_mask"].to(torch.device("cuda"))
                )
                eval_out = [tokenizer.decode(
                    out, skip_special_tokens=True) for out in output]
                eval_label = [tokenizer.decode(
                    label, skip_special_tokens=True) for label in batch["labels"]]
                all_out.extend(eval_out)
                all_label.extend(eval_label)

    return accuracy_score(all_label, all_out)


# @dataclass
# class TranslationOutput:
#     """Helper class for translation output"""
#     config: Dict
#     print_data: Dict

#     @classmethod
#     def from_output(cls, config, output):
#         """Loads from outputs

#         :param outputs: the outputs produced by the model
#         """
#         # retrieve data needed for printing
#         print_data = {}
#         print_out_keys = set(
#             itertools.chain(*[list(i["print_out"].keys()) for i in output])
#         )

#         for key_name in print_out_keys:
#             raw_data = [t["print_out"][key_name] for t in output]
#             print_data[key_name] = [t for t in itertools.chain(*raw_data)]

#         return cls(config=config, print_data=print_data)

#     @property
#     def prefixes(self):
#         return self.print_data.get("prefix", [])

#     @property
#     def label_scores(self):
#         return self.print_data.get("label_scores", [])

#     @property
#     def targets(self):
#         return self.print_data.get("text_out", [])

#     @property
#     def outputs(self):
#         return self.print_data.get("gen_out", [])

#     def _normalize_text(self, s):
#         """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
#         import string
#         import re

#         def remove_articles(text):
#             regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#             return re.sub(regex, " ", text)

#         def white_space_fix(text):
#             return " ".join(text.split())

#         def remove_punc(text):
#             exclude = set(string.punctuation)
#             return "".join(ch for ch in text if ch not in exclude)

#         def lower(text):
#             return text.lower()

#         return white_space_fix(remove_articles(remove_punc(lower(s))))

#     def _compute_f1(self, generation_pairs):
#         prediction, truth = generation_pairs
#         pred_tokens = self._normalize_text(prediction).split()
#         truth_tokens = self._normalize_text(truth).split()

#         # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
#         if len(pred_tokens) == 0 or len(truth_tokens) == 0:
#             return int(pred_tokens == truth_tokens)

#         common_tokens = set(pred_tokens) & set(truth_tokens)

#         # if there are no common tokens then f1 = 0
#         if len(common_tokens) == 0:
#             return 0

#         prec = len(common_tokens) / len(pred_tokens)
#         rec = len(common_tokens) / len(truth_tokens)

#         return 2 * (prec * rec) / (prec + rec)

#     def _jaccard(self, str_pair):
#         str1, str2 = str_pair
#         a = set(str1.lower().split())
#         b = set(str2.lower().split())
#         if (len(a) == 0) & (len(b) == 0):
#             return 0.5
#         c = a.intersection(b)

#         return float(len(c)) / (len(a) + len(b) - len(c))

#     def gen_f1(self, sort=False):
#         """Returns an f1 score for generation

#         :rtype: float or None
#         """
#         targets = self.targets
#         outputs = self.outputs

#         targets = [self._normalize_text(t.strip())
#                    for k, t in enumerate(targets)]
#         outputs = [self._normalize_text(t.strip())
#                    for k, t in enumerate(outputs)]

#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             batch_f1 = list(map(self._compute_f1, zip(targets, outputs)))
#             avg_f1 = sum(batch_f1) / len(batch_f1)
#             return avg_f1

#     def gen_jaccard(self, sort=False):
#         """Returns an jaccard text accuracy for generation

#         :rtype: float or None
#         """
#         targets = self.targets
#         outputs = self.outputs

#         targets = [self._normalize_text(t.strip())
#                    for k, t in enumerate(targets)]
#         outputs = [self._normalize_text(t.strip())
#                    for k, t in enumerate(outputs)]

#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             batch_jaccard = list(map(self._jaccard, zip(targets, outputs)))
#             avg_jaccard = sum(batch_jaccard) / len(batch_jaccard)
#             return avg_jaccard

#     def gen_em(self, sort=False):
#         """Returns an exact match accuracy for generation

#         :rtype: float or None
#         """
#         prefixes = self.prefixes
#         targets = self.targets
#         outputs = self.outputs

#         if prefixes:
#             targets = [t.strip() for k, t in enumerate(targets)
#                        if not prefixes[k] or prefixes[k] == "answer:"]
#             outputs = [t.strip() for k, t in enumerate(outputs)
#                        if not prefixes[k] or prefixes[k] == "answer:"]

#         # special sorting functionality for set processing
#         # assumes set items are delimited by `+`
#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             return sklearn_metrics.accuracy_score(targets, outputs)

#     @property
#     def generative(self):
#         return True

#     def enumerate_instances(self):
#         """Enumerate through instances for printing

#         """
#         guids = self.print_data["guid"]
#         text_in = self.print_data["text_in"]
#         prefixes = self.prefixes
#         targets = self.targets
#         outputs = self.outputs
#         label_scores = self.label_scores

#         total_outputs = []

#         for k, identifier in enumerate(guids):
#             instance_dict = {}
#             instance_dict["id"] = identifier
#             instance_dict["context"] = text_in[k]
#             instance_dict["gen_out"] = outputs[k]
#             if targets:
#                 instance_dict["answer"] = targets[k]
#             if prefixes:
#                 instance_dict["meta"] = {}
#                 instance_dict["meta"]["prefix"] = prefixes[k]
#             if label_scores:
#                 instance_dict["label_scores"] = label_scores[k]

#             total_outputs.append(instance_dict)

#         return total_outputs

#     def __iter__(self):
#         for item in self.enumerate_instances():
#             yield item
