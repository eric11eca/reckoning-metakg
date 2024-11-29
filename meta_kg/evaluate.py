import re
import string
# import evaluate

from typing import Dict
from dataclasses import dataclass

from .utils.py_io import *

ANSWER_PREFIX = "Answer:"
REASON_PREFIX = "Support"
REASON_SEP = "**\n"

# accuracy_metric = evaluate.load("accuracy")
# blue = evaluate.load("bleu")
# rouge = evaluate.load("rouge")
# meteor = evaluate.load("meteor")

def normalize_text(text):
    """
    Removing articles and punctuation, and
    standardizing whitespace are all typical
    text processing steps.

    :param text: text to normalize
    """
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_exact_match(prediction, truth):
    check = normalize_text(truth) == normalize_text(prediction)
    return int(check)

@dataclass
class GenerationOutput:
    """Data class for a unified format of generation output"""
    guid: str
    prompt: str
    response: str
    answer: str
    metrics: Dict
    metadata: Dict

    @classmethod
    def from_output(cls, output: Dict):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        :rtype: GenerationOutput
        """
        guid = output["guid"]
        prompt = output["prompt"]
        response = output["output"]
        answer = output["answer"]
        metrics = {}
        metadata = {}
        # if "metadata" in output["print_data"]:
        #     metadata.update(output["print_data"]["metadata"])

        return cls(
            guid=guid,
            prompt=prompt,
            response=response,
            answer=answer,
            metrics=metrics,
            metadata=metadata,
        )

    @property
    def to_dict(self):
        """Converts to a dictionary"""
        return {
            "guid": self.guid,
            "prompt": self.prompt,
            "response": self.response,
            "answer": self.answer,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def compute_exact_match(self, response, target):
        """
        Compute exact match between response and target

        :param response: response from the model
        :param target: target from the dataset
        :rtype: int
        """
        # response = response.split(ANSWER_PREFIX)[1].strip()
        norm_response = normalize_text(response)
        norm_target = normalize_text(target)
        return int(norm_response == norm_target)

    def compute_f1(self, response, target):
        """
        Compute F1 score between response and target

        :param response: response from the model
        :param target: target from the dataset
        :rtype: float
        """
        pred_tokens = normalize_text(response).split()
        truth_tokens = normalize_text(target).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    def reason_ratio(self, reason_steps_gen, reason_steps_gold):
        """
        Compute the ratio of the reason steps generated

        :param reason_steps_gen: generated reasoning steps from the model
        :param reason_steps_gold: gold reasoning steps from the dataset
        :rtype: float
        """
        gen_set = set(reason_steps_gen)
        gold_set = set(reason_steps_gold)
        coverage = len(gen_set & gold_set)
        ratio = coverage / len(gold_set)
        return ratio, coverage

    def extract_reason_steps(self, response, target):
        """Extract the reason steps from the response & target"""
        reason_gold = target.split(REASON_PREFIX)[1].strip()
        reason_gen = response.split(REASON_PREFIX)[1].strip()
        reason_steps_gen = reason_gen.split(REASON_SEP)
        reason_steps_gold = reason_gold.split(REASON_SEP)
        return reason_steps_gen, reason_steps_gold

    def compute_metrics(self):
        """Returns an exact match accuracy for generation"""
        response = self.response.lower()
        target = self.answer.lower()
        
        pred = response.split(REASON_PREFIX)[0].strip()
        gold = target.split(REASON_PREFIX)[0].strip()

        self.metrics["em"] = self.compute_exact_match(pred, gold)
        self.metrics["f1"] = self.compute_f1(response, target)

        # if REASON_PREFIX in response:
        #     reason_steps_gen, reason_steps_gold = self.extract_reason_steps(response, target)
        #     ratio, coverage = self.reason_ratio(reason_steps_gen, reason_steps_gold)
        #     self.metrics["reason_ratio"] = ratio
        #     self.metrics["reason_coverage"] = coverage
        #     self.metrics["num_gen_stpes"] = len(reason_steps_gen)

def eval(records):
    generations = [GenerationOutput.from_output(record) for record in records]
    for generation in generations:
        generation.compute_metrics()
    accuracy = sum([gen.metrics["em"] for gen in generations]) / len(generations)
    f1 = sum([gen.metrics["f1"] for gen in generations]) / len(generations)
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
    }

    if "reason_ratio" in generations[0].metrics:
        reason_ratio = sum([gen.metrics["reason_ratio"] for gen in generations]) / len(generations)
        reason_coverage = sum([gen.metrics["reason_coverage"] for gen in generations]) / sum([gen.metrics["num_gen_stpes"] for gen in generations])
        metrics["reason_ratio"] = reason_ratio
        metrics["reason_coverage"] = reason_coverage
    return metrics, generations

if __name__ == "__main__":
    import uuid
    from datetime import datetime

    records = []
    records.append({
        "print_data": {
            "guid": str(uuid.uuid4()),
            "prompt": "Question: Which of these branches of the trigeminal nerve contain somatic motor processes?\nOptions: A. The supraorbital nerve\nB. The infraorbital nerve\nC. The mandibular nerve\nD. None of the above\n\nAnswer:",
            "response": """Based on\n
1. The Supraorbital Nerve: This is a branch of the ophthalmic nerve (V1). The ophthalmic nerve primarily carries sensory fibers and does not contain somatic motor processes.**
2. The Infraorbital Nerve: This is a branch of the maxillary nerve (V2). Similar to the ophthalmic nerve, the maxillary nerve is primarily sensory and does not contain motor fibers.**
3. The Mandibular Nerve (V3): The mandibular nerve is unique among the three branches of the trigeminal nerve because it contains both sensory and motor fibers. The motor fibers of the mandibular nerve innervate the muscles of mastication (chewing) and a few other muscles, such as the mylohyoid and the anterior belly of the digastric.**
The answer is: C""",
            "answer": "C",
            "metadata": {
                "task": "mcat",
                "language": "en"
            }
        },
        "loss": 0.1
    })

    records.append({
        "print_data": {
            "guid": str(uuid.uuid4()),
            "prompt": "Question: If the effects of IAA treatment in nerve cells are the same as those observed in myocytes, which feature of an action potential would be most affected by IAA treatment?\nOptions: A. Initiation of depolarization\nB. Rising phase of depolarization\nC. Falling phase to undershoot\nD. Return to resting potential\n\nAnswer:",
            "response": """Based on\n
1. Role of Na⁺/K⁺ ATPase in Recovery: Restores Ion Gradient, Helps Re-establish Resting Potential, Energy-Dependent Process**
2. The inhibition of glycolysis by IAA is likely to lead to a decrease in cellular ATP concentration**
3. The membrane potential returns to its resting state, often assisted by the Na⁺/K⁺ ATPase pump.**
The answer is: B""",
            "answer": "D",
            "metadata": {
                "task": "mcat",
                "language": "en"
            }
        },
        "loss": 0.5
    })

    metrics, generations = eval(records)

    run_id = int(datetime.timestamp(datetime.now()))
    run_dir = f"./runs/icr-phi-2-mcat-{run_id}/evals"
    os.makedirs(run_dir, exist_ok=True)
    write_metrics(metrics, run_dir, f"epoch-{1}-val.json")

    generation_with_metrics = [gen_record.to_dict for gen_record in generations]
    run_dir = f"./runs/icr-phi-2-mcat-{run_id}/outputs"
    write_generations(generation_with_metrics, run_dir, f"epoch-{1}-val.jsonl")