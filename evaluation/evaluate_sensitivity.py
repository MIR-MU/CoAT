import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeInfoDIff

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
from evaluation.tasks.en.adversarialqa import AdversarialQATask

# task = AdversarialQATask("en")
task = GLUEDiagnostics("en")

evaluator = RougeInfoDIff(task)

# model_path = "gaussalgo/mt5-base-priming-QA_en-cs"
# model_path = "allenai/tk-instruct-3b-def-pos"
model_path = "google/t5-efficient-tiny"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

evaluation = evaluator(model, tokenizer, None)
print("%s performance difference: %s" % (model_path, evaluation))
