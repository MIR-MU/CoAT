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
model_path = "allenai/tk-instruct-3b-def-pos"
# model_path = "train_dir_random_large/checkpoint-5000/AQA-en-Priming"
# model_path = "gaussalgo/mt5-large-priming-QA_en-cs"
# model_path = "bigscience/T0_3B"
model_path = "trained_models/SQuAD+SQAD_hard_ch2000"
model_path = "allenai/tk-instruct-11b-def-pos-neg-expl"
model_path = "t5-large"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

evaluation = evaluator(model, tokenizer, None)
print("%s performance difference: %s" % (model_path, evaluation))
