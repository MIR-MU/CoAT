import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeInfoDIff

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
# from evaluation.tasks.en.qa import PrimedQATask
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="./saved_models/bert-base-uncased_finetuned_baseline", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
args = parser.parse_args()
results = {}

for model_name_or_path in args.model_names_or_paths.split(","):
    results[model_name_or_path] = {}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for template_id in DatasetTemplates('glue/mnli').all_template_names:
        task = GLUEDiagnostics("en", DatasetTemplates('glue/mnli')[template_id])

        evaluator = RougeInfoDIff(task)

        # model_path = "gaussalgo/mt5-base-priming-QA_en-cs"
        # model_path = "allenai/tk-instruct-3b-def-pos"

        random_selection_perf, info_selection_perf = evaluator(model, tokenizer, None)
        print("{}\t{:.5f}\t{:.5f}\t{:.5f}".format(model_name_or_path,
                                                  random_selection_perf,
                                                  info_selection_perf,
                                                  info_selection_perf - random_selection_perf))
        results[model_name_or_path][template_id] = {"random": random_selection_perf,
                                                    "info": info_selection_perf,
                                                    "diff": info_selection_perf - random_selection_perf}

print(results)
