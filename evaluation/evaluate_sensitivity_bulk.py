import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeInfoDIff, AccuracyInfoDIff

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
# from evaluation.tasks.en.qa import AdversarialQATask
import argparse

from evaluation.tasks.en.openbookqa import OpenBookQATask
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="gaussalgo/mt5-base-priming-QA_en-cs", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--eval_dataset_promptsource_id", default="glue/mnli", type=str,
                    help="Evaluation dataset. Must be one of the implemented datasets: "
                         "'glue/mnli', 'openbookqa/additional', 'hotpot_qa/fullwiki'")
parser.add_argument("--template_names", default=None, type=str,
                    help="Names of the templates to evaluate with")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")

args = parser.parse_args()
results = {}

for model_name_or_path in args.model_names_or_paths.split(","):
    results[model_name_or_path] = {}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # eval templates resolution
    if args.template_names is not None:
        eval_templates = args.template_names
    else:
        if args.eval_dataset_promptsource_id == 'hotpot_qa/fullwiki':
            # only two templates for hotpot_qa require answering questions, others are for different tasks
            eval_templates = ['generate_answer_interrogative', 'generate_answer_affirmative']
        else:
            eval_templates = DatasetTemplates(args.eval_dataset_promptsource_id).all_template_names

    for template_id in eval_templates:
        template = DatasetTemplates(args.eval_dataset_promptsource_id)[template_id]
        # eval task resolution - done in the loop to reset its state (deduplication)
        if args.eval_dataset_promptsource_id == "glue/mnli":
            task = GLUEDiagnostics("en", template)
        elif args.eval_dataset_promptsource_id == "openbookqa/additional":
            task = OpenBookQATask("en", template)
        elif args.eval_dataset_promptsource_id == 'hotpot_qa/fullwiki':
            task = R4CHotpotQATask("en", template)
        else:
            raise ValueError("Non-implemented dataset: %s" % args.eval_dataset_promptsource_id)

        # evaluation metric resolution
        if args.metric == "ROUGE":
            evaluator = RougeInfoDIff(task)
        elif args.metric == "Accuracy":
            evaluator = AccuracyInfoDIff(task)
        else:
            raise ValueError("Unknown metric: %s" % args.metric)

        random_selection_perf, info_selection_perf = evaluator(model, tokenizer, None)
        print("{}\t{:.5f}\t{:.5f}\t{:.5f}".format(model_name_or_path,
                                                  random_selection_perf,
                                                  info_selection_perf,
                                                  info_selection_perf - random_selection_perf))
        results[model_name_or_path][template_id] = {"random": random_selection_perf,
                                                    "info": info_selection_perf,
                                                    "diff": info_selection_perf - random_selection_perf}

print(results)
