import argparse
import json
import os

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeRandom
from evaluation.tasks.NI_task import NITask

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="gaussalgo/mt5-base-priming-QA_en-cs", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--eval_tasks",
                    default="task937,task202,task936,task641,task1344,task1615,task1385,task935,task199,task1388,task1554,task640,task534,task201,task1386,task463,task1387,task738,task1529,task190,task200,task1612,task970,task890,task464,task1516,task642,task1178,task391,task939,task392,task938,task1168,task828,task1628,task943,task1182,task1171,task968,task942,task1181,task1172,task1393,task1174,task1627,task1177,task1184,task1185,task1176,task614,task1629,task1175,task827,task1173,task1180,task1170,task1183,task969,task941,task1626,task940,task393,task1169,task1179,task1391,task1664,task304,task892,task891,task330,task401,task033,task133,task329,task249,task648,task1390,task893,task879,task362,task1533,task1534,task880,task1531,task1394,task020,task050,task1439,task233,task226,task396,task1640,task232,task1442,task242,task1624,task520,task290,task349,task1155,task1152,task1158,task1156,task1157,task1159,task1153,task1154,task039,task281,task613,task645,task620,task036,task623,task670,task121,task1195,task442,task1345,task035,task671,task1562,task1622,task034,task402,task1356,task1540,task1659,task569,task1342,task220,task1561,task418,task1358,task769,task219,task602,task1586,task743,task500,task619,task510,task288,task1161,task957,task1631,task1598,task1728,task102,task677,task1407,task1409,task760,task1557",
                    type=str,
                    help="Coma-separated list of evaluated task ids")
parser.add_argument("--NI_data_dir", default="evaluation/natural-instructions/tasks", type=str,
                    help="Coma-separated list of evaluated task ids")
parser.add_argument("--firstn", type=int, default=500,
                    help="If given, a number of samples from dataset to evaluate on.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")

args = parser.parse_args()

results = {}

# loading data
all_files = os.listdir(args.NI_data_dir)
all_tasks_files = [f for f in all_files if f.endswith("json")]

tasks_metadata = {}
for f in all_tasks_files:
    with open(os.path.join(args.NI_data_dir, f)) as f_obj:
        json_dict = json.load(f_obj)
        tasks_metadata[f.replace(".json", "")] = {k: v for k, v in json_dict.items()}

# eval iteration
for model_name_or_path in args.model_names_or_paths.split(","):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                  # device_map=device_map,
                                                  # device_map="auto",  # TODO
                                                  # max_memory=max_memory_mapping
                                                  )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    results[model_name_or_path] = {}
    for task_id in args.eval_tasks.split(","):
        task_key = next(k for k in tasks_metadata.keys() if task_id+"_" in k)
        task_demonstrations = tasks_metadata[task_key]["Instances"]
        task = NITask([(sample["input"], sample["output"][0]) for sample in task_demonstrations], label=task_key)
        evaluator = RougeRandom(task,
                                bootstrap=False,
                                max_input_length=args.max_input_length,
                                firstn=args.firstn if args.firstn else None)

        random_perf, info_perf = evaluator.get_per_sampling_performance(model, tokenizer, use_cache=False)
        results[model_name_or_path][task_key] = random_perf

        print("%s\t%s\t%s" % (model_name_or_path, task_key, random_perf))

    pd.DataFrame(results).to_csv("%s_NI_evaluation.tsv" % model_name_or_path, sep="\t")
