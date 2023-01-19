# Conceptual Few-Shot Evaluation & Training

Training and evaluation testbed for few-shot learners aimed to learn concepts of unseen tasks

## Conceptual Few-shot Evaluation

To run the Conceptual few-shot evaluation on a selected model, run `evaluate_sensitivity_bulk.py` script:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r evaluation/requirements.txt
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_sensitivity_bulk.py \ 
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard \
    --bootstrapping False
    --metric ROUGE
    --dataset_ids glue/mnli,openbookqa/additional,hotpot_qa/fullwiki,worldtree
    --firstn 100
```
All resources should be resolved automatically.

If you evaluate using `--bootstrapping True`, collect the stdout to a file and analyse the results using [this notebook](analyses/conceptual_few_shot_eval_viz.ipynb).

In order to perform evaluation on a full dataset, simply remove `--firstn` parameter.

## SuperGLUE evaluation

To reproduce our evaluation on SuperGLUE, run the following:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_sensitivity_bulk.py \ 
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard,allenai/tk-instruct-large-def-pos \
    --metric Accuracy
    --tasks axb,boolq,cb,wsc,copa,multirc,rte,wic,record,axg
```
All resources should be resolved automatically.

## Concept-Aware Training

The training of concept-aware model can be reproduced by running the following script.

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r training/requirements.txt
pip install -r evaluation/requirements.txt

CUDA_VISIBLE_DEVICES=0 python training/train_mt5teabreac+qa_hard.py
```

The script intentionally contains all parameters fixed, but if you need to change something,
e.g. due to the environment restrictions, do not hesitate to adjust `AdaptationArguments` or evaluations within the code.

## Baseline: Random Demonstrations Selection Training

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r training/requirements.txt
pip install -r evaluation/requirements.txt

CUDA_VISIBLE_DEVICES=0 python training/train_mt5teabreac+qa_random.py
```
