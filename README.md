# Conceptual Few-Shot Evaluation & Training

Training and evaluation sources to assess few-shot learners' ability 
to utlize **informative** concepts in prediction.

## Conceptual Few-shot Evaluation

To extract the concepts from explanations as proposed in the paper, 
and run the Conceptual few-shot evaluation on a selected model, 
run `conceptual_fewshot_evaluator.py` script:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r evaluation/requirements.txt
spacy download en_core_web_sm  # For OpenBookQA concepts extraction

CUDA_VISIBLE_DEVICES=0 python evaluation/conceptual_fewshot_evaluator.py \
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard \
    --bootstrap False \
    --metric ROUGE \
    --dataset_ids glue/mnli,openbookqa/additional,hotpot_qa/fullwiki,worldtree \
    --firstn 100
```
All resources and concepts extractions should be resolved automatically.

If you evaluate using `--bootstrapping True`, collect the stdout to a file and analyse the results using [this notebook](analyses/conceptual_few_shot_eval_viz.ipynb).

In order to perform evaluation on a full dataset, simply remove `--firstn` parameter.

## SuperGLUE evaluation

To reproduce our evaluation on SuperGLUE, run the following:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
CUDA_VISIBLE_DEVICES=0 python evaluation/superglue_evaluator.py \
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard,allenai/tk-instruct-large-def-pos \
    --metric Accuracy \
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

cd training
./download_teaberac_data.sh
cd ..

CUDA_VISIBLE_DEVICES=0 python training/train_mt5_teabreac+qa_hard.py
```

The script intentionally contains all parameters fixed, but if you need to change something,
e.g. due to the environment restrictions, do not hesitate to adjust `AdaptationArguments` or evaluations within the code.

The training scripts include evaluations on SuperGLUE and various TeaBReaC concepts.


### Baseline: Random Demonstrations Selection Training

In the sequence above, replace the python script path with `train_mt5_teabreac+qa_random.py`.

```shell
CUDA_VISIBLE_DEVICES=0 python training/train_mt5_teabreac+qa_random.py
```

### Citation

If you use Conceptual Few-shot Evaluation in scientific work, please cite this work as follows:

```bibtex
@inproceedings{stefanik2023incontext,
               author = {{{\v{S}}tef{\'a}nik}, Michal and {Kadl{\v{c}}{\'\i}k}, Marek},
               title={Can In-context Learners Learn a Reasoning Concept from Demonstrations?}, 
               booktitle = {Proceedings of ACL 2023: Natural Language Reasoning and Structured Explanations (NLRSE)},
               publisher = {ACL},
               numpages = {6},
               year={2023},
               url = {https://arxiv.org/abs/2212.01692},
}
```

If you'd like to reference Concept-Aware Training, please cite other paper that introduces it:

```bibtex
@article{stefanik2023conceptaware,
         title={Concept-aware Training Improves In-context Learning Ability of Language Models}, 
         author={{{\v{S}}tef{\'a}nik}, Michal and {Kadl{\v{c}}{\'\i}k}, Marek},
         year={2023},
         eprint={2305.13775},
         archivePrefix={arXiv},
         primaryClass={cs.CL},
         url = {https://arxiv.org/abs/2305.13775},
}
```

