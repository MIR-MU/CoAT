from typing import List, Dict

import numpy as np
import pandas as pd
from adaptor.evaluators.generative import ROUGE
from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective

# from training.all_evaluators import eval_examples
eval_examples = 200
from training.concepts_distances import concepts_edit_distances

# teabreac training resources with validation, split to ID and OOD concepts, with different distances to ID


def _get_answer(ans_object: List[Dict[str, str]]) -> str:
    # list([{'number': '121', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}])
    return " ".join(ans['number'] if ans['number']
                    else " ".join(ans['date'].values()) if " ".join(ans['date'].values())
    else " ".join(ans['spans']) for ans in ans_object)


def _get_concepts(program_modules: List[str]) -> str:
    return " ".join(program_modules)


def _concept_distance(c: str, all_concepts: List[str]) -> float:
    import nltk
    distances = [nltk.edit_distance(c, other_c) for other_c in all_concepts if other_c != c]
    return min(distances)


tea_train = pd.read_json("training/data/teabreac_v1.0_multihop_qa_train.jsonl", lines=True)
orig_len = len(tea_train)

tea_train = tea_train[tea_train["context_text"].apply(lambda text: len(text) < 1000)].sample(frac=1)
print("Reduced to %s percent of original samples by length." % (len(tea_train) / orig_len) * 100)

tea_train["context_text"] = tea_train["context_text"].apply(lambda c: c.replace(" -> ", ". "))
tea_train["answers_text"] = tea_train["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))
tea_train["program_modules_str"] = tea_train["program_modules"].apply(lambda modules: _get_concepts(modules))

concepts_distances = pd.Series(concepts_edit_distances)

print("Mutual concepts' distances: %s" % concepts_distances.value_counts())

easy_concepts = np.random.choice(concepts_distances[concepts_distances == 2].index, size=eval_examples).tolist()
mean_concepts = np.random.choice(concepts_distances[concepts_distances == 5].index, size=eval_examples).tolist()
hard_concepts = np.random.choice(concepts_distances[concepts_distances == 12].index, size=eval_examples).tolist()

train_concepts = set(tea_train["program_modules_str"].unique()) - set(easy_concepts + mean_concepts + hard_concepts)


tea_val = pd.read_json("training/data/teabreac_v1.0_multihop_qa_dev.jsonl", lines=True)
tea_val["context_text"] = tea_val["context_text"].apply(lambda c: c.replace(" -> ", ". "))
tea_val["answers_text"] = tea_val["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))
tea_val["program_modules_str"] = tea_val["program_modules"].apply(lambda modules: _get_concepts(modules))

tea_train_subset = tea_train[tea_train["program_modules_str"].isin(train_concepts)]
tea_val = tea_val[tea_val["program_modules_str"].isin(train_concepts)]  # subset of trained concepts

tea_val = tea_val[tea_val["answers_text"].apply(lambda ans: ans is not None
                                                            and isinstance(ans, str)
                                                            and len(ans.strip()) > 0)]


def per_concepts_eval_objective(lang_module: LangModule,
                                qa_df: pd.DataFrame,
                                concepts: List[str],
                                label: str) -> Objective:
    concept_df = qa_df[qa_df["program_modules_str"].isin(concepts)]

    from training.priming_objective import Priming
    eval_objective = Priming(lang_module,
                             max_eval_samples=eval_examples,
                             demos_selection_strategy="informative",  # TODO set
                             texts_or_path=[],
                             text_pair_or_path=[],
                             val_texts_or_path=concept_df["question_text"],
                             val_text_pair_or_path=concept_df["context_text"],
                             labels_or_path=[],
                             val_labels_or_path=concept_df["answers_text"],
                             train_question_categories=[],
                             val_question_categories=concept_df["program_modules_str"],
                             batch_size=1,
                             val_evaluators=[ROUGE(**{"additional_sep_char": "▁"})],
                             source_lang_id="en",
                             objective_id="teabreac_train-%s" % label)
    return eval_objective
