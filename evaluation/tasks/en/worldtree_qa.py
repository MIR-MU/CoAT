import logging
import os
import zipfile
from typing import List, Optional

import pandas as pd

from evaluation.tasks.task import Task

logger = logging.getLogger()

answers_ranking = ["(A)", "(B)", "(C)", "(D)", "(E)", "(XX)", "(0)", "(1)", "(2)", "(3)", "(4)", "(5)", "(XX)"]


class WorldTreeQA(Task):
    url: str = "https://cognitiveai.org/dist/WorldtreeExplanationCorpusV2.1_Feb2020.zip"

    @staticmethod
    def _explanations_pair_match(expls1: List[str], expls2: List[str]) -> float:
        coverage = len([expl1 in expls2 for expl1 in expls1])
        max_norm = max(len(expls1), len(expls2))
        return coverage / max_norm

    @staticmethod
    def _answer_by_key(full_question: str, answer_key: str) -> str:
        answer_rank = answers_ranking.index("(%s)" % answer_key)
        correct_option = full_question.split(" (%s) " % answer_key)[1].split(answers_ranking[answer_rank + 1])[
            0].strip()
        return correct_option

    @staticmethod
    def _context_from_explanations(expls_str: List[str]) -> str:
        return ". ".join(expls_str) + ". "

    def _load_explanations(self) -> pd.Series:
        explanations_root = os.path.join(self.cache_dir, "WorldtreeExplanationCorpusV2.1_Feb2020/tablestore/v2.1")
        index_f = os.path.join(explanations_root, "tableindex.txt")
        explanations_dfs = [pd.read_csv(os.path.join(explanations_root, "tables", fname.strip()), sep="\t").fillna("")
                            for fname in open(index_f).readlines()]
        expls = [df.apply(lambda row: " ".join(row[[k for k in row.keys() if "SKIP" not in k]].astype(str)), axis=1)
                 for df in explanations_dfs]
        explanations_contents_s = pd.concat(expls)
        explanations_contents_s.index = pd.concat([df["[SKIP] UID"] for df in explanations_dfs])
        # index duplicates cause inconsistent types of .loc[] access
        explanations_contents_s = explanations_contents_s[~explanations_contents_s.index.duplicated(keep='first')]

        return explanations_contents_s

    def __init__(self, *_, num_demonstrations: int = 3, difficulty: Optional[str] = None):
        # we ignore shared lang_id and template attributes
        super().__init__()

        with zipfile.ZipFile(self.data_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        expected_questions_path = "WorldtreeExplanationCorpusV2.1_Feb2020/questions/questions.train.tsv"
        assert os.path.exists(expected_questions_path), "Unexpected file structure"

        questions_df = pd.read_csv(expected_questions_path, sep="\t")
        questions_df = questions_df[~questions_df.explanation.isna()]
        questions_df = questions_df.set_index("QuestionID", drop=True)

        if difficulty is not None:
            questions_df = questions_df[questions_df.arcset == difficulty]
            assert len(questions_df), "Invalid difficulty. Options: Easy, Challenge."

        explanations = questions_df.explanation.apply(lambda expls: [expl.split()[-1] for expl in expls.split("|")
                                                                     if not expl.split()[-1].isupper()])
        questions_df.explanation = explanations
        expls_texts_s = self._load_explanations()
        questions_df["expl_texts"] = questions_df.explanation.apply(lambda expl_ids: [expls_texts_s.loc[expl_id]
                                                                                      for expl_id in expl_ids])

        # eagerly group the questions by the max intersection of the explanations
        non_assigned_expls_idx = questions_df.explanation.index.copy(deep=True).tolist()
        expls_idx_to_clusters = {}
        cluster_to_explanations = {}
        next_cluster = 0

        while non_assigned_expls_idx:
            expls_id_to_assign = non_assigned_expls_idx.pop()
            expls_idx_to_clusters[expls_id_to_assign] = next_cluster

            expls_to_assign = explanations.loc[expls_id_to_assign]

            for _ in range(num_demonstrations):
                score = 0
                best_demo_expls_id = None
                if not non_assigned_expls_idx:
                    logger.warning("Leaving %s demonstrations unpaired.", num_demonstrations)
                    break
                for cand_demo_idx in non_assigned_expls_idx:
                    cand_expls = explanations.loc[cand_demo_idx]
                    cand_score = self._explanations_pair_match(expls_to_assign, cand_expls)
                    if cand_score > score:
                        best_demo_expls_id = cand_demo_idx

                expls_idx_to_clusters[best_demo_expls_id] = next_cluster
                expls_to_assign = set(expls_to_assign).union(set(explanations.loc[best_demo_expls_id]))
                non_assigned_expls_idx.remove(best_demo_expls_id)

            cluster_to_explanations[next_cluster] = [expls_texts_s.loc[expl_id].strip() for expl_id in expls_to_assign]
            next_cluster += 1

        # reorder the clusters' assignment according to the shared index
        questions_df["cluster"] = pd.Series(expls_idx_to_clusters)
        questions_df["cluster_explanations"] = questions_df["cluster"].apply(lambda cluster_id:
                                                                             cluster_to_explanations[cluster_id])

        # finally, persist the synchronised data -- construct contexts from the union of explanations
        self.data = questions_df.apply(
            lambda sample: ("%s %s" % (self._context_from_explanations(sample.cluster_explanations),
                                       sample.question),
                            self._answer_by_key(sample.question, sample.AnswerKey),
                            str(sample.cluster)), axis=1).tolist()
