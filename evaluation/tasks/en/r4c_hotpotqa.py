import itertools
from typing import Optional, List, Union, Tuple

import pandas as pd
from datasets import load_dataset
from promptsource.templates import Template, DatasetTemplates

from evaluation.tasks.task import Task
from training.priming_objective import priming_formats


class R4CHotpotQATask(Task):

    def _construct_qa_prompt(self, question: str, context: str) -> str:
        return priming_formats["QA"][self.lang_id] % (question, context)

    @staticmethod
    def _informative_factor(r4c_explanations: List[List[Tuple[Union[str, int, Tuple[str, str, str]]]]]) -> List[str]:
        return [" ".join(explanation[-1][-2:]) for explanation in itertools.chain(*r4c_explanations)]

    def __init__(self,
                 lang_id: str,
                 template: Optional[Template] = DatasetTemplates("hotpot_qa/fullwiki")["generate_answer_affirmative"]):
        super().__init__()
        self.lang_id = lang_id
        self.template = template.name
        # train split is not used, since Tk-instruct uses it
        hotpot_qa_dataset = load_dataset("hotpot_qa", "fullwiki")["validation"]
        r4c_val = pd.read_json("https://raw.githubusercontent.com/naoya-i/r4c/master/corpus/dev_csf.json")
        hotpot_qa_annotated = hotpot_qa_dataset.filter(lambda sample: sample["id"] in r4c_val.columns)
        # hotpot_qa = pd.DataFrame(hotpot_qa_annotated).set_index("id", drop=True)

        info_factors = [self._informative_factor(r4c_val[sample_id]) for sample_id in r4c_val.columns]

        self.data = [(*template.apply(sample), info_factor)  # type: ignore
                     for sample, info_factor in zip(hotpot_qa_annotated, info_factors)]
