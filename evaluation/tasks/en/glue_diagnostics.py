from datasets import load_dataset
from promptsource.templates import DatasetTemplates

from evaluation.tasks.task import Task

seen_premises = set()


class GLUEDiagnostics(Task):

    def __init__(self, lang_id: str = "en"):
        super().__init__(".")
        self.lang_id = lang_id

        dataset = load_dataset("pietrolesci/glue_diagnostics")["test"]
        template = DatasetTemplates('glue/mnli')["GPT-3 style"]

        # this will remove the samples with the identical premise from the evaluation
        duplicates_map = []
        for sample in dataset:
            identifier = sample["logic"] + " ".join(sorted(sample["premise"].split()))
            duplicates_map.append(identifier in seen_premises)
            seen_premises.add(identifier)

        self.data = [(*template.apply(sample), sample["logic"]) for sample, is_duplicate  # type: ignore
                     in zip(dataset, duplicates_map) if sample["logic"] and not is_duplicate]
