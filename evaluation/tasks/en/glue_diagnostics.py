from datasets import load_dataset
from promptsource.templates import DatasetTemplates

from evaluation.tasks.task import Task


class GLUEDiagnostics(Task):

    def __init__(self, lang_id: str = "en"):
        super().__init__(".")
        self.lang_id = lang_id

        dataset = load_dataset("pietrolesci/glue_diagnostics")["test"]
        template = DatasetTemplates('glue/mnli')["GPT-3 style"]
        # TODO: demonstrations are too often very similar to the predicted sample
        self.data = [(*template.apply(sample), sample["logic"]) for sample in dataset]  # type: ignore
