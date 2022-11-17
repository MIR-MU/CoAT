from typing import Tuple, List, Union


def selection_criterion(predicted_example: Tuple[str, str, Union[str, List[str]]],
                        candidate_demonstration: Tuple[str, str, Union[str, List[str]]],
                        demo_selection_strategy: str) -> bool:
    if demo_selection_strategy == "random":
        return True
    elif demo_selection_strategy == "cluster-random":
        # any sample is fine for demonstration, with the random selection strategy
        pred_cat = predicted_example[2]
        cand_cat = candidate_demonstration[2]
        assert type(pred_cat) == type(cand_cat), "Informativeness factor of all samples must be of the same type."
        if isinstance(pred_cat, list):
            return any(cand_c in pred_cat for cand_c in cand_cat)
        else:
            return pred_cat == cand_cat
    else:
        raise ValueError("Demo selection strategy %s unknown." % demo_selection_strategy)


def construct_sample(demonstrations: List[Tuple[str, str, str]],
                     predicted_sample: Tuple[str, str, str]) -> str:
    demonstrations = "\n".join(["Input: %s Prediction: %s" % demo[:2] for demo in demonstrations])
    return demonstrations + "\nInput: %s Prediction:" % predicted_sample[0]
