"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b

For newer models:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --model_name llama3 --model_size 8b
"""

import json
from llama_wrapper import ModelWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_open_ended(
    item: Dict[str, str],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=500
    )
    return {
        "question": question,
        "model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    choices = item["choices"]
    answer = item["answer"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=500
    )
    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "model_output": model_output,
    }


def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    Test steering on a given behavior
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        "open_ended": get_open_ended_test_data(settings.behavior),
        "truthful_qa": get_truthful_qa_data(),
        "mmlu": get_mmlu_data(),
    }
    model = ModelWrapper(
        HUGGINGFACE_TOKEN, 
        model_name=settings.model_name,
        size=settings.model_size, 
        use_chat=not settings.use_base_model
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        if settings.override_vector is not None:
            vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
        else:
            vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
        if settings.model_size != "7b":
            vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--layers", nargs="+", type=int, default=[10])
    parser.add_argument(
        "--multipliers", nargs="+", type=float, default=[0.1, 0.5, 1, 2, 5, 10]
    )
    parser.add_argument(
        "--type", type=str, choices=["ab", "open_ended", "tqa", "mmlu"], default="ab"
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="llama2", 
                        help="Model name (e.g., llama2, llama3, gemma, mistral, mixtral)")
    parser.add_argument("--model_size", type=str, default="7b",
                        help="Model size (e.g., 7b, 13b, 8b, 2b, 8x7b)")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--normalized", action="store_true", default=False)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--use_pos_system_prompt", action="store_true", default=False)
    parser.add_argument("--use_neg_system_prompt", action="store_true", default=False)

    args = parser.parse_args()
    settings = SteeringSettings(
        behaviors=args.behaviors,
        type=args.type,
        use_base_model=args.use_base_model,
        model_name=args.model_name,
        model_size=args.model_size,
        normalized=args.normalized,
        system_prompt=args.system_prompt,
        use_pos_system_prompt=args.use_pos_system_prompt,
        use_neg_system_prompt=args.use_neg_system_prompt,
    )
    test_steering(args.layers, args.multipliers, settings, args.overwrite)
