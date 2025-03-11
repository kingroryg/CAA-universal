from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE

# Gemma chat format
GEMMA_B_INST, GEMMA_E_INST = "<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"
GEMMA_E_RESP = "<end_of_turn>"

# Mistral chat format
MISTRAL_B_INST, MISTRAL_E_INST = "[INST]", "[/INST]"
MISTRAL_B_SYS = "<s>[INST] "
MISTRAL_E_SYS = " [/INST]"

# Llama 3 chat format
LLAMA3_B_INST, LLAMA3_E_INST = "<|begin_of_text|><|user|>\n", "<|end_of_text|>\n<|assistant|>\n"
LLAMA3_E_RESP = "<|end_of_text|>"
LLAMA3_B_SYS, LLAMA3_E_SYS = "<|begin_of_text|><|system|>\n", "<|end_of_text|>\n"


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_gemma(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        # Gemma doesn't have a dedicated system prompt format, so we prepend it to the user input
        user_input = f"{system_prompt}\n\n{user_input}"
    
    input_content += f"{GEMMA_B_INST}{user_input.strip()}{GEMMA_E_INST}"
    if model_output is not None:
        input_content += f"{model_output.strip()}{GEMMA_E_RESP}"
    
    return tokenizer.encode(input_content)


def tokenize_mistral(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    if system_prompt is not None:
        input_content = f"{MISTRAL_B_SYS}{system_prompt}\n\n{user_input.strip()}{MISTRAL_E_SYS}"
    else:
        input_content = f"{MISTRAL_B_INST} {user_input.strip()} {MISTRAL_E_INST}"
    
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    
    return tokenizer.encode(input_content)


def tokenize_llama3(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += f"{LLAMA3_B_SYS}{system_prompt.strip()}{LLAMA3_E_SYS}"
    
    input_content += f"{LLAMA3_B_INST}{user_input.strip()}{LLAMA3_E_INST}"
    if model_output is not None:
        input_content += f"{model_output.strip()}{LLAMA3_E_RESP}"
    
    return tokenizer.encode(input_content)


def get_tokenizer_for_model(model_name: str):
    """
    Get the appropriate tokenization function for a given model.
    
    Args:
        model_name: The model name or path
        
    Returns:
        A tuple of (tokenize_function, add_from_pos_marker)
    """
    model_name = model_name.lower()
    
    if "llama-3" in model_name or "llama3" in model_name:
        return tokenize_llama3, LLAMA3_E_INST
    elif "llama-2" in model_name or "llama2" in model_name:
        if "chat" in model_name or "instruct" in model_name:
            return tokenize_llama_chat, ADD_FROM_POS_CHAT
        else:
            return tokenize_llama_base, ADD_FROM_POS_BASE
    elif "gemma" in model_name:
        return tokenize_gemma, GEMMA_E_INST
    elif "mistral" in model_name or "mixtral" in model_name:
        return tokenize_mistral, MISTRAL_E_INST
    else:
        # Default to Llama 2 chat format for unknown models
        return tokenize_llama_chat, ADD_FROM_POS_CHAT
