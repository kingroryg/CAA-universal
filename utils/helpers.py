import torch as t
import matplotlib.pyplot as plt

def set_plotting_settings():
    try:
        # Try modern seaborn style first
        plt.style.use('seaborn-v0_8')
    except:
        try:
            # Fall back to older seaborn style
            plt.style.use('seaborn')
        except:
            # If all else fails, use default style
            pass
    
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)



def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    mask = position_ids >= from_id
    mask = mask.unsqueeze(-1)

    matrix += mask.float() * vector
    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_path(model_name: str, size: str = None, is_base: bool = None):
    """
    Get the Huggingface model path for a given model name and configuration.
    
    Args:
        model_name: The model name (e.g., "llama2", "llama3", "gemma", etc.)
        size: The model size (e.g., "7b", "13b", "8b", etc.)
        is_base: Whether to use the base model or the chat/instruct model
        
    Returns:
        The Huggingface model path
    """
    model_name = model_name.lower()
    
    # Handle legacy format (for backward compatibility)
    if model_name == "llama2" or model_name == "llama-2":
        if is_base is None:
            is_base = False  # Default to chat model for Llama 2
        if size is None:
            size = "7b"  # Default to 7b
        
        if is_base:
            return f"meta-llama/Llama-2-{size}-hf"
        else:
            return f"meta-llama/Llama-2-{size}-chat-hf"
    
    # Handle Llama 3 models
    elif model_name == "llama3" or model_name == "llama-3":
        if size is None:
            size = "8b"  # Default to 8b
        if is_base is None:
            is_base = False  # Default to instruct model
            
        if is_base:
            return f"meta-llama/Llama-3-{size}-hf"
        else:
            return f"meta-llama/Llama-3-{size}-Instruct-hf"
    
    # Handle Gemma models
    elif model_name == "gemma":
        if size is None:
            size = "7b"  # Default to 7b
        if is_base is None:
            is_base = False  # Default to instruct model
            
        if is_base:
            return f"google/gemma-{size}"
        else:
            return f"google/gemma-{size}-it"
    
    # Handle Mistral models
    elif model_name == "mistral":
        if size is None:
            size = "7b"  # Default to 7b
        if is_base is None:
            is_base = False  # Default to instruct model
            
        if is_base:
            return f"mistralai/Mistral-{size}-v0.1"
        else:
            return f"mistralai/Mistral-{size}-Instruct-v0.1"
    
    # Handle Mixtral models
    elif model_name == "mixtral":
        if size is None:
            size = "8x7b"  # Default size
        if is_base is None:
            is_base = False  # Default to instruct model
            
        if is_base:
            return f"mistralai/Mixtral-{size}-v0.1"
        else:
            return f"mistralai/Mixtral-{size}-Instruct-v0.1"
    
    # For direct model paths (fallback)
    else:
        return model_name

def model_name_format(name: str) -> str:
    """
    Format the model name for display purposes.
    
    Args:
        name: The model path or name
        
    Returns:
        A formatted model name for display
    """
    name = name.lower()
    
    # Llama 2 models
    if "llama-2" in name or "llama2" in name:
        is_chat = "chat" in name or "instruct" in name
        is_7b = "7b" in name
        is_13b = "13b" in name
        is_70b = "70b" in name
        
        size = "7B" if is_7b else "13B" if is_13b else "70B" if is_70b else "Unknown"
        variant = "Chat" if is_chat else ""
        
        return f"Llama 2 {variant} {size}".strip()
    
    # Llama 3 models
    elif "llama-3" in name or "llama3" in name:
        is_instruct = "instruct" in name
        is_8b = "8b" in name
        is_70b = "70b" in name
        
        size = "8B" if is_8b else "70B" if is_70b else "Unknown"
        variant = "Instruct" if is_instruct else ""
        
        return f"Llama 3 {variant} {size}".strip()
    
    # Gemma models
    elif "gemma" in name:
        is_instruct = "it" in name or "instruct" in name
        is_2b = "2b" in name
        is_7b = "7b" in name
        
        size = "2B" if is_2b else "7B" if is_7b else "Unknown"
        variant = "Instruct" if is_instruct else ""
        
        return f"Gemma {variant} {size}".strip()
    
    # Mistral models
    elif "mistral" in name and "mixtral" not in name:
        is_instruct = "instruct" in name
        is_7b = "7b" in name
        
        size = "7B" if is_7b else "Unknown"
        variant = "Instruct" if is_instruct else ""
        
        return f"Mistral {variant} {size}".strip()
    
    # Mixtral models
    elif "mixtral" in name:
        is_instruct = "instruct" in name
        is_8x7b = "8x7b" in name
        
        size = "8x7B" if is_8x7b else "Unknown"
        variant = "Instruct" if is_instruct else ""
        
        return f"Mixtral {variant} {size}".strip()
    
    # For other models, just return a cleaned up version of the path
    else:
        # Extract the model name from the path
        if "/" in name:
            name = name.split("/")[-1]
        
        # Clean up common patterns
        name = name.replace("-hf", "").replace("-v0.1", "")
        
        # Capitalize words
        return " ".join(word.capitalize() for word in name.split("-"))
