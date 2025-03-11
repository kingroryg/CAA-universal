"""
Usage: python analyze_vectors.py

For newer models:
python analyze_vectors.py --model_name llama3 --model_size 8b
"""

import os
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from behaviors import ALL_BEHAVIORS, get_analysis_dir, HUMAN_NAMES, get_steering_vector, ANALYSIS_PATH
from utils.helpers import get_model_path, model_name_format, set_plotting_settings
from tqdm import tqdm

set_plotting_settings()

def get_caa_info(behavior: str, model_name: str, model_size: str, is_base: bool):
    all_vectors = []
    
    # Determine number of layers based on model
    if model_name == "llama2":
        if model_size == "7b":
            n_layers = 32
        elif model_size == "13b":
            n_layers = 40
        elif model_size == "70b":
            n_layers = 80
    elif model_name == "llama3":
        if model_size == "8b":
            n_layers = 32
        elif model_size == "70b":
            n_layers = 80
    elif model_name == "gemma":
        if model_size == "2b":
            n_layers = 18
        elif model_size == "7b":
            n_layers = 28
    elif model_name == "mistral":
        n_layers = 32
    elif model_name == "mixtral":
        n_layers = 32
    else:
        n_layers = 32  # Default
    
    model_path = get_model_path(model_name, model_size, is_base)
    
    # Check if vectors exist
    for layer in range(n_layers):
        try:
            vector = get_steering_vector(behavior, layer, model_path)
            all_vectors.append(vector)
        except:
            print(f"Warning: Vector not found for {behavior}, layer {layer}, model {model_path}")
            break
    
    if not all_vectors:
        return None
    
    return {
        "vectors": all_vectors,
        "n_layers": len(all_vectors),
        "model_name": model_name_format(model_path),
    }

def plot_per_layer_similarities(model_name: str, model_size: str, is_base: bool, behavior: str):
    analysis_dir = get_analysis_dir(behavior)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        
    caa_info = get_caa_info(behavior, model_name, model_size, is_base)
    if caa_info is None:
        print(f"Skipping {behavior} for {model_name} {model_size} (vectors not found)")
        return
        
    all_vectors = caa_info["vectors"]
    n_layers = caa_info["n_layers"]
    model_name_str = caa_info["model_name"]
    
    matrix = np.zeros((n_layers, n_layers))
    for layer1 in range(n_layers):
        for layer2 in range(n_layers):
            cosine_sim = t.nn.functional.cosine_similarity(all_vectors[layer1], all_vectors[layer2], dim=0).item()
            matrix[layer1, layer2] = cosine_sim
    
    plt.figure(figsize=(3, 3))
    sns.heatmap(matrix, annot=False, cmap='coolwarm')
    
    # Set ticks for every 5th layer
    tick_interval = max(1, n_layers // 8)  # Adjust interval based on number of layers
    plt.xticks(list(range(n_layers))[::tick_interval], list(range(n_layers))[::tick_interval])
    plt.yticks(list(range(n_layers))[::tick_interval], list(range(n_layers))[::tick_interval])
    
    plt.title(f"Layer similarity, {model_name_str}", fontsize=11)
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name_str.replace(' ', '_')}_{behavior}.svg"), format='svg')
    plt.close()

def plot_base_chat_similarities(model_name: str, model_size: str):
    plt.figure(figsize=(5, 3))
    
    for behavior in ALL_BEHAVIORS:
        base_caa_info = get_caa_info(behavior, model_name, model_size, True)
        chat_caa_info = get_caa_info(behavior, model_name, model_size, False)
        
        if base_caa_info is None or chat_caa_info is None:
            print(f"Skipping {behavior} for {model_name} {model_size} (vectors not found)")
            continue
            
        vectors_base = base_caa_info["vectors"]
        vectors_chat = chat_caa_info["vectors"]
        
        # Use the minimum number of layers between base and chat models
        n_layers = min(base_caa_info["n_layers"], chat_caa_info["n_layers"])
        
        cos_sims = []
        for layer in range(n_layers):
            cos_sim = t.nn.functional.cosine_similarity(vectors_base[layer], vectors_chat[layer], dim=0).item()
            cos_sims.append(cos_sim)
            
        plt.plot(list(range(n_layers)), cos_sims, label=HUMAN_NAMES[behavior], linestyle="solid", linewidth=2)
    
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Base vs. Chat model vector similarity ({model_name} {model_size})", fontsize=12)
    # legend in bottom right
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, f"base_chat_similarities_{model_name}_{model_size}.png"), format='png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2", 
                        help="Model name (e.g., llama2, llama3, gemma, mistral, mixtral)")
    parser.add_argument("--model_size", type=str, default="7b",
                        help="Model size (e.g., 7b, 13b, 8b, 2b, 8x7b)")
    args = parser.parse_args()
    
    for behavior in tqdm(ALL_BEHAVIORS):
        plot_per_layer_similarities(args.model_name, args.model_size, True, behavior)
        plot_per_layer_similarities(args.model_name, args.model_size, False, behavior)
    
    plot_base_chat_similarities(args.model_name, args.model_size)
