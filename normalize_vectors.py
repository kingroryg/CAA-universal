from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os
import argparse

def normalize_vectors(model_name: str, model_size: str, is_base: bool, n_layers: int):
    """
    Normalize vectors for a specific model configuration
    
    Args:
        model_name: Name of the model (e.g., llama2, llama3, gemma)
        model_size: Size of the model (e.g., 7b, 13b, 8b)
        is_base: Whether to use the base model or the chat/instruct model
        n_layers: Number of layers in the model
    """
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    
    for layer in range(n_layers):
        print(f"Processing layer {layer}")
        norms = {}
        vecs = {}
        new_paths = {}
        
        for behavior in ALL_BEHAVIORS:
            model_path = get_model_path(model_name, model_size, is_base=is_base)
            vec_path = get_vector_path(behavior, layer, model_path)
            
            # Check if vector exists
            if not os.path.exists(vec_path):
                print(f"Warning: Vector not found for {behavior}, layer {layer}, model {model_path}")
                continue
                
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        
        if not vecs:
            print(f"No vectors found for layer {layer}, skipping")
            continue
            
        print(f"Norms: {norms}")
        mean_norm = t.tensor(list(norms.values())).mean().item()
        
        # normalize all vectors to have the same norm
        for behavior in vecs.keys():
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        
        # save the normalized vectors
        for behavior in vecs.keys():
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
            print(f"Saved normalized vector for {behavior}, layer {layer}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2", 
                        help="Model name (e.g., llama2, llama3, gemma, mistral, mixtral)")
    parser.add_argument("--model_size", type=str, default="7b",
                        help="Model size (e.g., 7b, 13b, 8b, 2b, 8x7b)")
    parser.add_argument("--use_base_model", action="store_true", default=False,
                        help="Whether to use the base model instead of the chat/instruct model")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Number of layers in the model (if not specified, will use default for model)")
    
    args = parser.parse_args()
    
    # Set default number of layers based on model
    if args.n_layers is None:
        if args.model_name == "llama2":
            if args.model_size == "7b":
                args.n_layers = 32
            elif args.model_size == "13b":
                args.n_layers = 40
            elif args.model_size == "70b":
                args.n_layers = 80
        elif args.model_name == "llama3":
            if args.model_size == "8b":
                args.n_layers = 32
            elif args.model_size == "70b":
                args.n_layers = 80
        elif args.model_name == "gemma":
            if args.model_size == "2b":
                args.n_layers = 18
            elif args.model_size == "7b":
                args.n_layers = 28
        elif args.model_name == "mistral":
            args.n_layers = 32
        elif args.model_name == "mixtral":
            args.n_layers = 32
        else:
            args.n_layers = 32  # Default
    
    normalize_vectors(args.model_name, args.model_size, args.use_base_model, args.n_layers)