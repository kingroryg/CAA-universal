import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.helpers import add_vector_from_position, find_instruction_end_postion, get_model_path, model_name_format
from utils.tokenize import (
    tokenize_llama_chat,
    tokenize_llama_base,
    ADD_FROM_POS_BASE,
    ADD_FROM_POS_CHAT,
    get_tokenizer_for_model,
)
from typing import Optional, Dict, Any, Tuple


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class ModelWrapper:
    def __init__(
        self,
        hf_token: str,
        model_name: str = "llama2",
        size: str = "7b",
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.use_chat = use_chat
        self.model_name = model_name
        self.model_size = size
        
        # Get the model path and tokenization function
        self.model_name_path = get_model_path(model_name, size, not use_chat)
        self.tokenize_fn, self.end_str_marker = get_tokenizer_for_model(self.model_name_path)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, token=hf_token
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, token=hf_token
        )
        
        # Load custom weights if provided
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))
        
        # Convert to half precision for larger models to save memory
        if size not in ["7b", "2b", "8b"]:
            self.model = self.model.half()
        
        self.model = self.model.to(self.device)
        
        # Set up end string marker for finding instruction end position
        self.END_STR = t.tensor(self.tokenizer.encode(self.end_str_marker)[1:]).to(
            self.device
        )
        
        # Wrap model layers to capture activations
        self._wrap_model_layers()

    def _wrap_model_layers(self):
        """Wrap model layers to capture activations"""
        # Different models have different layer structures
        # We need to identify the correct attribute names
        
        # Identify model architecture and layer structure
        model_config = self.model.config
        
        # Get the main model component (transformer)
        if hasattr(self.model, "model"):
            transformer = self.model.model
        else:
            transformer = self.model
        
        # Find the layers attribute
        layers_attr = None
        for attr in ["layers", "h", "decoder.layers", "transformer.h"]:
            if hasattr(transformer, attr.split('.')[0]):
                current = transformer
                for part in attr.split('.'):
                    if hasattr(current, part):
                        current = getattr(current, part)
                        layers_attr = attr
                    else:
                        break
                if layers_attr:
                    break
        
        if not layers_attr:
            raise ValueError(f"Could not find layers in model {self.model_name_path}")
        
        # Find the normalization layer
        norm = None
        for attr in ["norm", "ln_f", "final_layer_norm"]:
            if hasattr(transformer, attr):
                norm = getattr(transformer, attr)
                break
        
        if norm is None:
            raise ValueError(f"Could not find normalization layer in model {self.model_name_path}")
        
        # Get the embedding/unembedding matrix
        if hasattr(self.model, "lm_head"):
            unembed_matrix = self.model.lm_head
        else:
            # For models where lm_head is tied to input embeddings
            unembed_matrix = self.model.get_input_embeddings()
        
        # Wrap each layer
        layers = current  # current points to the layers after the loop above
        for i in range(len(layers)):
            layer = layers[i]
            layers[i] = BlockOutputWrapper(
                layer, unembed_matrix, norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self._get_layers():
            layer.save_internal_decodings = value

    def _get_layers(self):
        """Get the model layers regardless of architecture"""
        if hasattr(self.model, "model"):
            transformer = self.model.model
        else:
            transformer = self.model
            
        # Find the layers attribute
        for attr in ["layers", "h", "decoder.layers", "transformer.h"]:
            if hasattr(transformer, attr.split('.')[0]):
                current = transformer
                for part in attr.split('.'):
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        break
                return current
        
        raise ValueError(f"Could not find layers in model {self.model_name_path}")

    def set_from_positions(self, pos: int):
        self.from_pos = pos

    def generate(self, tokens, max_new_tokens=100):
        return self.model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=False
        )

    def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        tokens = self.tokenize_fn(
            self.tokenizer, user_input, model_output, system_prompt
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        output = self.generate(tokens, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
        return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> t.Tensor:
        tokens = self.tokenize_fn(
            self.tokenizer, user_input, model_output, system_prompt
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self._get_layers()[layer].block.self_attn.activations

    def set_add_activations(self, layer, activations):
        self._get_layers()[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self._get_layers()[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self._get_layers()[layer].dot_products

    def reset_all(self):
        for layer in self._get_layers():
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self._get_layers()):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self._get_layers()[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


# For backward compatibility
LlamaWrapper = ModelWrapper
