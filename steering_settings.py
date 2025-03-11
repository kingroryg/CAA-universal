from typing import Optional, Literal, List, Union
import os
from dataclasses import dataclass, field
from behaviors import ALL_BEHAVIORS
from utils.helpers import model_name_format

@dataclass
class SteeringSettings:
    behaviors: List[str] = field(default_factory=lambda: ["sycophancy"])
    behavior: str = None
    type: Literal["open_ended", "ab", "tqa", "mmlu"] = "ab"
    system_prompt: Optional[str] = None
    use_pos_system_prompt: bool = False
    use_neg_system_prompt: bool = False
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = False
    model_name: str = "llama2"
    model_size: str = "7b"
    normalized: bool = False
    override_model_weights_path: Optional[str] = None

    def __post_init__(self):
        # Set behavior to the first behavior in the list if not specified
        if self.behavior is None and self.behaviors:
            self.behavior = self.behaviors[0]
        
        assert self.behavior in ALL_BEHAVIORS, f"Invalid behavior {self.behavior}"
        
    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[Union[int, float]] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "use_pos_system_prompt": self.use_pos_system_prompt if self.use_pos_system_prompt else None,
            "use_neg_system_prompt": self.use_neg_system_prompt if self.use_neg_system_prompt else None,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "normalized": self.normalized if self.normalized else None,
            "override_model_weights_path": self.override_model_weights_path,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[Union[int, float]] = None,
    ):
        elements = {
            "layer": str(layer)+"_",
            "multiplier": str(float(multiplier))+"_",
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "use_pos_system_prompt": self.use_pos_system_prompt if self.use_pos_system_prompt else None,
            "use_neg_system_prompt": self.use_neg_system_prompt if self.use_neg_system_prompt else None,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "normalized": self.normalized if self.normalized else None,
            "override_model_weights_path": self.override_model_weights_path,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]
    
    def get_formatted_model_name(self):
        """Get a formatted model name for display purposes"""
        from utils.helpers import get_model_path
        
        # Get the model path
        model_path = get_model_path(self.model_name, self.model_size, self.use_base_model)
        
        # Format the model name
        return model_name_format(model_path)
        
