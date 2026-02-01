import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so100_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(12),
        "observation/images.main.left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class S0100Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi05 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType 

    
    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI05  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        # For the SO100 the state is of size 6 so we pad
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        wrist_image_right = _parse_image(data["observation/images/right"])

        images = {
            "base_0_rgb": np.zeros_like(wrist_image_right),
            "left_wrist_0_rgb": np.zeros_like(wrist_image_right),
            "right_wrist_0_rgb": wrist_image_right,
        }

        image_masks = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.False_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }
        
        # Actions are only available during training.
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "pick up the banana and put it into the box"

        return inputs


@dataclasses.dataclass(frozen=True)
class S0100Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Make sure to only return the appropriate number of actions here
        # 6 for 1 robot, 12 for 2
        return {"actions": np.asarray(data["actions"][:, :6])}
    
if __name__ == "__main__":
    example = make_so100_example()
    print("Example input:", example)
    so100_inputs = S0100Inputs(action_dim=6, model_type=_model.ModelType.PI0)(example)
    print("SO100 model inputs:", so100_inputs)