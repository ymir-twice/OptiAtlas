"""Maximizing this reward minimizes per-image RGB MSE against the explicit GT."""
from .pixel import reference_mse


class MSEReward:
    requires_reference_images = True
    supports_valid_mask = True

    def __init__(self):
        self.metadata = {"kind": "negative_mse", "data_range": 1.0,
                         "reward": "-mean((generated_RGB_01 - GT_RGB_01)^2)",
                         "reduction": "mean over RGB and pixels, separately for each image",
                         "minimum": -1.0, "maximum": 0.0,
                         "reference_images": "required, detached; never generation conditioning"}

    def __call__(self, images, prompts, reference_images=None, *, valid_mask=None):
        return -reference_mse(images, prompts, reference_images, valid_mask=valid_mask)

    def reward_components(self, images, prompts, reference_images=None, *, valid_mask=None):
        score = self(images, prompts, reference_images, valid_mask=valid_mask)
        return {"negative_mse": score, "score": score}
