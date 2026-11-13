import json

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool

from ..protocol import DRAW_PROMPT_DESCRIPTION


@register_tool("draw")
class DrawTool(BaseTool):
    description = "Generate a new image from a text prompt. Returns the generated image for visual inspection. Text-to-image only: no editing and no input images."
    parameters = {"type": "object", "properties": {
        "prompt": {"type": "string", "description": DRAW_PROMPT_DESCRIPTION}},
        "required": ["prompt"]}

    def __init__(self, draw_callback, cfg=None):
        super().__init__(cfg)
        self.draw_callback = draw_callback

    def call(self, params, **kwargs):
        args = self._verify_json_format_args(params, strict_json=True)
        if set(args) != {"prompt"} or not isinstance(args["prompt"], str) or not 1 <= len(args["prompt"].strip()) <= 10000:
            raise ValueError("draw accepts only a nonempty text prompt")
        result = self.draw_callback(args["prompt"])
        return [ContentItem(text=json.dumps(result, ensure_ascii=False)), ContentItem(image=result["image_path"])]
