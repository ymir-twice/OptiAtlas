"""Qwen adapter for bounded local optical knowledge retrieval."""
import json

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool

from ..knowledge import knowledge_lookup, validate_lookup_args


@register_tool("knowledge_lookup")
class KnowledgeLookupTool(BaseTool):
    description = (
        "Look up conditional optical principles and suggested 2D image checks in a versioned local "
        "52-category knowledge base. This is guidance, not evidence about the current image. "
        "Use category_ids=[] and an empty query to list all category IDs and names; "
        "otherwise use exact IDs and/or a short English or Chinese query. Returns at most 5 cards."
    )
    # BaseTool validates this top-level shape; unknown arguments are rejected below.
    parameters = {
        "type": "object",
        "properties": {
            "category_ids": {
                "type": "array", "items": {"type": "string"}, "maxItems": 5,
                "description": "Exact taxonomy IDs in priority order, or [] to search/list categories.",
            },
            "query": {"type": "string", "maxLength": 1000, "default": ""},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        },
        "required": ["category_ids"],
    }

    def __init__(self, callback=knowledge_lookup, cfg=None):
        super().__init__(cfg)
        self.callback = callback

    def call(self, params, **kwargs):
        args = self._verify_json_format_args(params, strict_json=True)
        if not isinstance(args, dict) or set(args) - {"category_ids", "query", "max_results"} or "category_ids" not in args:
            raise ValueError("knowledge_lookup accepts category_ids, query, and max_results only")
        category_ids = args["category_ids"]
        query = args.get("query", "")
        max_results = args.get("max_results", 3)
        validate_lookup_args(category_ids, query, max_results)
        result = self.callback(category_ids=category_ids, query=query, max_results=max_results)
        return [ContentItem(text=json.dumps(result, ensure_ascii=False))]
