"""Lazy model adapters; importing the CLI does not load model libraries."""


def build_adapter(config):
    options = dict(config)
    name = options.pop("name")
    if name == "sensenova_u15":
        from .sensenova import SenseNovaAdapter
        return SenseNovaAdapter(options)
    if name == "qwen_image21":
        from .qwen_image21 import QwenImage21Adapter
        return QwenImage21Adapter(options)
    raise ValueError(f"Unknown model adapter: {name}")
