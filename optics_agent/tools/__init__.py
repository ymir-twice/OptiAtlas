"""Native tools; research integrations load only when explicitly enabled."""
from .draw import DrawTool

__all__ = ["DrawTool", "KnowledgeLookupTool", "ImageSearchTool"]


def __getattr__(name):
    if name == "KnowledgeLookupTool":
        from .knowledge import KnowledgeLookupTool
        return KnowledgeLookupTool
    if name == "ImageSearchTool":
        from .image_search import ImageSearchTool
        return ImageSearchTool
    raise AttributeError(name)
