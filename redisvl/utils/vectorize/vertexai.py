from typing import Any, List, Optional

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.genai import GenAIVectorizer


@deprecated_class("VertexAIVectorizer", "Use GenAIVectorizer instead.")
class VertexAIVectorizer(GenAIVectorizer):
    """A backwards-compatible alias for GenAIVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: str = "", text: Any = "", **kwargs) -> List[float]:
        """Generate a vector embedding for a single input using the VertexAI API.

        Deprecated: Use `GenAIVectorizer.embed` instead.
        """
        content = content or text
        return super().embed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: Optional[List[str]] = None,
        texts: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of inputs using the VertexAI API.

        Deprecated: Use `GenAIVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)
