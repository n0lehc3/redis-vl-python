import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.vectorize.base import BaseVectorizer


class GenAIVectorizer(BaseVectorizer):
    """The GenAIVectorizer uses Google's Gemini embedding models
    API to create embeddings via the Google Gen AI SDK.

    This vectorizer supports two modes of operation:

    1. **Vertex AI Mode** (default): For use with Google Cloud Platform (GCP).
       Requires an active GCP project and location (region). Credentials can be
       provided via GOOGLE_APPLICATION_CREDENTIALS environment variable or
       explicitly in api_config.

    2. **Gemini API Mode**: For use with Google's Gemini API. Requires a Google
       API key, which can be obtained from Google AI Studio
       (https://aistudio.google.com/api-keys).

    The google-genai library must be installed:
    `pip install google-genai>=1.47.0`

    You can optionally enable caching to improve performance when generating
    embeddings for repeated inputs.

    **Examples:**

    .. code-block:: python

        # Vertex AI Mode (default)
        vectorizer = GenAIVectorizer(
            model="gemini-embedding-001",
            api_config={
                "project_id": "your_gcp_project_id",  # OR set GCP_PROJECT_ID env var
                "location": "your_gcp_location",      # OR set GCP_LOCATION env var
                "use_vertexai": True  # Default
            }
        )
        embedding = vectorizer.embed("Hello, world!")

        # Gemini API Mode
        vectorizer = GenAIVectorizer(
            model="gemini-embedding-001",
            api_config={
                "api_key": "your_google_api_key",  # OR set GOOGLE_API_KEY env var
                "use_vertexai": False
            }
        )
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="genai_embeddings_cache")

        vectorizer = GenAIVectorizer(
            model="gemini-embedding-001",
            api_config={
                "api_key": "your_google_api_key",
                "use_vertexai": False
            },
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Batch embedding of multiple texts
        embeddings = vectorizer.embed_many(
            ["Hello, world!", "Goodbye, world!"],
            batch_size=2
        )

        # Multimodal usage - Only available in Vertex AI mode
        # (text, image, and video embeddings)
        from google.genai.types import Image, Video

        vectorizer = GenAIVectorizer(
            model="gemini-embedding-001",
            api_config={
                "project_id": "your_gcp_project_id",
                "location": "your_gcp_location",
                "use_vertexai": True  # Multimodal requires Vertex AI
            }
        )
        text_embedding = vectorizer.embed("Hello, world!")
        image_embedding = vectorizer.embed(Image.from_file("path/to/image.jpg"))
        video_embedding = vectorizer.embed(Video.from_file("path/to/video.mp4"))

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        dims: Optional[int] = 3072,
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the GenAI vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to
                'gemini-embedding-001'.
                supported models for text embeddings: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#supported-models
            api_config (Optional[Dict], optional): Dictionary containing API configuration.
                Defaults to None. Configuration depends on the mode:

                For Vertex AI  (use_vertexai=True):
                    - "use_vertexai" (bool): Set to True for Vertex AI . Defaults to True.
                    - "project_id" (str): GCP project ID. Can also use GCP_PROJECT_ID env var.
                    - "location" (str): GCP region/location. Can also use GCP_LOCATION env var.
                    - "credentials": Optional explicit credentials object.

                For Gemini API  (use_vertexai=False):
                    - "use_vertexai" (bool): Set to False for Gemini API.
                    - "api_key" (str): Google API key. Can also use GOOGLE_API_KEY env var.

            dims (Optional[int]): The dimensionality of the embeddings produced by the model.
                Defaults to 3072, which is the dimension for Google's Gemini embedding models.
                https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the google-genai library is not installed.
            ValueError: If required configuration (project_id/location for Vertex AI  or
                api_key for Gemini API ) is not provided.
            ValueError: If an invalid dtype is provided.

        References:
            - Vertex AI SDK Migration: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk#embeddings
            - Google Gen AI SDK: https://googleapis.github.io/python-genai/genai.html#genai.models.Models.embed_content
        """
        super().__init__(model=model, dtype=dtype, cache=cache, dims=dims)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    @property
    def is_multimodal(self) -> bool:
        """Whether multimodal embedding is supported.

        Multimodal embedding is only available in Vertex AI.
        """
        return (
            self._client.vertexai
        )  # technically not enough, you can still use a text-only model in vertexai, and having 'multimodal' in the model name is not a guarantee either.

    def embed_image(self, image_path: str, **kwargs) -> Union[List[float], bytes]:
        """Embed an image (from its path on disk) using a Google Gen AI multimodal model.

        Note: Multimodal embedding is only available in Vertex AI .
        """

        from google.genai.types import Image

        if not self.is_multimodal:
            raise NotImplementedError(
                "Image embedding is only supported in Vertex AI with multimodal models."
            )
        return self.embed(Image.from_file(location=image_path), **kwargs)

    def embed_video(self, video_path: str, **kwargs) -> Union[List[float], bytes]:
        """Embed a video (from its path on disk) using a Google Gen AI multimodal model.

        Note: Multimodal embedding is only available in Vertex AI.
        """
        from google.genai.types import Video

        if not self.is_multimodal:
            raise NotImplementedError(
                "Video embedding is only supported in Vertex AI with multimodal models."
            )
        return self.embed(Video.from_file(location=video_path), **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the Google Gen AI client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Initialize the Google Gen AI client in either Vertex AI or Gemini API.

        Args:
            api_config: Dictionary with GCP configuration options:
                For Vertex AI:
                    - use_vertexai: True (or omit, default is True)
                    - project_id: GCP project ID (or GCP_PROJECT_ID env var)
                    - location: GCP region (or GCP_LOCATION env var)
                    - credentials: Optional explicit credentials object

                For Gemini API:
                    - use_vertexai: False
                    - api_key: Google API key (or GOOGLE_API_KEY env var)

            **kwargs: Additional arguments for client initialization

        Raises:
            ImportError: If google-genai library is not installed
            ValueError: If required parameters are not provided for the chosen mode
        """

        # Fetch the project_id and location from api_config or environment variables
        use_vertexai = bool(
            api_config.get("use_vertexai")
            if api_config
            else os.getenv("USE_VERTEXAI", "true").lower() == "true"
        )  # default to true since this is replacing the previous VertexAIVectorizer which requires a GCP project.
        if use_vertexai:
            project_id = (
                api_config.get("project_id")
                if api_config
                else os.getenv("GCP_PROJECT_ID")
            )
            location = (
                api_config.get("location") if api_config else os.getenv("GCP_LOCATION")
            )

            if not project_id:
                raise ValueError(
                    "Missing project_id. "
                    "Provide the id in the api_config with key 'project_id' "
                    "or set the GCP_PROJECT_ID environment variable."
                )

            if not location:
                raise ValueError(
                    "Missing location. "
                    "Provide the location (region) in the api_config with key 'location' "
                    "or set the GCP_LOCATION environment variable."
                )

            # Check for credentials
            credentials = api_config.get("credentials") if api_config else None
            kwargs["project"] = (
                project_id  # genai sdk uses 'project' but keeping 'project_id' for backwards compatibility with VertexAIVectorizer
            )
            kwargs["location"] = location
            kwargs["credentials"] = credentials
        else:
            api_key = (
                api_config.get("api_key") if api_config else os.getenv("GOOGLE_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Missing API key. "
                    "Provide the key in the api_config with key 'api_key' or set the GOOGLE_API_KEY environment variable."
                )
            kwargs["api_key"] = api_key
        kwargs["vertexai"] = use_vertexai

        try:
            from google import genai

            self._client = genai.Client(
                **kwargs,
            )

        except ImportError:
            raise ImportError(
                "GenAI vectorizer requires the google-genai library. "
                "Please install with `pip install google-genai>=1.47.0`."
            )

    def _preprocess(self, content: Any) -> Any:
        """Preprocess content for embedding"""
        from google.genai.types import Image, Part, Video

        if isinstance(content, str):
            return content
        elif isinstance(content, Image):
            if not content.image_bytes or not content.mime_type:
                raise ValueError("Invalid Image content")
            content_to_embed = Part.from_bytes(
                data=content.image_bytes, mime_type=content.mime_type
            )
        elif isinstance(content, Video):
            if not content.video_bytes or not content.mime_type:
                raise ValueError("Invalid Video content")
            content_to_embed = Part.from_bytes(
                data=content.video_bytes, mime_type=content.mime_type
            )
        # TODO: consider supporting bytes since the embed() docstring's example is an image as bytes, but will require mime sniffing
        else:
            raise TypeError(
                "Invalid input type for multimodal embedding. "
                "Must be str, Image, or Video."
            )
        return content_to_embed

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ValueError)),
    )
    def _embed(self, content: Any, **kwargs) -> List[float]:
        """
        Generate a vector embedding for a single input using the Google Gen AI API.

        Supports multiple input types:
        - Text strings for standard embeddings
        - Images (Vertex AI only, for multimodal models)
        - Videos (Vertex AI only, for multimodal models)

        Args:
            content: Input to embed (str, Image, or Video)
            **kwargs: Additional parameters to pass to the API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If content type is not supported (str, Image, or Video)
            ValueError: If embedding fails or no embedding is returned
            NotImplementedError: If attempting to embed images/videos in Gemini API
        """
        result = self._embed_many([content], batch_size=1, **kwargs)
        if not result or not result[0]:
            raise ValueError("No embedding returned from the Google GenAI API.")
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ValueError)),
    )
    def _embed_many(
        self, contents: List[Any], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of texts using the Google Gen AI API.

        Args:
            contents: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If contents is not a list of strings
            NotImplementedError: If attempting batch embedding in Vertex AI with multimodal models
            ValueError: If embedding fails
        """

        from google.genai.types import EmbedContentConfig

        if not self.is_multimodal and any(not isinstance(c, str) for c in contents):
            raise NotImplementedError(
                "Batch multimodal embeddings is only supported on Vertex AI."
            )
        if not isinstance(contents, list):
            raise TypeError("Must pass in a list of values to embed.")

        try:
            embeddings: List = []
            for batch in self.batchify(
                contents, batch_size, preprocess=self._preprocess
            ):
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",  # TODO:https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types
                        output_dimensionality=self.dims,
                    ),
                )
                if not response.embeddings:
                    raise ValueError(
                        "No embeddings returned from the Google GenAI API."
                    )  # TODO: consider retrying or silently skipping if a single batch fails.
                embeddings.extend([r.values for r in response.embeddings])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ValueError)),
    )
    async def _aembed(self, content: Any, **kwargs) -> List[float]:
        """
        Asynchronously generate a vector embedding for a single input using the Google Gen AI API.

        Supports multiple input types:
        - Text strings for standard embeddings
        - Images (Vertex AI only, for multimodal models)
        - Videos (Vertex AI only, for multimodal models)

        Args:
            content: Input to embed (str, Image, or Video)
            **kwargs: Additional parameters to pass to the async API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If content type is not supported (str, Image, or Video)
            ValueError: If embedding fails or no embedding is returned
            NotImplementedError: If attempting to embed images/videos in Gemini API
        """
        result = await self._aembed_many([content], batch_size=1, **kwargs)
        if not result or not result[0]:
            raise ValueError("No embedding returned from the Google GenAI API.")
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ValueError)),
    )
    async def _aembed_many(
        self, contents: List[Any], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Asynchronously generate vector embeddings for a batch of texts using the Google Gen AI API.

        Args:
            contents: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the async API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If contents is not a list of strings
            NotImplementedError: If attempting batch embedding in Vertex AI with multimodal models
            ValueError: If embedding fails
        """
        from google.genai.types import EmbedContentConfig

        if not self.is_multimodal and any(not isinstance(c, str) for c in contents):
            raise NotImplementedError(
                "Batch multimodal embeddings is only supported on Vertex AI."
            )
        if not isinstance(contents, list):
            raise TypeError("Must pass in a list of values to embed.")

        try:
            embeddings: List = []
            for batch in self.batchify(
                contents, batch_size, preprocess=self._preprocess
            ):
                response = await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",  # TODO:https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types
                        output_dimensionality=self.dims,
                    ),
                )
                if not response.embeddings:
                    raise ValueError(
                        "No embeddings returned from the Google GenAI API."
                    )  # TODO: consider retrying or silently skipping if a single batch fails.
                embeddings.extend([r.values for r in response.embeddings])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def _serialize_for_cache(self, content: Any) -> Union[bytes, str]:
        """Convert content to a cacheable format."""
        from google.genai.types import Image, Video

        if isinstance(content, Image):
            if not content.image_bytes:
                raise ValueError("Invalid Image content for caching")
            return content.image_bytes
        elif isinstance(content, Video):
            if not content.video_bytes:
                raise ValueError("Invalid Video content for caching")
            return content.video_bytes
        return super()._serialize_for_cache(content)

    @property
    def type(self) -> str:
        return "genai"
