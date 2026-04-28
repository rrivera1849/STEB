# Adding Models

STEB supports four model types out of the box:

- **Encoder models** (`HFModel`): Bidirectional transformers (BERT, RoBERTa, etc.) using mean pooling.
- **Causal models** (`CausalModel`): Auto-regressive LMs (GPT-2, Llama, Mistral, etc.) using last-token pooling.
- **LUAR models** (`LUARModel`): Dedicated support for LUAR-CRUD and LUAR-MUD.
- **Random baseline** (`RandomModel`): Returns random vectors per episode; useful as a chance-level baseline. Invoke with `steb all random`.

Model type is auto-detected from the HuggingFace config. Encoder vs. causal routing happens automatically in `get_model()`.

## Adding a New Model

1. Create a new file in `steb/models/` (e.g., `steb/models/my_model.py`).
2. Inherit from `STEBModel` and implement `embed_multiple`:

    ```python
    from typing import List

    import numpy as np

    from .base import STEBModel


    class MyModel(STEBModel):
        """My custom embedding model."""

        supported_models = ["my-org/my-model"]

        def __init__(self, model_name_or_path: str):
            self.model_name_or_path = model_name_or_path
            # Load your model here

        def embed_multiple(
            self,
            episodes: List[List[str]],
            batch_size: int,
            show_progress: bool = False,
        ) -> np.ndarray:
            """Embeds a list of episodes."""
            # Your embedding logic here
            pass
    ```

3. Register in `steb/models/__init__.py` by adding to `MODEL_REGISTRY`:

    ```python
    from .my_model import MyModel

    MODEL_REGISTRY = {
        "hf": HFModel,
        "causal": CausalModel,
        "luar": LUARModel,
        "my_model": MyModel,
    }
    ```
