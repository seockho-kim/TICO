import importlib.metadata
import importlib.util

from packaging import version

MIN_VERSIONS = {
    "llama": "4.36.0",  # transformers==4.31.0 supports llama but without layer_idx and position_embeddings feature
    "qwen3-vl": "4.57.0",
}


def has_transformers_for(model_type: str) -> bool:
    if importlib.util.find_spec("transformers") is None:
        return False

    current_version = importlib.metadata.version("transformers")
    required_version = MIN_VERSIONS.get(model_type)

    if not required_version:
        raise ValueError(f"Invalid model_type {model_type}")

    if version.parse(current_version) >= version.parse(required_version):
        return True

    return False
