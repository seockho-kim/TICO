import threading

import torch
from packaging.version import Version

from tico.utils import logging
from tico.utils.installed_packages import is_transformers_installed

__all__ = ["register_dynamic_cache"]


def register_dynamic_cache():
    PyTreeRegistryHelper().register_dynamic_cache()


class PyTreeRegistryHelper:
    """
    Thread-safe singleton helper class for registering custom PyTree nodes.

    This class provides functionality to register DynamicCache as a PyTree node
    for torch.export compatibility. This registration is only needed for
    transformers versions below 4.50.0.

    Thread Safety:
    - Uses a class-level threading.Lock() to ensure thread-safe singleton instantiation
    - Uses the same lock to protect the registration process from concurrent calls
    """

    _instance = None  # Class variable to hold the singleton instance
    _has_called = False  # Flag to track if registration has been performed
    _lock = threading.Lock()  # Class-level lock for thread-safe operations

    def __init__(self):
        """Private constructor to prevent direct instantiation"""
        pass

    def __new__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation using double-checked locking pattern.

        Returns:
            PyTreeRegistryHelper: The singleton instance of this class
        """
        if not cls._instance:
            with cls._lock:  # Acquire lock for thread-safe instantiation
                if not cls._instance:  # Double-check after acquiring lock
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register_dynamic_cache(self):
        """
        Registers DynamicCache as a PyTree node for torch.export compatibility.

        This method is thread-safe and idempotent - it will only perform the
        registration once, even if called multiple times from different threads.

        Note:
            This registration is only needed for transformers versions below 4.50.0.

        Raises:
            ImportError: If transformers package is not installed
        """
        with self._lock:  # Acquire lock for thread-safe registration
            if self.__class__._has_called:
                logger = logging.getLogger(__name__)
                logger.debug("register_dynamic_cache already called, skipping")
                return

            self.__class__._has_called = True
            logger = logging.getLogger(__name__)
            logger.info("Registering DynamicCache PyTree node")

        if not is_transformers_installed:  # type: ignore[truthy-function]
            raise ImportError("transformers package is not installed")

        import transformers

        HAS_TRANSFORMERS_LESS_4_50_0 = Version(transformers.__version__) < Version(
            "4.50.0"
        )
        if not HAS_TRANSFORMERS_LESS_4_50_0:
            return

        from transformers.cache_utils import DynamicCache

        def _flatten_dynamic_cache(dynamic_cache: DynamicCache):
            if not isinstance(dynamic_cache, DynamicCache):
                raise RuntimeError(
                    "This pytree flattening function should only be applied to DynamicCache"
                )
            HAS_TORCH_2_6_0 = Version(torch.__version__) >= Version("2.6.0")
            if not HAS_TORCH_2_6_0:
                logger = logging.getLogger(__name__)
                logger.warning_once(
                    "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
                )
            dictionary = {
                "key_cache": getattr(dynamic_cache, "key_cache"),
                "value_cache": getattr(dynamic_cache, "value_cache"),
            }
            return torch.utils._pytree._dict_flatten(dictionary)

        def _flatten_with_keys_dynamic_cache(dynamic_cache: DynamicCache):
            dictionary = {
                "key_cache": getattr(dynamic_cache, "key_cache"),
                "value_cache": getattr(dynamic_cache, "value_cache"),
            }
            return torch.utils._pytree._dict_flatten_with_keys(dictionary)

        def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
            dictionary = torch.utils._pytree._dict_unflatten(values, context)
            cache = DynamicCache()
            for k, v in dictionary.items():
                setattr(cache, k, v)
            return cache

        def _flatten_dynamic_cache_for_fx(cache, spec):
            dictionary = {
                "key_cache": getattr(cache, "key_cache"),
                "value_cache": getattr(cache, "value_cache"),
            }
            return torch.fx._pytree._dict_flatten_spec(dictionary, spec)

        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            _flatten_dynamic_cache,
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
        )
        # TODO: This won't be needed in torch 2.7+.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache, _flatten_dynamic_cache_for_fx
        )
