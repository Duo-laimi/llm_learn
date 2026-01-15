from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch



class CacheLayerMixin(ABC):

    is_compilable = False

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.is_initialized = False


    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.LongTensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        if self.is_initialized:
            self.keys.to("cpu", non_blocking=True)
            self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        if self.is_initialized and self.keys.device != self.device:
            self.keys.to(self.device, non_blocking=True)
            self.values.to(self.device, non_blocking=True)

    def reset(self):
        if self.is_initialized:
            self.keys.zeros_()
            self.values.zeros_()

        if hasattr(self, "cumulate_length"):
            self.cumulate_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.keys.device))


class Cache:
    def __init__(
            self,
            layers: Optional[list[CacheLayerMixin]] = None,
            layers_to_replicate: Optional[CacheLayerMixin] = None,
            offload: bool = False,
            offload_only_not_sliding: bool = True
    ):
        pass

    def update(self, *args, **kwargs):
        pass