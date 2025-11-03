# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
from abc import ABC

# Third Party
import torch


class Accelerator(ABC):
    _subclasses = []

    def __init__(self):
        self.name = self.device_name()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Accelerator._subclasses.append(cls)

    @abc.abstractmethod
    def is_available(self):
        ...

    @abc.abstractmethod
    def device_name(self, device_index=None):
        ...

    @abc.abstractmethod
    def device(self, device_index=None):
        ...

    @abc.abstractmethod
    def set_device(self, device_index):
        ...

    @abc.abstractmethod
    def current_device(self):
        ...

    @abc.abstractmethod
    def current_device_name(self):
        ...

    @abc.abstractmethod
    def device_count(self):
        ...

    @abc.abstractmethod
    def get_device_properties(self, device_index):
        ...

    @abc.abstractmethod
    def synchronize(self, device_index=None):
        ...

    @property
    @abc.abstractmethod
    def Stream(self):
        ...

    @abc.abstractmethod
    def stream(self, stream):
        ...

    @abc.abstractmethod
    def current_stream(self, device_index=None):
        ...

    @abc.abstractmethod
    def default_stream(self, device_index=None):
        ...

    @abc.abstractmethod
    def get_device_properties(self, device_index=None):
        ...

    @abc.abstractmethod
    def support_pinned_memory(self):
        ...

    @abc.abstractmethod
    def has_c_ops(self):
        ...

class XPU(Accelerator):
    def __init__(self):
        super().__init__()

    def device_name(self, device_index=None):
        if device_index == None:
            return 'xpu'
        return 'xpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('xpu', device_index)

    def set_device(self, device_index):
        torch.xpu.set_device(device_index)

    def current_device(self):
        return torch.xpu.current_device()

    def current_device_name(self):
        return 'xpu:{}'.format(torch.xpu.current_device())

    def device_count(self):
        return torch.xpu.device_count()

    def get_device_properties(self, device_index):
        return torch.xpu.get_device_properties(device_index)

    def is_available(self):
        return torch.xpu.is_available()

    def synchronize(self, device_index=None):
        return torch.xpu.synchronize(device_index)

    @property
    def Stream(self):
        return torch.xpu.Stream

    def stream(self, stream):
        return torch.xpu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.xpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return torch.xpu.current_stream(device_index)

    def get_device_properties(self, device_index=None):
        return torch.xpu.get_device_properties(device_index)

    def support_pinned_memory(self):
        return True

    def has_c_ops(self):
        return True

class CUDA(Accelerator):
    def __init__(self):
        super().__init__()

    def device_name(self, device_index=None):
        if device_index == None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('cuda', device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def get_device_properties(self, device_index):
        return torch.cuda.get_device_properties(device_index)

    def is_available(self):
        return torch.cuda.is_available()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    @property
    def Stream(self):
        return torch.cuda.Stream

    def stream(self, stream):
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    def get_device_properties(self, device_index=None):
        return torch.cuda.get_device_properties(device_index)

    def support_pinned_memory(self):
        return True

    def has_c_ops(self):
        return True

accelerator = None
supported_devices = []
for cls in Accelerator._subclasses:
    _instance = cls()
    supported_devices.append(_instance.name)
    if not accelerator and _instance.is_available():
        accelerator = _instance

assert accelerator, f"CRITICAL ERROR: No supported devices [{', '.join(supported_devices)}] found!"
