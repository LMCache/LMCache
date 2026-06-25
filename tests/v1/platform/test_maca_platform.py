# SPDX-License-Identifier: Apache-2.0
"""Tests for MACA platform capability reporting."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import _registry
from lmcache.v1.platform.maca import (
    get_maca_platform_report,
    is_maca_available,
)


def test_report_defaults_to_not_maca_on_non_metax_without_vllm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA A100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))

    report = get_maca_platform_report(include_vllm=False)

    assert report.is_maca is False
    assert report.torch_device_type == "cuda"
    assert report.torch_device_name == "NVIDIA A100"
    assert report.vllm_platform_class is None


def test_report_detects_metax_from_torch_device_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "MetaX C500")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))

    report = get_maca_platform_report(include_vllm=False)

    assert report.is_maca is True
    assert report.torch_cuda_available is True
    assert report.torch_device_count == 4
    assert report.torch_device_type == "cuda"
    assert report.torch_cuda_capability == (8, 0)


def test_report_can_include_vllm_maca_platform(monkeypatch):
    class FakeMacaPlatform:
        device_name = "maca"
        device_type = "cuda"

        @classmethod
        def is_cuda(cls):
            return False

        @classmethod
        def is_cuda_alike(cls):
            return True

        @classmethod
        def get_device_capability(cls, device_id=0):
            return SimpleNamespace(major=9, minor=0)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm.platforms":
            return SimpleNamespace(current_platform=FakeMacaPlatform())
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("lmcache.v1.platform.maca.import_module", fake_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Unknown CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))

    report = get_maca_platform_report(include_vllm=True)

    assert report.is_maca is True
    assert report.vllm_platform_class == "FakeMacaPlatform"
    assert report.vllm_device_name == "maca"
    assert report.vllm_device_type == "cuda"
    assert report.vllm_is_cuda is False
    assert report.vllm_is_cuda_alike is True
    assert report.vllm_device_capability == "namespace(major=9, minor=0)"


def test_report_accepts_vllm_boolean_properties(monkeypatch):
    class FakeMacaPlatform:
        device_name = "maca"
        device_type = "cuda"
        is_cuda = False
        is_cuda_alike = True

    def fake_import(name, *args, **kwargs):
        if name == "vllm.platforms":
            return SimpleNamespace(current_platform=FakeMacaPlatform())
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("lmcache.v1.platform.maca.import_module", fake_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Unknown CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))

    report = get_maca_platform_report(include_vllm=True)

    assert report.is_maca is True
    assert report.vllm_is_cuda is False
    assert report.vllm_is_cuda_alike is True


def test_report_detects_maca_from_vllm_when_torch_is_unavailable(monkeypatch):
    class FakeMacaPlatform:
        device_name = "maca"
        device_type = "cuda"

    def fake_import_module(name, *args, **kwargs):
        if name == "vllm.platforms":
            return SimpleNamespace(current_platform=FakeMacaPlatform())
        raise ModuleNotFoundError(name)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("lmcache.v1.platform.maca.import_module", fake_import_module)
    monkeypatch.setattr("builtins.__import__", fake_import)

    report = get_maca_platform_report(include_vllm=True)

    assert report.is_maca is True
    assert report.torch_cuda_available is False
    assert report.vllm_platform_class == "FakeMacaPlatform"


def test_maca_availability_is_registered(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "MetaX C500")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))

    assert is_maca_available() is True
    assert _registry.is_available("maca") is True


def test_maca_does_not_register_a_separate_kv_wrapper():
    with pytest.raises(ValueError, match="device type 'maca'"):
        _registry.get_kv_wrapper_factory("maca")
