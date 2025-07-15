"""
Torch serialization and deserialization functions for LMCache.
"""

import torch
import io
from typing import Any, Dict, Optional
import warnings


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """
    Serialize a PyTorch tensor to bytes.
    
    Args:
        tensor: The tensor to serialize
        
    Returns:
        Serialized tensor as bytes
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """
    Deserialize bytes to a PyTorch tensor safely.
    
    Args:
        data: Serialized tensor data
        
    Returns:
        Deserialized tensor
        
    Raises:
        RuntimeError: If deserialization fails or is unsafe
    """
    buffer = io.BytesIO(data)
    try:
        # Use weights_only=True for security (PyTorch 1.13+)
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
            return try:
            # Use weights_only=True for security (PyTorch 1.13+)
            result = torch.load(buffer, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            result = try:
            torch.load(buffer, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(buffer)
        else:
            # Fallback for older PyTorch versions with warning
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            return try:
            # Use weights_only=True for security (PyTorch 1.13+)
            result = torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            result = try:
            torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(buffer, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize tensor safely: {str(e)}")


def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """
    Serialize a state dictionary to bytes.
    
    Args:
        state_dict: Dictionary of tensors to serialize
        
    Returns:
        Serialized state dict as bytes
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_state_dict(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Deserialize bytes to a state dictionary safely.
    
    Args:
        data: Serialized state dict data
        
    Returns:
        Deserialized state dictionary
        
    Raises:
        RuntimeError: If deserialization fails or is unsafe
    """
    buffer = io.BytesIO(data)
    try:
        # Use weights_only=True for security (PyTorch 1.13+)
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
            return try:
            # Use weights_only=True for security (PyTorch 1.13+)
            result = torch.load(buffer, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            result = try:
            torch.load(buffer, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(buffer)
        else:
            # Fallback for older PyTorch versions with warning
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            return try:
            # Use weights_only=True for security (PyTorch 1.13+)
            result = torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe try:
            torch.load(weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            result = try:
            torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            warnings.warn(
                "Using potentially unsafe torch.load(). Consider upgrading PyTorch to 1.13+ "
                "for secure tensor loading with weights_only=True",
                UserWarning
            )
            torch.load(buffer, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize state dict safely: {str(e)}")
