"""Pluggable vision adapters (YOLO or MOCK)."""

from .detector import Detection, VisionAdapter, create_adapter

__all__ = ["Detection", "VisionAdapter", "create_adapter"]
