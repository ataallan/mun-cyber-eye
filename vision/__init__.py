"""Pluggable vision adapters (activity, YOLO, or MOCK)."""

from .detector import Detection, VisionAdapter, create_adapter

__all__ = ["Detection", "VisionAdapter", "create_adapter"]
