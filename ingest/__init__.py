"""Video ingest for authorized camera sources and files."""

from .cameras import Camera, CameraStore, mask_uri, resolve_uri
from .errors import IngestError
from .sampler import FrameSampler, WebcamStub
from .source import iter_camera_frames

__all__ = [
    "Camera",
    "CameraStore",
    "FrameSampler",
    "IngestError",
    "WebcamStub",
    "iter_camera_frames",
    "mask_uri",
    "resolve_uri",
]
