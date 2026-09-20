"""Honest ingest failures. Never invent detections when a source is down."""

from __future__ import annotations


class IngestError(RuntimeError):
    """Camera source could not be opened or sampled.

    Callers must record this on the camera row and must not create
    synthetic detections to paper over the failure.
    """
