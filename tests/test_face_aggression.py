"""Gated face-expression assist: off by default, honest when enabled."""

import numpy as np

from vision.face_aggression import analyze_face_aggression, face_aggression_enabled


def test_face_path_off_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_FACE_AGGRESSION", raising=False)
    assert face_aggression_enabled() is False
    img = np.full((80, 80, 3), 180, dtype=np.uint8)
    result = analyze_face_aggression(img)
    assert result.status == "disabled"
    payload = result.to_dict()
    assert payload["face_aggression"] == "disabled"
    assert payload["identity"] is None
    assert payload["demographics"] is None
    assert payload["criminal_label"] is None
    assert payload["score"] is None


def test_enabled_blank_image_does_not_crash(monkeypatch):
    monkeypatch.setenv("ENABLE_FACE_AGGRESSION", "1")
    assert face_aggression_enabled() is True
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    result = analyze_face_aggression(blank, enabled=True)
    assert result.status in {"none_detected", "unavailable"}
    assert result.faces_found == 0
    payload = result.to_dict()
    assert payload["identity"] is None
    assert payload["criminal_label"] is None
    assert "criminal" not in (payload["note"] or "").lower() or "never" in (
        payload["note"] or ""
    ).lower() or "not a criminal" in (payload["note"] or "").lower()


def test_enabled_empty_image_none_detected():
    result = analyze_face_aggression(np.zeros((0, 0, 3), dtype=np.uint8), enabled=True)
    assert result.status in {"none_detected", "unavailable"}


def test_unavailable_when_cascade_missing(monkeypatch):
    import vision.face_aggression as fa

    monkeypatch.setattr(fa, "_cascade_path", lambda: None)
    result = analyze_face_aggression(np.zeros((48, 48, 3), dtype=np.uint8), enabled=True)
    assert result.status == "unavailable"
    assert "unavailable" in result.note.lower()
    assert result.to_dict()["face_aggression"] == "unavailable"
    assert result.to_dict()["score"] is None
