"""Object catalog, YOLO mapping, video frame extract, honest fallback."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.factory import create_app
from vision.object_inventory import (
    collect_object_inventory,
    detect_yolo_catalog_objects,
    merge_inventory_detections,
    objects_from_detections,
    reset_yolo_cache,
)
from vision.objects_catalog import (
    COCO_TO_CATALOG,
    OBJECTS,
    map_detector_label,
    resolve_object,
    validate_object_id,
)
from vision.detector import Detection
from vision.scene_context import infer_scene_place, place_prior_from_objects


def _tiny_jpeg_bytes(seed: int = 1) -> bytes:
    img = np.zeros((40, 48, 3), dtype=np.uint8)
    img[:] = (20 + seed, 40, 80)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _write_tiny_video(path: Path, n_frames: int = 10, fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (64, 48))
    assert writer.isOpened(), "OpenCV could not open VideoWriter for test clip"
    for i in range(n_frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = ((i * 18) % 255, 50, 90)
        writer.write(frame)
    writer.release()


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "siteadmin@localhost",
            "ADMIN_SYNC_PASSWORD": True,
            "ADMIN_ROLE": "admin",
            "OPERATOR_USERNAME": "reviewer",
            "OPERATOR_PASSWORD": "reviewpass",
            "OPERATOR_EMAIL": "reviewer@localhost",
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "ACTIVITY_DATA_ROOT": str(tmp_path / "activity"),
            "OBJECTS_DATA_ROOT": str(tmp_path / "objects"),
            "CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
            "ACTIVE_CHECKPOINT_FILE": str(tmp_path / "active_checkpoint.json"),
            "ACTIVITY_CHECKPOINT": str(tmp_path / "checkpoints" / "missing.joblib"),
            "VISION_BACKEND": "auto",
            "ALERT_NOTIFY_ON_CREATE": False,
            "SEED_DEMO_CAMERAS": False,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="siteadmin", password="test-pass-12"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def test_catalog_is_practical_and_maps_coco():
    assert 40 <= len(OBJECTS) <= 60
    groups = {e.group for e in OBJECTS}
    assert groups == {"home", "community"}
    assert resolve_object("fridge").id == "refrigerator"
    assert resolve_object("couch").id == "sofa"
    assert map_detector_label("dining table").id == "table"
    assert map_detector_label("tv").id == "television"
    assert map_detector_label("fire hydrant").id == "fire_hydrant"
    assert map_detector_label("person") is None
    assert map_detector_label("knife") is None
    for coco, catalog in COCO_TO_CATALOG.items():
        assert resolve_object(catalog) is not None, catalog
        _ = coco
    with pytest.raises(ValueError, match="Unknown object"):
        validate_object_id("../etc")
    with pytest.raises(ValueError, match="Unknown object"):
        validate_object_id("madison_square_garden")


def test_objects_from_detections_maps_yolo_labels():
    dets = [
        Detection("person", 0.9, (0, 0, 10, 10)),
        Detection("chair", 0.88, (1, 2, 3, 4)),
        Detection("refrigerator", 0.91, (5, 6, 7, 8)),
        Detection("knife", 0.7, (9, 9, 10, 10)),
    ]
    hits = objects_from_detections(dets, source="yolo")
    ids = {h.object_id for h in hits}
    assert ids == {"chair", "refrigerator"}
    assert all(h.source == "yolo" for h in hits)


def test_yolo_path_mocked(monkeypatch):
    reset_yolo_cache()

    class _Box:
        def __init__(self, cls_id, conf, xyxy):
            self.cls = [cls_id]
            self.conf = [conf]
            self.xyxy = [xyxy]

    class _Result:
        names = {0: "person", 56: "chair", 72: "refrigerator"}
        boxes = [
            _Box(0, 0.9, [0, 0, 10, 10]),
            _Box(56, 0.84, [10, 20, 30, 40]),
            _Box(72, 0.93, [50, 10, 90, 120]),
        ]

    class _Model:
        def predict(self, image, verbose=False):
            _ = image, verbose
            return [_Result()]

    monkeypatch.setattr(
        "vision.object_inventory.try_load_yolo", lambda model_name="yolov8n.pt": _Model()
    )
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    hits = detect_yolo_catalog_objects(img)
    assert hits is not None
    assert {h.object_id for h in hits} == {"chair", "refrigerator"}
    inv = collect_object_inventory(img, [], frame_index=0, mode="auto", adapter_name="activity")
    assert inv.backend == "yolo"
    assert {o.object_id for o in inv.objects} == {"chair", "refrigerator"}
    reset_yolo_cache()


def test_no_false_objects_when_backend_missing(monkeypatch):
    reset_yolo_cache()
    monkeypatch.setattr(
        "vision.object_inventory.try_load_yolo", lambda model_name="yolov8n.pt": None
    )
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    inv = collect_object_inventory(
        img,
        [Detection("person", 0.9, (1, 1, 8, 8))],
        frame_index=3,
        mode="auto",
        adapter_name="activity",
    )
    assert inv.backend == "unavailable"
    assert inv.objects == []
    assert "invent" in inv.note.lower() or "unavailable" in inv.note.lower()
    merged = merge_inventory_detections([Detection("person", 0.9)], inv)
    assert all(d.label != "refrigerator" for d in merged)
    reset_yolo_cache()


def test_mock_objects_only_when_mode_is_mock():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    mock_inv = collect_object_inventory(img, [], frame_index=3, mode="mock")
    assert mock_inv.backend == "mock"
    assert "refrigerator" in mock_inv.object_ids
    auto_inv = collect_object_inventory(
        img, [], frame_index=3, mode="auto", adapter_name="activity"
    )
    # auto never uses the MOCK script even on the same frame index
    assert auto_inv.backend != "mock"
    assert "refrigerator" not in auto_inv.object_ids


def test_object_place_prior_is_weak_and_does_not_override_camera():
    prior = place_prior_from_objects(["refrigerator", "sink"])
    assert prior is not None
    assert prior.place_type == "house_interior"
    assert prior.confidence < 0.50
    outdoor = place_prior_from_objects(["bench", "gate"])
    assert outdoor is not None
    assert outdoor.place_type in {"outdoor_plaza", "compound_courtyard"}
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    stamped = infer_scene_place(
        img, camera_place_type="street", object_ids=["refrigerator"]
    )
    assert stamped.place_type == "street"
    assert stamped.source == "camera"
    unknown = infer_scene_place(img, object_ids=["refrigerator"], allow_heuristic=False)
    assert unknown.place_type == "house_interior"
    assert unknown.source == "objects"


def test_admin_extract_video_writes_activity_and_object_frames(client, tmp_path):
    _login(client)
    video_path = tmp_path / "clip.avi"
    _write_tiny_video(video_path, n_frames=12, fps=6)
    payload = video_path.read_bytes()
    resp = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "activity",
            "category": "ordinary",
            "split": "train",
            "sample_fps": "3",
            "max_frames": "8",
            "video": (io.BytesIO(payload), "site_lab.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Sampled" in body
    assert "frame" in body.lower()
    dest = tmp_path / "activity" / "train" / "ordinary"
    frames = list(dest.glob("*.jpg"))
    assert len(frames) >= 1
    assert len(frames) <= 8

    obj_resp = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "object",
            "object_id": "refrigerator",
            "split": "train",
            "sample_fps": "3",
            "max_frames": "5",
            "video": (io.BytesIO(payload), "fridge_cam.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert obj_resp.status_code == 200
    obj_dir = tmp_path / "objects" / "train" / "refrigerator"
    assert list(obj_dir.glob("*.jpg"))
    page = client.get("/admin/train")
    html = page.get_data(as_text=True)
    assert "Extract frames from video" in html
    assert "refrigerator" in html.lower()


def test_extract_video_rejects_unknown_object_and_bad_type(client, tmp_path):
    _login(client)
    video_path = tmp_path / "ok.avi"
    _write_tiny_video(video_path, n_frames=6, fps=4)
    payload = video_path.read_bytes()
    bad = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "object",
            "object_id": "../etc",
            "split": "train",
            "video": (io.BytesIO(payload), "ok.avi"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Unknown object" in bad.get_data(as_text=True)
    assert not list((tmp_path / "objects").rglob("*.jpg"))

    not_video = client.post(
        "/admin/train/extract-video",
        data={
            "kind": "activity",
            "category": "ordinary",
            "split": "train",
            "video": (io.BytesIO(_tiny_jpeg_bytes()), "notes.exe"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Unsupported video type" in not_video.get_data(as_text=True)


def test_operator_cannot_extract_video(client):
    _login(client, "reviewer", "reviewpass")
    page = client.get("/admin/train")
    assert "Extract frames from video" not in page.get_data(as_text=True)
    denied = client.post(
        "/admin/train/extract-video",
        data={"kind": "activity", "category": "ordinary"},
        follow_redirects=True,
    )
    assert "Only the admin role" in denied.get_data(as_text=True)


def test_train_object_model_when_enough_images(client, tmp_path):
    _login(client)
    from vision.objects_catalog import ensure_objects_tree

    root = tmp_path / "objects"
    ensure_objects_tree(root, ["chair", "bench"])
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        for oid, color in (("chair", (10, 80, 200)), ("bench", (180, 40, 20))):
            folder = root / split / oid
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = np.zeros((48, 64, 3), dtype=np.uint8)
                img[:] = color
                img[i : i + 4, :, :] = 255
                cv2.imwrite(str(folder / f"{oid}_{i}.jpg"), img)
    resp = client.post(
        "/admin/train/objects",
        data={"model_type": "forest", "output_name": "objects_custom.joblib"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Object classifier wrote" in resp.get_data(as_text=True)
    assert (tmp_path / "checkpoints" / "objects_custom.joblib").is_file()
    empty = client.post(
        "/admin/train/objects",
        data={"model_type": "forest", "output_name": "objects_empty.joblib"},
        follow_redirects=True,
    )
    # second train still has data from the same tree — just assert first succeeded
    _ = empty
