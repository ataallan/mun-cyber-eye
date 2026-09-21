"""Scene place catalog, kit cues, and inference limits."""

import cv2
import numpy as np

from vision.dataset import (
    generate_demo_dataset,
    parse_folder_label,
    parse_folder_tags,
    render_demo_frame,
)
from vision.detector import Detection
from vision.scene_context import (
    DEMO_PLACE_SCENES,
    all_places,
    analyze_kit_cues,
    clear_scene_folder_index_cache,
    famous_arena_name_rejected,
    infer_place_heuristic,
    infer_scene_place,
    place_display_name,
    resolve_place,
    validate_place_type,
)


def test_place_catalog_is_a_solid_known_set():
    places = all_places()
    ids = [p.id for p in places]
    assert 20 <= len(places) <= 30
    assert len(ids) == len(set(ids))
    for required in (
        "sports_field",
        "basketball_court",
        "tennis_court",
        "volleyball_court",
        "indoor_arena",
        "gymnasium",
        "playground",
        "track",
        "swimming_pool",
        "skate_park",
        "street",
        "sidewalk",
        "parking_lot",
        "corridor_hallway",
        "lobby",
        "stairwell",
        "house_interior",
        "residential_yard",
        "compound_courtyard",
        "driveway",
        "classroom_or_office",
        "unknown",
    ):
        assert required in ids
    assert resolve_place("hallway").id == "corridor_hallway"
    assert resolve_place("compound").id == "compound_courtyard"
    assert resolve_place("home").id == "house_interior"
    assert resolve_place("pitch").id == "sports_field"
    assert resolve_place("madison_square_garden") is None
    assert "Basketball court" == place_display_name("basketball_court")
    assert validate_place_type("") == ""
    assert validate_place_type("Street") == "street"


def test_scene_folder_tags():
    tags = parse_folder_tags("scene__street")
    assert tags.category == "ordinary"
    assert tags.place_type == "street"
    assert tags.sport_context is None
    tags = parse_folder_tags("game_or_play__scene__street")
    assert tags.category == "game_or_play"
    assert tags.place_type == "street"
    tags = parse_folder_tags("scene__basketball_court")
    assert tags.category == "game_or_play"
    assert tags.place_type == "basketball_court"
    assert parse_folder_label("game_or_play__basketball") == ("game_or_play", "basketball")
    implied = parse_folder_tags("game_or_play__basketball")
    assert implied.place_type == "basketball_court"
    assert parse_folder_label("confrontation") == ("potential_fight", None)


def test_never_emits_a_famous_arena_name():
    for place in all_places():
        blob = f"{place.id} {place.display_name} {place.notes}"
        assert not famous_arena_name_rejected(blob)


def test_heuristic_place_on_demo_painters():
    expected = {
        "street": "street",
        "corridor_hallway": "corridor_hallway",
        "house_interior": "house_interior",
        "compound_courtyard": "compound_courtyard",
        "basketball_court": "basketball_court",
        "sports_field": "sports_field",
    }
    for place, want in expected.items():
        img = render_demo_frame(
            "ordinary" if place not in {"basketball_court", "sports_field"} else "game_or_play",
            seed=21,
            place_type=place,
        )
        found, conf = infer_place_heuristic(img)
        assert found == want, (place, found, conf)
        assert conf >= 0.55


def test_color_wash_does_not_invent_a_place():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :] = (200, 40, 60)
    found, _conf = infer_place_heuristic(img)
    assert found is None


def test_camera_stamp_wins_over_heuristic():
    img = render_demo_frame("ordinary", seed=3, place_type="street")
    assessment = infer_scene_place(img, camera_place_type="corridor_hallway")
    assert assessment.place_type == "corridor_hallway"
    assert assessment.source == "camera"
    assert assessment.confidence >= 0.9
    assert "Madison" not in assessment.display
    assert "Madison" not in assessment.note


def test_folder_index_matches_labeled_scene(tmp_path):
    root = tmp_path / "activity"
    dest = root / "train" / "scene__street"
    dest.mkdir(parents=True)
    for i in range(4):
        frame = render_demo_frame("ordinary", seed=30 + i, place_type="street")
        cv2.imwrite(str(dest / f"s{i}.jpg"), frame)
    clear_scene_folder_index_cache()
    probe = render_demo_frame("ordinary", seed=99, place_type="street")
    assessment = infer_scene_place(probe, data_root=root, allow_heuristic=False)
    assert assessment.place_type == "street"
    assert assessment.source == "folder"
    clear_scene_folder_index_cache()


def test_kit_similarity_on_matching_jerseys():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    img[40:95, 20:50] = (20, 20, 220)  # saturated red
    img[40:95, 100:130] = (25, 18, 210)
    people = [
        Detection("person", 0.9, (20, 40, 50, 95)),
        Detection("person", 0.88, (100, 40, 130, 95)),
    ]
    kit = analyze_kit_cues(img, people)
    assert kit.team_kit_similarity >= 0.55
    assert kit.jersey_like_colors is True
    assert "identity" in kit.note.lower()
    mixed = np.array(img)
    mixed[40:95, 100:130] = (200, 40, 20)  # blue vs red
    kit_mixed = analyze_kit_cues(mixed, people)
    assert kit_mixed.team_kit_similarity < kit.team_kit_similarity


def test_generate_demo_writes_scene_folders(tmp_path):
    root = tmp_path / "activity"
    generate_demo_dataset(root, n_train=6, n_val=0, n_test=0, seed=2, overwrite=True)
    assert (root / "train" / "scene__street").is_dir()
    assert any((root / "train" / f"scene__{p}").is_dir() for p in DEMO_PLACE_SCENES)
