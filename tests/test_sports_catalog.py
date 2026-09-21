"""Sports catalog, folder mapping, and demo sport painters."""

from vision.dataset import (
    canonicalize_category,
    generate_demo_dataset,
    iter_split_samples_with_context,
    parse_folder_label,
    render_demo_frame,
)
from vision.sports_catalog import (
    DEMO_SPORT_SCENES,
    all_sports,
    infer_sport_context,
    resolve_sport,
    sport_display_name,
)


def test_catalog_is_a_solid_known_set():
    sports = all_sports()
    ids = [s.id for s in sports]
    assert 25 <= len(sports) <= 40
    assert len(ids) == len(set(ids))
    for required in (
        "basketball",
        "soccer",
        "american_football",
        "volleyball",
        "tennis",
        "badminton",
        "baseball",
        "softball",
        "cricket",
        "rugby",
        "hockey",
        "track_athletics",
        "wrestling",
        "boxing",
        "martial_arts_training",
        "playground_games",
        "table_tennis",
        "gymnastics",
        "swimming",
        "skateboarding",
    ):
        assert required in ids
    assert resolve_sport("football").id == "soccer"
    assert resolve_sport("hoops").id == "basketball"
    assert resolve_sport("wrestling_sport").id == "wrestling"
    assert resolve_sport("ping_pong").id == "table_tennis"
    assert resolve_sport("sports") is None
    assert "Basketball" == sport_display_name("basketball")


def test_folder_mapping_double_underscore_and_aliases():
    assert parse_folder_label("game_or_play__basketball") == ("game_or_play", "basketball")
    assert parse_folder_label("game_or_play__soccer") == ("game_or_play", "soccer")
    assert parse_folder_label("game_or_play__football") == ("game_or_play", "soccer")
    assert parse_folder_label("basketball") == ("game_or_play", "basketball")
    assert parse_folder_label("confrontation") == ("potential_fight", None)
    assert parse_folder_label("sports") == ("game_or_play", None)
    assert canonicalize_category("game_or_play__tennis") == "game_or_play"
    assert canonicalize_category("boxing") == "game_or_play"


def test_nested_and_underscore_folders_load(tmp_path):
    import cv2

    root = tmp_path / "activity"
    (root / "train" / "game_or_play__basketball").mkdir(parents=True)
    (root / "train" / "game_or_play" / "soccer").mkdir(parents=True)
    (root / "train" / "tennis").mkdir(parents=True)
    frame = render_demo_frame("game_or_play", seed=1, sport_context="basketball")
    cv2.imwrite(str(root / "train" / "game_or_play__basketball" / "a.jpg"), frame)
    cv2.imwrite(
        str(root / "train" / "game_or_play" / "soccer" / "b.jpg"),
        render_demo_frame("game_or_play", seed=2, sport_context="soccer"),
    )
    cv2.imwrite(
        str(root / "train" / "tennis" / "c.jpg"),
        render_demo_frame("game_or_play", seed=3, sport_context="tennis"),
    )
    rows = list(iter_split_samples_with_context(root, "train"))
    pairs = {(cat, sport) for _path, cat, sport in rows}
    assert ("game_or_play", "basketball") in pairs
    assert ("game_or_play", "soccer") in pairs
    assert ("game_or_play", "tennis") in pairs


def test_infer_sport_context_on_demo_painters():
    for sport in DEMO_SPORT_SCENES:
        img = render_demo_frame("game_or_play", seed=11, sport_context=sport)
        found, conf = infer_sport_context(img)
        assert found == sport, (sport, found, conf)
        assert conf >= 0.50


def test_generate_demo_writes_sport_folders(tmp_path):
    root = tmp_path / "activity"
    generate_demo_dataset(root, n_train=6, n_val=0, n_test=0, seed=2, overwrite=True)
    sports = {
        sport
        for _p, _c, sport in iter_split_samples_with_context(root, "train")
        if sport
    }
    assert {"soccer", "tennis", "volleyball"} <= sports
