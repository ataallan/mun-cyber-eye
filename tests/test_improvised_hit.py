"""Unidentified object used to hit or thrown at another person."""

from __future__ import annotations

import numpy as np

from risk.engine import ActivityCategory, RiskEngine
from vision.assists import AssistState, enrich_detections
from vision.detector import Detection
from vision.improvised_hit import HIT_LABEL, analyze_improvised_hit
from vision.throw_assist import ObjectTrack, THROW_LABEL, analyze_throw


def _people():
    holder = Detection("person", 0.92, (20, 40, 100, 300))
    other = Detection("person", 0.90, (240, 40, 330, 300))
    return holder, other


def _hit_motion(label: str):
    holder, other = _people()
    prev = [ObjectTrack(label, (70, 150, 110, 190), (90.0, 170.0))]
    current = [
        holder,
        other,
        Detection(label, 0.8, (250, 150, 300, 200)),
    ]
    return current, prev


def test_unidentified_object_contact_is_a_hit_cue():
    current, prev = _hit_motion("backpack")
    assessment = analyze_improvised_hit(current, prev_tracks=prev)
    assert assessment.hit is True
    assert "unidentified_improvised" in assessment.cues
    assert assessment.object_label == "backpack"

    image = np.zeros((320, 400, 3), dtype=np.uint8)
    state = AssistState()
    holder, other = _people()
    enrich_detections(
        image,
        [holder, other, Detection("backpack", 0.8, (70, 150, 110, 190))],
        state,
    )
    merged, assist = enrich_detections(image, current, state)
    assert assist.improvised_hit.reported is True
    assert any(det.label == HIT_LABEL for det in merged)
    result = RiskEngine().assess(merged)
    assert result.should_alert is True
    assert result.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert "unidentified object used to hit" in result.rationale.lower()
    assert result.weapon_id == "unidentified_improvised"


def test_dead_or_still_frames_do_not_invent_a_hit():
    assert analyze_improvised_hit([], prev_tracks=[]).hit is False
    holder, other = _people()
    still = [holder, other, Detection("backpack", 0.8, (70, 150, 110, 190))]
    prev = [ObjectTrack("backpack", (70, 150, 110, 190), (90.0, 170.0))]
    assert analyze_improvised_hit(still, prev_tracks=prev).hit is False
    assert analyze_improvised_hit(still, prev_tracks=None).hit is False
    assert analyze_improvised_hit(
        [holder, Detection("backpack", 0.8, (250, 150, 300, 200))],
        prev_tracks=prev,
    ).hit is False

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    merged, assist = enrich_detections(image, [], AssistState())
    assert assist.improvised_hit.hit is False
    assert all(det.label != HIT_LABEL for det in merged)


def test_named_knife_is_not_relabeled_unidentified():
    current, prev = _hit_motion("knife")
    assessment = analyze_improvised_hit(current, prev_tracks=prev)
    assert assessment.hit is False
    image = np.zeros((320, 400, 3), dtype=np.uint8)
    state = AssistState()
    holder, other = _people()
    enrich_detections(
        image,
        [holder, other, Detection("knife", 0.8, (70, 150, 110, 190))],
        state,
    )
    merged, _assist = enrich_detections(image, current, state)
    assert all(det.label != HIT_LABEL for det in merged)


def test_unidentified_throw_is_called_out_and_not_sport_softened():
    holder = Detection("person", 0.92, (20, 40, 90, 280))
    target = Detection("person", 0.9, (400, 40, 470, 280))
    prev = [ObjectTrack("wrench", (70, 150, 95, 185), (82.5, 167.5))]
    dets = [holder, target, Detection("wrench", 0.8, (180, 150, 205, 185))]
    thrown = analyze_throw(dets, prev_tracks=prev)
    assert thrown.thrown_at_person is True
    assert thrown.harmful is True
    assert "unidentified_improvised" in thrown.cues
    assert analyze_improvised_hit(dets, prev_tracks=prev).hit is False

    extras = {
        "throw": thrown.to_dict(),
        "thrown_at_person": True,
        "sport_context": "basketball",
        "sport_confidence": 0.9,
        "place_type": "basketball_court",
        "place_confidence": 0.9,
    }
    result = RiskEngine().assess(
        [
            Detection("game_or_play", 0.84, extras=extras),
            Detection("person", 0.9, extras=extras),
            Detection("person", 0.88, extras=extras),
            Detection("wrench", 0.8, extras=extras),
            Detection(THROW_LABEL, thrown.confidence, extras=extras),
        ]
    )
    assert result.should_alert is True
    assert result.category == ActivityCategory.POTENTIAL_WEAPON_OBJECT
    assert "unidentified object thrown" in result.rationale.lower()
