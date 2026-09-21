# Object and structure catalog — Mun Cyber Eye

**AI detects and alerts. Humans verify and decide.**

Operators asked whether the system can **scan structures and objects** — refrigerators, chairs, tables, home equipment, community fixtures. Today’s activity model is trained on **image frames** (people/activity classes). Optional YOLO has limited COCO overlap. This module is an **assistive inventory** with honest limits. It is not a complete 3D scan of a house or a civic GIS, and it does **not** enforce, detain, or lock anything.

## Honesty

| Capability | What exists | What it is not |
|------------|-------------|----------------|
| Catalog | ~50 stable ids (`refrigerator`, `chair`, `table`, `sofa`, `bench`, `gate`, `playground_equipment`, …) | An infinite list of every SKU, brand, or unnamed object |
| Detection | Ultralytics YOLO mapped onto catalog ids **when installed**; bbox + confidence | Proof the room contains that appliance, or a property inventory for insurance/legal use |
| Missing YOLO | `objects_backend=unavailable`; empty object list | Inventing a fridge on a real upload |
| MOCK | Scripted demo objects on **synthetic** pipeline runs only | Used on authorized file uploads or registry cameras |
| Training | Admin can extract video frames into `data/objects/train/<object_id>/` and optionally fit a small CPU sklearn classifier | A second activity model; sklearn whole-frame scores are **not** used to invent detections at runtime |
| Place prior | Fridge/sink → small `house_interior` lean; bench/gate → outdoor/community. Confidence stays below the 0.50 fight-vs-play threshold | Overriding a camera `place_type` stamp or proving trespass |

Humans verify every consequential call.

## Catalog (practical, not infinite)

Stable ids live in `vision/objects_catalog.py`.

**Home / indoor:** refrigerator, chair, table, sofa, bed, desk, television, microwave, oven (stove), sink, toilet, door, window, lamp, bookshelf, wardrobe, washing_machine, dishwasher, coffee_maker, toaster, bathtub, shower, nightstand, dresser, mirror, cabinet, counter, fan, heater, air_conditioner, computer

**Community / outdoor:** bench, fence, gate, playground_equipment, trash_bin, street_light, fire_hydrant, mailbox, bus_stop, picnic_table, traffic_light, stop_sign, parking_meter, bicycle_rack, water_fountain, planter, dumpster, kiosk, outdoor_sign, bollard, flagpole

Aliases (`fridge` → `refrigerator`, `couch` → `sofa`, `dining table` → `table`) resolve through the catalog.

## COCO / YOLO overlap

YOLO (COCO 80) maps only where labels overlap, including: chair, couch, bed, dining table, tv, refrigerator, microwave, oven, toaster, sink, toilet, bench, fire hydrant, potted plant, traffic light, stop sign, parking meter, laptop.

Ids with **no** COCO mapping (door, window, washing_machine, playground_equipment, mailbox, …) need operator-labeled frames. They will not appear on Last run until a detector that knows them is present.

## Dataset layout

```
data/activity/{train,val,test}/<activity_or_scene_folder>/*.jpg   # activity / sport / place
data/objects/{train,val,test}/<object_id>/*.jpg                   # object class
```

Activity training still reads only `data/activity/`. Object folders are a sibling tree so they cannot break the six-class activity trainer.

## Video → frames (admin)

On **Train models** (`/admin/train`):

1. Choose kind: activity / game_or_play+sport / scene place / object class.
2. Upload an authorized mp4 / avi / mov / mkv.
3. `FrameSampler` writes JPEGs at the chosen FPS and max-frame cap.
4. Then **Train from labeled data** (activity) or **Train object model** (when enough object images exist).

Videos are sampled to frames; the sklearn activity trainer still learns from images. Path traversal and unknown catalog ids are rejected. See [ADMIN_TRAINING.md](ADMIN_TRAINING.md).

## Console

- **Run Pipeline → Last run** — “Objects / structures” with counts and `objects_backend` (`yolo` | `mock` | `unavailable`), plus fall / gunshot / aimed / thrown safety cues.
- **Alert detail** — optional nearby objects in metadata (assistive), plus thrown-object / aimed-firearm cues when present.
- **Cameras** — unchanged except a note that object inventory is pipeline metadata, not a camera setting.
- **Admin train** — extract-from-video + object catalog counts + optional object trainer.

## Limits (again)

This is not production instance segmentation, not a home IoT inventory, and not a claim that every community structure is recognized. False positives and false negatives are expected. Humans verify.
