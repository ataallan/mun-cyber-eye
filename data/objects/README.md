# Object dataset layout

Labeled **crops or frames** for the home / community object catalog (authorized sources only). Sibling of `data/activity/` so object labels cannot break the six-class activity trainer.

```
data/objects/
  train/<object_id>/*.jpg
  val/<object_id>/*.jpg
  test/<object_id>/*.jpg
```

Catalog ids (`refrigerator`, `chair`, `bench`, `playground_equipment`, …) are listed in [docs/OBJECTS_AND_STRUCTURES.md](../../docs/OBJECTS_AND_STRUCTURES.md). Runtime inventory prefers YOLO when installed; missing detectors report `objects_backend=unavailable` and do **not** invent objects.

Mun Cyber developers can extract frames from an authorized video on **Train models** (`/admin/train`). Customer accounts cannot.
