# Activity dataset layout

Place labeled frames here (authorized sources only). Generated demo images are gitignored.

```
data/activity/
  train/<category>/*.jpg
  val/<category>/*.jpg
  test/<category>/*.jpg
```

Categories: `ordinary`, `game_or_play`, `dance`, `potential_fight`, `potential_fall`, `potential_weapon_object`.

Optional sport context on play folders: `game_or_play__basketball/`, `game_or_play/soccer/`, or a catalog sport name (`basketball/`). Optional place tags: `scene__street/`, `scene__basketball_court/`. Object/structure frames live in the sibling tree `data/objects/` — see [docs/OBJECTS_AND_STRUCTURES.md](../../docs/OBJECTS_AND_STRUCTURES.md). Videos uploaded for labeling are sampled to JPEG frames; the sklearn activity trainer still learns from images.

```bash
python -m vision.train_activity --generate-demo
```

Admins can also upload frames and train from the console (**Train models**, `/admin/train`). See [docs/ADMIN_TRAINING.md](../../docs/ADMIN_TRAINING.md) and [docs/PHASE3.md](../../docs/PHASE3.md).
