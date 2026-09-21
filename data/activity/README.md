# Activity dataset layout

Place labeled frames here (authorized sources only). Generated demo images are gitignored.

```
data/activity/
  train/<category>/*.jpg
  val/<category>/*.jpg
  test/<category>/*.jpg
```

Categories: `ordinary`, `game_or_play`, `dance`, `potential_fight`, `potential_fall`, `potential_weapon_object`.

```bash
python -m vision.train_activity --generate-demo
```

See [docs/PHASE3.md](../../docs/PHASE3.md).
