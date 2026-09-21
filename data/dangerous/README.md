# Dangerous-object labeled frames

Authorized clips only. Sibling of `data/objects/` so weapon-like labels do not mix into the home/community furniture inventory.

```
data/dangerous/{train,val,test}/<id>/*.jpg
```

Catalog ids (`knife`, `baseball_bat`, `handgun`, `bottle`, …) are listed in [docs/DANGEROUS_OBJECTS.md](../../docs/DANGEROUS_OBJECTS.md). Admin **Extract frames from video** → kind **dangerous / weapon-like class**. Activity class `potential_weapon_object` remains a separate six-class trainer folder under `data/activity/`.
