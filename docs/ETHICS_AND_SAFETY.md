# Ethics and Safety — Mun Cyber Eye

## Core rule

**AI detects and alerts. Humans verify and decide.**

Mun Cyber Eye is an **assistance** system for authorized security personnel. It must not independently confront, arrest, punish, or label a person as a criminal.

## Authorized use only

- Process only **authorized** camera feeds and approved demo sources. Register those feeds in the camera registry; do not point RTSP or webcam ingest at unauthorized streams. Elevated-risk detection is automation from those cameras when the pipeline runs — not from a customer uploading incident clips for “live detection.” Video files are for **training** (extract frames) only.
- Deploy only where monitoring is lawful and appropriately authorized (e.g., schools, residential communities, businesses, public spaces with lawful infrastructure).
- Operators are responsible for permissions, notices, and local compliance.
- An offline camera or missing RTSP secret is an **error**, not a clear scene and not a fabricated threat. The system must not invent detections to fill the gap.

## Human-in-the-loop

- Every alert is provisional and requires human review.
- Console actions are **acknowledge / dismiss / escalate / reopen** only, and are audited. There is no UI that wipes an alert row. Cameras are enable / disable and detach-from-account only — no permanent camera delete. There is no video library delete UI. See [PRODUCT_OPS.md](PRODUCT_OPS.md).
- Developer train / activate / labeled uploads are written to `system_audit`.
- Confidence scores and rationales support judgment; they do not replace it.
- Weapon-related detections are **indicators**, not proof of possession or intent.
- Outbound email and webhooks notify **authorized personnel only**. They do not close a review, lock doors, or dispatch force.
- Recommended human actions in the payload are **advisory**. A missing Resend key is reported as queued or undelivered — never as a successful send.

## Privacy and security (prototype expectations)

- Local operator accounts on the console (hashed passwords; optional Resend reset email). Fresh installs create the first site admin via Create account — no known default password is shipped. That account cannot train models.
- Prefer least-privilege access to video and alert data.
- Plan for encryption, retention limits, and access audits before any pilot.
- Do not scrape or reuse biometric identity databases in this prototype.

## Accuracy and fairness

- Computer vision can produce false positives and false negatives.
- Heuristics in Phase 2 and the Phase 3 demo checkpoint are provisional; evaluate across lighting, angles, ages, clothing, and environments before pilots. Distinguishing game or play and dance from a real confrontation is difficult even for humans — do not treat a `potential_fight` label as proof.
- The sports catalog and `sport_context` metadata are an **assistive lookup** (folder labels + simple court-color proxies). They do **not** mean the model knows most sports or can referee a real game.
- Place / venue type (`place_type`) is a **catalog setting** (court, street, corridor, house, compound, …), optionally stamped from the camera registry or tagged folders. It is **not** recognition of a named arena, and **setting is not identity**.
- Home / community object inventory (refrigerator, chair, bench, playground equipment, …) is **assistive**. YOLO maps overlapping COCO labels when installed; otherwise the console reports `objects_backend=unavailable` and does **not** invent objects on a real camera feed. This is not a property inventory or a determination of what happened.
- Uniform / kit cues are **clothing-color clusters**. They must never be used as facial identity, demographics, gang labels, or proof of guilt.
- Body-aggression scores are OpenCV motion / pose **proxies**. High motion during sport is often intense play. Do not treat `aggressive_motion` as proof of assault.
- Optional face-expression assist (`ENABLE_FACE_AGGRESSION`, default **off**) is unreliable and assistive only. When enabled it may note a possible tense expression for a human to verify. It must **never** identify a person, infer demographics, or label a face as criminal.
- Fall **manner** (`sudden_collapse` / `accidental_fall` / `unknown_fall`) is a bbox / motion subtype on `potential_fall`. It is **not** a medical diagnosis of syncope, assault, or a trip. Person-down still needs review.
- Gunshot **video** proxies (`possible_gunshot_video_proxy`) are not ballistic proof. `ENABLE_GUNSHOT_AUDIO=0` never invents a bang. Fireworks and reflections false-fire.
- `firearm_aimed_at_person` is a bbox cone toward another person — not proof of a real firearm or intent. Sport context must **not** suppress it. One person with a gun-like object is brandish only.
- Dangerous-object catalog classes (`weapon_class`, `harm_potential`) and use-against-person intensity (`none` / `brandished` / `threatening_motion` / `possible_strike`) are **indicators only** — not proof of assault, possession, or intent. Toys, tools, and phones false-fire. A baseball bat on a matching field at low intensity may stay game or play; a bat used toward a person outside sport still alerts. Ordinary chairs and bottles are not weapons unless use cues are present.
- `object_thrown_at_person` (`thrown_projectile`) is a weak frame-to-frame translation toward another person. Sport balls on a court may stay game or play; a brick or bottle toward a person on a street / corridor / house does **not** get sport-softened. This is not proof of assault.
- The bundled activity model is trained on synthetic scenes. Retrain on authorized labeled video and publish per-class precision / recall / F1 before any field trial.
- Measure reviewer agreement with AI alerts (see proposal evaluation metrics).
- A higher F1 score does not authorize skipping human review.

## Prohibited autonomous enforcement

This system must **not**:

- Trigger locks, weapons, or physical interventions without a human decision path.
- Publish public “wanted” or guilt labels from detections alone.
- Label faces as criminal or attach identity-based guilt from a detection (including any face-expression score or clothing / kit color).
- Be marketed as infallible weapon or violence detection.
- Treat a live ingest hook as an enforcement trigger (no locks, dispatch, or detention on camera events).

## Children and vulnerable populations

Where the product may support school or community safety, extra care is required: parental/guardian policy alignment, limited retention, trained reviewers, and clear escalation to appropriate human responders — never AI-only discipline.

## Reporting concerns

Treat misuse of surveillance or bypass of human review as a product safety incident. Prefer stopping deployment over shipping unsafe automation.
