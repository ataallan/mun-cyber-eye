# Ethics and Safety — Mun Cyber Eye

## Core rule

**AI detects and alerts. Humans verify and decide.**

Mun Cyber Eye is an **assistance** system for authorized security personnel. It must not independently confront, arrest, punish, or label a person as a criminal.

## Authorized use only

- Process only **authorized** camera feeds and approved demo sources. Register those feeds in the camera registry; do not point RTSP or webcam ingest at unauthorized streams.
- Deploy only where monitoring is lawful and appropriately authorized (e.g., schools, residential communities, businesses, public spaces with lawful infrastructure).
- Operators are responsible for permissions, notices, and local compliance.
- An offline camera or missing RTSP secret is an **error**, not a clear scene and not a fabricated threat. The system must not invent detections to fill the gap.

## Human-in-the-loop

- Every alert is provisional and requires human review.
- Console actions (acknowledge / dismiss / escalate) are audited.
- Confidence scores and rationales support judgment; they do not replace it.
- Weapon-related detections are **indicators**, not proof of possession or intent.
- Outbound email and webhooks notify **authorized personnel only**. They do not close a review, lock doors, or dispatch force.
- Recommended human actions in the payload are **advisory**. A missing Resend key is reported as queued or undelivered — never as a successful send.

## Privacy and security (prototype expectations)

- Local operator accounts on the console (hashed passwords; optional Resend reset email).
- Prefer least-privilege access to video and alert data.
- Plan for encryption, retention limits, and access audits before any pilot.
- Do not scrape or reuse biometric identity databases in this prototype.

## Accuracy and fairness

- Computer vision can produce false positives and false negatives.
- Heuristics in Phase 2 and the Phase 3 demo checkpoint are provisional; evaluate across lighting, angles, ages, clothing, and environments before pilots. Distinguishing game or play and dance from a real confrontation is difficult even for humans — do not treat a `potential_fight` label as proof.
- The bundled activity model is trained on synthetic scenes. Retrain on authorized labeled video and publish per-class precision / recall / F1 before any field trial.
- Measure reviewer agreement with AI alerts (see proposal evaluation metrics).
- A higher F1 score does not authorize skipping human review.

## Prohibited autonomous enforcement

This system must **not**:

- Trigger locks, weapons, or physical interventions without a human decision path.
- Publish public “wanted” or guilt labels from detections alone.
- Label faces as criminal or attach identity-based guilt from a detection.
- Be marketed as infallible weapon or violence detection.
- Treat a live ingest hook as an enforcement trigger (no locks, dispatch, or detention on camera events).

## Children and vulnerable populations

Where the product may support school or community safety, extra care is required: parental/guardian policy alignment, limited retention, trained reviewers, and clear escalation to appropriate human responders — never AI-only discipline.

## Reporting concerns

Treat misuse of surveillance or bypass of human review as a product safety incident. Prefer stopping deployment over shipping unsafe automation.
