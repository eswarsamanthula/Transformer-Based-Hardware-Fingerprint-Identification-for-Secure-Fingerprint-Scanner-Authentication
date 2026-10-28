# 20 — Web Application User Guide

How to run and use every page of the THFSA web application.

## 1. Start the server

```bash
python web/app.py
```

Open **http://127.0.0.1:5001** in a browser.

- After editing a route/template, **restart** the server (`Ctrl+C` then run it again).
- After editing CSS/JS, hard-refresh the page (`Ctrl+Shift+R`).
- Theme toggle (☀︎/☾︎) in the top bar; the choice persists per browser.

## 2. Top-bar navigation

| Link | Page | What it is for |
|---|---|---|
| `results` | Dashboard `/` | Multiclass scanner-ID leaderboard, charts, training curves |
| `dashboard` (on `/results` page) | `results` | Links back to the dashboard |
| `archive` | `/archive` | Raw inventory on disk: tables, metrics, graphs, checkpoints, logs |
| `auth` | `/auth` | Register a user, four-gate MFA sign-in with live gate trace, and the inference lab (identify / verify / mutual-HMAC replay). The old `/infer` and `/login` paths redirect here |
| `RESULTS` page (nav "results" target) | `/results` | Final measured results & authentication design |

Note: the nav shows two "results"-like items because the dashboard (`/`) and the
final-results page (`/results`) are different pages. The dashboard focuses on the
multiclass benchmark; `/results` focuses on the final verification models and MFA.

## 3. Dashboard — `/` (nav: `results`)

Sections:

01 **Leaderboard — multiclass scanner ID** — every 14 models × channel, val vs
   Dataset 2 test accuracy/F1/AUC. Data from `/api/multiclass`.
02 **Test accuracy by architecture** — grayscale vs rgb bar chart.
03 **Training curves** — pick a run in the dropdown; loss and accuracy curves
   from `/logs/training/*.csv`.
04 **One-vs-all scanner verification** — mobilenet binary per scanner
   (acc/F1/AUC/FAR/FRR) from `/api/binary`.

Read this page for the model sweep; for the *final* result story use `/results`.

## 4. Final results page — `/results`

Model cards (VGG16 baseline, EfficientFormer-L1 proposed) with all metrics,
comparison graphs, scanner-wise table + graphs, threshold-calibration table and
graph, dataset-evaluation notes, MFA architecture and interpretation. Every number
is read live from `results/metrics/` + `results/tables/` (`/api/results`). This is
the page to cite for final numbers.

## 5. Archive — `/archive`

Raw inventory, exactly as on disk:

01 **Result tables** — every CSV in `results/tables/`.
02 **Metric files** — every CSV in `results/metrics/` (columns listed; full view
   on `/results`).
03 **Graphs** — thumbnails of `results/graphs/*.png`.
04 **Checkpoint inventory** — `.pt` files with size and kind.
05 **Training logs** — per-epoch runs with final val accuracy.

## 6. Auth page — `/auth`

One page, three sections: **01 enroll (register)**, **sign in** (side-by-side panels),
and **02 live inference lab**.

### Register (left panel)

1. Username (3–32 characters, letters/digits/`_ . -`).
2. Password + confirm (min 4 chars).
3. **Enrollment fingerprints · 1+ images** — upload images, or click a sample
   chip; picking a chip auto-fills the bound scanner.
   Tip for the demo: pick 3 samples from the **same** scanner (e.g. `S6 · …`)
   because enrolment must match the scanner you bind.
4. **Bind scanner** — set to the scanner of the enrolled samples.
5. Click **register**.

After success, a **one-time overlay** shows a **QR code** for the TOTP setup
(plus the base32 secret and `otpauth://` URI below it). Scan the QR with your
authenticator app (Google Authenticator / Aegis / etc.), or enter the secret
manually — do it now, the secret is shown once. Click **I have added it**.

### Sign in (right panel)

1. Username, password, 6-digit **passkey code** from your authenticator.
2. **Live fingerprint capture** — upload or click a sample chip. Its scanner must
   match the **declared scanner** and the bound scanner you registered with.
3. **Declared scanner** — set to the scanner of the chosen sample.
4. Click **authenticate**.

The sign-in result shows `AUTHENTICATED`, the predicted scanner, the first 16
chars of the **session key**, and a **live gate trace** with four entries: password, TOTP, fingerprint-template cosine, and scanner verification. The HMAC session-key agreement is checked within the scanner gate. On failure the trace stops at the failing gate.

### Live inference lab

Two panels:

**A · capture / probe**
- `choose` file — upload a fingerprint image, **or**
- click a sample chip (`S2 · Sensor2(Test1)3` …) to use that Dataset 2 image.

**B · control deck**
- `Model` — 3 verifier pairs with binary checkpoints:
  `mobilenet · grayscale`, `vgg16 · grayscale`, `efficientformer_l1 · rgb`.
- `Channel` — auto-synced to the selected model's binary channel, read-only
  (the model↔channel link is fixed by the frozen checkpoints).
- `Scanner` — target/claimed scanner 1–10.
- `Server password` (optional) — only used by the MFA button.

Buttons:
- `identify` — multiclass scanner ID (top-k + scores).
- `verify` — one-vs-all verification of the sample against the chosen scanner.
- `MFA` — password + biometric + declared-scanner gate (`/api/mfa`) rendered as
  the animated device/server protocol trace.

### Demo tips

- **Cold start:** the first login loads the 20 binary one-vs-all checkpoints
  (~30–60 s CPU). Subsequent logins are fast. A TOTP code can expire during that
  wait — after the first successful call, re-enter a fresh code.
- If the login fails on the scanner gate, the sample you chose is not accepted
  by the frozen verifiers for that scanner — pick a sample whose scanner matches
  the bound one and that the accept matrix accepts
  (see `docs/current/18_VERIFY_ACCEPT_MATRIX.md`; the chips shown are pre-filtered to
  the 53 samples accepted on all 3 models).
- Reset users: delete `authentication/data/thfsa_users.db` (users are wiped).

## 8. HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/repo` | GET | archive inventory (tables, metrics, graphs, checkpoints, logs) |
| `/api/results` | GET | final results: models, scanner, thresholds, auth, graphs |
| `/api/multiclass` | GET | multiclass leaderboard data |
| `/api/binary` | GET | one-vs-all verification data |
| `/api/curves` | GET | training curves |
| `/api/options` | GET | models, verification models, binary models, channels, scanners |
| `/api/samples` | GET | Dataset 2 sample list |
| `/api/identify` | POST | multiclass identification (image or sample) |
| `/api/verify` | POST | one-vs-all verification (image or sample + scanner) |
| `/api/mfa` | POST | password + biometric + declared scanner |
| `/api/auth/register` | POST | create user (username/password/fingerprints/scanner) |
| `/api/auth/login` | POST | four-gate login (HMAC agreement folded into scanner verification) |
| `/results/graphs/<file>` | GET | generated result graphs |

## 9. Where the data lives

| What | Where |
|---|---|
| Measured results | `results/tables/*.csv` (authoritative) |
| Presentation metrics | `results/metrics/*.csv` |
| Graphs | `results/graphs/*.png` |
| Training curves | `logs/training/*.csv` |
| Checkpoints | `checkpoints/` |
| User store | `authentication/data/thfsa_users.db` |