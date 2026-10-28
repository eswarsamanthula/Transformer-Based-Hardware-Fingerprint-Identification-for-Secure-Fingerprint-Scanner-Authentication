# 09 — Authentication System (HMAC-MFA)

> The old project's `mfa\` was scaffold-only (`[Old]`, doc 06). The rebuilt project ships a
> **working**, verified multi-factor authentication path. `[New]`

## Modules — `authentication\`

| File | Role |
|---|---|
| `scanner_identification.py` | `ScannerIdentifier` — loads a multiclass checkpoint, returns top-k scanner predictions + scores for an image/bytes; `available_models()` |
| `scanner_verification.py` | `ScannerVerifier` — loads a one-vs-all binary checkpoint for a target scanner; `verify_bytes` -> genuine/impostor decision + probability; `available_verification_models()` |
| `authentication_flow.py` | `authenticate(...)` — orchestrates the MFA check: password + biometric (captured image) + declared scanner |
| `scanner_identification_binary.py` | `ScannerIdentifierBinary` — scanner-hardware identification with the **frozen** VGG16-grayscale + EfficientFormer-L1-RGB one-vs-all checkpoints and Dataset-1-only calibrated thresholds |
| `user_biometric.py` | user fingerprint enrollment + cosine-similarity verification on a derived 64×64 vector template (JSON/base64 only, never pickle) |
| `user_store.py` | SQLite (bcrypt-hashed passwords, TOTP secret, template, scanner binding) at `authentication/data/thfsa_users.db` |
| `passkey.py` | RFC-6238 TOTP second factor (pyotp), labelled "passkey / OTP" |
| `auth_flow.py` | `register()` + `authenticate()` — the four-gate flow over user_store + user_biometric + scanner verification, with mutual HMAC agreement folded into gate 4 |
| `__init__.py` | exports |
| `server_secret.bin` | server-held key material for the HMAC-SHA256 mutual-authentication/session tie-in |

## Flow (as wired into `web\app.py::/api/mfa`) `[New]`

1. Client supplies: fingerprint image (upload or validated Dataset 2 sample), a password,
   the declared scanner, and an optional server password.
2. Server runs `ScannerIdentifier` biometric on the image **plus** `ScannerVerifier` against
   the declared scanner and the password check (HMAC-tied session semantics).
3. Response = structured result (genuine/impostor + identity evidence, auth outcome).

Endpoints in `web\app.py`:
- `GET /auth` — the merged **auth + inference lab** page (`templates\auth.html`):
  register/sign-in panels with the live gate trace, plus the identify/verify/MFA
  deck. `GET /login` and `GET /infer` are 302 redirects to `/auth`.
- `GET /api/options`, `/api/samples` — model/scanner/channel options + Dataset 2 sample picker.
- `POST /api/identify` — multiclass prediction (top-k + scores).
- `POST /api/verify` — one-vs-all verification for a claimed scanner.
- `POST /api/mfa` — full password+biometric+declared-scanner flow.
- `GET /`, `/archive`, `/results` — dashboard + archive (results API: `/api/repo`, `/api/multiclass`,
  `/api/binary`, `/api/curves`, `/api/results`; reads the single consolidated summary — see doc 10);
  `/results` renders the final-results page reading `results/metrics` + `results/graphs` (doc 19).
- Model checkpoints/verifiers are cached (LRU, limit 4) to keep inference responsive.

## Full MFA register + login `[New]`

New endpoints in `web\app.py` back the auth panel of the merged `GET /auth` page
(directories: register / sign in, incl. the one-time passkey-confirmation QR overlay):

- `POST /api/auth/register` — username + password (+ confirm) + 1..N enrollment
  fingerprints (upload or Dataset 2 sample names) + declared scanner. Stores:
  bcrypt hash, TOTP secret, derived template (JSON), scanner binding. Returns the
  **one-time** base32 secret + `otpauth://` URI.
- `POST /api/auth/login` — password + TOTP code + one live fingerprint + declared
  scanner. Runs four recorded gates and returns the shared HMAC session key; HMAC agreement is checked within gate 4.

`auth_flow.authenticate` gate order (documented trace — every intermediate
`FALSE` aborts):

1. **stored password** ? `bcrypt.checkpw` -> wrong password -> FALSE
2. **passkey** ? TOTP ?1 window -> wrong passkey code -> FALSE
3. **user fingerprint** ? cosine template -> fingerprint match failed -> FALSE
4. **scanner verification + mutual HMAC-SHA256 exchange** ? the selected scanner
   verifier must accept the image for the bound scanner and the session-key
   agreement must succeed. HMAC is folded into this gate, so it is not a fifth
   trace entry.

`/api/auth/login` returns `trace`, a per-gate record `{gate, label, ok, detail}`
that the `/auth` page renders as the live gate trace (PASS/FAIL per factor).

No model is retrained here; `scanner_identification_binary` only *uses* the
existing 10+10 binary checkpoints and reads `binary_threshold_calibration.csv`.

## Standalone tests (no pytest) `[New]`

```bash
python tests/test_user_biometric.py
python tests/test_scanner_identification_binary.py
python tests/test_auth_web.py
python tests/test_authentication_flow.py
```

`tests/test_authentication_flow.py` additionally proves the tricky case: a
device whose registration password differs fails the final session-key
agreement even when every other factor is correct (mutual-auth FALSE).

## Verification evidence `[New]`

Verified in the write phase: live identify/verify/MFA calls against real Dataset 2 images
return the expected shapes and scores; the System1 grayscale verifier genuinely-rejects at
p≈0.114 as expected from `results_binary_summary.csv` (doc 08). HMAC mutual-auth handshake
works end-to-end (`server_secret.bin` present and used). All four test files above pass.

## Known caveats

- Binary decisions use the per-scanner thresholds from
  `results/tables/binary_threshold_calibration.csv` (Dataset-1 validation calibration,
  frozen; 0.5 only as a fallback if a model/scanner row is missing) — doc 08/19.
- The user-fingerprint factor is a **derived-vector cosine template** (academic
  prototype, not a minutiae matcher): same-scanner same-finger scores ≈0.74-0.86 vs
  cross-scanner ≈0.69-0.79 at threshold 0.70. The margin is real but thin; engineer a
  proven veiner/alignment matcher before any production claim.
- First login warm-up loads all 20 binary models (~30-60 s, CPU); a slow TOTP code can
  expire mid-request, so users should enter a fresh code after the first call. Cached
  thereafter.
- Demo authenticates a *scanner fingerprint + password*, not a named user profile; user
  identity is not modelled (no user labels exist in Dataset 1).
- Coverage: `available_models()`/`available_verification_models()` enumerate exactly the
  checkpoints physically present (doc 11); swin/deit/convnext/vit_b16 now live in
  `checkpoints\transformer\` and are available live. `[New]`