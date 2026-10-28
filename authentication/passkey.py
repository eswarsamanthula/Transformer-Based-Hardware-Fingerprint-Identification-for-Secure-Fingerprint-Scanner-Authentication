"""
passkey.py — secure second-factor abstraction for the academic prototype.

True WebAuthn/FIDO2 hardware attestation is NOT implemented; pretending a
plain text box is a real FIDO passkey would be dishonest. Instead this is a
standard TOTP-style second factor (RFC 6238) via `pyotp`, labelled in the UI
as "Passkey / OTP factor".

Only the base32 secret is stored (server-side, in the user DB). The
one-time passcode is never stored and never logged. Comparison is delegated
to pyotp's constant-time verification.
"""

import pyotp

DIGITS = 6
PERIOD = 30  # seconds


def create_passkey_secret() -> str:
    """Generate a fresh base32 TOTP secret (pyotp's default random source)."""
    return pyotp.random_base32()


def generate_passcode(secret: str) -> str:
    """Current one-time passcode for `secret` (for testing / authenticator)."""
    if not secret:
        raise ValueError("empty passkey secret")
    return pyotp.TOTP(secret, digits=DIGITS, interval=PERIOD).now()


def verify_passcode(secret: str, passcode: str, valid_window: int = 1) -> bool:
    """Verify a submitted passcode, allowing ±`valid_window` periods of drift."""
    if not secret or not passcode:
        return False
    return pyotp.TOTP(secret, digits=DIGITS, interval=PERIOD).verify(
        passcode, valid_window=valid_window)