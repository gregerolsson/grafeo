//! Tamper-evident wrapper around WASM snapshot bytes.
//!
//! Prepends a `GSN1` magic header and appends an HMAC-SHA256 tag so the
//! receiver can detect modifications or substitutions before the payload
//! reaches the bincode deserializer. Verification runs in constant time
//! via the `hmac` crate.
//!
//! Format:
//!
//! ```text
//! [4 bytes "GSN1"] [payload bytes ...] [32 bytes HMAC-SHA256(magic || payload)]
//! ```
//!
//! The tag covers the magic bytes as well as the payload so an attacker
//! cannot strip the header and downgrade a signed snapshot to unsigned.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

/// Magic header identifying a signed snapshot.
pub(crate) const MAGIC: &[u8; 4] = b"GSN1";

/// Size of the HMAC-SHA256 tag appended to every signed snapshot.
pub(crate) const TAG_LEN: usize = 32;

/// Minimum legal length for a signed snapshot: header + tag with empty payload.
pub(crate) const MIN_LEN: usize = MAGIC.len() + TAG_LEN;

/// Upper bound on bytes accepted by import paths.
///
/// WASM tabs already run into heap pressure well before this limit; the
/// cap keeps a malicious payload from triggering an out-of-memory abort
/// inside bincode deserialisation.
pub(crate) const MAX_SNAPSHOT_BYTES: usize = 128 * 1024 * 1024;

type HmacSha256 = Hmac<Sha256>;

/// Returns `true` if `data` starts with the `GSN1` signed-snapshot magic.
pub(crate) fn looks_signed(data: &[u8]) -> bool {
    data.len() >= MAGIC.len() && &data[..MAGIC.len()] == MAGIC
}

/// Wraps `payload` with a signed-snapshot header and HMAC-SHA256 tag.
pub(crate) fn wrap(key: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(MAGIC.len() + payload.len() + TAG_LEN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(payload);

    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(&out);
    let tag = mac.finalize().into_bytes();
    out.extend_from_slice(&tag);
    out
}

/// Verifies the HMAC-SHA256 tag and returns the payload slice on success.
///
/// Errors use plain strings so the caller can wrap them in `JsError` without
/// leaking structural details about why a snapshot was rejected.
pub(crate) fn unwrap<'a>(key: &[u8], data: &'a [u8]) -> Result<&'a [u8], String> {
    if data.len() < MIN_LEN {
        return Err(format!(
            "snapshot too small: {} bytes, need at least {}",
            data.len(),
            MIN_LEN
        ));
    }
    if !looks_signed(data) {
        return Err(
            "snapshot is missing the GSN1 signed-format header; use importSnapshot() for \
             unsigned exports"
                .into(),
        );
    }

    let split = data.len() - TAG_LEN;
    let (body, tag) = data.split_at(split);

    let mut mac = HmacSha256::new_from_slice(key).map_err(|e| e.to_string())?;
    mac.update(body);
    mac.verify_slice(tag).map_err(|_| {
        "HMAC verification failed: snapshot was tampered with or signed with a different key"
            .to_string()
    })?;

    Ok(&body[MAGIC.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_returns_original_payload() {
        let key = b"0123456789abcdef0123456789abcdef";
        let payload = b"some snapshot bytes";
        let signed = wrap(key, payload);
        let unwrapped = unwrap(key, &signed).expect("verify ok");
        assert_eq!(unwrapped, payload);
    }

    #[test]
    fn wrong_key_fails_verification() {
        let signed = wrap(b"key-a", b"payload");
        let err = unwrap(b"key-b", &signed).unwrap_err();
        assert!(err.contains("HMAC"));
    }

    #[test]
    fn tampered_payload_fails_verification() {
        let key = b"secret";
        let mut signed = wrap(key, b"payload");
        // Flip a byte in the middle of the payload.
        let mid = signed.len() / 2;
        signed[mid] ^= 0x01;
        assert!(unwrap(key, &signed).is_err());
    }

    #[test]
    fn truncated_tag_fails_verification() {
        let key = b"secret";
        let mut signed = wrap(key, b"payload");
        signed.truncate(signed.len() - 1);
        assert!(unwrap(key, &signed).is_err());
    }

    #[test]
    fn stripped_magic_fails_verification() {
        let key = b"secret";
        let signed = wrap(key, b"payload");
        // Drop the magic header: the caller should not be able to pass this
        // off as a raw-unsigned snapshot via importSnapshotSigned.
        let body_and_tag = &signed[MAGIC.len()..];
        assert!(unwrap(key, body_and_tag).is_err());
    }

    #[test]
    fn empty_snapshot_rejected() {
        let err = unwrap(b"k", &[]).unwrap_err();
        assert!(err.contains("too small"));
    }

    #[test]
    fn looks_signed_detects_magic() {
        let signed = wrap(b"k", b"payload");
        assert!(looks_signed(&signed));
        assert!(!looks_signed(b"payload"));
        assert!(!looks_signed(b""));
        assert!(!looks_signed(b"GSN"));
    }
}
