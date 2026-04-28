//! Column codecs for CompactStore.
//!
//! Wraps Grafeo's existing storage primitives into a unified enum with
//! random access and `Value` decoding. CompactStore owns these types:
//! the underlying primitives are not modified.

use std::sync::Arc;

use arcstr::ArcStr;
use bytes::{Bytes, BytesMut};
use grafeo_common::types::Value;

use crate::codec::{BitPackedInts, BitVector, BlockEntry, DictionaryEncoding};

// ── Phase 3a: Bytes-backed read helpers ──────────────────────────────
//
// Fixed-width codec variants (RawI64, Float64, Int8Vector, Float32Vector)
// store their raw bytes as `bytes::Bytes` rather than `Vec<T>`. This is
// the long-term storage abstraction for the entire columnar layer
// (revised D7): a heap-owned column and a mmap-backed column have the
// same type; the `Bytes` constructor decides. Phase 3c adds the mmap
// constructor (`Bytes::from_owner`).
//
// Read helpers use safe `from_le_bytes` (no `unsafe` required). On x86,
// modern compilers fold `try_into().unwrap()` + `from_le_bytes` into a
// single `mov` for aligned reads; unaligned reads cost nothing extra.
// Phase 3 endianness is locked to little-endian on disk; readers
// validate this contract at the section header, not per-element.

#[inline]
fn read_le_i64(bytes: &Bytes, byte_idx: usize) -> Option<i64> {
    let end = byte_idx.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(i64::from_le_bytes(chunk))
}

#[inline]
fn read_le_f64(bytes: &Bytes, byte_idx: usize) -> Option<f64> {
    let end = byte_idx.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(f64::from_le_bytes(chunk))
}

#[inline]
fn read_le_f32(bytes: &Bytes, byte_idx: usize) -> Option<f32> {
    let end = byte_idx.checked_add(4)?;
    let chunk: [u8; 4] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(f32::from_le_bytes(chunk))
}

#[inline]
fn read_i8(bytes: &Bytes, byte_idx: usize) -> Option<i8> {
    bytes.get(byte_idx).copied().map(u8::cast_signed)
}

fn vec_to_bytes_i64(values: &[i64]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 8);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_f64(values: &[f64]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 8);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_f32(values: &[f32]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_i8(values: &[i8]) -> Bytes {
    // i8 and u8 have identical 1-byte layout; the cast is exact.
    let mut buf = BytesMut::with_capacity(values.len());
    for &v in values {
        buf.extend_from_slice(&[v.cast_unsigned()]);
    }
    buf.freeze()
}

/// A single column of data backed by one of Grafeo's storage codecs.
///
/// Each variant wraps an existing primitive via composition: the
/// primitives themselves are never modified. Use [`get`](Self::get) for
/// `Value`-typed access and the specialised accessors when you know the
/// underlying codec.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ColumnCodec {
    /// Fixed-width bit-packed unsigned integers.
    ///
    /// Used for `Value::Int64` columns whose values are all `>= 0`. Preserves
    /// the `Int64` type on decode via `v as i64`.
    BitPacked(BitPackedInts),
    /// Dictionary-encoded strings.
    Dict(DictionaryEncoding),
    /// Null/boolean bitmap.
    Bitmap(BitVector),
    /// Int8 quantized vectors (flat array with stride). Bytes-backed
    /// (Phase 3a): each row occupies `dimensions` consecutive bytes.
    Int8Vector {
        /// Flat byte array; `len() / dimensions` logical rows.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// Native IEEE 754 double-precision floats. Bytes-backed (Phase 3a):
    /// LE `f64` values, 8 bytes per logical row.
    Float64(Bytes),
    /// Float32 vectors (flat array with stride), for embedding / vector search.
    /// Bytes-backed (Phase 3a): LE `f32` components, `4 * dimensions`
    /// bytes per row.
    Float32Vector {
        /// Flat byte array; `len() / (4 * dimensions)` logical rows.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// Native signed 64-bit integers.
    ///
    /// Used for `Value::Int64` columns when at least one value is negative.
    /// `BitPacked` can't represent negatives correctly in its ordered
    /// operations (`find_eq`, `find_in_range`, zone maps) because it operates
    /// on `u64`; `RawI64` stores the values natively to preserve both the
    /// GQL type and signed ordering semantics on round-trip. Bytes-backed
    /// (Phase 3a): LE `i64` values, 8 bytes per logical row.
    RawI64(Bytes),
}

impl ColumnCodec {
    // ── Phase 3a: Bytes-backed constructors ─────────────────────────

    /// Constructs a [`RawI64`](Self::RawI64) column from a `Vec<i64>`.
    ///
    /// The values are encoded as little-endian `i64` bytes and stored in
    /// a refcounted [`Bytes`] buffer. Phase 3c will add a parallel
    /// constructor that builds from a mmap slice without copying.
    #[must_use]
    pub fn raw_i64(values: Vec<i64>) -> Self {
        Self::RawI64(vec_to_bytes_i64(&values))
    }

    /// Constructs a [`Float64`](Self::Float64) column from a `Vec<f64>`.
    #[must_use]
    pub fn float64(values: Vec<f64>) -> Self {
        Self::Float64(vec_to_bytes_f64(&values))
    }

    /// Constructs an [`Int8Vector`](Self::Int8Vector) column from a flat
    /// `Vec<i8>` and a per-vector dimension count.
    ///
    /// `data.len()` must be a multiple of `dimensions` (or 0 when
    /// `dimensions == 0`); the codec doesn't enforce this on construction
    /// to match the previous tuple-style API, but `len()` and `get` use
    /// integer division so a misaligned tail is silently truncated.
    #[must_use]
    pub fn int8_vector(data: Vec<i8>, dimensions: u16) -> Self {
        Self::Int8Vector {
            bytes: vec_to_bytes_i8(&data),
            dimensions,
        }
    }

    /// Constructs a [`Float32Vector`](Self::Float32Vector) column from a
    /// flat `Vec<f32>` and a per-vector dimension count. Each component
    /// is stored as 4 LE bytes.
    #[must_use]
    pub fn float32_vector(data: Vec<f32>, dimensions: u16) -> Self {
        Self::Float32Vector {
            bytes: vec_to_bytes_f32(&data),
            dimensions,
        }
    }

    /// Decodes the value at `index` into a [`Value`].
    ///
    /// - [`BitPacked`](Self::BitPacked) → `Value::Int64(v as i64)`
    /// - [`Dict`](Self::Dict) → `Value::String(ArcStr::from(s))`
    /// - [`Bitmap`](Self::Bitmap) → `Value::Bool(b)`
    /// - [`Int8Vector`](Self::Int8Vector) → `Value::List(...)` of `Int64` values
    /// - [`Float64`](Self::Float64) → `Value::Float64(f)`
    /// - [`RawI64`](Self::RawI64) → `Value::Int64(n)`
    ///
    /// Returns `None` when `index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            // The builder validates all values <= i64::MAX, so this cast is lossless.
            // reason: values validated <= i64::MAX during build
            Self::BitPacked(bp) => bp.get(index).map(|v| {
                // reason: values validated <= i64::MAX during build
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v as i64);
                val
            }),
            Self::Dict(dict) => dict.get(index).map(|s| Value::String(ArcStr::from(s))),
            Self::Bitmap(bv) => bv.get(index).map(Value::Bool),
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > bytes.len() {
                    return None;
                }
                let values: Vec<Value> = (start..end)
                    .map(|i| Value::Int64(read_i8(bytes, i).unwrap_or(0) as i64))
                    .collect();
                Some(Value::List(Arc::from(values)))
            }
            Self::Float64(bytes) => read_le_f64(bytes, index.checked_mul(8)?).map(Value::Float64),
            Self::RawI64(bytes) => read_le_i64(bytes, index.checked_mul(8)?).map(Value::Int64),
            Self::Float32Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start_byte = index.checked_mul(dims)?.checked_mul(4)?;
                let end_byte = start_byte.checked_add(dims.checked_mul(4)?)?;
                if end_byte > bytes.len() {
                    return None;
                }
                let values: Vec<f32> = (0..dims)
                    .map(|d| read_le_f32(bytes, start_byte + d * 4).unwrap_or(0.0))
                    .collect();
                Some(Value::Vector(Arc::from(values.as_slice())))
            }
        }
    }

    /// Returns the raw `u64` stored at `index` (useful for FK columns).
    ///
    /// Only meaningful for [`BitPacked`](Self::BitPacked) columns; all other
    /// variants return `None`.
    #[inline]
    #[must_use]
    pub fn get_raw_u64(&self, index: usize) -> Option<u64> {
        match self {
            Self::BitPacked(bp) => bp.get(index),
            _ => None,
        }
    }

    /// Returns a slice over the int8 vector at `index`.
    ///
    /// Only meaningful for [`Int8Vector`](Self::Int8Vector) columns; all other
    /// variants return `None`.
    #[must_use]
    pub fn get_int8_vector(&self, index: usize) -> Option<&[i8]> {
        match self {
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > bytes.len() {
                    return None;
                }
                let u8_slice: &[u8] = &bytes[start..end];
                // SAFETY: `i8` and `u8` have identical layout (both are 1
                // byte, no padding, no niche). Reinterpreting `&[u8]` as
                // `&[i8]` is valid: the slice metadata (ptr + len) is the
                // same; only the element type changes for the caller.
                // This is the same operation `bytemuck::cast_slice` would
                // perform; we inline it to avoid a dependency.
                #[allow(unsafe_code)]
                let i8_slice: &[i8] = unsafe {
                    std::slice::from_raw_parts(u8_slice.as_ptr().cast::<i8>(), u8_slice.len())
                };
                Some(i8_slice)
            }
            _ => None,
        }
    }

    /// Returns the number of logical values in this column.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::BitPacked(bp) => bp.len(),
            Self::Dict(dict) => dict.len(),
            Self::Bitmap(bv) => bv.len(),
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                bytes.len().checked_div(dims).unwrap_or(0)
            }
            Self::Float64(bytes) => bytes.len() / 8,
            Self::Float32Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                bytes.len().checked_div(dims * 4).unwrap_or(0)
            }
            Self::RawI64(bytes) => bytes.len() / 8,
        }
    }

    /// Returns `true` if the column contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of logical blocks in this column.
    ///
    /// Empty columns report `1` (a single zero-row block) so downstream
    /// serializers and iterators can treat block emission uniformly.
    /// Non-empty columns return `ceil(len / DEFAULT_BLOCK_ROWS)`.
    #[must_use]
    pub fn block_count(&self) -> usize {
        if self.is_empty() {
            return 1;
        }
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        self.len().div_ceil(block_rows)
    }

    /// Descriptor for the block at index `i`, or `None` if `i` is out of
    /// range.
    ///
    /// Block `i` covers rows `[i * DEFAULT_BLOCK_ROWS, min((i+1) *
    /// DEFAULT_BLOCK_ROWS, len()))`. Empty columns return a single
    /// zero-row block at index 0.
    ///
    /// Phase 2c will fill in per-block statistics (`min`, `max`,
    /// `null_count`, optional `bloom`).
    #[must_use]
    pub fn block_at(&self, i: usize) -> Option<BlockEntry> {
        if i >= self.block_count() {
            return None;
        }
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        let start = i * block_rows;
        let end = (start + block_rows).min(self.len());
        // reason: block lengths fit in u32 since DEFAULT_BLOCK_ROWS itself is u32.
        #[allow(clippy::cast_possible_truncation)]
        let row_count = (end - start) as u32;
        Some(BlockEntry::new(row_count))
    }

    /// Iterator over this column's block descriptors.
    ///
    /// Phase 2a yields exactly one block (covering all rows). The
    /// invariant that each block's `row_count`s sum to `self.len()` will
    /// remain true through Phase 2b's multi-block layout.
    pub fn block_iter(&self) -> impl Iterator<Item = BlockEntry> + '_ {
        (0..self.block_count()).filter_map(move |i| self.block_at(i))
    }

    /// Returns the offsets of all rows whose value equals `target`.
    ///
    /// Operates in the codec's native domain to avoid per-row `Value`
    /// allocation:
    /// - [`BitPacked`](Self::BitPacked): compares raw `u64` values via
    ///   [`BitPackedInts::get`]
    /// - [`Dict`](Self::Dict): resolves the target string to a dictionary
    ///   code once, then scans integer codes via
    ///   [`DictionaryEncoding::filter_by_code`]
    /// - [`Bitmap`](Self::Bitmap): checks bits directly
    ///
    /// Falls back to [`get`](Self::get)-based comparison for type mismatches.
    pub fn find_eq(&self, target: &Value) -> Vec<usize> {
        match (self, target) {
            (Self::BitPacked(bp), &Value::Int64(v)) => {
                if v < 0 {
                    return Vec::new();
                }
                // reason: v >= 0 checked above
                #[allow(clippy::cast_sign_loss)]
                let target_u64 = v as u64;
                (0..bp.len())
                    .filter(|&i| bp.get(i) == Some(target_u64))
                    .collect()
            }
            (Self::Dict(dict), Value::String(s)) => match dict.encode(s.as_str()) {
                Some(code) => dict.filter_by_code(|c| c == code),
                None => Vec::new(),
            },
            (Self::Bitmap(bv), &Value::Bool(target_bool)) => (0..bv.len())
                .filter(|&i| bv.get(i) == Some(target_bool))
                .collect(),
            (Self::Float64(bytes), &Value::Float64(target)) => (0..bytes.len() / 8)
                .filter(|&i| read_le_f64(bytes, i * 8) == Some(target))
                .collect(),
            (Self::RawI64(bytes), &Value::Int64(target)) => (0..bytes.len() / 8)
                .filter(|&i| read_le_i64(bytes, i * 8) == Some(target))
                .collect(),
            _ => (0..self.len())
                .filter(|&i| self.get(i).as_ref() == Some(target))
                .collect(),
        }
    }

    /// Returns the offsets of all rows whose value falls within the given range.
    ///
    /// Like [`find_eq`](Self::find_eq), operates in the codec's native domain
    /// to avoid per-row `Value` allocation for integer columns.
    pub fn find_in_range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<usize> {
        if let Self::BitPacked(bp) = self {
            let min_u64 = match min {
                // reason: v >= 0 guard ensures no sign loss
                #[allow(clippy::cast_sign_loss)]
                Some(&Value::Int64(v)) if v >= 0 => Some(v as u64),
                Some(&Value::Int64(_)) => Some(0),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };
            let max_u64 = match max {
                // reason: v >= 0 guard ensures no sign loss
                #[allow(clippy::cast_sign_loss)]
                Some(&Value::Int64(v)) if v >= 0 => Some(v as u64),
                Some(&Value::Int64(v)) if v < 0 => return Vec::new(),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };

            return (0..bp.len())
                .filter(|&i| {
                    if let Some(v) = bp.get(i) {
                        let above_min = match min_u64 {
                            Some(lo) if min_inclusive => v >= lo,
                            Some(lo) => v > lo,
                            None => true,
                        };
                        let below_max = match max_u64 {
                            Some(hi) if max_inclusive => v <= hi,
                            Some(hi) => v < hi,
                            None => true,
                        };
                        above_min && below_max
                    } else {
                        false
                    }
                })
                .collect();
        }

        if let Self::RawI64(bytes) = self {
            // Native i64 comparison — signed ordering semantics "for free".
            let min_i64 = match min {
                Some(&Value::Int64(v)) => Some(v),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };
            let max_i64 = match max {
                Some(&Value::Int64(v)) => Some(v),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };

            return (0..bytes.len() / 8)
                .filter(|&i| {
                    let v = read_le_i64(bytes, i * 8).unwrap_or(0);
                    let above_min = match min_i64 {
                        Some(lo) if min_inclusive => v >= lo,
                        Some(lo) => v > lo,
                        None => true,
                    };
                    let below_max = match max_i64 {
                        Some(hi) if max_inclusive => v <= hi,
                        Some(hi) => v < hi,
                        None => true,
                    };
                    above_min && below_max
                })
                .collect();
        }

        self.find_in_range_fallback(min, max, min_inclusive, max_inclusive)
    }

    /// Fallback range scan via per-row `Value` decode.
    fn find_in_range_fallback(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<usize> {
        use super::zone_map::compare_values;

        (0..self.len())
            .filter(|&i| {
                let Some(v) = self.get(i) else {
                    return false;
                };
                if let Some(min_val) = min {
                    match compare_values(&v, min_val) {
                        Some(std::cmp::Ordering::Less) => return false,
                        Some(std::cmp::Ordering::Equal) if !min_inclusive => return false,
                        None => return false,
                        _ => {}
                    }
                }
                if let Some(max_val) = max {
                    match compare_values(&v, max_val) {
                        Some(std::cmp::Ordering::Greater) => return false,
                        Some(std::cmp::Ordering::Equal) if !max_inclusive => return false,
                        None => return false,
                        _ => {}
                    }
                }
                true
            })
            .collect()
    }

    // ── Serialization (for CompactStoreSection) ───────────────────

    /// Serializes this codec to a byte buffer.
    ///
    /// Format: `[discriminant: u8][codec-specific data]`
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            Self::BitPacked(bp) => {
                buf.push(0); // discriminant
                buf.push(bp.bits_per_value());
                write_usize_as_u32(buf, bp.len());
                let data = bp.data();
                write_usize_as_u32(buf, data.len());
                for &word in data {
                    buf.extend_from_slice(&word.to_le_bytes());
                }
            }
            Self::Dict(dict) => {
                buf.push(1); // discriminant
                let dict_entries = dict.dictionary();
                write_usize_as_u32(buf, dict_entries.len());
                for entry in dict_entries.iter() {
                    let s = entry.as_ref().as_bytes();
                    write_usize_as_u32(buf, s.len());
                    buf.extend_from_slice(s);
                }
                let codes = dict.codes();
                write_usize_as_u32(buf, codes.len());
                for &code in codes {
                    buf.extend_from_slice(&code.to_le_bytes());
                }
            }
            Self::Bitmap(bv) => {
                buf.push(2); // discriminant
                write_usize_as_u32(buf, bv.len());
                let data = bv.data();
                write_usize_as_u32(buf, data.len());
                for &word in data {
                    buf.extend_from_slice(&word.to_le_bytes());
                }
            }
            Self::Int8Vector { bytes, dimensions } => {
                buf.push(3); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                write_usize_as_u32(buf, bytes.len());
                buf.extend_from_slice(bytes);
            }
            Self::Float64(bytes) => {
                buf.push(4); // discriminant
                write_usize_as_u32(buf, bytes.len() / 8);
                buf.extend_from_slice(bytes);
            }
            Self::Float32Vector { bytes, dimensions } => {
                buf.push(5); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims_bytes = (*dimensions as usize) * 4;
                let total_components = bytes.len().checked_div(4).unwrap_or(0);
                write_usize_as_u32(buf, total_components);
                // Block-pad: ensure rows align to dims_bytes for read_from.
                let _ = dims_bytes;
                buf.extend_from_slice(bytes);
            }
            Self::RawI64(bytes) => {
                buf.push(6); // discriminant
                write_usize_as_u32(buf, bytes.len() / 8);
                buf.extend_from_slice(bytes);
            }
        }
    }

    /// Deserializes a codec from a byte buffer at the given offset.
    ///
    /// Returns the decoded codec and advances `pos` past the consumed bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is truncated or the discriminant is unknown.
    pub fn read_from(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let discriminant = *data.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;

        match discriminant {
            0 => {
                // BitPacked
                let bits = *data.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let count = read_u32_le(data, pos)? as usize;
                let data_len = read_u32_le(data, pos)? as usize;
                let mut words = Vec::with_capacity(data_len);
                for _ in 0..data_len {
                    words.push(read_u64_le(data, pos)?);
                }
                Ok(Self::BitPacked(BitPackedInts::from_raw_parts(
                    words, bits, count,
                )))
            }
            1 => {
                // Dict
                let dict_len = read_u32_le(data, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(data, pos)? as usize;
                    if *pos + slen > data.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&data[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let codes_len = read_u32_le(data, pos)? as usize;
                let mut codes = Vec::with_capacity(codes_len);
                for _ in 0..codes_len {
                    codes.push(read_u32_le(data, pos)?);
                }
                Ok(Self::Dict(DictionaryEncoding::new(
                    Arc::from(entries.into_boxed_slice()),
                    codes,
                )))
            }
            2 => {
                // Bitmap
                let bit_len = read_u32_le(data, pos)? as usize;
                let data_len = read_u32_le(data, pos)? as usize;
                let mut words = Vec::with_capacity(data_len);
                for _ in 0..data_len {
                    words.push(read_u64_le(data, pos)?);
                }
                Ok(Self::Bitmap(BitVector::from_raw_parts(words, bit_len)))
            }
            3 => {
                // Int8Vector
                let dimensions = read_u16_le(data, pos)?;
                let data_len = read_u32_le(data, pos)? as usize;
                if *pos + data_len > data.len() {
                    return Err("truncated Int8Vector data");
                }
                let bytes = Bytes::copy_from_slice(&data[*pos..*pos + data_len]);
                *pos += data_len;
                Ok(Self::Int8Vector { bytes, dimensions })
            }
            4 => {
                // Float64: count of f64 values; storage is 8 bytes each.
                let count = read_u32_le(data, pos)? as usize;
                let byte_need = count.checked_mul(8).ok_or("Float64 length overflow")?;
                if *pos + byte_need > data.len() {
                    return Err("truncated Float64 data");
                }
                let bytes = Bytes::copy_from_slice(&data[*pos..*pos + byte_need]);
                *pos += byte_need;
                Ok(Self::Float64(bytes))
            }
            5 => {
                // Float32Vector: total component count (rows * dims).
                let dimensions = read_u16_le(data, pos)?;
                let component_count = read_u32_le(data, pos)? as usize;
                let byte_need = component_count
                    .checked_mul(4)
                    .ok_or("Float32Vector length overflow")?;
                if *pos + byte_need > data.len() {
                    return Err("truncated Float32Vector data");
                }
                let bytes = Bytes::copy_from_slice(&data[*pos..*pos + byte_need]);
                *pos += byte_need;
                Ok(Self::Float32Vector { bytes, dimensions })
            }
            6 => {
                // RawI64: count of i64 values; storage is 8 bytes each.
                let count = read_u32_le(data, pos)? as usize;
                let byte_need = count.checked_mul(8).ok_or("RawI64 length overflow")?;
                if *pos + byte_need > data.len() {
                    return Err("truncated RawI64 data");
                }
                let bytes = Bytes::copy_from_slice(&data[*pos..*pos + byte_need]);
                *pos += byte_need;
                Ok(Self::RawI64(bytes))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Serializes this codec to v2 format with a per-block index.
    ///
    /// Layout:
    /// ```text
    /// [disc:u8]
    /// [global_params...]                  // codec-specific (empty for RawI64/Bitmap/Float64)
    /// [block_count:u32]
    /// [block_index]                       // block_count x [byte_offset:u32, byte_len:u32, row_count:u32]
    /// [block_data]                        // concatenated per-block bodies
    /// ```
    ///
    /// Block bodies carry just the row data for their range; codec
    /// parameters that apply to the whole column (bit width, dictionary,
    /// vector dimensions) live in `global_params`. Phase 2c will extend
    /// the per-block index entry with stats fields by bumping section
    /// version to 3.
    ///
    /// # Panics
    ///
    /// Panics if any per-block byte length or the block count exceeds
    /// `u32::MAX`. CompactStore columns are bounded by `u32::MAX` rows
    /// (the section format's hard limit), so this is unreachable for
    /// any column built through the public APIs.
    pub fn write_to_v2(&self, buf: &mut Vec<u8>) {
        let (metas, bodies) = self.emit_blocked_codec(buf);
        write_block_index_and_bodies(buf, &metas, &bodies);
    }

    /// Serializes this codec to v3 format: like v2, but each block index
    /// entry is followed by an inline per-block `ZoneMap` for skip
    /// pruning.
    ///
    /// `stats_hint` may pass pre-computed per-block stats from
    /// `NodeTable::block_zone_maps_for`; when `None` (or wrong shape),
    /// per-block stats are computed inline during write.
    ///
    /// # Panics
    ///
    /// Same conditions as [`write_to_v2`](Self::write_to_v2).
    pub fn write_to_v3(&self, buf: &mut Vec<u8>, stats_hint: Option<&[super::zone_map::ZoneMap]>) {
        let (metas, bodies) = self.emit_blocked_codec(buf);
        let computed;
        let stats: &[super::zone_map::ZoneMap] = match stats_hint {
            Some(hint) if hint.len() == metas.len() => hint,
            _ => {
                computed = super::zone_map::compute_block_zone_maps(self);
                &computed
            }
        };
        write_block_index_and_bodies_with_stats(buf, &metas, &bodies, stats);
    }

    /// Pushes the discriminant + global params for this codec into
    /// `buf`, then collects per-block bodies and metadata. Shared by
    /// `write_to_v2` and `write_to_v3`.
    fn emit_blocked_codec(&self, buf: &mut Vec<u8>) -> (Vec<BlockMeta>, Vec<u8>) {
        let block_count = self.block_count();
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        let mut bodies: Vec<u8> = Vec::new();
        let mut metas: Vec<BlockMeta> = Vec::with_capacity(block_count);

        match self {
            Self::BitPacked(bp) => {
                buf.push(0);
                buf.push(bp.bits_per_value());
                let bits_per_value = bp.bits_per_value();
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(bp.len());
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end - start) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    let row_values: Vec<u64> = (start..end)
                        .map(|j| bp.get(j).expect("row in range"))
                        .collect();
                    let block_packed =
                        crate::codec::BitPackedInts::pack_with_bits(&row_values, bits_per_value);
                    write_usize_as_u32(&mut bodies, block_packed.data().len());
                    for &word in block_packed.data() {
                        bodies.extend_from_slice(&word.to_le_bytes());
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Dict(dict) => {
                buf.push(1);
                let entries = dict.dictionary();
                write_usize_as_u32(buf, entries.len());
                for entry in entries.iter() {
                    let s = entry.as_ref().as_bytes();
                    write_usize_as_u32(buf, s.len());
                    buf.extend_from_slice(s);
                }
                let codes = dict.codes();
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(codes.len());
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end - start) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    for &code in &codes[start..end] {
                        bodies.extend_from_slice(&code.to_le_bytes());
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Bitmap(bv) => {
                buf.push(2);
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(bv.len());
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end - start) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    let bits: Vec<bool> = (start..end)
                        .map(|j| bv.get(j).expect("row in range"))
                        .collect();
                    let block_bv = crate::codec::BitVector::from_bools(&bits);
                    write_usize_as_u32(&mut bodies, block_bv.data().len());
                    for &word in block_bv.data() {
                        bodies.extend_from_slice(&word.to_le_bytes());
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Int8Vector { bytes, dimensions } => {
                buf.push(3);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims = *dimensions as usize;
                let row_count_total = bytes.len().checked_div(dims).unwrap_or(0);
                for i in 0..block_count {
                    let start_row = i * block_rows;
                    let end_row = (start_row + block_rows).min(row_count_total);
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end_row - start_row) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    if dims > 0 {
                        let start = start_row * dims;
                        let end = end_row * dims;
                        bodies.extend_from_slice(&bytes[start..end]);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Float64(bytes) => {
                buf.push(4);
                let total_rows = bytes.len() / 8;
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(total_rows);
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end - start) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    bodies.extend_from_slice(&bytes[start * 8..end * 8]);
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Float32Vector { bytes, dimensions } => {
                buf.push(5);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims = *dimensions as usize;
                let row_byte_size = dims.checked_mul(4).unwrap_or(0);
                let row_count_total = bytes.len().checked_div(row_byte_size.max(1)).unwrap_or(0);
                for i in 0..block_count {
                    let start_row = i * block_rows;
                    let end_row = (start_row + block_rows).min(row_count_total);
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end_row - start_row) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    if row_byte_size > 0 {
                        let start = start_row * row_byte_size;
                        let end = end_row * row_byte_size;
                        bodies.extend_from_slice(&bytes[start..end]);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::RawI64(bytes) => {
                buf.push(6);
                let total_rows = bytes.len() / 8;
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(total_rows);
                    #[allow(clippy::cast_possible_truncation)]
                    let row_count = (end - start) as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_offset = bodies.len() as u32;
                    bodies.extend_from_slice(&bytes[start * 8..end * 8]);
                    #[allow(clippy::cast_possible_truncation)]
                    let byte_len = (bodies.len() as u32) - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
        }

        (metas, bodies)
    }

    /// Deserializes a codec from v2 format.
    ///
    /// Inverse of [`write_to_v2`](Self::write_to_v2). The reader is
    /// strict: it requires the block index and concatenated bodies to
    /// match exactly, and refuses unknown discriminants.
    ///
    /// # Errors
    ///
    /// Returns a static-string error on truncation, unknown discriminant,
    /// or block-index inconsistency (offset + len out of bounds).
    pub fn read_from_v2(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let discriminant = *data.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        match discriminant {
            0 => {
                // BitPacked: bits_per_value, then block index + bodies.
                let bits = *data.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let mut all_values: Vec<u64> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > data.len() {
                        return Err("BitPacked block body out of bounds");
                    }
                    let mut bp = body_start;
                    let word_count = read_u32_le(data, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(data, &mut bp)?);
                    }
                    let block_bp = crate::codec::BitPackedInts::from_raw_parts(
                        words,
                        bits,
                        meta.row_count as usize,
                    );
                    for j in 0..meta.row_count as usize {
                        all_values.push(
                            block_bp
                                .get(j)
                                .ok_or("BitPacked block index out of range")?,
                        );
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::BitPacked(
                    crate::codec::BitPackedInts::pack_with_bits(&all_values, bits),
                ))
            }
            1 => {
                // Dict: global dictionary, then block index of u32 codes.
                let dict_len = read_u32_le(data, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(data, pos)? as usize;
                    if *pos + slen > data.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&data[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let mut codes: Vec<u32> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    for _ in 0..meta.row_count {
                        codes.push(read_u32_le(data, &mut bp)?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Dict(DictionaryEncoding::new(
                    Arc::from(entries.into_boxed_slice()),
                    codes,
                )))
            }
            2 => {
                // Bitmap
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let mut all_bits: Vec<bool> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    let word_count = read_u32_le(data, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(data, &mut bp)?);
                    }
                    let block_bv =
                        crate::codec::BitVector::from_raw_parts(words, meta.row_count as usize);
                    for j in 0..meta.row_count as usize {
                        all_bits.push(block_bv.get(j).ok_or("Bitmap block index out of range")?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Bitmap(crate::codec::BitVector::from_bools(&all_bits)))
            }
            3 => {
                // Int8Vector: each block carries row_count * dims raw bytes.
                let dimensions = read_u16_le(data, pos)?;
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let dims = dimensions as usize;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > data.len() {
                        return Err("Int8Vector block body out of bounds");
                    }
                    let expected = (meta.row_count as usize) * dims;
                    if (meta.byte_len as usize) != expected {
                        return Err("Int8Vector block byte_len mismatch");
                    }
                    buf.extend_from_slice(&data[body_start..body_end]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Int8Vector {
                    bytes: buf.freeze(),
                    dimensions,
                })
            }
            4 => {
                // Float64: each block carries row_count * 8 LE bytes.
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * 8;
                    if body_start + need > data.len() {
                        return Err("truncated Float64 block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Float64(buf.freeze()))
            }
            5 => {
                // Float32Vector: each block carries row_count * dims * 4 LE bytes.
                let dimensions = read_u16_le(data, pos)?;
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let row_byte_size = (dimensions as usize).checked_mul(4).unwrap_or(0);
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * row_byte_size;
                    if body_start + need > data.len() {
                        return Err("truncated Float32Vector block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Float32Vector {
                    bytes: buf.freeze(),
                    dimensions,
                })
            }
            6 => {
                // RawI64: each block carries row_count * 8 LE bytes.
                let (metas, bodies_start) = read_block_index(data, pos)?;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * 8;
                    if body_start + need > data.len() {
                        return Err("truncated RawI64 block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::RawI64(buf.freeze()))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Deserializes a codec from v3 format, returning the codec and
    /// per-block zone maps for skip pruning.
    ///
    /// The body of each block uses the same layout as v2; only the
    /// block index is extended with an inline `ZoneMap` per entry.
    ///
    /// # Errors
    ///
    /// Returns a static-string error on truncation, unknown
    /// discriminant, or block-index inconsistency.
    pub fn read_from_v3(
        data: &[u8],
        pos: &mut usize,
    ) -> Result<(Self, Vec<super::zone_map::ZoneMap>), &'static str> {
        let discriminant = *data.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        match discriminant {
            0 => {
                let bits = *data.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let mut all_values: Vec<u64> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > data.len() {
                        return Err("BitPacked block body out of bounds");
                    }
                    let mut bp = body_start;
                    let word_count = read_u32_le(data, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(data, &mut bp)?);
                    }
                    let block_bp = crate::codec::BitPackedInts::from_raw_parts(
                        words,
                        bits,
                        meta.row_count as usize,
                    );
                    for j in 0..meta.row_count as usize {
                        all_values.push(
                            block_bp
                                .get(j)
                                .ok_or("BitPacked block index out of range")?,
                        );
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::BitPacked(crate::codec::BitPackedInts::pack_with_bits(
                        &all_values,
                        bits,
                    )),
                    stats,
                ))
            }
            1 => {
                let dict_len = read_u32_le(data, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(data, pos)? as usize;
                    if *pos + slen > data.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&data[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let mut codes: Vec<u32> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    for _ in 0..meta.row_count {
                        codes.push(read_u32_le(data, &mut bp)?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::Dict(DictionaryEncoding::new(
                        Arc::from(entries.into_boxed_slice()),
                        codes,
                    )),
                    stats,
                ))
            }
            2 => {
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let mut all_bits: Vec<bool> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    let word_count = read_u32_le(data, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(data, &mut bp)?);
                    }
                    let block_bv =
                        crate::codec::BitVector::from_raw_parts(words, meta.row_count as usize);
                    for j in 0..meta.row_count as usize {
                        all_bits.push(block_bv.get(j).ok_or("Bitmap block index out of range")?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::Bitmap(crate::codec::BitVector::from_bools(&all_bits)),
                    stats,
                ))
            }
            3 => {
                let dimensions = read_u16_le(data, pos)?;
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let dims = dimensions as usize;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > data.len() {
                        return Err("Int8Vector block body out of bounds");
                    }
                    let expected = (meta.row_count as usize) * dims;
                    if (meta.byte_len as usize) != expected {
                        return Err("Int8Vector block byte_len mismatch");
                    }
                    buf.extend_from_slice(&data[body_start..body_end]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::Int8Vector {
                        bytes: buf.freeze(),
                        dimensions,
                    },
                    stats,
                ))
            }
            4 => {
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * 8;
                    if body_start + need > data.len() {
                        return Err("truncated Float64 block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((Self::Float64(buf.freeze()), stats))
            }
            5 => {
                let dimensions = read_u16_le(data, pos)?;
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let row_byte_size = (dimensions as usize).checked_mul(4).unwrap_or(0);
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * row_byte_size;
                    if body_start + need > data.len() {
                        return Err("truncated Float32Vector block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::Float32Vector {
                        bytes: buf.freeze(),
                        dimensions,
                    },
                    stats,
                ))
            }
            6 => {
                let (metas, stats, bodies_start) = read_block_index_v3(data, pos)?;
                let mut buf = BytesMut::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let need = (meta.row_count as usize) * 8;
                    if body_start + need > data.len() {
                        return Err("truncated RawI64 block body");
                    }
                    buf.extend_from_slice(&data[body_start..body_start + need]);
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((Self::RawI64(buf.freeze()), stats))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Returns an estimate of heap memory used by this column in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::BitPacked(bp) => bp.data().len() * std::mem::size_of::<u64>(),
            Self::Dict(d) => {
                let codes_bytes = d.codes().len() * std::mem::size_of::<u32>();
                let dict_bytes: usize = d.dictionary().iter().map(|s| s.len()).sum();
                codes_bytes + dict_bytes
            }
            Self::Bitmap(bv) => bv.data().len() * std::mem::size_of::<u64>(),
            Self::Int8Vector { bytes, .. } => bytes.len(),
            Self::Float64(bytes) => bytes.len(),
            Self::Float32Vector { bytes, .. } => bytes.len(),
            Self::RawI64(bytes) => bytes.len(),
        }
    }
}

// ── v2 block-index helpers ──────────────────────────────────────

/// Per-block metadata in the v2 column index.
///
/// Phase 2c will extend this with stats fields by bumping the section
/// version to 3; the current 12-byte layout pins the v2 format.
#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    byte_offset: u32,
    byte_len: u32,
    row_count: u32,
}

const BLOCK_META_BYTES: usize = 12;

/// Total byte length of all bodies described by `metas`.
fn total_bodies_len(metas: &[BlockMeta]) -> usize {
    metas
        .last()
        .map_or(0, |m| (m.byte_offset + m.byte_len) as usize)
}

/// Writes the v2 block index followed by the concatenated block bodies.
fn write_block_index_and_bodies(buf: &mut Vec<u8>, metas: &[BlockMeta], bodies: &[u8]) {
    write_usize_as_u32(buf, metas.len());
    for meta in metas {
        buf.extend_from_slice(&meta.byte_offset.to_le_bytes());
        buf.extend_from_slice(&meta.byte_len.to_le_bytes());
        buf.extend_from_slice(&meta.row_count.to_le_bytes());
    }
    buf.extend_from_slice(bodies);
}

/// Writes the v3 block index (v2 layout + inline per-block ZoneMap)
/// followed by the concatenated block bodies.
fn write_block_index_and_bodies_with_stats(
    buf: &mut Vec<u8>,
    metas: &[BlockMeta],
    bodies: &[u8],
    stats: &[super::zone_map::ZoneMap],
) {
    debug_assert_eq!(metas.len(), stats.len(), "stats must align with metas");
    write_usize_as_u32(buf, metas.len());
    for (meta, zm) in metas.iter().zip(stats.iter()) {
        buf.extend_from_slice(&meta.byte_offset.to_le_bytes());
        buf.extend_from_slice(&meta.byte_len.to_le_bytes());
        buf.extend_from_slice(&meta.row_count.to_le_bytes());
        zm.write_inline(buf);
    }
    buf.extend_from_slice(bodies);
}

/// Reads the v2 block index and returns the parsed metas and the byte
/// position where the concatenated bodies start.
fn read_block_index(data: &[u8], pos: &mut usize) -> Result<(Vec<BlockMeta>, usize), &'static str> {
    let block_count = read_u32_le(data, pos)? as usize;
    let index_bytes = block_count
        .checked_mul(BLOCK_META_BYTES)
        .ok_or("block index overflow")?;
    if *pos + index_bytes > data.len() {
        return Err("truncated block index");
    }
    let mut metas = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let byte_offset = read_u32_le(data, pos)?;
        let byte_len = read_u32_le(data, pos)?;
        let row_count = read_u32_le(data, pos)?;
        metas.push(BlockMeta {
            byte_offset,
            byte_len,
            row_count,
        });
    }
    let bodies_start = *pos;
    Ok((metas, bodies_start))
}

/// Reads the v3 block index (v2 layout + inline per-block ZoneMap) and
/// returns the parsed metas, per-block stats, and the byte position
/// where the concatenated bodies start.
fn read_block_index_v3(
    data: &[u8],
    pos: &mut usize,
) -> Result<(Vec<BlockMeta>, Vec<super::zone_map::ZoneMap>, usize), &'static str> {
    let block_count = read_u32_le(data, pos)? as usize;
    let mut metas = Vec::with_capacity(block_count);
    let mut stats = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let byte_offset = read_u32_le(data, pos)?;
        let byte_len = read_u32_le(data, pos)?;
        let row_count = read_u32_le(data, pos)?;
        let zm = super::zone_map::ZoneMap::read_inline(data, pos)?;
        metas.push(BlockMeta {
            byte_offset,
            byte_len,
            row_count,
        });
        stats.push(zm);
    }
    let bodies_start = *pos;
    Ok((metas, stats, bodies_start))
}

// ── Binary read helpers ─────────────────────────────────────────

/// Writes a usize as u32 LE, panicking on overflow (data >4 GiB).
fn write_usize_as_u32(buf: &mut Vec<u8>, v: usize) {
    let n = u32::try_from(v).expect("value exceeds u32::MAX in compact codec serialization");
    buf.extend_from_slice(&n.to_le_bytes());
}

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16, &'static str> {
    if *pos + 2 > data.len() {
        return Err("truncated u16");
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, &'static str> {
    if *pos + 8 > data.len() {
        return Err("truncated u64");
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

#[cfg(test)]
// reason: test values are small known constants
#[allow(clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use crate::codec::{BitPackedInts, BitVector, DictionaryBuilder};

    #[test]
    fn test_bitpacked_round_trip() {
        // 4-bit values (max = 15)
        let values = vec![0u64, 5, 10, 15, 3, 7];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.len(), 6);
        assert!(!col.is_empty());

        for (i, &expected) in values.iter().enumerate() {
            let v = col.get(i).unwrap();
            assert_eq!(v, Value::Int64(expected as i64));
        }
    }

    #[test]
    fn test_dict_round_trip() {
        let mut builder = DictionaryBuilder::new();
        builder.add("alpha");
        builder.add("beta");
        builder.add("alpha");
        let dict = builder.build();

        let col = ColumnCodec::Dict(dict);
        assert_eq!(col.len(), 3);

        assert_eq!(col.get(0), Some(Value::String(ArcStr::from("alpha"))));
        assert_eq!(col.get(1), Some(Value::String(ArcStr::from("beta"))));
        assert_eq!(col.get(2), Some(Value::String(ArcStr::from("alpha"))));
    }

    #[test]
    fn test_bitmap_round_trip() {
        let bools = vec![true, false, true, true, false];
        let bv = BitVector::from_bools(&bools);
        let col = ColumnCodec::Bitmap(bv);

        assert_eq!(col.len(), 5);
        assert_eq!(col.get(0), Some(Value::Bool(true)));
        assert_eq!(col.get(1), Some(Value::Bool(false)));
        assert_eq!(col.get(2), Some(Value::Bool(true)));
        assert_eq!(col.get(3), Some(Value::Bool(true)));
        assert_eq!(col.get(4), Some(Value::Bool(false)));
    }

    #[test]
    fn test_int8_vector_round_trip() {
        // 2 vectors of dimension 3
        let data = vec![1i8, 2, 3, -4, -5, -6];
        let col = ColumnCodec::int8_vector(data, 3);

        assert_eq!(col.len(), 2);

        let v0 = col.get(0).unwrap();
        let expected0: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert_eq!(v0, Value::List(Arc::from(expected0)));

        let v1 = col.get(1).unwrap();
        let expected1: Vec<Value> = vec![Value::Int64(-4), Value::Int64(-5), Value::Int64(-6)];
        assert_eq!(v1, Value::List(Arc::from(expected1)));
    }

    #[test]
    fn test_get_raw_u64_on_bitpacked() {
        let values = vec![100u64, 200, 300];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.get_raw_u64(0), Some(100));
        assert_eq!(col.get_raw_u64(1), Some(200));
        assert_eq!(col.get_raw_u64(2), Some(300));
        assert_eq!(col.get_raw_u64(3), None);

        // Non-BitPacked variant returns None.
        let bv = BitVector::from_bools(&[true]);
        let bm_col = ColumnCodec::Bitmap(bv);
        assert_eq!(bm_col.get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_slice() {
        let data = vec![10i8, 20, 30, 40, 50, 60];
        let col = ColumnCodec::int8_vector(data, 3);

        assert_eq!(col.get_int8_vector(0), Some(&[10i8, 20, 30][..]));
        assert_eq!(col.get_int8_vector(1), Some(&[40i8, 50, 60][..]));
        assert_eq!(col.get_int8_vector(2), None);

        // Non-Int8Vector variant returns None.
        let bp = BitPackedInts::pack(&[1u64]);
        let bp_col = ColumnCodec::BitPacked(bp);
        assert_eq!(bp_col.get_int8_vector(0), None);
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let bp = BitPackedInts::pack(&[1u64, 2, 3]);
        let col = ColumnCodec::BitPacked(bp);
        assert_eq!(col.get(999), None);
        assert_eq!(col.get_raw_u64(999), None);

        let bv = BitVector::from_bools(&[true]);
        let bm = ColumnCodec::Bitmap(bv);
        assert_eq!(bm.get(5), None);

        let mut builder = DictionaryBuilder::new();
        builder.add("x");
        let dict = builder.build();
        let dc = ColumnCodec::Dict(dict);
        assert_eq!(dc.get(10), None);

        let vec_col = ColumnCodec::int8_vector(vec![1, 2], 2);
        assert_eq!(vec_col.get(1), None);
        assert_eq!(vec_col.get_int8_vector(1), None);
    }

    // -----------------------------------------------------------------------
    // find_eq tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_eq_bitpacked() {
        let values = vec![0u64, 5, 10, 5, 3, 5];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.find_eq(&Value::Int64(5)), vec![1, 3, 5]);
        assert_eq!(col.find_eq(&Value::Int64(0)), vec![0]);
        assert_eq!(col.find_eq(&Value::Int64(99)), Vec::<usize>::new());
        // Negative target: BitPacked stores unsigned values, no matches.
        assert_eq!(col.find_eq(&Value::Int64(-1)), Vec::<usize>::new());
    }

    #[test]
    fn test_find_eq_dict() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Vincent", "Jules", "Vincent", "Mia", "Jules"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        assert_eq!(col.find_eq(&Value::String("Vincent".into())), vec![0, 2]);
        assert_eq!(col.find_eq(&Value::String("Mia".into())), vec![3]);
        assert_eq!(
            col.find_eq(&Value::String("Butch".into())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_find_eq_bitmap() {
        let bools = vec![true, false, true, true, false];
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));

        assert_eq!(col.find_eq(&Value::Bool(true)), vec![0, 2, 3]);
        assert_eq!(col.find_eq(&Value::Bool(false)), vec![1, 4]);
    }

    #[test]
    fn test_find_eq_type_mismatch_uses_fallback() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // String target on BitPacked column: type mismatch, falls back.
        assert_eq!(
            col.find_eq(&Value::String("hello".into())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_find_eq_int8_vector_uses_fallback() {
        // Int8Vector has no specialised find_eq path, so it uses the fallback.
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);
        let target_vec: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        let target = Value::List(Arc::from(target_vec));
        let matches = col.find_eq(&target);
        assert_eq!(matches, vec![0]);
    }

    // -----------------------------------------------------------------------
    // Int8Vector zero-dimension edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_int8_vector_zero_dimensions_get() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        // Zero dimensions: get() should return None.
        assert_eq!(col.get(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_get_int8_vector() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        // Zero dimensions: get_int8_vector() should return None.
        assert_eq!(col.get_int8_vector(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_len_and_is_empty() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    // -----------------------------------------------------------------------
    // heap_bytes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_heap_bytes_bitpacked() {
        let values = vec![0u64, 5, 10, 15];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);
        // Should report nonzero heap usage.
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_dict() {
        let mut builder = DictionaryBuilder::new();
        builder.add("Amsterdam");
        builder.add("Berlin");
        builder.add("Paris");
        let dict = builder.build();
        let col = ColumnCodec::Dict(dict);
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_bitmap() {
        let bools = vec![true, false, true, true, false];
        let bv = BitVector::from_bools(&bools);
        let col = ColumnCodec::Bitmap(bv);
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_int8_vector() {
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);
        // Heap usage equals data length (1 byte per i8).
        assert_eq!(col.heap_bytes(), 6);
    }

    // -----------------------------------------------------------------------
    // find_in_range tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_bitpacked_inclusive() {
        // values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // [3, 6] inclusive
        let result = col.find_in_range(Some(&Value::Int64(3)), Some(&Value::Int64(6)), true, true);
        assert_eq!(result, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_find_in_range_bitpacked_exclusive() {
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // (3, 6) exclusive
        let result =
            col.find_in_range(Some(&Value::Int64(3)), Some(&Value::Int64(6)), false, false);
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn test_find_in_range_bitpacked_open_ended() {
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // > 7 (no upper bound)
        let result = col.find_in_range(Some(&Value::Int64(7)), None, false, false);
        assert_eq!(result, vec![8, 9]);

        // <= 2 (no lower bound)
        let result = col.find_in_range(None, Some(&Value::Int64(2)), false, true);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_find_in_range_fallback_for_dict() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // String range ["Berlin", "Prague"] inclusive: Berlin, Paris, Prague
        let result = col.find_in_range(
            Some(&Value::String("Berlin".into())),
            Some(&Value::String("Prague".into())),
            true,
            true,
        );
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_find_in_range_negative_max() {
        // All values are unsigned (>= 0), so a negative max should yield no results.
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(None, Some(&Value::Int64(-1)), false, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_in_range_negative_min() {
        // Negative min is clamped to 0 internally: all values (0..10) should pass.
        let values: Vec<u64> = (0..5).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(Some(&Value::Int64(-10)), None, true, true);
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_find_in_range_type_mismatch_uses_fallback() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // String bounds on a BitPacked column: type mismatch, falls back.
        let result = col.find_in_range(
            Some(&Value::String("a".into())),
            Some(&Value::String("z".into())),
            true,
            true,
        );
        // Fallback uses compare_values which returns None for Int vs String,
        // so no rows match.
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_in_range_int8_vector_uses_fallback() {
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);

        // Int8Vector is not BitPacked, so it goes through the fallback path.
        // Range scan on list values uses compare_values, which returns None
        // for lists, so nothing matches.
        let result = col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(10)), true, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_out_of_bounds_all_codecs() {
        // BitPacked
        let bp = BitPackedInts::pack(&[1u64, 2, 3]);
        let col = ColumnCodec::BitPacked(bp);
        assert_eq!(col.get(3), None);

        // Dict
        let mut builder = DictionaryBuilder::new();
        builder.add("Alix");
        let col = ColumnCodec::Dict(builder.build());
        assert_eq!(col.get(1), None);

        // Bitmap
        let bv = BitVector::from_bools(&[true]);
        let col = ColumnCodec::Bitmap(bv);
        assert_eq!(col.get(1), None);

        // Int8Vector
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 3);
        assert_eq!(col.get(1), None);
        assert_eq!(col.get_int8_vector(1), None);
    }

    // -----------------------------------------------------------------------
    // Large Int8Vector column: serde round-trip and random access at scale
    // -----------------------------------------------------------------------

    /// Build a 100-vector Int8Vector column of dim 384, serialize it via
    /// `write_to`, deserialize via `read_from`, and verify bit-exact roundtrip
    /// at representative indices. Covers the Int8Vector branches in `get`,
    /// `get_int8_vector`, and both `write_to` / `read_from` (discriminant 3).
    #[test]
    fn test_column_int8_vector_roundtrip() {
        let dims: u16 = 384;
        let rows = 100usize;
        // Deterministic values: row r, dim d -> ((r * 7 + d) mod 251) - 120.
        // reason: intentional modular wrap to produce i8 values across the range
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let data: Vec<i8> = (0..rows * dims as usize)
            .map(|idx| (((idx * 7) % 251) as i64 - 120) as i8)
            .collect();
        let col = ColumnCodec::int8_vector(data.clone(), dims);
        assert_eq!(col.len(), rows);

        // Serialize and deserialize.
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len(), "read_from should consume the full buffer");
        assert_eq!(decoded.len(), rows);

        // Check representative indices after the round-trip.
        for &row in &[0usize, 1, 50, 99] {
            let decoded_slice = decoded.get_int8_vector(row).unwrap();
            let start = row * dims as usize;
            assert_eq!(decoded_slice, &data[start..start + dims as usize]);
            // Also verify the Value::List decoding path is consistent.
            let decoded_value = decoded.get(row).unwrap();
            if let Value::List(items) = decoded_value {
                assert_eq!(items.len(), dims as usize);
                assert_eq!(items[0], Value::Int64(i64::from(decoded_slice[0])));
            } else {
                panic!("expected Value::List for Int8Vector element");
            }
        }
    }

    /// Index-out-of-bounds and zero-dimensions edge cases for `Int8Vector`
    /// exercised together. Covers the early-return guards in both `get`
    /// (lines 62-70) and `get_int8_vector` (lines 100-110).
    #[test]
    fn test_column_vector_oob_and_zero_dim() {
        // OOB: 2 vectors of dim 3, index 5 is well past the end.
        let col = ColumnCodec::int8_vector(vec![1i8, 2, 3, 4, 5, 6], 3);
        assert_eq!(col.len(), 2);
        assert!(col.get(2).is_none());
        assert!(col.get(5).is_none());
        assert!(col.get_int8_vector(2).is_none());
        assert!(col.get_int8_vector(5).is_none());

        // Zero-dim column: len == 0 by construction and no element is accessible.
        let zero = ColumnCodec::int8_vector(Vec::new(), 0);
        assert_eq!(zero.len(), 0);
        assert!(zero.is_empty());
        assert!(zero.get(0).is_none());
        assert!(zero.get_int8_vector(0).is_none());
    }

    /// A `Dict` column queried with `Int64` bounds: both bounds are a type
    /// mismatch, so `find_in_range` takes the fallback path. Inside the
    /// fallback, `compare_values(String, Int64)` returns `None` for every row,
    /// so the result is empty. Covers the non-BitPacked entry into
    /// `find_in_range_fallback` with a type-mismatched range.
    #[test]
    fn test_find_in_range_incompatible_types() {
        let mut builder = DictionaryBuilder::new();
        for city in ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"] {
            builder.add(city);
        }
        let col = ColumnCodec::Dict(builder.build());

        let result =
            col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(100)), true, true);
        assert!(
            result.is_empty(),
            "Int64 bounds on a Dict column should yield no matches"
        );
    }

    /// A buffer that is truncated mid-codec must return an error, never panic.
    /// Serializes a valid BitPacked column, then truncates the buffer at each
    /// interior byte and verifies `read_from` reports an error.
    #[test]
    fn test_column_serde_truncated_buffer() {
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&[1u64, 2, 3, 4, 5]));
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        assert!(buf.len() > 4);

        // Truncate to zero: missing discriminant.
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&[], &mut pos).is_err());

        // Truncate after the discriminant but before bits_per_value.
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&buf[..1], &mut pos).is_err());

        // Truncate mid-header (before the row count u32 is complete).
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&buf[..3], &mut pos).is_err());

        // Truncate mid-payload (drop the last byte of the packed u64 words).
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&buf[..buf.len() - 1], &mut pos).is_err());

        // Unknown discriminant: must return an error, not panic.
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&[0xFFu8], &mut pos).is_err());

        // Truncated Int8Vector payload (dimensions OK, length promises more
        // bytes than exist). Build a minimal header: discriminant=3, dims=2,
        // data_len=4, but only provide 2 bytes of payload.
        let mut bad = vec![3u8];
        bad.extend_from_slice(&2u16.to_le_bytes());
        bad.extend_from_slice(&4u32.to_le_bytes());
        bad.extend_from_slice(&[0u8, 0u8]);
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&bad, &mut pos).is_err());
    }

    // -----------------------------------------------------------------------
    // Serde round-trip tests: write_to + read_from
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_read_round_trip_bitpacked() {
        let bp = BitPackedInts::pack(&[3u64, 7, 12, 5]);
        let col = ColumnCodec::BitPacked(bp);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len(), "read should consume entire buffer");

        assert_eq!(decoded.len(), 4);
        for i in 0..4 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_dict() {
        let mut b = DictionaryBuilder::new();
        for s in ["Amsterdam", "Berlin", "Amsterdam", "Paris"] {
            b.add(s);
        }
        let col = ColumnCodec::Dict(b.build());

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 4);
        for i in 0..4 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_bitmap() {
        let bv = BitVector::from_bools(&[true, false, true, true, false, false, true]);
        let col = ColumnCodec::Bitmap(bv);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 7);
        for i in 0..7 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_int8_vector() {
        let data: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let col = ColumnCodec::int8_vector(data, 4);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.get_int8_vector(0), Some(&[1i8, -2, 3, -4][..]));
        assert_eq!(decoded.get_int8_vector(1), Some(&[5i8, -6, 7, -8][..]));
    }

    // -----------------------------------------------------------------------
    // Serde error cases: truncation and unknown discriminants
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_from_empty_buffer_errors() {
        let mut pos = 0;
        let err = ColumnCodec::read_from(&[], &mut pos).unwrap_err();
        assert_eq!(err, "truncated codec discriminant");
    }

    #[test]
    fn test_read_from_unknown_discriminant_errors() {
        // Discriminant values 0..=3 are valid; 99 is unknown.
        let buf = vec![99u8];
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "unknown codec discriminant");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_bits() {
        // Discriminant 0 (BitPacked) but no bits_per_value byte following.
        let buf = vec![0u8];
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated bits_per_value");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_count() {
        // Discriminant + bits byte, but truncated before u32 count.
        let buf = vec![0u8, 4, 0, 0]; // only 2 of 4 bytes for u32 count
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u32");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_words() {
        // Discriminant + bits + count + data_len=2 but no u64 data words.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes()); // count=1
        buf.extend_from_slice(&2u32.to_le_bytes()); // data_len=2
        // no u64 words
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u64");
    }

    #[test]
    fn test_read_from_truncated_dict_string() {
        // Discriminant=1 (Dict), dict_len=1, slen=5, but only 3 bytes of string.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len=1
        buf.extend_from_slice(&5u32.to_le_bytes()); // slen=5
        buf.extend_from_slice(b"abc"); // only 3 bytes, need 5
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated dict string");
    }

    #[test]
    fn test_read_from_invalid_utf8_in_dict() {
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len=1
        buf.extend_from_slice(&2u32.to_le_bytes()); // slen=2
        buf.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "invalid UTF-8 in dict");
    }

    #[test]
    fn test_read_from_truncated_bitmap_words() {
        // Discriminant=2 (Bitmap), bit_len=64, data_len=1, but no u64 data.
        let mut buf = vec![2u8];
        buf.extend_from_slice(&64u32.to_le_bytes()); // bit_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // data_len
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u64");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_dimensions() {
        // Discriminant=3 (Int8Vector), only 1 byte of the 2-byte dimensions.
        let buf = vec![3u8, 0];
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u16");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_data() {
        // Discriminant=3, dimensions=2, data_len=4, but only 2 data bytes.
        let mut buf = vec![3u8];
        buf.extend_from_slice(&2u16.to_le_bytes()); // dimensions=2
        buf.extend_from_slice(&4u32.to_le_bytes()); // data_len=4
        buf.extend_from_slice(&[10u8, 20]); // only 2 of 4 bytes
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated Int8Vector data");
    }

    // -----------------------------------------------------------------------
    // Empty (zero-length) columns: all codec variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_bitpacked_round_trip() {
        let bp = BitPackedInts::pack(&[]);
        let col = ColumnCodec::BitPacked(bp);
        assert!(col.is_empty());
        assert_eq!(col.len(), 0);

        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_dict_round_trip() {
        let builder = DictionaryBuilder::new();
        let dict = builder.build();
        let col = ColumnCodec::Dict(dict);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_bitmap_round_trip() {
        let bv = BitVector::from_bools(&[]);
        let col = ColumnCodec::Bitmap(bv);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_int8_vector_round_trip() {
        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_string_in_dict() {
        let mut b = DictionaryBuilder::new();
        b.add("");
        b.add("Alix");
        b.add("");
        let col = ColumnCodec::Dict(b.build());

        assert_eq!(col.get(0), Some(Value::String(ArcStr::from(""))));
        assert_eq!(col.get(1), Some(Value::String(ArcStr::from("Alix"))));
        assert_eq!(col.get(2), Some(Value::String(ArcStr::from(""))));

        // Round-trip to exercise the len=0 branch of write/read dict string.
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).unwrap();
        assert_eq!(decoded.get(0), Some(Value::String(ArcStr::from(""))));
        assert_eq!(decoded.get(2), Some(Value::String(ArcStr::from(""))));
    }

    // -----------------------------------------------------------------------
    // find_eq / find_in_range boundary exactness
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_exact_boundaries_inclusive_vs_exclusive() {
        // values: 10, 20, 30, 40, 50
        let values = vec![10u64, 20, 30, 40, 50];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // [20, 40] inclusive on both ends, boundary values included.
        let inclusive =
            col.find_in_range(Some(&Value::Int64(20)), Some(&Value::Int64(40)), true, true);
        assert_eq!(inclusive, vec![1, 2, 3]);

        // (20, 40) exclusive on both ends, boundaries excluded.
        let exclusive = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            false,
            false,
        );
        assert_eq!(exclusive, vec![2]);

        // [20, 40) inclusive-exclusive mix.
        let mixed_a = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            true,
            false,
        );
        assert_eq!(mixed_a, vec![1, 2]);

        // (20, 40] exclusive-inclusive mix.
        let mixed_b = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            false,
            true,
        );
        assert_eq!(mixed_b, vec![2, 3]);
    }

    #[test]
    fn test_find_in_range_bitpacked_fallback_on_float_min() {
        // Float min triggers the fallback path on BitPacked. The fallback
        // still knows how to compare Int64 <-> Float64, so values >= 2.5
        // out of [1, 2, 3] match.
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(Some(&Value::Float64(2.5)), None, true, true);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_find_in_range_bitpacked_fallback_on_float_max() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // Float max also routes through the fallback, comparing Int <-> Float.
        let result = col.find_in_range(None, Some(&Value::Float64(2.5)), true, true);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_find_in_range_open_both_ends_returns_all() {
        let values = vec![1u64, 2, 3, 4, 5];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let all = col.find_in_range(None, None, true, true);
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_find_in_range_fallback_dict_exclusive() {
        let mut b = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            b.add(name);
        }
        let col = ColumnCodec::Dict(b.build());

        // Exclusive on lower bound: "Berlin" excluded.
        let result = col.find_in_range(Some(&Value::String("Berlin".into())), None, false, true);
        assert_eq!(result, vec![2, 3]); // Paris, Prague

        // Exclusive on upper bound: "Prague" excluded.
        let result = col.find_in_range(None, Some(&Value::String("Prague".into())), true, false);
        assert_eq!(result, vec![0, 1, 2]); // Amsterdam, Berlin, Paris
    }

    #[test]
    fn test_find_in_range_fallback_mismatch_returns_none_for_row() {
        // Target uses None from compare_values by mixing codec types.
        // Bitmap column + Int64 range -> compare returns None -> rows excluded.
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false, true]));
        let result = col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(5)), true, true);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // get_raw_u64 / get_int8_vector on all non-matching variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_raw_u64_returns_none_for_all_non_bitpacked() {
        let mut b = DictionaryBuilder::new();
        b.add("x");
        assert_eq!(ColumnCodec::Dict(b.build()).get_raw_u64(0), None);

        assert_eq!(ColumnCodec::int8_vector(vec![1i8], 1).get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_returns_none_for_all_non_vector() {
        let bp = BitPackedInts::pack(&[1u64]);
        assert_eq!(ColumnCodec::BitPacked(bp).get_int8_vector(0), None);

        let mut b = DictionaryBuilder::new();
        b.add("x");
        assert_eq!(ColumnCodec::Dict(b.build()).get_int8_vector(0), None);

        let bv = BitVector::from_bools(&[true]);
        assert_eq!(ColumnCodec::Bitmap(bv).get_int8_vector(0), None);
    }

    // -----------------------------------------------------------------------
    // heap_bytes on empty columns
    // -----------------------------------------------------------------------

    #[test]
    fn test_heap_bytes_empty_columns() {
        let bp = BitPackedInts::pack(&[]);
        assert_eq!(ColumnCodec::BitPacked(bp).heap_bytes(), 0);

        let builder = DictionaryBuilder::new();
        assert_eq!(ColumnCodec::Dict(builder.build()).heap_bytes(), 0);

        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        assert_eq!(col.heap_bytes(), 0);
    }

    // -----------------------------------------------------------------------
    // find_eq for Dict where target string is not in the dictionary
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_eq_dict_target_not_in_dictionary() {
        let mut b = DictionaryBuilder::new();
        b.add("Amsterdam");
        b.add("Berlin");
        let col = ColumnCodec::Dict(b.build());

        // Target "Prague" does not exist in the dictionary, so encode returns None.
        let result = col.find_eq(&Value::String(ArcStr::from("Prague")));
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Accessor coverage for non-matching variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_raw_u64_on_dict_and_int8_vector_returns_none() {
        // Dict variant: get_raw_u64 is meaningless.
        let mut builder = DictionaryBuilder::new();
        builder.add("Vincent");
        let dict_col = ColumnCodec::Dict(builder.build());
        assert_eq!(dict_col.get_raw_u64(0), None);

        // Int8Vector variant: get_raw_u64 is meaningless.
        let vec_col = ColumnCodec::int8_vector(vec![1i8, 2, 3], 3);
        assert_eq!(vec_col.get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_on_dict_and_bitmap_returns_none() {
        // Dict variant: not an Int8Vector.
        let mut builder = DictionaryBuilder::new();
        builder.add("Jules");
        let dict_col = ColumnCodec::Dict(builder.build());
        assert_eq!(dict_col.get_int8_vector(0), None);

        // Bitmap variant: not an Int8Vector.
        let bm_col = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false]));
        assert_eq!(bm_col.get_int8_vector(0), None);
    }

    // -----------------------------------------------------------------------
    // find_in_range fallback branches: Dict exclusive/open-ended, Equal with
    // !inclusive, and uncomparable (None) values.
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_dict_exclusive_bounds() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // (Amsterdam, Prague) exclusive should yield Berlin and Paris.
        let result = col.find_in_range(
            Some(&Value::String("Amsterdam".into())),
            Some(&Value::String("Prague".into())),
            false,
            false,
        );
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_find_in_range_dict_open_bounds() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // No lower bound, upper inclusive = Berlin.
        let result = col.find_in_range(None, Some(&Value::String("Berlin".into())), true, true);
        assert_eq!(result, vec![0, 1]);

        // Lower inclusive = Paris, no upper bound.
        let result = col.find_in_range(Some(&Value::String("Paris".into())), None, true, true);
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn test_find_in_range_fallback_uncomparable_skips_rows() {
        // Int8Vector rows compare to None against any Value, so the fallback
        // filters them out when any bound is supplied.
        let data = vec![1i8, 2, 3];
        let col = ColumnCodec::int8_vector(data, 3);
        let min = Value::Int64(0);
        let max = Value::Int64(10);

        // min alone: the None (Uncomparable) branch returns false.
        let result = col.find_in_range(Some(&min), None, true, true);
        assert!(result.is_empty());

        // max alone: same story for the max arm.
        let result = col.find_in_range(None, Some(&max), true, true);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Serialization round-trips for each codec variant
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_to_read_from_bitpacked_round_trip() {
        let values = vec![0u64, 5, 10, 15, 3, 7];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_dict_round_trip() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Vincent", "Jules", "Vincent", "Mia"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_bitmap_round_trip() {
        let bools = vec![true, false, true, true, false, false, true];
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_int8_vector_round_trip() {
        // 3 vectors of dimension 4, mixing positive and negative values.
        let data = vec![1i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
        let col = ColumnCodec::int8_vector(data, 4);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get_int8_vector(i), col.get_int8_vector(i));
        }
    }

    // -----------------------------------------------------------------------
    // read_from error paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_from_truncated_discriminant() {
        let data: &[u8] = &[];
        let mut pos = 0;
        let err = ColumnCodec::read_from(data, &mut pos).unwrap_err();
        assert_eq!(err, "truncated codec discriminant");
    }

    #[test]
    fn test_read_from_unknown_discriminant() {
        let data: &[u8] = &[42];
        let mut pos = 0;
        let err = ColumnCodec::read_from(data, &mut pos).unwrap_err();
        assert_eq!(err, "unknown codec discriminant");
    }

    #[test]
    fn test_read_from_truncated_bits_per_value() {
        // Discriminant 0 (BitPacked) with no following byte for bits_per_value.
        let data: &[u8] = &[0];
        let mut pos = 0;
        let err = ColumnCodec::read_from(data, &mut pos).unwrap_err();
        assert_eq!(err, "truncated bits_per_value");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_word() {
        // Discriminant 0 + bits_per_value=4 + count=1 + data_len=1, then
        // truncated 8-byte word.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8, 0, 0]); // only 3 bytes of the 8-byte word
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u64");
    }

    #[test]
    fn test_read_from_dict_truncated_string() {
        // Discriminant 1 (Dict) + dict_len=1 + slen=5, but no string bytes follow.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len = 1
        buf.extend_from_slice(&5u32.to_le_bytes()); // slen = 5
        // Only add 2 of the 5 needed bytes.
        buf.extend_from_slice(b"ab");
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated dict string");
    }

    #[test]
    fn test_read_from_dict_invalid_utf8() {
        // Discriminant 1 + dict_len=1 + slen=2 + invalid UTF-8 bytes.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len = 1
        buf.extend_from_slice(&2u32.to_le_bytes()); // slen = 2
        buf.extend_from_slice(&[0xFFu8, 0xFE]); // invalid UTF-8
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "invalid UTF-8 in dict");
    }

    #[test]
    fn test_read_from_int8_vector_truncated_data() {
        // Discriminant 3 + dimensions=2 + data_len=6, but only 3 bytes follow.
        let mut buf = vec![3u8];
        buf.extend_from_slice(&2u16.to_le_bytes()); // dimensions = 2
        buf.extend_from_slice(&6u32.to_le_bytes()); // data_len = 6
        buf.extend_from_slice(&[1u8, 2, 3]);
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated Int8Vector data");
    }

    #[test]
    fn test_read_from_int8_vector_truncated_dimensions() {
        // Discriminant 3 + only 1 byte (u16 needs 2).
        let buf = vec![3u8, 0];
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u16");
    }

    #[test]
    fn test_read_from_bitmap_truncated() {
        // Discriminant 2 + truncated u32 (bit_len).
        let buf = vec![2u8, 0, 0];
        let mut pos = 0;
        let err = ColumnCodec::read_from(&buf, &mut pos).unwrap_err();
        assert_eq!(err, "truncated u32");
    }

    // -----------------------------------------------------------------------
    // Empty-column round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_to_read_from_empty_bitpacked() {
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&[]));
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_write_to_read_from_empty_bitmap() {
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&[]));
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_write_to_read_from_empty_int8_vector() {
        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
    }

    // -----------------------------------------------------------------------
    // RawI64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_raw_i64_get_decodes_as_int64() {
        let col = ColumnCodec::raw_i64(vec![-100, 0, 42, i64::MIN, i64::MAX]);
        assert_eq!(col.len(), 5);
        assert_eq!(col.get(0), Some(Value::Int64(-100)));
        assert_eq!(col.get(1), Some(Value::Int64(0)));
        assert_eq!(col.get(2), Some(Value::Int64(42)));
        assert_eq!(col.get(3), Some(Value::Int64(i64::MIN)));
        assert_eq!(col.get(4), Some(Value::Int64(i64::MAX)));
        assert_eq!(col.get(5), None);
    }

    #[test]
    fn test_raw_i64_find_eq() {
        let col = ColumnCodec::raw_i64(vec![-50, 10, -50, 20, 0, -50]);
        assert_eq!(col.find_eq(&Value::Int64(-50)), vec![0, 2, 5]);
        assert_eq!(col.find_eq(&Value::Int64(10)), vec![1]);
        assert_eq!(col.find_eq(&Value::Int64(0)), vec![4]);
        assert_eq!(col.find_eq(&Value::Int64(999)), Vec::<usize>::new());
        // Cross-type matches are not supported: a Float64 target should
        // not match any i64 row via the fast path. Fallback may coerce,
        // but for now the equality predicate is type-strict.
        assert_eq!(col.find_eq(&Value::Float64(10.0)), Vec::<usize>::new());
    }

    #[test]
    fn test_raw_i64_find_in_range_signed_ordering() {
        // Values span the sign boundary; native i64 ordering must apply.
        let col = ColumnCodec::raw_i64(vec![-10, -5, 0, 5, 10, -100, 100]);

        // [-5, 5] inclusive
        let result = col.find_in_range(Some(&Value::Int64(-5)), Some(&Value::Int64(5)), true, true);
        assert_eq!(result, vec![1, 2, 3]);

        // (-5, 5) exclusive
        let result = col.find_in_range(
            Some(&Value::Int64(-5)),
            Some(&Value::Int64(5)),
            false,
            false,
        );
        assert_eq!(result, vec![2]);

        // < 0 (no lower bound)
        let result = col.find_in_range(None, Some(&Value::Int64(0)), false, false);
        assert_eq!(result, vec![0, 1, 5]);

        // >= 10 (no upper bound)
        let result = col.find_in_range(Some(&Value::Int64(10)), None, true, true);
        assert_eq!(result, vec![4, 6]);
    }

    #[test]
    fn test_write_to_read_from_raw_i64_round_trip() {
        let col = ColumnCodec::raw_i64(vec![-42, 0, 1, i64::MIN, i64::MAX, -1_000_000_000]);

        let mut buf = Vec::new();
        col.write_to(&mut buf);

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_empty_raw_i64() {
        let col = ColumnCodec::raw_i64(Vec::new());
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&buf, &mut pos).expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_raw_i64_heap_bytes() {
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3]);
        assert_eq!(col.heap_bytes(), 3 * std::mem::size_of::<i64>());

        let empty = ColumnCodec::raw_i64(Vec::new());
        assert_eq!(empty.heap_bytes(), 0);
    }

    // ── Phase 2a: Block API ────────────────────────────────────────────
    //
    // Phase 2a treats every column as a single block (block_count == 1).
    // Phase 2b will introduce multi-block layouts. These tests pin the
    // contract so the API is stable before the format changes.

    #[test]
    fn alix_block_count_is_one_for_every_codec() {
        // BitPacked
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3]));
        assert_eq!(bp.block_count(), 1);

        // Dict
        let mut b = DictionaryBuilder::new();
        b.add("x");
        let dict = ColumnCodec::Dict(b.build());
        assert_eq!(dict.block_count(), 1);

        // Bitmap
        let bm = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false]));
        assert_eq!(bm.block_count(), 1);

        // Int8Vector
        let iv = ColumnCodec::int8_vector(vec![1i8, 2, 3, 4], 2);
        assert_eq!(iv.block_count(), 1);

        // Float64
        let f64 = ColumnCodec::float64(vec![1.0, 2.0]);
        assert_eq!(f64.block_count(), 1);

        // Float32Vector
        let fv = ColumnCodec::float32_vector(vec![1.0f32, 2.0, 3.0, 4.0], 2);
        assert_eq!(fv.block_count(), 1);

        // RawI64
        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3]);
        assert_eq!(r64.block_count(), 1);
    }

    #[test]
    fn gus_block_at_zero_carries_full_row_count() {
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3, 4, 5]));
        let entry = bp.block_at(0).expect("block 0 exists");
        assert_eq!(entry.row_count, 5);

        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3, 4]);
        let entry = r64.block_at(0).expect("block 0 exists");
        assert_eq!(entry.row_count, 4);
    }

    #[test]
    fn vincent_empty_column_has_one_zero_row_block() {
        // Phase 2a convention: empty columns still report block_count == 1
        // so downstream serializers can treat block emission uniformly.
        let empty = ColumnCodec::raw_i64(Vec::new());
        assert_eq!(empty.block_count(), 1);
        let entry = empty.block_at(0).expect("zero-row block at 0");
        assert_eq!(entry.row_count, 0);
    }

    #[test]
    fn jules_block_at_out_of_bounds_returns_none() {
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3]));
        assert!(bp.block_at(1).is_none());
        assert!(bp.block_at(usize::MAX).is_none());
    }

    #[test]
    fn mia_block_iter_yields_block_count_entries() {
        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3, 4]);
        let entries: Vec<_> = r64.block_iter().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_count, 4);
    }

    #[test]
    fn butch_block_row_counts_sum_to_column_len() {
        // Phase 2a: trivially true (one block, row_count == len).
        // Phase 2b: this contract is the migration test — multi-block
        // serializers must preserve total row count.
        for col in [
            ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3, 4, 5, 6, 7])),
            ColumnCodec::float64(vec![1.0, 2.0, 3.0]),
            ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]),
        ] {
            let total: u32 = col.block_iter().map(|b| b.row_count).sum();
            assert_eq!(total as usize, col.len());
        }
    }

    // ── Phase 2b: v2 multi-block on-disk format ───────────────────────
    //
    // v2 layout per column (after the section header):
    //   [disc:u8][global_params...]
    //   [block_count:u32]
    //   [block_index: block_count x (byte_offset:u32, byte_len:u32, row_count:u32)]
    //   [block_data: concatenated per-block bodies]
    //
    // The runtime API (block_count(), block_at(), len(), get(...)) is
    // unchanged after a round-trip — only the on-disk layout differs.

    fn assert_round_trip_v2_equals(col: &ColumnCodec) {
        let mut buf = Vec::new();
        col.write_to_v2(&mut buf);
        let mut pos = 0;
        let recovered = ColumnCodec::read_from_v2(&buf, &mut pos).expect("v2 round-trip");
        assert_eq!(pos, buf.len(), "v2 reader should consume entire buffer");
        assert_eq!(recovered.len(), col.len(), "len after v2 round-trip");
        for i in 0..col.len() {
            assert_eq!(
                recovered.get(i),
                col.get(i),
                "value at row {i} after v2 round-trip"
            );
        }
    }

    #[test]
    fn django_v2_round_trip_raw_i64_single_block() {
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]);
        assert_eq!(col.block_count(), 1);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_raw_i64_multi_block() {
        // 2049 rows: 1024 + 1024 + 1 → 3 blocks at default block size.
        // reason: i is bounded by 2049 which fits in i64.
        #[allow(clippy::cast_possible_wrap)]
        let values: Vec<i64> = (0..2049i64).map(|i| i - 1024).collect();
        let col = ColumnCodec::raw_i64(values);
        assert_eq!(col.block_count(), 3);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_bitpacked_multi_block() {
        let values: Vec<u64> = (0..2500u64).map(|i| i % 16).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));
        assert!(col.block_count() >= 2, "expect multi-block at 2500 rows");
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_dict_multi_block() {
        let mut b = DictionaryBuilder::new();
        for i in 0..1500u32 {
            b.add(if i % 3 == 0 {
                "alpha"
            } else if i % 3 == 1 {
                "beta"
            } else {
                "gamma"
            });
        }
        let col = ColumnCodec::Dict(b.build());
        assert_eq!(col.block_count(), 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_bitmap_multi_block() {
        let bools: Vec<bool> = (0..1100u32).map(|i| i % 2 == 0).collect();
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_float64_multi_block() {
        let vals: Vec<f64> = (0..1100u32).map(|i| f64::from(i) * 0.5).collect();
        let col = ColumnCodec::float64(vals);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_int8_vector_multi_block() {
        // 1100 vectors of dim 4 = 4400 i8 entries.
        // reason: known small integer literals
        #[allow(clippy::cast_possible_wrap)]
        let data: Vec<i8> = (0..4400u32).map(|i| (i % 200) as i8).collect();
        let col = ColumnCodec::int8_vector(data, 4);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_float32_vector_multi_block() {
        // 1100 vectors of dim 4 = 4400 f32 entries.
        let data: Vec<f32> = (0..4400u32).map(|i| i as f32 * 0.5).collect();
        let col = ColumnCodec::float32_vector(data, 4);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn shosanna_v2_round_trip_empty_column() {
        // Empty columns should still serialize/deserialize cleanly with v2.
        let col = ColumnCodec::raw_i64(Vec::new());
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn hans_v1_and_v2_produce_different_bytes() {
        // Sanity: the v1 (flat) and v2 (block-indexed) layouts differ
        // even for a small single-block column. This pins the format
        // separation so a future "did we forget to switch?" regression
        // is caught.
        let col = ColumnCodec::raw_i64(vec![1, 2, 3, 4, 5]);
        let mut v1 = Vec::new();
        col.write_to(&mut v1);
        let mut v2 = Vec::new();
        col.write_to_v2(&mut v2);
        assert_ne!(v1, v2, "v1 and v2 layouts must differ");
    }

    #[test]
    fn beatrix_v1_round_trip_still_works() {
        // v1 (flat) reader/writer must keep working for one release as
        // the section reader's compat path. We round-trip via the
        // unchanged write_to / read_from pair.
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]);
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&buf, &mut pos).expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }

    // ── Phase 3a: Bytes-backed fixed-width codecs ─────────────────────

    #[test]
    fn test_raw_i64_constructor_round_trip() {
        let values = vec![-100i64, -1, 0, 1, 100, i64::MIN, i64::MAX];
        let col = ColumnCodec::raw_i64(values.clone());

        assert_eq!(col.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(col.get(i), Some(Value::Int64(expected)));
        }
        assert_eq!(col.get(values.len()), None);
    }

    #[test]
    fn test_float64_constructor_round_trip() {
        let values = vec![-1.5_f64, 0.0, 100.25, f64::MIN, f64::MAX];
        let col = ColumnCodec::float64(values.clone());

        assert_eq!(col.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(col.get(i), Some(Value::Float64(expected)));
        }
    }

    #[test]
    fn test_int8_vector_constructor_round_trip() {
        let col = ColumnCodec::int8_vector(vec![1i8, 2, 3, -4, -5, -6], 3);

        assert_eq!(col.len(), 2);
        let v0 = col.get(0).unwrap();
        let expected0: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert_eq!(v0, Value::List(Arc::from(expected0)));

        // get_int8_vector slice access
        assert_eq!(col.get_int8_vector(0), Some(&[1i8, 2, 3][..]));
        assert_eq!(col.get_int8_vector(1), Some(&[-4i8, -5, -6][..]));
    }

    #[test]
    fn test_float32_vector_constructor_round_trip() {
        let col = ColumnCodec::float32_vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3);

        assert_eq!(col.len(), 2);
        match col.get(0) {
            Some(Value::Vector(v)) => {
                assert_eq!(&*v, &[1.0_f32, 2.0, 3.0]);
            }
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn test_bytes_backed_zero_copy_clone() {
        // Cloning a ColumnCodec should be cheap: the underlying Bytes
        // is refcounted, not deep-copied. Verify by building a large
        // RawI64 column and asserting clone is fast and shares storage
        // (smoke test via len equivalence; full reference-counting is
        // tested by `bytes` itself).
        let big: Vec<i64> = (0..10_000).collect();
        let col = ColumnCodec::raw_i64(big);
        let cloned = col.clone();
        assert_eq!(col.len(), cloned.len());
        for i in (0..10_000).step_by(1024) {
            assert_eq!(col.get(i), cloned.get(i));
        }
    }

    #[test]
    fn test_raw_i64_v1_round_trip_with_bytes_storage() {
        let col = ColumnCodec::raw_i64(vec![-7, 0, 7, 42]);
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&buf, &mut pos).expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }

    #[test]
    fn test_float64_v1_round_trip_with_bytes_storage() {
        let col = ColumnCodec::float64(vec![-2.5, 0.0, 1.0, std::f64::consts::PI]);
        let mut buf = Vec::new();
        col.write_to(&mut buf);
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&buf, &mut pos).expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }
}
