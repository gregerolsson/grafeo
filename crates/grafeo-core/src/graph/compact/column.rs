//! Column codecs for CompactStore.
//!
//! Wraps Grafeo's existing storage primitives into a unified enum with
//! random access and `Value` decoding. CompactStore owns these types:
//! the underlying primitives are not modified.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::Value;

use crate::codec::{BitPackedInts, BitVector, DictionaryEncoding};

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
    BitPacked(BitPackedInts),
    /// Dictionary-encoded strings.
    Dict(DictionaryEncoding),
    /// Null/boolean bitmap.
    Bitmap(BitVector),
    /// Int8 quantized vectors (flat array with stride).
    Int8Vector {
        /// Flat array of int8 components.
        data: Vec<i8>,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// Native IEEE 754 double-precision floats.
    Float64(Vec<f64>),
    /// Float32 vectors (flat array with stride), for embedding / vector search.
    Float32Vector {
        /// Flat array of f32 components.
        data: Vec<f32>,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
}

impl ColumnCodec {
    /// Decodes the value at `index` into a [`Value`].
    ///
    /// - [`BitPacked`](Self::BitPacked) → `Value::Int64(v as i64)`
    /// - [`Dict`](Self::Dict) → `Value::String(ArcStr::from(s))`
    /// - [`Bitmap`](Self::Bitmap) → `Value::Bool(b)`
    /// - [`Int8Vector`](Self::Int8Vector) → `Value::List(...)` of `Int64` values
    /// - [`Float64`](Self::Float64) → `Value::Float64(f)`
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
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > data.len() {
                    return None;
                }
                let values: Vec<Value> = data[start..end]
                    .iter()
                    .map(|&v| Value::Int64(v as i64))
                    .collect();
                Some(Value::List(Arc::from(values)))
            }
            Self::Float64(vec) => vec.get(index).copied().map(Value::Float64),
            Self::Float32Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > data.len() {
                    return None;
                }
                Some(Value::Vector(Arc::from(&data[start..end])))
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
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > data.len() {
                    return None;
                }
                Some(&data[start..end])
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
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                data.len().checked_div(dims).unwrap_or(0)
            }
            Self::Float64(vec) => vec.len(),
            Self::Float32Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                data.len().checked_div(dims).unwrap_or(0)
            }
        }
    }

    /// Returns `true` if the column contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            (Self::Float64(vec), &Value::Float64(target)) => vec
                .iter()
                .enumerate()
                .filter(|&(_, v)| *v == target)
                .map(|(i, _)| i)
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
            Self::Int8Vector { data, dimensions } => {
                buf.push(3); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                write_usize_as_u32(buf, data.len());
                for &v in data {
                    buf.push(v.to_le_bytes()[0]);
                }
            }
            Self::Float64(vec) => {
                buf.push(4); // discriminant
                write_usize_as_u32(buf, vec.len());
                for &v in vec {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            Self::Float32Vector { data, dimensions } => {
                buf.push(5); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                write_usize_as_u32(buf, data.len());
                for &v in data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
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
                let bytes = &data[*pos..*pos + data_len];
                // Safe: u8 and i8 have the same layout.
                let i8_data: Vec<i8> = bytes.iter().map(|&b| i8::from_le_bytes([b])).collect();
                *pos += data_len;
                Ok(Self::Int8Vector {
                    data: i8_data,
                    dimensions,
                })
            }
            4 => {
                // Float64
                let count = read_u32_le(data, pos)? as usize;
                let mut vec = Vec::with_capacity(count);
                for _ in 0..count {
                    vec.push(read_f64_le(data, pos)?);
                }
                Ok(Self::Float64(vec))
            }
            5 => {
                // Float32Vector
                let dimensions = read_u16_le(data, pos)?;
                let data_len = read_u32_le(data, pos)? as usize;
                let byte_need = data_len
                    .checked_mul(4)
                    .ok_or("Float32Vector length overflow")?;
                if *pos + byte_need > data.len() {
                    return Err("truncated Float32Vector data");
                }
                let mut f32_data = Vec::with_capacity(data_len);
                for _ in 0..data_len {
                    f32_data.push(read_f32_le(data, pos)?);
                }
                Ok(Self::Float32Vector {
                    data: f32_data,
                    dimensions,
                })
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
            Self::Int8Vector { data, .. } => data.len(),
            Self::Float64(vec) => vec.len() * std::mem::size_of::<f64>(),
            Self::Float32Vector { data, .. } => data.len() * std::mem::size_of::<f32>(),
        }
    }
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

fn read_f64_le(data: &[u8], pos: &mut usize) -> Result<f64, &'static str> {
    if *pos + 8 > data.len() {
        return Err("truncated f64");
    }
    let v = f64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_f32_le(data: &[u8], pos: &mut usize) -> Result<f32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated f32");
    }
    let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };

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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };

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

        let vec_col = ColumnCodec::Int8Vector {
            data: vec![1, 2],
            dimensions: 2,
        };
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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };
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
        let col = ColumnCodec::Int8Vector {
            data: vec![1, 2, 3],
            dimensions: 0,
        };
        // Zero dimensions: get() should return None.
        assert_eq!(col.get(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_get_int8_vector() {
        let col = ColumnCodec::Int8Vector {
            data: vec![1, 2, 3],
            dimensions: 0,
        };
        // Zero dimensions: get_int8_vector() should return None.
        assert_eq!(col.get_int8_vector(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_len_and_is_empty() {
        let col = ColumnCodec::Int8Vector {
            data: vec![1, 2, 3],
            dimensions: 0,
        };
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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };
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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };

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
        let col = ColumnCodec::Int8Vector {
            data: vec![1, 2, 3],
            dimensions: 3,
        };
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
        let col = ColumnCodec::Int8Vector {
            data: data.clone(),
            dimensions: dims,
        };
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
        let col = ColumnCodec::Int8Vector {
            data: vec![1i8, 2, 3, 4, 5, 6],
            dimensions: 3,
        };
        assert_eq!(col.len(), 2);
        assert!(col.get(2).is_none());
        assert!(col.get(5).is_none());
        assert!(col.get_int8_vector(2).is_none());
        assert!(col.get_int8_vector(5).is_none());

        // Zero-dim column: len == 0 by construction and no element is accessible.
        let zero = ColumnCodec::Int8Vector {
            data: Vec::new(),
            dimensions: 0,
        };
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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 4,
        };

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
        let col = ColumnCodec::Int8Vector {
            data: Vec::new(),
            dimensions: 4,
        };
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

        assert_eq!(
            ColumnCodec::Int8Vector {
                data: vec![1i8],
                dimensions: 1
            }
            .get_raw_u64(0),
            None
        );
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

        let col = ColumnCodec::Int8Vector {
            data: Vec::new(),
            dimensions: 4,
        };
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
}
