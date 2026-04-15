//! WAL write throughput and recovery replay benchmarks.
//!
//! These benchmarks cover the hot paths in the storage crate:
//! - Single-record write (NoSync, Batch, Sync)
//! - Batch write throughput (1K records in a committed transaction)
//! - Recovery replay of a pre-populated WAL

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::{NodeId, TransactionId, Value};
use grafeo_storage::wal::{DurabilityMode, WalConfig, WalManager, WalRecord, WalRecovery};

/// Creates a representative WAL record for benchmarking.
fn make_record(i: u64) -> WalRecord {
    WalRecord::SetNodeProperty {
        id: NodeId::new(i),
        key: "name".to_string(),
        value: Value::String(format!("node_{i}").into()),
    }
}

fn bench_wal_write_nosync(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let config = WalConfig {
        durability: DurabilityMode::NoSync,
        ..WalConfig::default()
    };
    let wal = WalManager::with_config(dir.path(), config).unwrap();

    let record = make_record(1);

    c.bench_function("wal_write_nosync", |b| {
        b.iter(|| {
            wal.log(black_box(&record)).unwrap();
        });
    });
}

fn bench_wal_write_batch(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let config = WalConfig {
        durability: DurabilityMode::Batch {
            max_delay_ms: 100,
            max_records: 1000,
        },
        ..WalConfig::default()
    };
    let wal = WalManager::with_config(dir.path(), config).unwrap();

    let record = make_record(1);

    c.bench_function("wal_write_batch", |b| {
        b.iter(|| {
            wal.log(black_box(&record)).unwrap();
        });
    });
}

fn bench_wal_write_sync(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let config = WalConfig {
        durability: DurabilityMode::Sync,
        ..WalConfig::default()
    };
    let wal = WalManager::with_config(dir.path(), config).unwrap();

    let record = make_record(1);

    c.bench_function("wal_write_sync", |b| {
        b.iter(|| {
            wal.log(black_box(&record)).unwrap();
        });
    });
}

fn bench_wal_batch_commit(c: &mut Criterion) {
    let config = WalConfig {
        durability: DurabilityMode::NoSync,
        ..WalConfig::default()
    };

    c.bench_function("wal_batch_commit_1000", |b| {
        b.iter(|| {
            // Fresh directory per iteration so the log never grows across runs
            let dir = tempfile::tempdir().unwrap();
            let wal = WalManager::with_config(dir.path(), config.clone()).unwrap();

            for i in 0..1000u64 {
                wal.log(&make_record(i)).unwrap();
            }

            wal.log(black_box(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            }))
            .unwrap();
        });
    });
}

fn bench_wal_recovery_replay(c: &mut Criterion) {
    // Pre-populate a WAL with 10K committed records
    let dir = tempfile::tempdir().unwrap();
    let config = WalConfig {
        durability: DurabilityMode::NoSync,
        ..WalConfig::default()
    };
    let wal = WalManager::with_config(dir.path(), config).unwrap();

    for i in 0..10_000u64 {
        wal.log(&make_record(i)).unwrap();
    }
    wal.log(&WalRecord::TransactionCommit {
        transaction_id: TransactionId::new(1),
    })
    .unwrap();
    wal.sync().unwrap();

    // Keep dir alive but drop the wal so files are closed
    let wal_dir = dir.path().to_path_buf();
    drop(wal);

    c.bench_function("wal_recovery_replay_10k", |b| {
        b.iter(|| {
            let recovery = WalRecovery::new(&wal_dir);
            let records = recovery.recover().unwrap();
            black_box(records.len());
        });
    });
}

criterion_group!(
    benches,
    bench_wal_write_nosync,
    bench_wal_write_batch,
    bench_wal_write_sync,
    bench_wal_batch_commit,
    bench_wal_recovery_replay,
);
criterion_main!(benches);
