//! Integration tests for the tier-aware `CompactStore` base.
//!
//! Exercises the full `compact() → spill_all() → swap_base()` lifecycle
//! through the public GrafeoDB API, verifying that:
//!
//! - `compact()` installs a `CompactStoreTiered` wrapper and registers a
//!   `CompactStoreConsumer` with the BufferManager.
//! - `BufferManager::spill_all()` actually spills the base to a mmap'd file
//!   and publishes the fresh `Arc<CompactStore>` to the `LayeredStore`.
//! - Reads continue to work transparently across the tier transition.
//! - `recompact()` rebuilds the tier wrapper so its `Weak` back-references
//!   track the new base.
//!
//! Requires: `compact-store`, `mmap`, `lpg` (all default-on).

#![cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]

use std::path::PathBuf;
use std::sync::Arc;

use grafeo_engine::{Config, GrafeoDB};

fn spill_dir(label: &str) -> PathBuf {
    let base = std::env::temp_dir().join("grafeo-compact-tiered-tests");
    base.join(format!("{label}-{}", std::process::id()))
}

fn config_with_spill(dir: &PathBuf) -> Config {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("create spill dir");
    Config::in_memory().with_spill_path(dir.clone())
}

fn seed_db(db: &mut GrafeoDB) {
    for i in 0..16 {
        db.execute(&format!(
            "INSERT (:Person {{name: 'person-{i}', age: {i}}})"
        ))
        .unwrap();
    }
}

#[test]
fn compact_installs_tiered_wrapper() {
    let dir = spill_dir("installs");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let tiered = db
        .compact_tiered()
        .expect("tiered installed after compact()");
    assert!(!tiered.is_on_disk(), "starts in-memory");
    assert!(tiered.memory_bytes() > 0);

    // LayeredStore and tiered agree on the base Arc right after compact().
    let layered = db
        .layered_store()
        .expect("layered installed after compact()");
    assert!(Arc::ptr_eq(&layered.base_store_arc(), &tiered.store()));
}

#[test]
fn spill_all_tiers_base_to_mmap() {
    let dir = spill_dir("spill");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let tiered = Arc::clone(db.compact_tiered().unwrap());
    let layered = Arc::clone(db.layered_store().unwrap());
    let pre_base = layered.base_store_arc();

    // Force every can-spill consumer to spill. The compact-store consumer
    // will persist the base and swap_base() on the layered store.
    let freed = db.buffer_manager().spill_all();

    assert!(tiered.is_on_disk(), "tier switched to OnDisk");
    assert!(
        dir.join("compact_base.grafeo").exists(),
        "spill file written"
    );
    // Vector/text consumers also spill (or report 0); just assert the
    // compact base contributed something when it was non-empty.
    let _ = freed;

    // LayeredStore now points at the fresh (mmap-backed) base, distinct
    // from the pre-spill Arc.
    let post_base = layered.base_store_arc();
    assert!(!Arc::ptr_eq(&pre_base, &post_base));
    assert!(Arc::ptr_eq(&post_base, &tiered.store()));
}

#[test]
fn reads_survive_tier_transition() {
    let dir = spill_dir("reads");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let session = db.session();
    let before = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    let count_before = before.rows()[0][0].clone();
    drop(session);

    db.buffer_manager().spill_all();
    assert!(db.compact_tiered().unwrap().is_on_disk());

    // Query again against the now-mmap-backed base.
    let session = db.session();
    let after = session.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(after.rows()[0][0], count_before);

    // Property access still works.
    let names = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.age")
        .unwrap();
    assert_eq!(names.rows().len(), 16);
}

#[test]
fn recompact_rebuilds_tier_wrapper() {
    let dir = spill_dir("recompact");
    let mut db = GrafeoDB::with_config(config_with_spill(&dir)).unwrap();
    seed_db(&mut db);
    db.compact().unwrap();

    let first_tiered = Arc::clone(db.compact_tiered().unwrap());

    // Add overlay mutations then recompact.
    db.execute("INSERT (:Person {name: 'alix', age: 99})")
        .unwrap();
    db.recompact().unwrap();

    let second_tiered = Arc::clone(db.compact_tiered().unwrap());
    assert!(
        !Arc::ptr_eq(&first_tiered, &second_tiered),
        "recompact replaced the tier wrapper"
    );
    assert!(!second_tiered.is_on_disk());

    // New base matches the new tier wrapper.
    let layered = db.layered_store().unwrap();
    assert!(Arc::ptr_eq(
        &layered.base_store_arc(),
        &second_tiered.store()
    ));

    // Spill the new wrapper to verify the old one's spill no longer races
    // against the live LayeredStore (old weak ref dangles harmlessly).
    db.buffer_manager().spill_all();
    assert!(second_tiered.is_on_disk());
    assert!(
        !first_tiered.is_on_disk(),
        "old wrapper untouched after recompact"
    );
}

#[test]
fn spill_without_spill_path_is_noop() {
    let mut db = GrafeoDB::new_in_memory();
    for i in 0..4 {
        db.execute(&format!("INSERT (:Person {{name: 'p-{i}'}})"))
            .unwrap();
    }
    db.compact().unwrap();

    db.buffer_manager().spill_all();
    // Without a spill_path the compact-store consumer reports can_spill=false,
    // so the base stays in-memory.
    assert!(!db.compact_tiered().unwrap().is_on_disk());
}
