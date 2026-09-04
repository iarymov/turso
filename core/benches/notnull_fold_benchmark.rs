//! Microbenchmark for the NOT-NULL constant-fold optimization (tursodatabase/turso#4921):
//! constant-folds `col IS [NOT] NULL` / `COUNT(col)` at plan time when `col` is provably
//! non-null. `COUNT` reuses the O(1) btree-count fast path; `IS NULL` folds to an
//! impossible condition; `IS NOT NULL` folds to an unconditional scan.
//!
//! Run:  cargo bench -p turso_core --bench notnull_fold_benchmark

#[cfg(feature = "codspeed")]
use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
#[cfg(not(feature = "codspeed"))]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use turso_core::{Database, PlatformIO, SqliteDialect, StepResult};

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const N: usize = 1_000_000;

fn drain_turso(db: &Database, stmt: &mut turso_core::Statement) {
    loop {
        match stmt.step().unwrap() {
            StepResult::Row => {
                black_box(stmt.row());
            }
            StepResult::IO | StepResult::Yield | StepResult::Sleep { .. } => {
                db.io.step().unwrap();
            }
            StepResult::Done => break,
            StepResult::Interrupt | StepResult::Busy => unreachable!(),
        }
    }
    stmt.reset().unwrap();
}

fn open_conn(path: &std::path::Path) -> (Arc<Database>, Arc<turso_core::Connection>) {
    #[allow(clippy::arc_with_non_send_sync)]
    let io = Arc::new(PlatformIO::new().unwrap());
    let db = Database::open_file(io, path.to_str().unwrap(), Arc::new(SqliteDialect)).unwrap();
    let conn = db.connect().unwrap();
    {
        let mut p = conn.prepare("PRAGMA cache_size=-65536").unwrap();
        while !matches!(p.step().unwrap(), StepResult::Done) {
            db.io.step().unwrap();
        }
    }
    (db, conn)
}

/// All rows populated (never NULL) -- the point of this table is that `notnull_text`/
/// `notnull_blob` carry a real `NOT NULL` schema constraint, making them eligible for
/// the constant-fold.
fn seed_notnull_db(n: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("notnull_fold.db");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=DELETE;
         CREATE TABLE t(
             id INTEGER PRIMARY KEY,
             g TEXT,
             notnull_text TEXT NOT NULL,
             notnull_blob BLOB NOT NULL
         );",
    )
    .unwrap();
    let tx = conn.unchecked_transaction().unwrap();
    {
        let mut ins = tx
            .prepare("INSERT INTO t(id, g, notnull_text, notnull_blob) VALUES (?1, ?2, ?3, ?4)")
            .unwrap();
        for i in 1..=n as i64 {
            let g = format!("group-{:02}", i % 16);
            let text = format!("row-text-payload-{i}");
            let blob = vec![(i % 251) as u8; 64];
            ins.execute((i, &g, text, blob)).unwrap();
        }
    }
    tx.commit().unwrap();
    dir
}

const NOTNULL_QUERY_SHAPES: &[(&str, &str)] = &[
    ("count_col", "SELECT COUNT({col}) FROM t"),
    ("where_is_null", "SELECT id FROM t WHERE {col} IS NULL"),
    (
        "where_is_not_null",
        "SELECT id FROM t WHERE {col} IS NOT NULL",
    ),
];

const NOTNULL_COLUMNS: &[&str] = &["notnull_text", "notnull_blob"];

// Set BENCH_SCALE_SWEEP=1 to additionally run at 10K and 100M rows, on top of the
// default 1M. Gated behind an env var so a plain `cargo bench` invocation stays at the
// original single-scale runtime.
fn notnull_fold_scales() -> Vec<usize> {
    if std::env::var("BENCH_SCALE_SWEEP").as_deref() == Ok("1") {
        vec![10_000, 1_000_000, 100_000_000]
    } else {
        vec![N]
    }
}

#[turso_macros::codspeed_criterion_benchmark]
fn bench_notnull_fold(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("notnull_fold");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for n in notnull_fold_scales() {
        let dir = seed_notnull_db(n);
        let path = dir.path().join("notnull_fold.db");

        for col in NOTNULL_COLUMNS {
            let (db, conn) = open_conn(&path);
            for (shape_label, sql_template) in NOTNULL_QUERY_SHAPES {
                let sql = sql_template.replace("{col}", col);
                let bench_id = format!("{shape_label}/{col}");
                group.bench_with_input(BenchmarkId::new(bench_id, n), &n, |b, _| {
                    let mut stmt = conn.prepare(&sql).unwrap();
                    b.iter(|| drain_turso(&db, &mut stmt));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_notnull_fold);
criterion_main!(benches);
