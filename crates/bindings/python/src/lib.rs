//! Use Grafeo from Python with native Rust performance.
//!
//! You get full access to the graph database through a Pythonic API - same
//! query speed, same durability, with the convenience of Python's ecosystem.
//!
//! ## Quick Start
//!
//! ```python
//! from grafeo import GrafeoDB
//!
//! # Create an in-memory database (or pass a path for persistence)
//! db = GrafeoDB()
//!
//! # Create some people
//! db.execute("INSERT (:Person {name: 'Alix', role: 'Engineer'})")
//! db.execute("INSERT (:Person {name: 'Gus', role: 'Manager'})")
//! db.execute("""
//!     MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
//!     INSERT (a)-[:REPORTS_TO]->(b)
//! """)
//!
//! # Query the graph
//! result = db.execute("MATCH (p:Person)-[:REPORTS_TO]->(m) RETURN p.name, m.name")
//! for row in result:
//!     print(f"{row['p.name']} reports to {row['m.name']}")
//! ```
//!
//! ## Data Science Integration
//!
//! | Library | How to use | Best for |
//! | ------- | ---------- | -------- |
//! | pandas | `result.to_pandas()` or `db.nodes_df()` | Tabular operations |
//! | polars | `result.to_polars()` | Fast columnar analytics |
//! | NetworkX | `db.as_networkx().to_networkx()` | Graph visualization, analysis |
//! | solvOR | `db.as_solvor()` | Operations research algorithms |

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pyo3::prelude::*;

mod bridges;
mod database;
mod error;
mod graph;
mod quantization;
mod query;
mod stream;
mod types;

#[cfg(feature = "algos")]
use bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
use database::{AsyncQueryResult, AsyncQueryResultIter, PyGrafeoDB, PyIsolationLevel};
use graph::{PyEdge, PyNode};
use query::PyQueryResult;
use stream::PyResultStream;
use types::PyValue;

/// Returns the active SIMD instruction set for vector operations.
///
/// Useful for debugging and verifying that SIMD acceleration is being used.
///
/// Returns one of: "avx2", "sse", "neon", or "scalar"
///
/// Example:
///     import grafeo
///     print(f"SIMD support: {grafeo.simd_support()}")  # e.g., "avx2"
#[pyfunction]
fn simd_support() -> &'static str {
    grafeo_core::index::vector::simd_support()
}

/// Grafeo Python module.
#[pymodule]
fn grafeo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrafeoDB>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyResultStream>()?;
    m.add_class::<AsyncQueryResult>()?;
    m.add_class::<AsyncQueryResultIter>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyIsolationLevel>()?;
    #[cfg(feature = "algos")]
    {
        m.add_class::<PyAlgorithms>()?;
        m.add_class::<PyNetworkXAdapter>()?;
        m.add_class::<PySolvORAdapter>()?;
    }

    // Register quantization types
    quantization::register(m)?;

    // Add module-level functions
    m.add_function(wrap_pyfunction!(simd_support, m)?)?;
    m.add_function(wrap_pyfunction!(types::vector, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
