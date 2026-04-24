# Contributing to Grafeo

Thanks for wanting to help out! Here's what you need to know.

## Setup

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --workspace
```

You'll need **Rust 1.91.1+** and optionally **Python 3.12+** / **Node.js 20+** for the bindings.

## Branching

We use feature branches off `main`:

- `feature/<description>` for new functionality
- `fix/<description>` for bug fixes
- `release/<version>` for release stabilization

Create your branch from `main`, open a PR back to `main` when ready.

## Making Changes

1. Create a branch: `git checkout -b feature/my-thing`
2. Write code and tests
3. Run checks: `./scripts/ci-local.sh` (or `.\scripts\ci-local.ps1` on Windows)
4. Push and open a PR

You can also run checks individually:

```bash
cargo fmt --all              # Format
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo test --all-features --workspace     # Test
```

### Commit Messages

We use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`.

## Architecture

| Crate | What it does |
| ----- | ------------ |
| `grafeo` | Top-level facade, re-exports public API |
| `grafeo-common` | Foundation types, memory, utilities |
| `grafeo-core` | Graph storage, indexes, execution |
| `grafeo-adapters` | Query parsers (GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ) |
| `grafeo-engine` | Database facade, sessions, transactions |
| `grafeo-cli` | CLI with interactive shell, query execution, import/export, backup, WAL management |
| `grafeo-bindings-common` | Shared library for all language bindings |
| `grafeo-python` | Python bindings (PyO3) |
| `grafeo-node` | Node.js/TypeScript bindings (napi-rs) |
| `grafeo-c` | C FFI layer (also used by Go via CGO) |
| `grafeo-wasm` | WebAssembly bindings (wasm-bindgen) |
| `grafeo-csharp` | C# / .NET 8 bindings (P/Invoke, wraps grafeo-c) |
| `grafeo-dart` | Dart bindings (dart:ffi, wraps grafeo-c) |

## Spec Tests (gtests)

Declarative, cross-language integration tests live in `tests/spec/` as `.gtest` files:

```text
tests/spec/
├── lpg/           # Labeled Property Graph tests
│   ├── gql/       # GQL (ISO 39075)
│   ├── cypher/    # openCypher
│   ├── gremlin/   # TinkerPop Gremlin
│   ├── graphql/   # GraphQL over LPG
│   └── sql_pgq/   # SQL/PGQ (SQL:2023)
├── rdf/           # RDF model tests
│   ├── sparql/    # SPARQL 1.1
│   └── graphql/   # GraphQL over RDF
├── common/        # Language-agnostic tests
├── datasets/      # Shared test fixtures (.setup files)
├── regression/    # Issue-mapped regression tests
└── rosetta/       # Cross-language equivalence tests
```

Each `.gtest` file is YAML-like with a `meta:` header and `tests:` list. A build script generates Rust `#[test]` functions at compile time. Run them with:

```bash
cargo test -p grafeo-spec-tests                          # All spec tests
cargo test -p grafeo-spec-tests -- gremlin               # Filter by keyword
cargo test -p grafeo-spec-tests -- rdf_sparql             # SPARQL tests only
```

### Writing a spec test

```yaml
meta:
  language: gql
  model: lpg
  section: "my-feature"
  title: My Feature Tests
  dataset: social_network

tests:
  - name: basic_query
    query: MATCH (p:Person) RETURN p.name
    expect:
      rows:
        - [Mia]
        - [Jules]
        - [Vincent]
```

Tests can use `skip: "reason"` to mark known gaps, `expect: { count: N }` for row count checks, `expect: { ordered: true }` for order-sensitive assertions, and `expect: { error: "substring" }` for expected errors.

### Persistent variants for index-dependent cases

Any `.gtest` file or case that declares `requires: [text-index]` or `requires: [vector-index]` automatically gets a second generated test, suffixed `_persistent`, that opens `GrafeoDB::open(tempdir)` instead of `GrafeoDB::new_in_memory()`. This exercises the WAL-wrapped read path that every on-disk session uses, so regressions in wrapper-layer delegation fail here instead of slipping past the in-memory suite.

Contributors don't need to do anything special: write the test once, name it as you would any other case, and the harness emits both variants. If a test case passes in-memory but fails `_persistent`, the bug is almost certainly in a store wrapper (WAL, CDC, Layered) missing a delegation.

## Code Style

- Standard Rust conventions: `rustfmt` and `clippy` are enforced in CI
- Use `thiserror` for error types
- Tests go in the same file under `#[cfg(test)]`
- Descriptive test names: `test_<function>_<scenario>`

## Python Bindings

```bash
cd crates/bindings/python
maturin develop
pytest tests/ -v --ignore=tests/benchmark_phases.py
```

## Node.js Bindings

```bash
cd crates/bindings/node
npm install
npm run build
npm test
```

## Ecosystem Projects

These companion projects live in separate repositories under the [GrafeoDB](https://github.com/GrafeoDB) organization:

| Project | Description |
| ------- | ----------- |
| [grafeo-server](https://github.com/GrafeoDB/grafeo-server) | HTTP server & web UI |
| [grafeo-web](https://github.com/GrafeoDB/grafeo-web) | Browser-based Grafeo (WASM) |
| [gwp](https://github.com/GrafeoDB/gql-wire-protocol) | GQL Wire Protocol (gRPC) |
| [boltr](https://github.com/GrafeoDB/boltr) | Bolt v5.x Wire Protocol |
| [grafeo-memory](https://github.com/GrafeoDB/grafeo-memory) | AI memory layer for LLM applications |
| [grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain) | LangChain graph + vector store |
| [grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex) | LlamaIndex PropertyGraphStore |
| [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp) | MCP server for LLM agents |
| [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph) | Graph visualization widget |
| [anywidget-vector](https://github.com/GrafeoDB/anywidget-vector) | Vector visualization widget |
| [graph-bench](https://github.com/GrafeoDB/graph-bench) | Benchmark suite |
| [ann-benchmarks](https://github.com/GrafeoDB/ann-benchmarks) | Vector search benchmarking |

## Benchmarks and Performance Regressions

PRs opened from this repository are benchmarked on
[CodSpeed](https://codspeed.io/), which runs the Criterion microbenchmarks
under Callgrind for <1% variance. Results post as a PR comment with a diff
vs `main`. PRs from forks are skipped: the CodSpeed token isn't exposed to
fork workflows; after an initial review a maintainer can push the branch to
this repo to trigger a run.

The following suites are tracked:

- `grafeo-core/benches/index_bench.rs`: adjacency, HashIndex, HNSW
  insert/search, distance kernels, quantisation, CompactStore point queries
- `grafeo-common/benches/arena_bench.rs`: epoch arena, bump allocator,
  object pool
- `grafeo-storage/benches/wal_bench.rs`: WAL write throughput, recovery
  replay
- `grafeo-engine/benches/query_bench.rs`: end-to-end GQL + SPARQL
- `grafeo-engine/benches/serialization_bench.rs`: snapshot + Value codecs
- `grafeo-engine/benches/regression_bench.rs`: multi-hop, repeated-parse,
  edge-type filter
- `grafeo-engine/benches/memory_bench.rs`: memory footprint snapshot

Reproduce locally:

```bash
# Pin matches .github/workflows/codspeed.yml; bump both in lock-step.
cargo install cargo-codspeed --version 4.5.0
cargo codspeed build --package grafeo-core \
    --features "vector-index compact-store" --bench index_bench
cargo codspeed run --package grafeo-core
```

If a PR flags a >5% regression on a hot path, include a brief explanation in
the PR description. The trade-off is often acceptable (e.g. a correctness
fix that costs some throughput), but the maintainer should know it was
deliberate rather than accidental.

Adding a new Criterion bench: import `use criterion::{...}` as usual. The
workspace renames `criterion` to the `codspeed-criterion-compat` package
(see root `Cargo.toml`), so there is nothing to change per crate: add the
`[[bench]]` entry and any features in the owning crate's `Cargo.toml`, and
add the suite to `.github/workflows/codspeed.yml` so it lands in the PR
comment. The adapter is a drop-in replacement: plain `cargo bench` continues
to work unchanged.

## Pre-commit Hooks (Optional)

```bash
cargo install prek
prek install
```

This runs format, lint and license checks automatically before each commit.

## Links

- [Repository](https://github.com/GrafeoDB/grafeo)
- [Issues](https://github.com/GrafeoDB/grafeo/issues)
- [Documentation](https://grafeo.dev)

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
