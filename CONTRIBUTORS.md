# Join the GrafeoDB Community

We're building a modern graph database ecosystem in Rust, and we'd love your help.

## Why Contribute?

**Learn by doing**: Work with cutting-edge tech including Rust, WebAssembly, Arrow/Polars vectorization, MVCC transactions and multiple query language parsers.

**Shape the project**: We're early-stage and open to ideas. Your contributions can influence the direction of the entire ecosystem.

**Build your portfolio**: Graph databases are in demand. Contributing here gives you real experience with database internals, query optimization and systems programming.

## Ways to Get Involved

### Code Contributions

| Area | Skills | Projects |
|------|--------|----------|
| **Core Database** | Rust, database internals | [grafeo](https://github.com/GrafeoDB/grafeo) |
| **Query Languages** | Parsing, compilers | GQL, Cypher, SPARQL, Gremlin, GraphQL support |
| **Graph Algorithms** | Algorithms, math | PageRank, community detection, centrality |
| **Python Bindings** | Rust + Python, PyO3 | [grafeo](https://github.com/GrafeoDB/grafeo) |
| **Browser Runtime** | TypeScript, WebAssembly | [grafeo-web](https://github.com/GrafeoDB/grafeo-web) |
| **Visualization** | Three.js, Sigma.js | [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph), [anywidget-vector](https://github.com/GrafeoDB/anywidget-vector) |
| **Server** | Rust, Axum, REST APIs | [grafeo-server](https://github.com/GrafeoDB/grafeo-server) |
| **Benchmarking** | Python, data analysis | [graph-bench](https://github.com/GrafeoDB/graph-bench) |

### Non-Code Contributions

- **Documentation**: Tutorials, examples, API docs
- **Testing**: Bug reports, edge cases, stress testing
- **Design**: Logo, diagrams, UI/UX for widgets
- **Community**: Answer questions, write blog posts, give talks

## Getting Started

1. **Pick a project** that matches your interests
2. **Read the README** and try running it locally
3. **Browse issues** labeled `good first issue` or `help wanted`
4. **Ask questions** by opening a discussion or issue

### Good First Issues

Look for issues tagged with:
- `good first issue`: Beginner-friendly tasks
- `help wanted`: We'd appreciate help here
- `documentation`: Docs improvements needed
- `testing`: Test coverage improvements

## Development Setup

Most projects use similar tooling:

```bash
# Rust projects
cargo build --workspace
cargo test --workspace

# Python projects
uv sync
uv run pytest
```

See each project's CONTRIBUTING.md for specific instructions.

## Our Stack

| Layer | Technology |
|-------|------------|
| Core | Rust (custom columnar storage, MVCC) |
| Python | PyO3, maturin |
| Server | Axum, Tower, Docker |
| Browser | WebAssembly, IndexedDB |
| Visualization | Three.js, Sigma.js, anywidget |
| Build | Cargo, uv, hatch |
| CI | GitHub Actions |

## Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

We aim to respond within a few days. Be patient with us, and we'll be patient with you.

## Contributors

Thank you to everyone who has contributed to Grafeo!

- **CorvusYe** ([@CorvusYe](https://github.com/CorvusYe)): Dart bindings ([#138](https://github.com/GrafeoDB/grafeo/pull/138)), single-file `.grafeo` format feature request ([#139](https://github.com/GrafeoDB/grafeo/issues/139))
- **temporaryfix** ([@temporaryfix](https://github.com/temporaryfix)): CompactStore columnar read-optimized store, RFC ([#199](https://github.com/GrafeoDB/grafeo/issues/199)), implementation ([#204](https://github.com/GrafeoDB/grafeo/pull/204)); Float32Vector codec and post-compact index fix ([#286](https://github.com/GrafeoDB/grafeo/pull/286)); unified hybrid queries proposa, BM25 `TextScanOperator`, planner pushdown for vector + text predicates, top-K rewrite, cost model ([#287](https://github.com/GrafeoDB/grafeo/pull/287))
- **Imaclean74** ([@Imaclean74](https://github.com/Imaclean74)): CompactStore multi-label-pair edge type fix ([#225](https://github.com/GrafeoDB/grafeo/pull/225))
- **Michaelzag** ([@Michaelzag](https://github.com/Michaelzag)): `cypher` feature `gql` dependency fix ([#233](https://github.com/GrafeoDB/grafeo/pull/233)), schema type extraction proposal ([#234](https://github.com/GrafeoDB/grafeo/issues/234)), Python named graph management ([#243](https://github.com/GrafeoDB/grafeo/pull/243)), Python per-transaction CDC ([#244](https://github.com/GrafeoDB/grafeo/pull/244)), graph/schema context validation ([#246](https://github.com/GrafeoDB/grafeo/pull/246)). **grafeo-server**: backup/restore endpoints with ArcSwap database handle ([server#53](https://github.com/GrafeoDB/grafeo-server/pull/53)), sync pull CDC error handling and test fixes ([server#55](https://github.com/GrafeoDB/grafeo-server/pull/55)), backup chain API adoption and database file rename ([server#60](https://github.com/GrafeoDB/grafeo-server/pull/60)), admin UI redesign with backup labels, studio auth, restore-to-epoch, cross-database restore, and graph view fix ([server#63](https://github.com/GrafeoDB/grafeo-server/pull/63))

## Recognition

Contributors are recognized in:
- Release notes
- Project documentation
- This file

## Current Maintainers

- **S.T. Grond** ([@StevenBtw](https://github.com/StevenBtw)): Architect

## License

All contributions are licensed under Apache-2.0.

---

**Ready to contribute?** Pick a repo, find an issue and send a PR. We're excited to have you.
