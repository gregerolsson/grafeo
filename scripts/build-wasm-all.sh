#!/usr/bin/env bash
# Build all WASM variants for grafeo-web.
#
# Produces four binaries:
#   pkg/       - Full variant (all query languages + AI search) for main export
#   pkg-lite/  - Browser variant (GQL only) for /lite export
#   pkg-lpg/   - Full LPG (all query languages + AI search)
#   pkg-rdf/   - RDF variant (GQL + SPARQL/RDF)
#
# Usage:
#   ./scripts/build-wasm-all.sh

set -euo pipefail

WASM_DIR="crates/bindings/wasm"

echo "=== Building WASM full variant (main export) ==="
./scripts/build-wasm.sh --features full --name "@grafeo-db/wasm"

echo ""
echo "=== Building WASM lite variant (/lite export) ==="
./scripts/build-wasm.sh --out-dir "$WASM_DIR/pkg-lite" --name "@grafeo-db/wasm-lite"

echo ""
echo "=== Building WASM LPG variant (all LPG languages + AI) ==="
./scripts/build-wasm.sh --features lpg --out-dir "$WASM_DIR/pkg-lpg" --name "@grafeo-db/wasm-lpg"

echo ""
echo "=== Building WASM RDF variant (GQL + SPARQL/RDF) ==="
./scripts/build-wasm.sh --features rdf --out-dir "$WASM_DIR/pkg-rdf" --name "@grafeo-db/wasm-rdf"

echo ""
echo "All variants built successfully."
echo "  Full variant: $WASM_DIR/pkg/       (used by @grafeo-db/web)"
echo "  Lite variant: $WASM_DIR/pkg-lite/  (used by @grafeo-db/web/lite)"
echo "  LPG variant:  $WASM_DIR/pkg-lpg/   (used by @grafeo-db/web/lpg)"
echo "  RDF variant:  $WASM_DIR/pkg-rdf/   (used by @grafeo-db/web/rdf)"
