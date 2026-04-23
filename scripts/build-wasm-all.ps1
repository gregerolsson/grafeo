# Build all WASM variants for grafeo-web.
#
# Produces four binaries:
#   pkg\       - Full variant (all query languages + AI search) for main export
#   pkg-lite\  - Browser variant (GQL only) for /lite export
#   pkg-lpg\   - Full LPG (all query languages + AI search)
#   pkg-rdf\   - RDF variant (GQL + SPARQL/RDF)
#
# Usage:
#   .\scripts\build-wasm-all.ps1

$ErrorActionPreference = "Stop"

$WasmDir = "crates\bindings\wasm"

Write-Host "=== Building WASM full variant (main export) ==="
& .\scripts\build-wasm.ps1 -Features full -Name "@grafeo-db/wasm"

Write-Host ""
Write-Host "=== Building WASM lite variant (/lite export) ==="
& .\scripts\build-wasm.ps1 -OutDir "$WasmDir\pkg-lite" -Name "@grafeo-db/wasm-lite"

Write-Host ""
Write-Host "=== Building WASM LPG variant (all LPG languages + AI) ==="
& .\scripts\build-wasm.ps1 -Features lpg -OutDir "$WasmDir\pkg-lpg" -Name "@grafeo-db/wasm-lpg"

Write-Host ""
Write-Host "=== Building WASM RDF variant (GQL + SPARQL/RDF) ==="
& .\scripts\build-wasm.ps1 -Features rdf -OutDir "$WasmDir\pkg-rdf" -Name "@grafeo-db/wasm-rdf"

Write-Host ""
Write-Host "All variants built successfully."
Write-Host "  Full variant: $WasmDir\pkg\       (used by @grafeo-db/web)"
Write-Host "  Lite variant: $WasmDir\pkg-lite\  (used by @grafeo-db/web/lite)"
Write-Host "  LPG variant:  $WasmDir\pkg-lpg\   (used by @grafeo-db/web/lpg)"
Write-Host "  RDF variant:  $WasmDir\pkg-rdf\   (used by @grafeo-db/web/rdf)"
