//! Shared formatting utilities for CLI output.

use grafeo_engine::memory_usage::MemoryUsage;

/// Prints a hierarchical memory breakdown to stdout.
///
/// Mirrors the structure of `db.memory_usage()`: total on top, then one block
/// per component (store, indexes, MVCC, caches, string pool, buffer manager,
/// RDF, CDC). Zero-valued or feature-disabled components are skipped so
/// default embedded builds stay terse.
pub fn format_memory(usage: &MemoryUsage) {
    println!("Total:           {}", format_bytes(usage.total_bytes));

    if usage.store.total_bytes > 0 {
        println!();
        println!("Store:           {}", format_bytes(usage.store.total_bytes));
        if usage.store.nodes_bytes > 0 {
            println!("  nodes:         {}", format_bytes(usage.store.nodes_bytes));
        }
        if usage.store.edges_bytes > 0 {
            println!("  edges:         {}", format_bytes(usage.store.edges_bytes));
        }
        if usage.store.node_properties_bytes > 0 {
            println!(
                "  node props:    {}",
                format_bytes(usage.store.node_properties_bytes)
            );
        }
        if usage.store.edge_properties_bytes > 0 {
            println!(
                "  edge props:    {}",
                format_bytes(usage.store.edge_properties_bytes)
            );
        }
    }

    if usage.indexes.total_bytes > 0 {
        println!();
        println!(
            "Indexes:         {}",
            format_bytes(usage.indexes.total_bytes)
        );
        for idx in &usage.indexes.vector_indexes {
            println!(
                "  vector[{}]: {} ({} items)",
                idx.name,
                format_bytes(idx.bytes),
                idx.item_count
            );
        }
        for idx in &usage.indexes.text_indexes {
            println!(
                "  text[{}]: {} ({} items)",
                idx.name,
                format_bytes(idx.bytes),
                idx.item_count
            );
        }
    }

    if usage.mvcc.total_bytes > 0 {
        println!();
        println!("MVCC:            {}", format_bytes(usage.mvcc.total_bytes));
        if usage.mvcc.average_chain_depth > 0.0 {
            println!("  avg chain:     {:.2}", usage.mvcc.average_chain_depth);
        }
    }

    if usage.caches.total_bytes > 0 || usage.caches.cached_plan_count > 0 {
        println!();
        println!(
            "Caches:          {} ({} plans)",
            format_bytes(usage.caches.total_bytes),
            usage.caches.cached_plan_count
        );
    }

    if usage.string_pool.total_bytes > 0 {
        println!();
        println!(
            "String pool:     {}",
            format_bytes(usage.string_pool.total_bytes)
        );
    }

    if usage.buffer_manager.allocated_bytes > 0 || usage.buffer_manager.budget_bytes > 0 {
        println!();
        println!(
            "Buffer manager:  {} / {} budget",
            format_bytes(usage.buffer_manager.allocated_bytes),
            format_bytes(usage.buffer_manager.budget_bytes)
        );
    }

    if !usage.rdf.is_empty() {
        println!();
        println!(
            "RDF:             {} ({} triples, {} named graphs)",
            format_bytes(usage.rdf.total_bytes),
            usage.rdf.triple_count,
            usage.rdf.named_graph_count
        );
        if usage.rdf.term_dictionary_bytes > 0 {
            println!(
                "  term dict:     {}",
                format_bytes(usage.rdf.term_dictionary_bytes)
            );
        }
        if usage.rdf.ring_index_bytes > 0 {
            println!(
                "  ring index:    {}",
                format_bytes(usage.rdf.ring_index_bytes)
            );
        }
    }

    if !usage.cdc.is_empty() {
        println!();
        println!(
            "CDC:             {} ({} events, {} entities)",
            format_bytes(usage.cdc.total_bytes),
            usage.cdc.event_count,
            usage.cdc.entity_count
        );
    }
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Format a duration in milliseconds as a human-readable string.
pub fn format_duration_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.1}ms", ms)
    } else {
        format!("{:.0}us", ms * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(1), "1 bytes");
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(10240), "10.00 KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.00 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 100), "100.00 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 2), "2.00 GB");
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.50 GB"
        );
    }

    #[test]
    fn test_format_duration_microseconds() {
        assert_eq!(format_duration_ms(0.5), "500us");
        assert_eq!(format_duration_ms(0.001), "1us");
    }

    #[test]
    fn test_format_duration_milliseconds() {
        assert_eq!(format_duration_ms(1.0), "1.0ms");
        assert_eq!(format_duration_ms(42.5), "42.5ms");
        assert_eq!(format_duration_ms(999.9), "999.9ms");
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration_ms(1000.0), "1.00s");
        assert_eq!(format_duration_ms(2500.0), "2.50s");
    }
}
