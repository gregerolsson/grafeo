package grafeo

import (
	"strings"
	"testing"
)

func seedPeople(t *testing.T) *Database {
	t.Helper()
	db, err := OpenInMemory()
	if err != nil {
		t.Fatalf("OpenInMemory: %v", err)
	}
	for _, stmt := range []string{
		"INSERT (:Person {name: 'Alix', age: 32})",
		"INSERT (:Person {name: 'Gus', age: 28})",
		"INSERT (:Person {name: 'Vincent', age: 45})",
		"INSERT (:Person {name: 'Jules', age: 40})",
		"INSERT (:Person {name: 'Mia', age: 24})",
	} {
		if _, err := db.Execute(stmt); err != nil {
			t.Fatalf("seed %q: %v", stmt, err)
		}
	}
	return db
}

func TestExecuteStream_MatchesExecute(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	const q = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"

	mat, err := db.Execute(q)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	stream, err := db.ExecuteStream(q)
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()
	rows, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(rows) != len(mat.Rows) {
		t.Fatalf("row count mismatch: stream=%d, materialized=%d", len(rows), len(mat.Rows))
	}
}

func TestExecuteStream_ExposesColumns(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream(
		"MATCH (p:Person) RETURN p.name AS name, p.age AS age",
	)
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()

	cols := stream.Columns()
	if len(cols) != 2 || cols[0] != "name" || cols[1] != "age" {
		t.Fatalf("unexpected columns: %v", cols)
	}
}

func TestExecuteStream_YieldsRowMaps(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()

	count := 0
	for {
		row, err := stream.Next()
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if row == nil {
			break
		}
		if _, ok := row["p.name"]; !ok {
			t.Fatalf("row missing p.name: %v", row)
		}
		count++
	}
	if count != 5 {
		t.Fatalf("expected 5 rows, got %d", count)
	}
}

func TestExecuteStream_NextReturnsNilAfterExhaustion(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()

	for {
		row, _ := stream.Next()
		if row == nil {
			break
		}
	}
	row, err := stream.Next()
	if err != nil {
		t.Fatalf("Next after exhaustion: unexpected error %v", err)
	}
	if row != nil {
		t.Fatalf("Next after exhaustion should return nil, got %v", row)
	}
}

func TestExecuteStream_HonorsFilter(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream(
		"MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name",
	)
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()

	rows, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("expected 3 rows with age > 30, got %d", len(rows))
	}
	want := map[string]bool{"Alix": true, "Vincent": true, "Jules": true}
	for _, row := range rows {
		name, _ := row["name"].(string)
		if !want[name] {
			t.Fatalf("unexpected name %q", name)
		}
	}
}

func TestExecuteStream_EmptyResult(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream(
		"MATCH (p:Person) WHERE p.age > 999 RETURN p.name",
	)
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	defer stream.Close()

	rows, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("expected empty result, got %d rows", len(rows))
	}
}

func TestExecuteStream_CloseShortCircuits(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	stream, err := db.ExecuteStream("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatalf("ExecuteStream: %v", err)
	}
	first, err := stream.Next()
	if err != nil || first == nil {
		t.Fatalf("first Next unexpected: err=%v row=%v", err, first)
	}
	stream.Close()
	if _, err := stream.Next(); err != ErrStreamClosed {
		t.Fatalf("Next after Close should return ErrStreamClosed, got %v", err)
	}
}

func TestExecuteStream_RejectsMutations(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	_, err := db.ExecuteStream("INSERT (:Person {name: 'Butch'})")
	if err == nil {
		t.Fatal("expected error for mutation; got nil")
	}
}

func TestExecuteStream_RejectsOrderBy(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	_, err := db.ExecuteStream("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
	if err == nil {
		t.Fatal("expected error for ORDER BY; got nil")
	}
}

func TestExecuteStream_RejectsSessionCommand(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatalf("OpenInMemory: %v", err)
	}
	defer db.Close()

	_, err = db.ExecuteStream("SESSION SET GRAPH analytics")
	if err == nil {
		t.Fatal("expected error for session command; got nil")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "session") {
		t.Fatalf("expected session error message, got %v", err)
	}
}

func TestExecuteStream_RejectsExplain(t *testing.T) {
	db := seedPeople(t)
	defer db.Close()

	_, err := db.ExecuteStream("EXPLAIN MATCH (p:Person) RETURN p.name")
	if err == nil {
		t.Fatal("expected error for EXPLAIN; got nil")
	}
}
