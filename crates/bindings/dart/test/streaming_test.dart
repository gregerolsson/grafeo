import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory();
    db.execute("INSERT (:Person {name: 'Alix', age: 32})");
    db.execute("INSERT (:Person {name: 'Gus', age: 28})");
    db.execute("INSERT (:Person {name: 'Vincent', age: 45})");
    db.execute("INSERT (:Person {name: 'Jules', age: 40})");
    db.execute("INSERT (:Person {name: 'Mia', age: 24})");
  });

  tearDown(() {
    db.close();
  });

  group('executeStream', () {
    test('returns same rows as execute()', () {
      const query = 'MATCH (p:Person) RETURN p.name AS name, p.age AS age';
      final materialized = db.execute(query).rows;
      final stream = db.executeStream(query);
      try {
        final streamed = stream.toList();
        expect(streamed.length, equals(materialized.length));
        expect(
          streamed.map((r) => r['name']).toSet(),
          equals(materialized.map((r) => r['name']).toSet()),
        );
      } finally {
        stream.close();
      }
    });

    test('exposes columns before iteration', () {
      final stream = db.executeStream(
        'MATCH (p:Person) RETURN p.name AS name, p.age AS age',
      );
      try {
        expect(stream.columns, equals(['name', 'age']));
      } finally {
        stream.close();
      }
    });

    test('yields maps keyed by column name', () {
      final stream = db.executeStream('MATCH (p:Person) RETURN p.name');
      try {
        final rows = stream.toList();
        expect(rows.length, equals(5));
        for (final row in rows) {
          expect(row, contains('p.name'));
        }
      } finally {
        stream.close();
      }
    });

    test('returns null after exhaustion', () {
      final stream = db.executeStream('MATCH (p:Person) RETURN p.name');
      try {
        while (stream.next() != null) {
          // drain
        }
        expect(stream.next(), isNull);
      } finally {
        stream.close();
      }
    });

    test('respects WHERE filters', () {
      final stream =
          db.executeStream('MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name');
      try {
        final names = stream.toList().map((r) => r['name'] as String).toSet();
        expect(names, equals({'Alix', 'Vincent', 'Jules'}));
      } finally {
        stream.close();
      }
    });

    test('empty result yields no rows', () {
      final stream = db.executeStream(
        'MATCH (p:Person) WHERE p.age > 999 RETURN p.name',
      );
      try {
        expect(stream.toList(), isEmpty);
      } finally {
        stream.close();
      }
    });

    test('close() short-circuits iteration', () {
      final stream = db.executeStream('MATCH (p:Person) RETURN p.name');
      expect(stream.next(), isNotNull);
      stream.close();
      expect(() => stream.next(), throwsA(isA<DatabaseException>()));
    });

    test('rejects mutations', () {
      expect(
        () => db.executeStream("INSERT (:Person {name: 'Butch'})"),
        throwsA(isA<DatabaseException>()),
      );
    });

    test('rejects ORDER BY (push-based pipeline)', () {
      expect(
        () => db.executeStream('MATCH (p:Person) RETURN p.name AS n ORDER BY n'),
        throwsA(isA<DatabaseException>()),
      );
    });

    test('rejects session commands', () {
      expect(
        () => db.executeStream('SESSION SET GRAPH analytics'),
        throwsA(isA<DatabaseException>()),
      );
    });

    test('rejects EXPLAIN', () {
      expect(
        () => db.executeStream('EXPLAIN MATCH (p:Person) RETURN p.name'),
        throwsA(isA<DatabaseException>()),
      );
    });

    test('supports for-in over rows()', () {
      final stream = db.executeStream('MATCH (p:Person) RETURN p.name');
      try {
        var count = 0;
        for (final _ in stream.rows()) {
          count++;
        }
        expect(count, equals(5));
      } finally {
        stream.close();
      }
    });
  });
}
