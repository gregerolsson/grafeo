using System.Text.Json;
using Xunit;

namespace Grafeo.Tests;

/// <summary>Tests for GrafeoDB.ExecuteStream cursor-based results.</summary>
public sealed class StreamingTests : IDisposable
{
    private readonly GrafeoDB _db = GrafeoDB.Memory();

    public StreamingTests()
    {
        _db.Execute("INSERT (:Person {name: 'Alix', age: 32})");
        _db.Execute("INSERT (:Person {name: 'Gus', age: 28})");
        _db.Execute("INSERT (:Person {name: 'Vincent', age: 45})");
        _db.Execute("INSERT (:Person {name: 'Jules', age: 40})");
        _db.Execute("INSERT (:Person {name: 'Mia', age: 24})");
    }

    public void Dispose() => _db.Dispose();

    [Fact]
    public void StreamsSameRowsAsExecute()
    {
        const string query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age";
        var materialized = _db.Execute(query);
        using var stream = _db.ExecuteStream(query);
        var streamed = stream.ToList();

        Assert.Equal(materialized.Rows.Count, streamed.Count);
        var matNames = materialized.Rows
            .Select(r => JsonElementToString(r["name"]))
            .ToHashSet();
        var strNames = streamed.Select(r => r["name"]?.ToString() ?? "").ToHashSet();
        Assert.Equal(matNames, strNames);
    }

    [Fact]
    public void ExposesColumnsBeforeIteration()
    {
        using var stream = _db.ExecuteStream(
            "MATCH (p:Person) RETURN p.name AS name, p.age AS age");
        Assert.Equal(new[] { "name", "age" }, stream.Columns);
    }

    [Fact]
    public void YieldsDictionaryPerRow()
    {
        using var stream = _db.ExecuteStream("MATCH (p:Person) RETURN p.name");
        var rows = stream.ToList();
        Assert.Equal(5, rows.Count);
        Assert.All(rows, r => Assert.True(r.ContainsKey("p.name")));
    }

    [Fact]
    public void ReturnsNullAfterExhaustion()
    {
        using var stream = _db.ExecuteStream("MATCH (p:Person) RETURN p.name");
        while (stream.Next() is not null) { /* drain */ }
        Assert.Null(stream.Next());
    }

    [Fact]
    public void HonorsWhereFilter()
    {
        using var stream = _db.ExecuteStream(
            "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name");
        var names = stream.ToList()
            .Select(r => r["name"]?.ToString() ?? "")
            .ToHashSet();
        Assert.Equal(new[] { "Alix", "Vincent", "Jules" }.ToHashSet(), names);
    }

    [Fact]
    public void EmptyResultYieldsNoRows()
    {
        using var stream = _db.ExecuteStream(
            "MATCH (p:Person) WHERE p.age > 999 RETURN p.name");
        Assert.Empty(stream.ToList());
    }

    [Fact]
    public void DisposeShortCircuitsIteration()
    {
        var stream = _db.ExecuteStream("MATCH (p:Person) RETURN p.name");
        Assert.NotNull(stream.Next());
        stream.Dispose();
        Assert.Throws<ObjectDisposedException>(() => stream.Next());
    }

    [Fact]
    public void RejectsMutations()
    {
        Assert.ThrowsAny<GrafeoException>(() =>
            _db.ExecuteStream("INSERT (:Person {name: 'Butch'})"));
    }

    [Fact]
    public void RejectsOrderBy()
    {
        Assert.ThrowsAny<GrafeoException>(() =>
            _db.ExecuteStream("MATCH (p:Person) RETURN p.name AS n ORDER BY n"));
    }

    [Fact]
    public void RejectsSessionCommands()
    {
        Assert.ThrowsAny<GrafeoException>(() =>
            _db.ExecuteStream("SESSION SET GRAPH analytics"));
    }

    [Fact]
    public void RejectsExplain()
    {
        Assert.ThrowsAny<GrafeoException>(() =>
            _db.ExecuteStream("EXPLAIN MATCH (p:Person) RETURN p.name"));
    }

    [Fact]
    public void SupportsForEachOverRows()
    {
        using var stream = _db.ExecuteStream("MATCH (p:Person) RETURN p.name");
        int count = 0;
        foreach (var _ in stream.Rows()) count++;
        Assert.Equal(5, count);
    }

    private static string JsonElementToString(object? value)
    {
        if (value is JsonElement je)
        {
            return je.ValueKind switch
            {
                JsonValueKind.String => je.GetString() ?? "",
                _ => je.ToString()
            };
        }
        return value?.ToString() ?? "";
    }
}
