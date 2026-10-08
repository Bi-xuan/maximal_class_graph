"""Check both algorithms against an independent reachability oracle."""

from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import pytest

from maximal_class_graph import graph_to_maximal_class, list_graphs_in_maximal_class


def family_key(family):
    return frozenset(frozenset(group) for group in family)


def nonloop_edges(graph):
    return frozenset((u, v) for u, v in graph.edges if u != v)


def oracle_family(nodes, edges):
    """Inclusion-maximal reachable sets via plain DFS, without SCCs/NetworkX."""
    adjacency = {node: set() for node in nodes}
    for u, v in edges:
        adjacency[u].add(v)
    reachable_sets = set()
    for source in nodes:
        seen = {source}
        pending = [source]
        while pending:
            for target in adjacency[pending.pop()]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        reachable_sets.add(frozenset(seen))
    return frozenset(group for group in reachable_sets if not any(group < other for other in reachable_sets))


def assert_loops_and_family(graph, family, nodes):
    assert isinstance(graph, nx.DiGraph)
    assert not graph.is_multigraph()
    assert set(graph) == set(nodes)
    assert set(nx.selfloop_edges(graph)) == {(node, node) for node in nodes}
    assert family_key(graph_to_maximal_class(graph.edges, nodes=nodes)) == family_key(family)


@pytest.mark.parametrize("size", range(5))
def test_exhaustive_forward_and_reverse_through_four_vertices(size):
    """All 4,166 loopless graphs on 0-4 vertices, grouped by oracle family."""
    nodes = list(range(size))
    possible = [(u, v) for u in nodes for v in nodes if u != v]
    expected = defaultdict(set)
    for bits in product((False, True), repeat=len(possible)):
        edges = frozenset(edge for edge, present in zip(possible, bits) if present)
        family = oracle_family(nodes, edges)
        expected[family].add(edges)
        assert family_key(graph_to_maximal_class(edges, nodes=nodes)) == family
    for family, edge_sets in expected.items():
        generated = list_graphs_in_maximal_class(family)
        actual = [nonloop_edges(graph) for graph in generated]
        assert len(actual) == len(set(actual)), "Duplicate graph returned"
        assert set(actual) == edge_sets
        for graph in generated:
            assert_loops_and_family(graph, family, nodes)


def test_r_example_and_overlapping_classes():
    edges = [(2, 1), (3, 1), (6, 4), (4, 5), (5, 6)]
    assert graph_to_maximal_class(edges, nodes=range(1, 7)) == (
        frozenset({1, 2}), frozenset({1, 3}), frozenset({4, 5, 6}),
    )


def test_scc_discovery_order_regression():
    # DFS discovery order is not the finishing order required by Kosaraju.
    edges = [(1, 2), (1, 3), (3, 2), (4, 3)]
    assert family_key(graph_to_maximal_class(edges)) == family_key([{1, 2, 3}, {2, 3, 4}])


@pytest.mark.parametrize("matrix", [
    [[1, 0, 1], [0, 0, 1], [0, 0, 1]],
    np.array([[1, 0, 1], [0, 0, 1], [0, 0, 1]]),
    np.array([[1, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
])
def test_matrix_equivalence_and_label_order(matrix):
    labels = ["b", "a", "c"]
    result = graph_to_maximal_class(matrix, input_format="matrix", nodes=labels)
    assert result == (frozenset({"b", "c"}), frozenset({"a", "c"}))
    assert family_key(result) == family_key(graph_to_maximal_class([("b", "c"), ("a", "c")]))


def test_default_matrix_labels_and_isolates():
    assert graph_to_maximal_class([[0, 1, 0], [0, 0, 0], [0, 0, 0]], input_format="matrix") == (
        frozenset({0, 1}), frozenset({2}),
    )
    assert graph_to_maximal_class([], nodes=[1, 2]) == (frozenset({1}), frozenset({2}))


def test_self_loops_and_repeated_edges_do_not_change_family():
    edges = [(1, 3), (2, 3)]
    assert graph_to_maximal_class(edges) == graph_to_maximal_class(edges * 2 + [(i, i) for i in range(1, 4)])


def test_empty_inputs():
    assert graph_to_maximal_class([]) == ()
    assert graph_to_maximal_class([], input_format="matrix") == ()
    assert graph_to_maximal_class(np.empty((0, 0)), input_format="matrix") == ()
    result = list_graphs_in_maximal_class(())
    assert len(result) == 1
    assert len(result[0]) == 0
    assert result[0].number_of_edges() == 0


def test_iterator_matches_list_and_yields_independent_graphs():
    family = [{1, 2, 3}]
    result = list_graphs_in_maximal_class(family)
    stream = list_graphs_in_maximal_class(family, as_iterator=True)
    assert iter(stream) is stream
    streamed = list(stream)
    assert [set(g.edges) for g in streamed] == [set(g.edges) for g in result]
    assert len({id(g) for g in streamed}) == len(streamed)
    streamed[0].clear()
    assert len(streamed[1]) == 3


def test_class_and_member_order_do_not_change_graphs():
    a = list_graphs_in_maximal_class([[1, 3], [2, 3]])
    b = list_graphs_in_maximal_class([[3, 2], [3, 1]])
    assert [set(g.edges) for g in a] == [set(g.edges) for g in b]
    assert len(a) == 1
    assert set(a[0].edges) == {(1, 1), (2, 2), (3, 3), (1, 3), (2, 3)}


def test_mixed_hashable_labels_and_generators():
    nodes = ["root", 7, ("sink", 1), "isolated"]
    edges = [("root", 7), (7, ("sink", 1))]
    family = graph_to_maximal_class((edge for edge in edges), nodes=iter(nodes))
    assert family_key(family) == family_key([nodes[:3], ["isolated"]])
    for graph in list_graphs_in_maximal_class((iter(group) for group in family)):
        assert_loops_and_family(graph, family, nodes)


def test_ten_vertex_sparse_family_is_enumerable():
    family = [{i} for i in range(10)]
    graphs = list_graphs_in_maximal_class(family)
    assert len(graphs) == 1
    assert_loops_and_family(graphs[0], family, range(10))


@pytest.mark.parametrize("stream", [False, True])
def test_search_limit_is_checked_immediately(stream):
    with pytest.raises(ValueError, match="90 allowed non-loop edges"):
        list_graphs_in_maximal_class([range(10)], as_iterator=stream)
    with pytest.raises(ValueError, match="64 candidates"):
        list_graphs_in_maximal_class([range(3)], max_candidates=63, as_iterator=stream)
    assert len(list_graphs_in_maximal_class([range(3)], max_candidates=64)) > 0
    assert len(list_graphs_in_maximal_class([range(3)], max_candidates=None)) > 0


@pytest.mark.parametrize("limit", [0, -1, 1.5, "64", True])
def test_invalid_search_limits(limit):
    with pytest.raises(ValueError, match="max_candidates"):
        list_graphs_in_maximal_class([{1}], max_candidates=limit)


@pytest.mark.parametrize("family", [
    [set()], [{1}, {1}], [{1}, {1, 2}], [{1, 2}, {2, 3}, {1, 3}],
    [1, 2], ["abc"], [[None]], [[[1]]], None,
])
def test_invalid_or_unrealizable_families(family):
    with pytest.raises(ValueError):
        list_graphs_in_maximal_class(family, as_iterator=True)


@pytest.mark.parametrize("graph, kwargs", [
    ([(1, 2)], {"input_format": "unknown"}),
    ([(1, 2, 3)], {}), ([1], {}), (["ab"], {}),
    ([(1, None)], {}), ([([1], 2)], {}),
    ([(1, 2)], {"nodes": [1]}), ([], {"nodes": [1, 1]}),
    ([[0, 1]], {"input_format": "matrix"}),
    ([[0, 1], [0]], {"input_format": "matrix"}),
    ([[2]], {"input_format": "matrix"}),
    ([[float("nan")]], {"input_format": "matrix"}),
    ([["1"]], {"input_format": "matrix"}),
    ([[[1]]], {"input_format": "matrix"}),
    ([[0]], {"input_format": "matrix", "nodes": [1, 2]}),
    (np.empty((0, 3)), {"input_format": "matrix"}),
    (np.array([0, 1]), {"input_format": "matrix"}),
    (None, {}),
])
def test_invalid_graph_inputs(graph, kwargs):
    with pytest.raises(ValueError):
        graph_to_maximal_class(graph, **kwargs)


def test_invalid_iterator_option():
    with pytest.raises(ValueError, match="as_iterator"):
        list_graphs_in_maximal_class([{1}], as_iterator="yes")
