"""Encoding, optional dependency isolation, and bounded output conversion."""

from itertools import islice
from pathlib import Path
import random
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from maximal_class_graph import (
    BitmaskGraphs,
    convert_bitmask_graphs,
    graph_to_maximal_class,
    list_graphs_in_maximal_class,
)


@pytest.mark.parametrize("size", range(11))
def test_each_bit_has_fixed_row_major_edge_position(size):
    nodes = [f"v{i}" for i in range(size)]
    positions = [(u, v) for u in range(size) for v in range(size) if u != v]
    for bit, (u, v) in enumerate(positions):
        matrix = convert_bitmask_graphs([1 << bit], nodes=nodes)[0]
        expected = np.eye(size, dtype=bool)
        expected[u, v] = True
        assert np.array_equal(matrix, expected)
        graph = convert_bitmask_graphs([1 << bit], nodes=nodes, output_format="digraph")[0]
        assert list(graph) == nodes
        assert set(graph.edges) == {(node, node) for node in nodes} | {(nodes[u], nodes[v])}
    complete = (1 << len(positions)) - 1
    assert np.array_equal(convert_bitmask_graphs([complete], nodes=nodes)[0], np.ones((size, size), dtype=bool))


@pytest.mark.parametrize("size", range(5, 11))
def test_larger_graphs_against_independent_networkx_sccs(size):
    rng = random.Random(size)
    nodes = list(range(size))
    possible = [(u, v) for u in nodes for v in nodes if u != v]
    for trial in range(30):
        probability = (0.1, 0.35, 0.7)[trial % 3]
        mask = sum(1 << k for k in range(len(possible)) if rng.random() < probability)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edge for k, edge in enumerate(possible) if mask & (1 << k))
        condensed = nx.condensation(graph)
        expected = frozenset(
            frozenset(node for c in nx.descendants(condensed, root) | {root} for node in condensed.nodes[c]["members"])
            for root, degree in condensed.in_degree() if degree == 0
        )
        actual = graph_to_maximal_class(mask, input_format="bitmask", nodes=nodes)
        assert frozenset(actual) == expected


def test_core_runs_without_any_third_party_packages():
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from maximal_class_graph import graph_to_maximal_class, list_graphs_in_maximal_class, convert_bitmask_graphs
family = (frozenset({1, 3}), frozenset({2, 3}))
assert graph_to_maximal_class([(1, 3), (2, 3)]) == family
assert graph_to_maximal_class([[0, 0, 1], [0, 0, 1], [0, 0, 0]], input_format='matrix', nodes=[1, 2, 3]) == family
result = list_graphs_in_maximal_class(family)
assert result.nodes == (1, 2, 3) and result.masks == [10]
assert graph_to_maximal_class(10, input_format='bitmask', nodes=result.nodes) == family
assert 'networkx' not in sys.modules and 'numpy' not in sys.modules
for output, extra in [('matrix', '[matrix]'), ('digraph', '[digraph]')]:
    try:
        convert_bitmask_graphs(result, output_format=output, as_iterator=True)
    except ImportError as exc:
        assert extra in str(exc)
    else:
        raise AssertionError('Missing dependency should be reported immediately')
"""
    subprocess.run([sys.executable, "-B", "-S", "-c", code, src], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("output", ["matrix", "digraph"])
def test_all_subset_and_single_graph_conversion(output):
    result = list_graphs_in_maximal_class([{1, 2, 3}])
    all_graphs = convert_bitmask_graphs(result, output_format=output)
    selected = convert_bitmask_graphs(result.masks[:2], nodes=result.nodes, output_format=output)
    single = convert_bitmask_graphs([result.masks[1]], nodes=result.nodes, output_format=output)
    assert len(all_graphs) == len(result.masks)
    assert len(selected) == 2 and len(single) == 1
    if output == "matrix":
        assert all(graph.dtype == np.bool_ for graph in all_graphs)
        assert np.array_equal(selected[1], single[0])
        selected[0][:] = False
        assert selected[1].diagonal().all()
    else:
        assert set(selected[1].edges) == set(single[0].edges)
        selected[0].clear()
        assert len(selected[1]) == 3


@pytest.mark.parametrize("output", ["matrix", "digraph"])
def test_conversion_limit_prevents_output_allocation(output, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Output allocated before conversion-limit check")
    monkeypatch.setattr(np, "eye", unexpected)
    monkeypatch.setattr(nx, "DiGraph", unexpected)
    with pytest.raises(ValueError, match="storage"):
        convert_bitmask_graphs([0] * 1001, nodes=[1], output_format=output)
    consumed = []
    def masks():
        for i in range(100):
            consumed.append(i)
            yield 0
    with pytest.raises(ValueError, match="max_graphs=2"):
        convert_bitmask_graphs(masks(), nodes=[1], output_format=output, max_graphs=2)
    assert consumed == [0, 1, 2]


@pytest.mark.parametrize("limit", [2, None])
def test_explicit_conversion_limit_override(limit):
    converted = convert_bitmask_graphs([0, 0], nodes=[1], max_graphs=limit)
    assert len(converted) == 2


@pytest.mark.parametrize("output", ["matrix", "digraph"])
def test_streaming_conversion_is_lazy_and_not_cumulatively_capped(output):
    consumed = []
    def masks():
        for i in range(4):
            consumed.append(i)
            yield 0
    stream = convert_bitmask_graphs(masks(), nodes=[1], output_format=output, max_graphs=1, as_iterator=True)
    assert iter(stream) is stream
    assert consumed == []
    next(stream)
    assert consumed == [0]
    assert len(list(stream)) == 3


def test_streamed_enumeration_can_be_converted_in_parts():
    result = list_graphs_in_maximal_class([{1, 2, 3}], as_iterator=True)
    first = convert_bitmask_graphs(islice(result.masks, 2), nodes=result.nodes)
    assert len(first) == 2
    remaining = convert_bitmask_graphs(result)
    assert len(first) + len(remaining) == len(list_graphs_in_maximal_class([{1, 2, 3}]).masks)


@pytest.mark.parametrize("mask", [-1, 64, True, False, 1.5, "1", None])
def test_invalid_masks_in_forward_and_conversion(mask):
    with pytest.raises(ValueError):
        graph_to_maximal_class(mask, input_format="bitmask", nodes=[1, 2, 3])
    with pytest.raises(ValueError):
        convert_bitmask_graphs([mask], nodes=[1, 2, 3])
    stream = convert_bitmask_graphs([mask], nodes=[1, 2, 3], as_iterator=True)
    with pytest.raises(ValueError):
        next(stream)


def test_all_eager_masks_are_validated_before_allocation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Output allocated before all masks were validated")
    monkeypatch.setattr(np, "eye", unexpected)
    with pytest.raises(ValueError):
        convert_bitmask_graphs([0, -1], nodes=[1])


def test_integer_protocol_and_more_than_64_bits():
    matrix = convert_bitmask_graphs([np.uint64(1)], nodes=["a", "b"])[0]
    assert np.array_equal(matrix, [[True, True], [False, True]])
    high = 1 << 89
    graph = convert_bitmask_graphs([high], nodes=range(10), output_format="digraph")[0]
    assert (9, 8) in graph.edges and graph.number_of_edges() == 11
    assert graph_to_maximal_class(high, input_format="bitmask", nodes=range(10)) == graph_to_maximal_class([(9, 8)], nodes=range(10))


@pytest.mark.parametrize("nodes", [[], [1]])
def test_zero_and_one_vertex_masks_have_no_nonloop_bits(nodes):
    with pytest.raises(ValueError):
        convert_bitmask_graphs([1], nodes=nodes)
    matrix = convert_bitmask_graphs([0], nodes=nodes)[0]
    assert np.array_equal(matrix, np.eye(len(nodes), dtype=bool))


@pytest.mark.parametrize("kwargs", [
    {"output_format": "edges"}, {"as_iterator": "yes"},
    {"max_graphs": 0}, {"max_graphs": -1}, {"max_graphs": True},
    {"max_graphs": "2"}, {"max_graphs": 1.5},
    {"nodes": [1, 1]}, {"nodes": [None]}, {"nodes": [[1]]},
])
def test_invalid_conversion_options(kwargs):
    options = {"nodes": [1]}
    options.update(kwargs)
    with pytest.raises(ValueError):
        convert_bitmask_graphs([0], **options)


def test_missing_or_conflicting_metadata_and_invalid_mask_collections():
    with pytest.raises(ValueError, match="nodes is required"):
        convert_bitmask_graphs([0])
    with pytest.raises(ValueError, match="already supplied"):
        convert_bitmask_graphs(BitmaskGraphs((1,), [0]), nodes=[1])
    with pytest.raises(ValueError, match="nodes is required"):
        graph_to_maximal_class(0, input_format="bitmask")
    for invalid in (1, "123", b"123", None):
        with pytest.raises(ValueError, match="iterable"):
            convert_bitmask_graphs(invalid, nodes=[1])
