"""Adaptations of the algorithms in Bi-xuan/maximal_class/Algorithms.Rmd."""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterable, Iterator
from typing import Optional, Union

import networkx as nx

_Family = tuple[frozenset[Hashable], ...]
_DEFAULT_MAX_CANDIDATES = 1 << 15


def _materialize(values: Iterable, name: str) -> list:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable, not a string.")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable.") from exc


def _validate_node(node: Hashable) -> None:
    if node is None:
        raise ValueError("None is not a supported vertex label.")
    try:
        hash(node)
    except TypeError as exc:
        raise ValueError("Vertex labels must be hashable.") from exc


def _unique_nodes(values: Iterable[Hashable]) -> list[Hashable]:
    unique = {}
    for node in values:
        _validate_node(node)
        unique[node] = None
    return list(unique)


def _ordered_nodes(values: Iterable[Hashable]) -> list[Hashable]:
    nodes = _unique_nodes(values)
    try:
        return sorted(nodes)
    except TypeError:
        # Mixed or custom labels need not have an ordering relation.
        return nodes


def _explicit_nodes(values: Iterable[Hashable]) -> list[Hashable]:
    nodes = _materialize(values, "nodes")
    unique = _unique_nodes(nodes)
    if len(unique) != len(nodes):
        raise ValueError("nodes must contain distinct vertex labels.")
    return nodes


def _from_edges(graph: Iterable, nodes: Optional[Iterable[Hashable]]) -> nx.DiGraph:
    edges = []
    for edge in _materialize(graph, "graph"):
        pair = _materialize(edge, "Each edge")
        if len(pair) != 2:
            raise ValueError("Each edge must contain exactly (source, target).")
        source, target = pair
        _validate_node(source)
        _validate_node(target)
        edges.append((source, target))

    labels = (
        _ordered_nodes(node for edge in edges for node in edge)
        if nodes is None
        else _explicit_nodes(nodes)
    )
    allowed = set(labels)
    if any(source not in allowed or target not in allowed for source, target in edges):
        raise ValueError("Every edge endpoint must be included in nodes.")
    result = nx.DiGraph()
    result.add_nodes_from(labels)
    result.add_edges_from(edges)
    return result


def _from_matrix(graph: Iterable, nodes: Optional[Iterable[Hashable]]) -> nx.DiGraph:
    shape = getattr(graph, "shape", None)
    if shape is not None and (len(shape) != 2 or shape[0] != shape[1]):
        raise ValueError("The adjacency matrix must be square and two-dimensional.")
    rows = [_materialize(row, "Each matrix row") for row in _materialize(graph, "graph")]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("The adjacency matrix must be square.")
    labels = list(range(size)) if nodes is None else _explicit_nodes(nodes)
    if len(labels) != size:
        raise ValueError("nodes must have one label per matrix row.")

    result = nx.DiGraph()
    result.add_nodes_from(labels)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            try:
                valid = not isinstance(value, (str, bytes)) and not hasattr(value, "__len__")
                valid = valid and value in (0, 1)
                if not valid:
                    raise ValueError("The adjacency matrix must contain only scalar 0 or 1 values.")
                if value == 1:
                    result.add_edge(labels[i], labels[j])
            except (TypeError, ValueError) as exc:
                raise ValueError("The adjacency matrix must contain only scalar 0 or 1 values.") from exc
    return result


def _family(graph: nx.DiGraph) -> _Family:
    if not graph:
        return ()
    condensed = nx.condensation(graph)
    classes = []
    for source, degree in condensed.in_degree():
        if degree == 0:
            reachable = nx.descendants(condensed, source) | {source}
            classes.append(frozenset(
                node for component in reachable for node in condensed.nodes[component]["members"]
            ))
    rank = {node: index for index, node in enumerate(graph)}
    classes.sort(key=lambda group: tuple(sorted(rank[node] for node in group)))
    return tuple(classes)


def graph_to_maximal_class(
    graph: Iterable,
    *,
    input_format: str = "edges",
    nodes: Optional[Iterable[Hashable]] = None,
) -> _Family:
    """Return the full maximal-class family of a labeled directed graph.

    A class consists of a source strongly connected component together with
    every vertex reachable from it. Different classes may overlap.

    Parameters
    ----------
    graph
        Iterable of directed (source, target) pairs, or a square binary matrix.
        NumPy arrays and nested lists are accepted for matrix input.
    input_format
        "edges" (default) or "matrix". Matrix entry [i, j] denotes i -> j.
    nodes
        Optional distinct, hashable vertex labels, excluding None. For edges,
        include isolated vertices here; all endpoints must be listed. For a
        matrix, provide labels in row/column order, or use default labels 0..n-1.

    Returns
    -------
    tuple of frozenset
        The complete family. Ordering follows the supplied node order, or
        sorted inferred labels when comparable (first appearance otherwise).
        Self-loops and repeated edge pairs do not affect the result.

    Raises
    ------
    ValueError
        If the input format, edges, labels, or binary matrix are invalid.
    """
    if input_format == "edges":
        normalized = _from_edges(graph, nodes)
    elif input_format == "matrix":
        normalized = _from_matrix(graph, nodes)
    else:
        raise ValueError('input_format must be "edges" or "matrix".')
    return _family(normalized)


def _normalize_family(maximal_class: Iterable) -> tuple[_Family, list[Hashable], list[int]]:
    classes = []
    all_nodes = []
    for group in _materialize(maximal_class, "maximal_class"):
        labels = _unique_nodes(_materialize(group, "Each class"))
        if not labels:
            raise ValueError("A maximal class cannot be empty; use () for the empty graph's family.")
        classes.append(frozenset(labels))
        all_nodes.extend(labels)
    if len(set(classes)) != len(classes):
        raise ValueError("The family must not contain duplicate classes.")
    nodes = _ordered_nodes(all_nodes)
    membership = [sum(1 << k for k, group in enumerate(classes) if node in group) for node in nodes]
    for k in range(len(classes)):
        if (1 << k) not in membership:
            raise ValueError(
                "The family is not realizable: every class must contain a vertex "
                "that belongs to no other class."
            )
    return tuple(classes), nodes, membership


def _can_realize(adjacency: list[int], requirements: list[tuple[int, tuple[int, ...]]]) -> bool:
    """Necessary reachability condition on selected plus undecided edges.

    Every target class must be reachable from some vertex exclusive to that
    class. Deleting edges cannot repair failure of this condition. Checking
    the optimistic graph therefore cannot prune a valid completion.
    """
    for target, sources in requirements:
        for source in sources:
            seen = frontier = 1 << source
            while frontier and seen & target != target:
                bit = frontier & -frontier
                frontier ^= bit
                new = adjacency[bit.bit_length() - 1] & ~seen
                seen |= new
                frontier |= new
            if seen & target == target:
                break
        else:
            return False
    return True


def _enumerate_graphs(
    nodes: list[Hashable],
    edges: list[tuple[int, int]],
    classes: _Family,
    membership: list[int],
) -> Iterator[nx.DiGraph]:
    possible = [0] * len(nodes)
    for source, target in edges:
        possible[source] |= 1 << target
    requirements = [
        (
            sum(1 << i for i, node in enumerate(nodes) if node in group),
            tuple(i for i, mask in enumerate(membership) if mask == 1 << k),
        )
        for k, group in enumerate(classes)
    ]
    expected = frozenset(classes)
    # The explicit stack avoids Python's recursion limit. Adjacency lists are
    # copied before modification; siblings never share mutable search state.
    stack = [(0, [0] * len(nodes), possible)]
    while stack:
        index, selected, optimistic = stack.pop()
        if not _can_realize(optimistic, requirements):
            continue
        if index == len(edges):
            candidate = nx.DiGraph()
            candidate.add_nodes_from(nodes)
            candidate.add_edges_from((node, node) for node in nodes)
            candidate.add_edges_from(
                (nodes[source], nodes[target])
                for source, target in edges if selected[source] & (1 << target)
            )
            # Verify with the same standard SCC algorithm as the public
            # forward function, independently of the pruning condition.
            if frozenset(_family(candidate)) == expected:
                yield candidate
            continue
        source, target = edges[index]
        included = selected.copy()
        included[source] |= 1 << target
        excluded = optimistic.copy()
        excluded[source] &= ~(1 << target)
        stack.append((index + 1, included, optimistic))
        stack.append((index + 1, selected, excluded))


def list_graphs_in_maximal_class(
    maximal_class: Iterable[Iterable[Hashable]],
    *,
    max_candidates: Optional[int] = _DEFAULT_MAX_CANDIDATES,
    as_iterator: bool = False,
) -> Union[list[nx.DiGraph], Iterator[nx.DiGraph]]:
    """Enumerate all labeled directed graphs having exactly the given family.

    The vertex set is the union of the input classes. Every output includes
    exactly one self-loop at each vertex; no parallel edges are generated.
    Input class order and member order do not affect family equality.

    Parameters
    ----------
    maximal_class
        The complete family of nonempty vertex sets, for example the result
        of graph_to_maximal_class(). Each class must have an exclusive vertex.
        The empty family () describes the graph with zero vertices.
    max_candidates
        Maximum 2**p before pruning, where p is the number of allowed non-loop
        edges. Default: 32768 (the original R limit of 15 allowed edges).
        Supply a positive integer to change it, or None to disable the limit.
    as_iterator
        False returns a complete list. True yields graphs incrementally.
        Validation and the search-limit check happen immediately in both modes.

    Raises
    ------
    ValueError
        For invalid/unrealizable families or options, or if the candidate
        count exceeds max_candidates. No partial list is returned.

    Notes
    -----
    Runtime and the number of outputs can be exponential. Streaming reduces
    memory use but does not make all families on 3-10 vertices enumerable.
    """
    if not isinstance(as_iterator, bool):
        raise ValueError("as_iterator must be a boolean.")
    if max_candidates is not None:
        try:
            if isinstance(max_candidates, bool):
                raise TypeError
            max_candidates = operator.index(max_candidates)
        except TypeError as exc:
            raise ValueError("max_candidates must be a positive integer or None.") from exc
        if max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer or None.")
    classes, nodes, membership = _normalize_family(maximal_class)
    # R's leakage rule: every class containing u must also contain v.
    # Its source rule is redundant under this rule on the union of classes.
    edges = [
        (u, v) for u in range(len(nodes)) for v in range(len(nodes))
        if u != v and membership[u] & membership[v] == membership[u]
    ]
    if max_candidates is not None:
        candidate_count = 1 << len(edges)
        if candidate_count > max_candidates:
            raise ValueError(
                f"Enumeration has {len(edges)} allowed non-loop edges and up to "
                f"{candidate_count} candidates before pruning, exceeding "
                f"max_candidates={max_candidates}. Increase max_candidates or "
                "set it to None explicitly; as_iterator=True reduces memory use "
                "but does not remove the search limit."
            )
    graphs = _enumerate_graphs(nodes, edges, classes, membership)
    return graphs if as_iterator else list(graphs)
