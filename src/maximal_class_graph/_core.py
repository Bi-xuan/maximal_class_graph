"""Bitmask adaptations of the algorithms in Bi-xuan/maximal_class."""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
from typing import Optional, Union

_Family = tuple[frozenset[Hashable], ...]
_DEFAULT_MAX_CANDIDATES = 1 << 15


@dataclass(frozen=True)
class BitmaskGraphs:
    """Graph masks and their shared, ordered vertex labels.

    ``masks`` is a list by default, or a one-pass iterator when enumeration
    uses ``as_iterator=True``. Bit positions enumerate non-loop edges in
    row-major order using ``nodes``. Self-loops at every vertex are implicit.
    """

    nodes: tuple[Hashable, ...]
    masks: Union[list[int], Iterator[int]]


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
        return nodes


def _explicit_nodes(values: Iterable[Hashable]) -> list[Hashable]:
    nodes = _materialize(values, "nodes")
    if len(_unique_nodes(nodes)) != len(nodes):
        raise ValueError("nodes must contain distinct vertex labels.")
    return nodes


def _positive_limit(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer or None.") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer or None.")
    return result


def _validate_mask(value: int, size: int) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        mask = operator.index(value)
    except TypeError as exc:
        raise ValueError("A graph bitmask must be a nonnegative integer, not a boolean.") from exc
    if mask < 0 or mask.bit_length() > size * (size - 1):
        raise ValueError(
            f"Graph bitmask has negative or out-of-range bits: "
            f"{size} vertices allow {size * (size - 1)} non-loop bits."
        )
    return mask


def _edge_bit(source: int, target: int, size: int) -> int:
    """The caller excludes loops; skip the diagonal in row-major order."""
    return 1 << (source * (size - 1) + target - (target > source))


def _iter_edge_indices(mask: int, size: int) -> Iterator[tuple[int, int]]:
    while mask:
        bit = mask & -mask
        source, offset = divmod(bit.bit_length() - 1, size - 1)
        yield source, offset + (offset >= source)
        mask ^= bit


def _from_edges(graph: Iterable, nodes: Optional[Iterable[Hashable]]) -> tuple[int, list[Hashable]]:
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
        if nodes is None else _explicit_nodes(nodes)
    )
    indices = {node: i for i, node in enumerate(labels)}
    mask = 0
    for source, target in edges:
        if source not in indices or target not in indices:
            raise ValueError("Every edge endpoint must be included in nodes.")
        u, v = indices[source], indices[target]
        if u != v:
            mask |= _edge_bit(u, v, len(labels))
    return mask, labels


def _from_matrix(graph: Iterable, nodes: Optional[Iterable[Hashable]]) -> tuple[int, list[Hashable]]:
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
    mask = 0
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            try:
                valid = not isinstance(value, (str, bytes)) and not hasattr(value, "__len__")
                if not (valid and value in (0, 1)):
                    raise ValueError
                if i != j and value == 1:
                    mask |= _edge_bit(i, j, size)
            except (TypeError, ValueError) as exc:
                raise ValueError("The adjacency matrix must contain only scalar 0 or 1 values.") from exc
    return mask, labels


def _reachability(mask: int, size: int) -> list[int]:
    """Reflexive transitive closure using vertex bitsets and Warshall's rule."""
    width = max(size - 1, 0)
    row_mask = (1 << width) - 1
    reachable = []
    for source in range(size):
        row = (mask >> (source * width)) & row_mask
        # Insert the implicit diagonal bit into the packed non-loop row.
        lower = row & ((1 << source) - 1)
        reachable.append(lower | ((row >> source) << (source + 1)) | (1 << source))
    for intermediate in range(size):
        bit = 1 << intermediate
        for source in range(size):
            if reachable[source] & bit:
                reachable[source] |= reachable[intermediate]
    return reachable


def _family_bits(mask: int, size: int) -> frozenset[int]:
    # The inclusion-maximal reachable sets are precisely the reachable sets
    # of source SCCs. No graph objects or explicit SCC graph are necessary.
    reachable = set(_reachability(mask, size))
    return frozenset(
        group for group in reachable
        if not any(group != other and group & other == group for other in reachable)
    )


def graph_to_maximal_class(
    graph: Union[Iterable, int],
    *,
    input_format: str = "edges",
    nodes: Optional[Iterable[Hashable]] = None,
) -> _Family:
    """Return the full maximal-class family using only bitmask calculations.

    ``graph`` accepts directed edge pairs, a square binary matrix, or an
    integer mask; set ``input_format`` to "edges", "matrix", or "bitmask".
    Matrix entry [i, j] denotes i -> j. NumPy is not required for nested lists.
    ``nodes`` supplies distinct hashable labels (excluding None), including
    isolated vertices. Matrix labels default to 0..n-1. Bitmask input requires
    ``nodes`` to specify both the vertex count and bit ordering.

    Bits enumerate non-loop edges in row-major order, starting at bit 0.
    Self-loops are ignored on input and implicit in masks. Repeated edges do
    not change the result. Returns a tuple of frozensets in supplied node
    order, or sorted inferred labels when comparable (first appearance otherwise).
    Invalid inputs raise ValueError. No NetworkX or NumPy objects are created
    internally, and inputs are not modified.
    """
    if input_format == "edges":
        mask, labels = _from_edges(graph, nodes)
    elif input_format == "matrix":
        mask, labels = _from_matrix(graph, nodes)
    elif input_format == "bitmask":
        if nodes is None:
            raise ValueError("nodes is required for bitmask input.")
        labels = _explicit_nodes(nodes)
        mask = _validate_mask(graph, len(labels))
    else:
        raise ValueError('input_format must be "edges", "matrix", or "bitmask".')
    groups = sorted(
        _family_bits(mask, len(labels)),
        key=lambda group: tuple(i for i in range(len(labels)) if group & (1 << i)),
    )
    return tuple(
        frozenset(node for i, node in enumerate(labels) if group & (1 << i))
        for group in groups
    )


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


def _can_realize(mask: int, size: int, requirements: list[tuple[int, tuple[int, ...]]]) -> bool:
    """Only prune when even all remaining edges cannot realize a class."""
    reachable = _reachability(mask, size)
    return all(
        any(reachable[source] & target == target for source in sources)
        for target, sources in requirements
    )


def _enumerate_masks(
    size: int,
    edge_bits: list[int],
    requirements: list[tuple[int, tuple[int, ...]]],
) -> Iterator[int]:
    possible = sum(edge_bits)
    expected = frozenset(target for target, _ in requirements)
    # Every search state consists of an edge index and two integer graph masks.
    stack = [(0, 0, possible)]
    while stack:
        index, selected, optimistic = stack.pop()
        if index == len(edge_bits):
            if _family_bits(selected, size) == expected:
                yield selected
            continue
        if not _can_realize(optimistic, size, requirements):
            continue
        bit = edge_bits[index]
        stack.append((index + 1, selected | bit, optimistic))
        stack.append((index + 1, selected, optimistic & ~bit))


def list_graphs_in_maximal_class(
    maximal_class: Iterable[Iterable[Hashable]],
    *,
    max_candidates: Optional[int] = _DEFAULT_MAX_CANDIDATES,
    as_iterator: bool = False,
) -> BitmaskGraphs:
    """Return all matching graphs as integer masks with shared vertex labels.

    The entire family is required; its union defines the vertex set. Each
    distinct class must be nonempty and have at least one exclusive vertex.
    The empty family describes the zero-vertex graph (mask 0, nodes ()).

    Returns ``BitmaskGraphs(nodes=..., masks=...)``. Masks use the fixed
    row-major non-loop encoding, independent of the input family. Self-loops
    at every vertex are implicit. By default ``masks`` is a complete list;
    ``as_iterator=True`` makes it a one-pass iterator. No graph objects or
    matrices are created during enumeration.

    ``max_candidates`` limits 2**p before pruning, where p is the number of
    allowed non-loop edges. Default: 32768, equivalent to the R limit of 15
    allowed edges. Set a positive integer to change it, or None to disable it.
    Family and option validation, including the guard, happens at call time
    in both modes. Invalid inputs or excessive searches raise ValueError.

    Enumeration can still have exponentially many outputs. Use
    ``convert_bitmask_graphs`` to convert all or selected masks separately.
    """
    if not isinstance(as_iterator, bool):
        raise ValueError("as_iterator must be a boolean.")
    max_candidates = _positive_limit(max_candidates, "max_candidates")
    classes, nodes, membership = _normalize_family(maximal_class)
    size = len(nodes)
    # Every class containing u must also contain v (the R leakage rule).
    edge_bits = [
        _edge_bit(u, v, size) for u in range(size) for v in range(size)
        if u != v and membership[u] & membership[v] == membership[u]
    ]
    if max_candidates is not None:
        candidate_count = 1 << len(edge_bits)
        if candidate_count > max_candidates:
            raise ValueError(
                f"Enumeration has {len(edge_bits)} allowed non-loop edges and up to "
                f"{candidate_count} candidates before pruning, exceeding "
                f"max_candidates={max_candidates}. Increase max_candidates or "
                "set it to None explicitly; as_iterator=True reduces memory use "
                "but does not remove the search limit."
            )
    requirements = [
        (
            sum(1 << i for i, node in enumerate(nodes) if node in group),
            tuple(i for i, mask in enumerate(membership) if mask == 1 << k),
        )
        for k, group in enumerate(classes)
    ]
    masks = _enumerate_masks(size, edge_bits, requirements)
    return BitmaskGraphs(tuple(nodes), masks if as_iterator else list(masks))
