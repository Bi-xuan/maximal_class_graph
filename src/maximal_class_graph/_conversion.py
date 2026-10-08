"""Optional, bounded conversion of integer graph masks to larger formats."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sized
from typing import Any, Callable, Optional, Union

from ._core import BitmaskGraphs, _explicit_nodes, _iter_edge_indices, _positive_limit, _validate_mask


def _builder(output_format: str, nodes: tuple[Hashable, ...]) -> Callable[[int], Any]:
    size = len(nodes)
    if output_format == "matrix":
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("Matrix conversion requires NumPy; install the package's [matrix] extra.") from exc

        def build(mask: int) -> Any:
            matrix = np.eye(size, dtype=bool)
            for u, v in _iter_edge_indices(mask, size):
                matrix[u, v] = True
            return matrix

    else:
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("DiGraph conversion requires NetworkX; install the package's [digraph] extra.") from exc

        def build(mask: int) -> Any:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from((node, node) for node in nodes)
            graph.add_edges_from((nodes[u], nodes[v]) for u, v in _iter_edge_indices(mask, size))
            return graph

    return build


def _storage_error(limit: int) -> ValueError:
    return ValueError(
        f"Conversion exceeds max_graphs={limit}. Matrices and DiGraph objects "
        "use more storage than bitmasks. Convert a subset, use as_iterator=True, "
        "or explicitly increase max_graphs (None disables the limit)."
    )


def _converted(masks: Iterator[int], size: int, build: Callable[[int], Any]) -> Iterator[Any]:
    for mask in masks:
        yield build(_validate_mask(mask, size))


def convert_bitmask_graphs(
    masks: Union[BitmaskGraphs, Iterable[int]],
    *,
    nodes: Optional[Iterable[Hashable]] = None,
    output_format: str = "matrix",
    max_graphs: Optional[int] = 1000,
    as_iterator: bool = False,
) -> Union[list[Any], Iterator[Any]]:
    """Convert all or selected masks to Boolean matrices or NetworkX DiGraphs.

    Pass a BitmaskGraphs result directly, or pass an iterable of masks with
    ``nodes`` giving the exact vertex order. For one mask, use ``[mask]``.
    ``output_format`` is "matrix" (NumPy Boolean arrays) or "digraph".
    Every result includes self-loops at every vertex. Matrix rows and columns
    follow ``nodes``; DiGraphs preserve those labels. Results are independent.

    By default at most 1000 graphs can be converted eagerly. Exceeding
    ``max_graphs`` raises ValueError with a storage explanation before any
    output objects are allocated. For unknown-length iterators, at most
    max_graphs+1 masks are consumed during this check (they cannot be put back).
    Set a positive integer to raise the limit or None to explicitly disable it.

    ``as_iterator=True`` returns a one-pass conversion iterator, without a
    cumulative graph-count limit. Only one output is created per iteration;
    callers retaining outputs are responsible for their storage. Options,
    labels, and dependency availability are checked at call time. Masks are
    checked before allocation, eagerly in list mode or as consumed in iterator
    mode. Invalid masks (negative, noninteger, boolean, or out-of-range) raise
    ValueError. Missing optional dependencies raise ImportError.
    """
    if output_format not in ("matrix", "digraph"):
        raise ValueError('output_format must be "matrix" or "digraph".')
    if not isinstance(as_iterator, bool):
        raise ValueError("as_iterator must be a boolean.")
    max_graphs = _positive_limit(max_graphs, "max_graphs")
    if isinstance(masks, BitmaskGraphs):
        if nodes is not None:
            raise ValueError("nodes is already supplied by BitmaskGraphs; omit nodes or pass results.masks.")
        labels = tuple(_explicit_nodes(masks.nodes))
        values = masks.masks
    else:
        if nodes is None:
            raise ValueError("nodes is required when converting a raw iterable of masks.")
        labels = tuple(_explicit_nodes(nodes))
        values = masks
    if isinstance(values, (str, bytes)):
        raise ValueError("masks must be an iterable of integers; wrap a single mask in a list.")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError("masks must be an iterable of integers; wrap a single mask in a list.") from exc
    if as_iterator:
        return _converted(iterator, len(labels), _builder(output_format, labels))
    if max_graphs is not None and isinstance(values, Sized) and len(values) > max_graphs:
        raise _storage_error(max_graphs)
    if max_graphs is None:
        buffered = list(iterator)
    else:
        buffered = []
        for mask in iterator:
            buffered.append(mask)
            if len(buffered) > max_graphs:
                raise _storage_error(max_graphs)
    checked = [_validate_mask(mask, len(labels)) for mask in buffered]
    build = _builder(output_format, labels)
    return [build(mask) for mask in checked]
