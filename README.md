# maximal_class_graph

## Installation and imports

Requires Python 3.9 or newer. Install the package directly from GitHub:

```sh
python -m pip install git+https://github.com/Bi-xuan/maximal_class_graph.git
```

Import the three public functions:

```python
from maximal_class_graph import (
    graph_to_maximal_class,
    list_graphs_in_maximal_class,
    convert_bitmask_graphs,
)
```

- `graph_to_maximal_class`: compute a directed graph's maximal-class family.
- `list_graphs_in_maximal_class`: enumerate graphs with a given maximal-class family.
- `convert_bitmask_graphs`: convert graph bitmasks to NumPy matrices or NetworkX graphs.

The result container type is also available as `from maximal_class_graph import BitmaskGraphs`.

## `graph_to_maximal_class()`

```python
graph_to_maximal_class(graph, *, input_format="edges", nodes=None)
```

### Input

`graph` represents a labeled directed graph in the format selected by
`input_format`:

| `input_format` | Format of `graph` | Meaning of `nodes` |
| --- | --- | --- |
| `"edges"` (default) | An iterable of `(source, target)` pairs, such as `[(1, 3), (2, 3)]`. | Optional ordered vertex labels. Include every edge endpoint and any isolated vertices. If omitted, labels are inferred from the edges. |
| `"matrix"` | A square binary adjacency matrix, as nested lists or a NumPy array. Entry `[i, j] == 1` means an edge from vertex `i` to vertex `j`; Boolean entries are accepted. | Optional labels in row/column order, one per row. Defaults to `0, ..., n-1`. |
| `"bitmask"` | A nonnegative integer encoding non-loop edges as described below. | Required ordered vertex labels, defining the vertex count and bit positions. |

Explicit `nodes` must contain distinct hashable labels, excluding `None`.
Repeated edges and self-loops do not affect the result. Nested-list matrix
input does not require NumPy.

A **bitmask** uses one bit per possible non-loop edge, in row-major order,
starting at bit 0 and skipping the diagonal. For `n` ordered vertices, the
edge from index `i` to index `j` (`i != j`) occupies bit
`i * (n - 1) + j - int(j > i)`. Valid masks satisfy
`0 <= mask < 2**(n * (n - 1))`; booleans are not accepted as masks.
For `nodes=(1, 2, 3)`, bits 0 through 5 represent
`1 -> 2`, `1 -> 3`, `2 -> 1`, `2 -> 3`, `3 -> 1`, `3 -> 2`.
Thus mask `10` encodes `1 -> 3` and `2 -> 3`.

### Output

A `tuple` of `frozenset` objects containing the **complete maximal-class
family**. Collapse each strongly connected component into one vertex. Each
source component (a component with no incoming edges from other components)
defines one class consisting of its vertices and all vertices reachable from
it. Classes can overlap.

Class ordering follows the supplied vertex order. Inferred edge labels are
sorted when comparable; otherwise their first appearance determines the
order. An empty graph with no vertices returns `()`.

```python
family = graph_to_maximal_class([(1, 3), (2, 3)])
# (frozenset({1, 3}), frozenset({2, 3}))

family = graph_to_maximal_class(10, input_format="bitmask", nodes=[1, 2, 3])
# The same family.
```

## `list_graphs_in_maximal_class()`

```python
list_graphs_in_maximal_class(
    maximal_class, *, max_candidates=32768, as_iterator=False,
)
```

### Input

- `maximal_class`: an iterable of iterables of vertex labels, such as
  `[{1, 3}, {2, 3}]`. Supply the **entire family** defined above; its union is
  the vertex set. Labels must be hashable and cannot be `None`. Classes must
  be nonempty and distinct, and each class must contain at least one vertex
  belonging to no other class. Use `()` for the zero-vertex graph's family.
- `max_candidates`: a positive integer or `None`. Limits the candidate count
  `2**p` before pruning, where `p` is the number of allowed non-loop edges.
  An edge `u -> v` is allowed when every input class containing `u` also
  contains `v`. The default is `32768`; `None` disables the limit. Exceeding
  the limit raises `ValueError`, including in iterator mode.
- `as_iterator`: a Boolean, default `False`, selecting whether the output
  masks are stored in a list or yielded by a one-pass iterator.

### Output

A `BitmaskGraphs` container with two attributes:

| Attribute | Format and meaning |
| --- | --- |
| `nodes` | A tuple of all vertex labels, sorted when comparable; otherwise in first-appearance order in the input. This order defines the bit positions. |
| `masks` | A `list[int]` by default, or a one-pass iterator of integers when `as_iterator=True`. Each mask represents one graph whose complete maximal-class family equals the input, using the encoding above. |

Each distinct labeled non-loop edge set appears once. Self-loops at every
vertex are implicit. No matrices or NetworkX objects are returned. The empty
family returns `nodes=()` and `masks=[0]` in list mode.

```python
results = list_graphs_in_maximal_class([{1, 3}, {2, 3}])
# BitmaskGraphs(nodes=(1, 2, 3), masks=[10])
```

## `convert_bitmask_graphs()`

```python
convert_bitmask_graphs(
    masks, *, nodes=None, output_format="matrix", max_graphs=1000,
    as_iterator=False,
)
```

### Input

- `masks`: a `BitmaskGraphs` container, or an iterable of integer graph masks
  using the encoding above. Wrap a single mask in a list, such as `[10]`.
- `nodes`: required for raw mask iterables, in the exact order used to encode
  the masks. Labels must be distinct and hashable, excluding `None`. Omit
  this argument when passing a `BitmaskGraphs` container, which supplies its
  own labels.
- `output_format`: `"matrix"` (default, requires NumPy) or `"digraph"`
  (requires NetworkX).
- `max_graphs`: a positive integer or `None`, default `1000`. Caps the number
  of graphs converted in list mode; exceeding it raises `ValueError` before
  output objects are allocated. With an unknown-length mask iterator, the
  check consumes up to `max_graphs + 1` masks. `None` disables the cap.
- `as_iterator`: a Boolean, default `False`. If `True`, returns a one-pass
  iterator that converts one graph at a time, without a cumulative
  graph-count cap.

### Output

A list by default, or a one-pass iterator when `as_iterator=True`, preserving
input mask order. Each element is an independent graph representation:

| `output_format` | Format and meaning of each element |
| --- | --- |
| `"matrix"` | A NumPy Boolean array of shape `(n, n)`, with rows and columns in `nodes` order. Entry `[i, j]` is `True` exactly when the graph contains the edge from `nodes[i]` to `nodes[j]`. Every diagonal entry is `True`. |
| `"digraph"` | A NetworkX `DiGraph` preserving all vertex labels, including isolated vertices, and containing a self-loop at every vertex. |

For the zero-vertex mask, conversion produces a `(0, 0)` matrix or an empty
`DiGraph`. Missing conversion dependencies raise `ImportError`.

```python
matrices = convert_bitmask_graphs([10], nodes=[1, 2, 3])
# matrices[0].tolist() == [
#     [True, False, True],
#     [False, True, True],
#     [False, False, True],
# ]

graphs = convert_bitmask_graphs(results, output_format="digraph")
# set(graphs[0].edges) == {(1, 1), (2, 2), (3, 3), (1, 3), (2, 3)}
```
