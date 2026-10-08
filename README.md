# maximal_class_graph

Convert a labeled directed graph into its complete maximal-class family and
enumerate all graphs having exactly that family. Both calculations use integer
bitmasks throughout. Graphs are stored compactly, with optional conversion to
NumPy Boolean matrices or NetworkX `DiGraph` objects.

Adapted from the [original R algorithms](https://github.com/Bi-xuan/maximal_class).

## Installation

Python 3.9 or newer is required. The calculations have no third-party dependencies:

```sh
python -m pip install .
```

Install optional dependencies for matrix and/or DiGraph conversion:

```sh
python -m pip install '.[matrix]'          # NumPy
python -m pip install '.[digraph]'         # NetworkX
python -m pip install '.[matrix,digraph]'  # Both
```

For development and tests:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

## Definition

Collapse each strongly connected component (SCC) into one vertex. In the
resulting condensation DAG, each source SCC defines a class: its own vertices
and every original vertex reachable from it. The complete family of these sets
is the graph's maximal-class family. Classes can overlap.

For example, edges `1 -> 3` and `2 -> 3` give `{1, 3}, {2, 3}`.
Cycles are allowed. Self-loops do not change the family.

## `graph_to_maximal_class()`

```python
from maximal_class_graph import graph_to_maximal_class

family = graph_to_maximal_class([(1, 3), (2, 3)], input_format="edges")
assert family == (frozenset({1, 3}), frozenset({2, 3}))
```

Specify `nodes` to include isolated vertices in edge input. When supplied, it
must list all vertices, including all edge endpoints:

```python
assert graph_to_maximal_class([(1, 2)], nodes=[1, 2, 3]) == (
    frozenset({1, 2}), frozenset({3}),
)
```

For matrix input, `[i, j] == 1` denotes `i -> j`. Matrices must be square and
binary; Boolean values are accepted. Nested lists require no NumPy. Labels
default to `0, ..., n-1`, or follow `nodes` in row/column order:

```python
matrix = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
assert graph_to_maximal_class(matrix, input_format="matrix", nodes=[1, 2, 3]) == family
```

Integer masks can be passed directly, with the vertex order specified:

```python
assert graph_to_maximal_class(10, input_format="bitmask", nodes=[1, 2, 3]) == family
```

Labels can be any hashable objects except `None`. Repeated edge pairs and
self-loops do not affect the family. Inputs are not modified. Class ordering
follows supplied node order; otherwise inferred comparable labels are sorted,
with first appearance used for labels that cannot be sorted. Compare families
without ordering using `frozenset(family)`.

## Bitmask encoding

A graph is one nonnegative Python integer. Bits enumerate **all** possible
non-loop edges in row-major order, starting at the least significant bit.
The encoding is independent of the maximal-class family and its allowed edges.
For `nodes = (1, 2, 3)`:

| Bit position | Edge |
| --- | --- |
| 0 | `1 -> 2` |
| 1 | `1 -> 3` |
| 2 | `2 -> 1` |
| 3 | `2 -> 3` |
| 4 | `3 -> 1` |
| 5 | `3 -> 2` |

Thus `10 == (1 << 1) | (1 << 3)` encodes `1 -> 3` and `2 -> 3`.
For indexed vertices `i != j`, the bit position is
`i * (n - 1) + j - (j > i)`.

The ordered vertex labels must be retained alongside the masks; the integer
alone does not specify labels, isolated vertices, or even the vertex count.
At 10 vertices a mask needs at most 90 bits. Self-loops at every vertex are
implicit and always restored during conversion. Zero denotes a graph with no
non-loop edges. It describes different graphs for different vertex metadata.

## `list_graphs_in_maximal_class()`

```python
from maximal_class_graph import list_graphs_in_maximal_class

results = list_graphs_in_maximal_class([{1, 3}, {2, 3}])
assert results.nodes == (1, 2, 3)
assert results.masks == [10]
```

The input is the **entire family**. Its union defines the vertex set. The return
value is a `BitmaskGraphs` container holding shared `nodes` and integer `masks`.
By default `masks` is the complete list of matching graphs. Each distinct
labeled non-loop edge set appears once. Isomorphic graphs with different
labeled edges remain distinct. No matrices or DiGraph objects are created.

For a one-pass mask iterator:

```python
results = list_graphs_in_maximal_class([{1, 2, 3}], as_iterator=True)
for mask in results.masks:
    assert isinstance(mask, int)
```

Family validation and the candidate-search limit are checked immediately in
both modes. An iterator is consumed as it is used; stopping early gives only
the masks consumed so far.

An edge `u -> v` is allowed only if every input class containing `u` also
contains `v`. With `p` allowed edges, there are at most `2**p` candidates before
pruning. The default `max_candidates=32768` matches the original R algorithm's
limit of 15 possible non-loop edges. Raise this positive integer explicitly,
or use `None` to disable the guard. Exceeding it raises `ValueError`; streaming
does not remove the search limit.

```python
results = list_graphs_in_maximal_class(
    [{0, 1, 2, 3}], max_candidates=4096, as_iterator=True,
)
```

Compact storage does not eliminate exponential output size. With 10 vertices
and one class containing them all, fixing a directed cycle still leaves 80
optional edges, giving at least `2**80` valid outputs. In contrast, ten singleton
classes give just one mask: zero, with ten implicit self-loops.

## `convert_bitmask_graphs()`

Convert a complete result or a chosen subset separately from enumeration:

```python
from maximal_class_graph import convert_bitmask_graphs

results = list_graphs_in_maximal_class([{1, 3}, {2, 3}])

# NumPy Boolean matrices, with rows/columns in results.nodes order.
matrices = convert_bitmask_graphs(results)
assert matrices[0].tolist() == [
    [True, False, True], [False, True, True], [False, False, True],
]

# NetworkX objects preserving the labels, including every self-loop.
graphs = convert_bitmask_graphs(results, output_format="digraph")
assert set(graphs[0].edges) == {(1, 1), (2, 2), (3, 3), (1, 3), (2, 3)}

# Convert a subset, or one mask wrapped in a list.
selected = convert_bitmask_graphs(results.masks[:100], nodes=results.nodes)
one = convert_bitmask_graphs([results.masks[0]], nodes=results.nodes)
```

Output objects are independent. `output_format` is `"matrix"` (default) or
`"digraph"`. Passing a `BitmaskGraphs` container supplies its own labels;
passing raw masks requires `nodes` in the exact encoding order. Do not supply
`nodes` again alongside a container. Even a single mask is passed in an iterable.

### Conversion storage limit

Eager conversion defaults to **at most 1,000 graphs** (`max_graphs=1000`). If
exceeded, a `ValueError` explains the storage concern before any matrix or
DiGraph is allocated. This is a graph-count cap, not an exact byte budget;
object sizes also depend on vertex count and density.

For unknown-length mask iterators, the guard consumes at most `max_graphs + 1`
masks. These cannot be put back if the cap is exceeded. To avoid consuming a
large stream accidentally, select a bounded part explicitly with `islice`.

```python
from itertools import islice

results = list_graphs_in_maximal_class([{1, 2, 3}], as_iterator=True)
first_ten = convert_bitmask_graphs(islice(results.masks, 10), nodes=results.nodes)

# Raise the cap explicitly, or use max_graphs=None to disable it.
remaining = convert_bitmask_graphs(results, max_graphs=2000)
```

For conversion without accumulating all outputs:

```python
results = list_graphs_in_maximal_class([{1, 2, 3}], as_iterator=True)
for graph in convert_bitmask_graphs(results, output_format="digraph", as_iterator=True):
    assert all(graph.has_edge(node, node) for node in graph)
```

Streaming conversion has no cumulative graph-count cap: it allocates one
output per iteration. Accumulating the outputs yourself still uses storage.
Options, labels, and dependency availability are checked at call time. All
eager masks are checked before allocating outputs; streamed masks are checked
as consumed. Negative, boolean, noninteger, or out-of-range masks raise
`ValueError`. Missing conversion dependencies raise an informative `ImportError`.

## Validation and empty graphs

Input classes must be nonempty and distinct. Each must contain a vertex
belonging to no other class; otherwise the family is not realizable and raises
`ValueError`. This condition is sufficient as well: select one exclusive vertex
per class and connect it to every other member of that class.

The zero-vertex graph has family `()`. Enumerating it returns
`BitmaskGraphs(nodes=(), masks=[0])`; conversion gives a `(0, 0)` matrix or an
empty DiGraph. A one-vertex graph also has mask zero, but has one implicit loop.

## Changes in version 0.2

Enumeration now returns `BitmaskGraphs`, rather than a list or iterator of
NetworkX graphs. Read `results.masks` for the compact results, or call
`convert_bitmask_graphs(results, output_format="digraph")` for the former output
objects. NetworkX and NumPy are optional conversion dependencies. The forward
function retains its edge/matrix inputs and adds direct bitmask input.

## Implementation and verification

Reachability uses bitset transitive closure. Inclusion-maximal reachable sets
give the source-SCC maximal classes. Reverse enumeration holds only integer
graph masks, prunes branches that cannot provide required reachability, and
checks exact families before yielding a mask.

Tests exhaustively compare both algorithms against an independent DFS oracle
for **all 4,166 directed graphs on zero through four vertices**, and verify the
converted edges and matrices. Additional tests cover fixed bit positions up to
10 vertices, graphs with masks beyond 64 bits, larger examples against NetworkX
SCCs, dependency-free calculations, conversion limits, streaming, and validation.
