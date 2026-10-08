# maximal_class_graph

A Python package for converting a labeled directed graph into its complete
maximal-class family and enumerating all graphs having exactly that family.
Adapted from the [original R algorithms](https://github.com/Bi-xuan/maximal_class).

## Installation

Requires Python 3.9 or newer and NetworkX. From this repository:

```sh
python -m pip install .
```

For development and tests (including NumPy matrix inputs):

```sh
python -m pip install -e '.[test]'
python -m pytest
```

NumPy is optional for normal use; nested Python lists work as matrices.

## Definition

Collapse each strongly connected component (SCC) into one vertex. In the
resulting condensation DAG, each source SCC defines a class: its own vertices
and every original vertex reachable from it. The complete family of these sets
is the graph's maximal-class family. Classes can overlap.

For example, edges `1 -> 3` and `2 -> 3` give the family `{1, 3}, {2, 3}`.
Cycles are allowed. Self-loops do not change the family.

## `graph_to_maximal_class()`

```python
from maximal_class_graph import graph_to_maximal_class

family = graph_to_maximal_class(
    [(1, 3), (2, 3)],
    input_format="edges",
)
assert family == (frozenset({1, 3}), frozenset({2, 3}))
```

Specify `nodes` when edge input must include isolated vertices. If supplied,
it must list all vertices, including all edge endpoints:

```python
family = graph_to_maximal_class([(1, 2)], nodes=[1, 2, 3])
assert family == (frozenset({1, 2}), frozenset({3}))
```

Matrix entry `[i, j] == 1` denotes the directed edge `i -> j`. The matrix must
be square and contain only scalar binary values. Boolean matrices also work.
The default labels are `0, ..., n-1`. Supply `nodes` to label rows and columns:

```python
mask = [
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
]
family = graph_to_maximal_class(mask, input_format="matrix", nodes=[1, 2, 3])
assert family == (frozenset({1, 3}), frozenset({2, 3}))
```

Labels can be any hashable objects except `None`. Repeated edge pairs are
collapsed, and self-loops are accepted. Inputs are not modified. The return
value is a tuple of frozensets, usable directly by the reverse function.
Class ordering follows supplied node order; otherwise inferred comparable
labels are sorted, with first appearance used for labels that cannot be sorted.
Compare families without ordering using `frozenset(family)`.

## `list_graphs_in_maximal_class()`

```python
from maximal_class_graph import list_graphs_in_maximal_class

graphs = list_graphs_in_maximal_class([{1, 3}, {2, 3}])
assert len(graphs) == 1
assert set(graphs[0].edges) == {
    (1, 1), (2, 2), (3, 3), (1, 3), (2, 3),
}
```

The input is the **entire family**, not one selected member of a larger family.
The union of its classes defines the vertex set. Every result is an independent
NetworkX `DiGraph` with exactly one self-loop at every vertex. Other edges are
enumerated exhaustively, and only graphs having exactly the input family are
returned. Class/member order is irrelevant, and graphs are not deduplicated up
to isomorphism: vertex labels matter.

To convert a result to other formats:

```python
import networkx as nx

edge_list = list(graphs[0].edges)
# Requires NumPy; nodelist specifies the matrix's row/column order.
mask = nx.to_numpy_array(graphs[0], nodelist=[1, 2, 3], dtype=int)
```

For streaming rather than building the full result list:

```python
for graph in list_graphs_in_maximal_class([{1, 2, 3}], as_iterator=True):
    assert all(graph.has_edge(node, node) for node in graph)
```

All validation, including the search limit, happens at function call time,
also in iterator mode. Exhausting the iterator gives every matching graph;
stopping iteration early gives only the graphs consumed so far.

## Enumeration limits

An edge `u -> v` is allowed only if every input class containing `u` also
contains `v`. With `p` allowed non-loop edges, there are at most `2**p`
candidates before pruning. The default limit is **32,768 candidates**,
equivalent to the original R algorithm's limit of 15 possible edges.
Self-loops are fixed and do not contribute to this count.

```python
# Permit a larger search explicitly; streaming limits retained output memory.
graphs = list_graphs_in_maximal_class(
    [{0, 1, 2, 3}], max_candidates=4096, as_iterator=True,
)
# max_candidates=None explicitly disables the guard.
```

The limit is conservative: it is checked before pruning and enumeration.
Exceeding it raises `ValueError`, including in iterator mode. A complete-list
call never returns a silently truncated result.

Graphs on 3-10 vertices can be practical for some families, but the number of
outputs itself can be enormous. With 10 vertices and one class containing them
all, there are 90 possible non-loop edges. Fixing a directed cycle still leaves
80 optional edges, giving at least `2**80` valid results. Streaming and pruning
cannot make full enumeration of such a family practical. In contrast, ten
singleton classes give one graph, consisting entirely of self-loops.

## Validation and empty graphs

Invalid edges, labels, matrices, options, and families raise `ValueError`.
Classes must be nonempty and distinct. Each must contain a vertex belonging to
no other class; otherwise the family is not realizable by a directed graph.
This condition is also sufficient: select one exclusive vertex per class and
connect it to every other vertex of that class.

The zero-vertex graph has family `()`. Enumerating that family returns a list
containing one empty graph. An empty class inside a family is invalid.

## Implementation and tests

The forward algorithm uses NetworkX's SCC condensation and source reachability,
without global variables or randomized traversal. The reverse algorithm adapts
the R edge restrictions, prunes branches whose remaining possible edges cannot
provide the required reachability from exclusive vertices, and verifies every
retained graph with the forward SCC algorithm.

The tests compare both functions against an independent DFS reachability oracle
for **all 4,166 directed graphs on zero through four labeled vertices**, ignoring
optional input self-loops. They also check matrix/edge equivalence, cyclic SCCs,
overlapping classes, self-loops, isolated vertices, streaming, invalid families,
search limits, and a sparse ten-vertex example.
