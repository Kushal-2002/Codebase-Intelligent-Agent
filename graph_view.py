"""
Turn parts of the call graph into Graphviz DOT for st.graphviz_chart.

Whole-repo graphs are unreadable (Lua has ~1,300 functions), so everything
here draws a small subgraph: the functions behind one answer, or the
neighbourhood of one function.
"""

# one fill colour per source file, light enough for dark text
FILE_COLOURS = [
    "#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#e0e7ff",
    "#ccfbf1", "#ffedd5", "#ede9fe", "#fee2e2", "#e5e7eb",
]

MAX_NODES = 60


def split_node(node):

    path, name = node.split("::", 1)

    return path, name


def escape(text):

    return text.replace("\\", "\\\\").replace('"', '\\"')


def quote(text):

    return f'"{escape(text)}"'


def to_dot(G, nodes, focus=(), rankdir="LR"):

    # nodes: which nodes to draw; edges are whatever G has between them
    # focus: nodes drawn with a thick border (the selected function, seeds)

    nodes = list(dict.fromkeys(nodes))

    files = list(dict.fromkeys(split_node(n)[0] for n in nodes))
    colour = {f: FILE_COLOURS[i % len(FILE_COLOURS)] for i, f in enumerate(files)}

    lines = [
        "digraph calls {",
        f"  rankdir={rankdir};",
        '  graph [bgcolor="transparent", nodesep=0.25, ranksep=0.5];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#64748b"];',
        '  edge [color="#94a3b8", arrowsize=0.7];',
    ]

    for node in nodes:
        path, name = split_node(node)
        file_name = path.rsplit("/", 1)[-1]
        border = ', penwidth=2.5, color="#1d4ed8"' if node in focus else ""
        # two-line label: function name, then its file
        label = f'"{escape(name)}\\n{escape(file_name)}"'

        lines.append(
            f'  {quote(node)} [label={label}, fillcolor="{colour[path]}", tooltip={quote(node)}{border}];'
        )

    drawn = set(nodes)

    for a, b in G.subgraph(drawn).edges():
        lines.append(f"  {quote(a)} -> {quote(b)};")

    lines.append("}")

    return "\n".join(lines)


def neighbourhood(G, node, depth=1, direction="both", limit=MAX_NODES):

    # breadth-first out from node; returns (nodes, truncated)

    if node not in G:
        return [node], False

    seen = [node]
    frontier = [node]

    for _ in range(depth):
        next_frontier = []

        for current in frontier:
            neighbours = []

            if direction in ("callers", "both"):
                neighbours += list(G.predecessors(current))

            if direction in ("callees", "both"):
                neighbours += list(G.successors(current))

            for n in neighbours:
                if n not in seen:
                    if len(seen) >= limit:
                        return seen, True

                    seen.append(n)
                    next_frontier.append(n)

        frontier = next_frontier

    return seen, False


def answer_graph(G, nodes, limit=MAX_NODES):

    # the functions an answer used; nodes with no edge to the others are
    # still drawn so the picture matches the "Functions used" list

    nodes = list(dict.fromkeys(nodes))[:limit]

    return to_dot(G, nodes)
