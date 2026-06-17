from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import networkx as nx
import pickle
import os

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

call_graph = {}
repository_functions = set()

function_to_nodes = {}


def find_calls(node):

    calls = []

    def dfs(n):

        if n.type == "call_expression":
            function_node = n.child_by_field_name("function")

            if function_node:
                calls.append(function_node.text.decode("utf8"))

        for child in n.children:
            dfs(child)

    dfs(node)

    return calls


def collect_functions(node):

    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")

        if declarator:
            function_name = declarator.text.decode("utf8")

            function_name = function_name.split("(")[0]

            repository_functions.add(function_name)

            node_id = f"{path}::{function_name}"

            if function_name not in function_to_nodes:
                function_to_nodes[function_name] = []

            function_to_nodes[function_name].append(node_id)

    for child in node.children:
        collect_functions(child)


def traverse(node, path):

    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")

        if declarator:
            function_name = declarator.text.decode("utf8")

            short_name = function_name.split("(")[0]

            current_node = f"{path}::{short_name}"

            calls = find_calls(node)

            filtered_calls = []

            for call in calls:
                call = call.split("(")[0]

                if call in function_to_nodes:
                    local_node = f"{path}::{call}"

                    if local_node in function_to_nodes[call]:
                        filtered_calls.append(local_node)

                    else:
                        for target_node in function_to_nodes[call]:
                            filtered_calls.append(target_node)

            filtered_calls = list(set(filtered_calls))

            call_graph[current_node] = filtered_calls

    for child in node.children:
        traverse(child, path)


# first pass for getting the repository functions so that we can differentiate from the
# library functions


for root, dirs, files in os.walk("repository"):
    if ".git" in root:
        continue

    for file in files:
        if not file.endswith(".cpp"):
            continue

        path = os.path.join(root, file)

        with open(path, "r", errors="ignore") as f:
            code = f.read()

            tree = parser.parse(bytes(code, "utf8"))

            root_node = tree.root_node

            collect_functions(root_node)


# we can build the build the graph once we have the repo functions

for root, dirs, files in os.walk("repository"):
    if ".git" in root:
        continue

    for file in files:
        if not file.endswith(".cpp"):
            continue

        path = os.path.join(root, file)

        with open(path, "r", errors="ignore") as f:
            code = f.read()

            tree = parser.parse(bytes(code, "utf8"))

            root_node = tree.root_node

            traverse(root_node, path)

print("\nCALL GRAPH\n")

for func, calls in call_graph.items():
    print(func)

    for c in calls:
        print("   ->", c)

    print()


G = nx.DiGraph()

for func, calls in call_graph.items():
    for call in calls:
        G.add_edge(func, call)


with open("call_graph.pkl", "wb") as f:
    pickle.dump(call_graph, f)

with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)

for func in call_graph:
    if "main" in func:
        print(func)

print(list(G.predecessors("repository/client.cpp::connect_to_addr")))

print(list(G.successors("repository/client.cpp::connect_any_tracker")))
