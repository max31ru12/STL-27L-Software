from collections import deque


class Graph:

    def __init__(self):
        self.__graph_dict = dict()

    def __add_neighbour(self, node: str, neighbour: str) -> None:
        if neighbour not in self.__graph_dict.keys():
            self.__graph_dict[neighbour] = [node]
        else:
            self.__graph_dict[neighbour].append(node)

    def add_edge(self, node: str, neighbour: str) -> None:
        if node not in self.__graph_dict.keys():
            self.__graph_dict[node] = [neighbour]
        else:
            self.__graph_dict[node].append(neighbour)
        self.__add_neighbour(node, neighbour)

    @property
    def graph(self):
        return self.__graph_dict

    def show_graph(self):
        return print(self.__graph_dict)


def bfs(graph: Graph, start_node: str, search_value: str):
    visited = set()
    queue = deque()
    queue.append(start_node)
    while queue:
        vertex = queue.popleft()
        if vertex == search_value:
            return vertex
        visited.add(vertex)
        for neighbour in graph.graph[vertex]:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.add(neighbour)


g = Graph()
g.add_edge('1', '2')
g.add_edge('1', '3')

g.show_graph()
