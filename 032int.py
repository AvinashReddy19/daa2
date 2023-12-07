








































#allpairs, floyd

def floyd_warshall(graph):
    n = len(graph)

    # Create a copy of the graph to store the shortest distances
    distance = [row[:] for row in graph]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] != float('inf') and distance[k][j] != float('inf'):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        print("\nThrough ",k+1," : ")
        for row in distance:
            print(row)

    return distance


n=int(input("Enter number of vertices: "))
adjacency_matrix= [[float('inf')] * n for _ in range(n)]

print("Enter the graph (enter 'inf' for infinity):")
for i in range(n):
        for j in range(n):
            while True:
                try:
                    value = input(f"Weight from vertex {i+1} to {j+1}: ")
                    if value.lower() == 'inf':
                        adjacency_matrix[i][j] = float('inf')
                    else:
                        adjacency_matrix[i][j] = int(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer or 'inf'.")
'''adjacency_matrix = [
        [0, 3, float('inf'), 7],
        [8, 0, 2, float('inf')],
        [5, float('inf'), 0,1],
        [2, float('inf'), float('inf'), 0, ]
    ]'''

print("Initial Distance Matrix:")
for row in adjacency_matrix:
        print(row)
result = floyd_warshall(adjacency_matrix)
print("\nFinal Distance Matrix:")
for row in result:
        print(row)


#bellman
def bellman_ford(adj_matrix, source):
    vertices = len(adj_matrix)
    dist = [float('inf')] * vertices
    dist[source] = 0
    for _ in range(vertices - 1):
        print("\n\nRelation iteration ",_+1,":")
        for u in range(vertices):
            for v in range(vertices):
                if adj_matrix[u][v] != 0: 
                    if dist[u] + adj_matrix[u][v] < dist[v]:
                        dist[v] = dist[u] + adj_matrix[u][v]
                    print("After relaxation of (",u+1,",",v+1,") edge: ",dist)
    for u in range(vertices):
        for v in range(vertices):
            if adj_matrix[u][v] != 0: 
                if dist[u] + adj_matrix[u][v] < dist[v]:
                    raise ValueError("Graph contains a negative cycle")

    return dist

'''graph_matrix = [
    [0, -1, 4, 0, 0],
    [0, 0, 3, 2, 2],
    [0, 0, 0, 0, 0],
    [0, 1, 5, 0, 0],
    [0, 0, 0, -3, 0]
]'''
N=int(input())
G=[]
print("Enter adjacency matrix:")
for _ in range(N):
    row=list(map(int,input().split()))
    G.append(row)
source_vertex = 0
result = bellman_ford(G, source_vertex)
for i in range(len(result)):
        print(f"Vertex {i + 1}: Distance = {result[i]}")

#krushkals
def kruskal_algo(n, graph):
    def find(component):
        if parent[component] == component:
            return component
        temp = find(parent[component])
        parent[component] = temp
        return temp

    def union(vertex1, vertex2):
        parent_of_vertex1 = find(vertex1)
        parent_of_vertex2 = find(vertex2)

        if parent_of_vertex1 == parent_of_vertex2:
            return True

        if rank[parent_of_vertex1] > rank[parent_of_vertex2]:
            parent[parent_of_vertex2] = parent_of_vertex1
        elif rank[parent_of_vertex1] < rank[parent_of_vertex2]:
            parent[parent_of_vertex1] = parent_of_vertex2
        else:
            parent[parent_of_vertex1] = parent_of_vertex2
            rank[parent_of_vertex2] += 1

        return False

    print("Minimum Spanning Tree is :-")
    print("V1", "V2", "Wt")

    ans = 0
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    edges.sort(key=lambda x: x[2])

    parent = [i for i in range(n)]
    rank = [1] * n

    for edge in edges:
        vertex1, vertex2, weight = edge
        flag = union(vertex1, vertex2)

        if not flag:
            print(vertex1, vertex2, weight)
            ans += weight

    return ans


adjacency_matrix = [[0, 28, 0, 0 ,0, 10, 0],
[28, 0, 16, 0 ,0 ,0 ,14],
[0 ,16 ,0 ,12 ,0 ,0 ,0],
[0, 0 ,12 ,0 ,22 ,0 ,18],
[0, 0 ,0 ,22 ,0 ,25, 24],
[10, 0 ,0 ,0 ,25 ,0 ,0],
[0 ,14 ,0 ,18 ,24 ,0 ,0]]

ans = kruskal_algo(7, adjacency_matrix)
print("The min cost is",ans)

#prims mst
INF = 9999999
#N = int(input("Enter number of vertices: "))

N=7
'''
N=int(input())
G=[]
print("Enter adjacency matrix:")
for _ in range(N):
    row=list(map(int,input().split()))
    G.append(row)'''

G=[[0, 28, 0, 0 ,0, 10, 0],
[28, 0, 16, 0 ,0 ,0 ,14],
[0 ,16 ,0 ,12 ,0 ,0 ,0],
[0, 0 ,12 ,0 ,22 ,0 ,18],
[0, 0 ,0 ,22 ,0 ,25, 24],
[10, 0 ,0 ,0 ,25 ,0 ,0],
[0 ,14 ,0 ,18 ,24 ,0 ,0]]
visited = [False]*N
no_edge = 0
visited[0] = True
l=[]
cost=0
row=[]
while (no_edge < N - 1):
    minimum = INF
    a = 0
    b = 0
    for m in range(N):
        if visited[m]:
            for n in range(N):
                    
                if ((not visited[n]) and G[m][n]):
                    row.append([m+1,n+1,G[m][n]])
                    if minimum > G[m][n]:
                        minimum = G[m][n]
                        a = m
                        b = n
    print("List of nodes connected to ",a+1)
    for i in row:
        if i[0]==a+1:
            print(i[1],"-",i[2])
    row=[]
    cost+=G[a][b]
    l.append(str(a+1) + "-" + str(b+1) + ":" + str(G[a][b]))
    print("Minimum edge is ",b+1,"-",G[a][b],"\n")
    visited[b] = True
    no_edge += 1
print("Edge : Weight")
for i in l:
    print(i)
print("Cost of Tree : ",cost)



#dijkstra , sssp
import heapq
def dijkstra(graph, start):
    num_vertices = len(graph)
    distance = [float('inf')] * num_vertices
    predecessor = [None] * num_vertices
    visited = [False] * num_vertices

    distance[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        dist_u, u = heapq.heappop(priority_queue)

        if visited[u]:
            continue

        visited[u] = True

        for v, weight in enumerate(graph[u]):
            if not visited[v] and weight != 0:
                new_distance = distance[u] + weight

                if new_distance < distance[v]:
                    distance[v] = new_distance
                    predecessor[v] = u
                    heapq.heappush(priority_queue, (distance[v], v))

    return distance, predecessor

def print_results(distance, predecessor, start):
    print("\nFinal Results:")
    for i in range(len(distance)):
        print(f"Vertex {i + 1}: Shortest Distance = {distance[i]}, Predecessor = {predecessor[i] + 1 if predecessor[i] is not None else None}")
    print(f"\nShortest Paths from Vertex {start + 1} to other vertices:")
    for i in range(len(distance)):
        path = [i + 1]
        current = i
        while predecessor[current] is not None:
            path.insert(0, predecessor[current] + 1)
            current = predecessor[current]
        print(f"To Vertex {i + 1}: {path}")
n = int(input("Enter the number of vertices: "))
start_vertex = int(input("Enter the start vertex (1 to N): ")) - 1
'''graph=[]
print("Enter adjacency matrix:")
for _ in range(n):
    row=list(map(int,input().split()))
    graph.append(row)'''
graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
               [4, 0, 8, 0, 0, 0, 0, 11, 0],
               [0, 8, 0, 7, 0, 4, 0, 0, 2],
               [0, 0, 7, 0, 9, 14, 0, 0, 0],
               [0, 0, 0, 9, 0, 10, 0, 0, 0],
               [0, 0, 4, 14, 10, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 1, 6],
               [8, 11, 0, 0, 0, 0, 1, 0, 7],
               [0, 0, 2, 0, 0, 0, 6, 7, 0]
               ]
print("\nGraph:")
for row in graph:
        print(row)
distances, predecessors = dijkstra(graph, start_vertex)
print_results(distances, predecessors, start_vertex)
