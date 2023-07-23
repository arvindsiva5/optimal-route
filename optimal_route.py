import math  # math module imported to use infinity value

# Problem 1

class Graph:
    """
    A data structure that represents a weighted directed graph using Adjacency List.
    Uses O(V + E) space, where V is number of vertices & E is number of edges
    """
    def __init__(self, s: int) -> None:
        """
        Function description: Initializes an Graph object. Creates an adjacency list
                              representing the graph.

        :Input:
            s: An integer representing size of the graph (number of vertices)
        :Output:
            No output
            A graph of size s is created
        :Time complexity: O(s), where s is the size of the graph (number of vertices)
        :Aux space complexity: O(s), where s is the size of the graph (number of vertices)
        """
        # initialize vertices with a list of s empty lists
        self.vertices = [[] for i in range(s)]

    def insert_edge(self, v: int, u: int, w: int) -> None:
        """
        Function description: Adds a weighted edge to the graph

        :Input:
            v: An integer representing start vertex of an edge
            u: An integer representing end vertex of an edge
            w: An integer representing weight of the edge
        :Output:
            No output
            An edge (v, u) of weight w is added to the graph
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # at vertices[v] a tuple (u, w) is appended, (u,v) is an edge with weight w
        self.vertices[v].append((u, w))

    def get_edges(self, v: int) -> list[int]:
        """
        Function description: Returns a list of directed edges that start from vertex v

        :Input:
            v: An integer representing the vertex
        :Output:
            A list of tuples of 2 integer (i, j) , where i is a vertex adjacent to v
            , where j is the weight of edge (v, i)
        :Time complexity: O(1)
        :Aux space complexity: O(E), where E is the number of adjacent vertices to v
        """
        # list of adjacent vertices and the weights of the edges returned
        return self.vertices[v]

    def __len__(self) -> None:
        """
        Function description: Returns the number of vertices in the graph (size of graph)

        :Input:
            No input
        :Output:
            An integer that represnt the size of the graph (number of vertices in graph)
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return len(self.vertices)

class PriorityQueue:
    """
    A priority queue data structure implemented using min-heap data structure.
    Uses keys of elements to ensure that the queue always satisfy min-heap property
    Uses O(n) space, where n is the number of elements in the queue
    """
    def __init__(self, n: int, keys: list[int]) -> None:
        """
        Function description: Initialize a PriorityQueue object. Takes in a list of integers
                              and its corresponding keys and heapify based on key values

        :Input:
            n: An integer representing the number of values in a list 1...n
            keys: A list of integers that are keys (values) for the numbers in the
                  list of size n with 1...n values. Every i-th key is the value for
                  the i-th in the list 1...n values that is used in heapifying the list.
        :Output:
            No output
            A queue is created and all the values 1..n are heapified
        :Time complexity: O(n), the number of elements in queue 1...n
        :Aux space complexity: O(n), the number of elements in queue 1...n
        """

        # tuples (i, keys[i]), where i = 0...n-1 is added
        self.queue = [None] + [(i, keys[i]) for i in range(n)]

        # stores index of i in the queue
        self.queue_key_index = [i for i in range(1, n + 1)]

        # queue is heapified to satisfy min-heap property
        self.heapify()

    def rise(self, i: int) -> None:
        """
        Function description: Move an element higher up the queue until it satisfies the min-heap property

        :Input: 
            i: An integer representing the element in the queue
        :Output: 
            No output
            i is moved higher up in the heap represented by queue until it satisfies the min-heap property
        :Time complexity: O(log(n)), the number of elements in queue 1...n
        :Aux space complexity: O(1)

        Reference: Refered FIT2004 Course Notes
        """
        parent = i // 2 # find parent index

        # move up queue until root node at top of queue
        while parent >= 1:
            # executed if parent key value higher than i (child) key value
            if self.queue[parent][1] > self.queue[i][1]:

                # values of parent and child stored
                parent_val = self.queue[parent]
                i_val = self.queue[i]

                # parent and child swapped to satisfy min-heap property
                self.queue[parent], self.queue[i] = i_val, parent_val

                # queue_key_index updated after swap
                (
                    self.queue_key_index[parent_val[0]],
                    self.queue_key_index[i_val[0]],
                ) = (i, parent)

                # i and parent updated to move up min-heap
                i = parent
                parent = i // 2

            # executed if parent key value lower or equal than child key value
            # min-heap property satisfied
            else:
                break

    def fall(self, i: int) -> None:
        """
        Function description: Move an element down the queue until it satisfies the min-heap property

        :Input: 
            i: An integer representing the element in the queue
        :Output: 
            No output
            i is moved down the heap represented by queue until it satisfies the min-heap property
        :Time complexity: O(log(n)), the number of elements in queue 1...n
        :Aux space complexity: O(1)

        Reference: Refered FIT2004 Course Notes
        """
        child = 2 * i  # find child of i
        n = len(self.queue) - 1  # index of last element in queue

        # move down queue until reach last element
        while child <= n:
            if child < n and self.queue[child + 1][1] < self.queue[child][1]:
                child += 1

            # executed if key of value of child is lesser than i (parent)
            if self.queue[i][1] > self.queue[child][1]:

                # values of parent and child stored
                i_val = self.queue[i]
                child_val = self.queue[child]

                # parent and child swapped to satisfy min-heap property
                self.queue[i], self.queue[child] = child_val, i_val

                # queue_key_index updated after swap
                (
                    self.queue_key_index[i_val[0]],
                    self.queue_key_index[child_val[0]],
                ) = (child, i)

                # i and child updated to down up min-heap
                i = child
                child = 2 * i

            # executed if child key value higher or equal to parent key value
            # min-heap property satisfied
            else:
                break

    def heapify(self) -> None:
        """
        Function description: Converts queue list to min-heap based on the keys stored in queue_key_index list

        :Input: 
            No input
        :Output: 
            No output
            queue is converted to a to min-heap based on the keys stored in queue_key_index list
        :Time complexity: O(n), the number of elements in queue 1...n
        :Aux space complexity: O(1)

        Reference: Refered FIT2004 Course Notes
        """

        # number of elements in queue is retrieved
        n = len(self.queue) - 1

        # every parent node in the heap is moved down the heap until it satisfies min-heap property based on keys
        for i in range(n // 2, 0, -1):
            self.fall(i)

    def update(self, v: int, k: int) -> None:
        """
        Function description: Move an element higher up the queue until it satisfies the min-heap property
                              based on the new updated key value

        :Input: 
            v: An integer representing the element in the queue
            k: An integer representing the key for the element v
        :Output: 
            No output
            v is moved higher up in the heap represented by queue until it satisfies the min-heap property
            based on the key value k
        :Time complexity: O(log(n)), the number of elements in queue 1...n
        :Aux space complexity: O(1)

        Reference: Refered FIT2004 Course Notes
        """
        # the index of element v in queue is retrieved
        i = self.queue_key_index[v]

        # the new key value k is updated for element v
        self.queue[i] = (v, k)

        # element v is moved higher up till it satisfies min-heap property based on key k
        self.rise(i)

    def extract_min(self) -> int:
        """
        Function description: Removes and returns the integer in queue (min-heap) that has the minimum 
                              key value in queue_key_index
        
        :Input:
            No input
        :Output:
            An integer that is an element in n queue (min-heap) that has the minimum key value in queue_key_index
        :Time complexity: O(log(n)), the number of elements in queue 1...n
        :Aux space complexity: O(1)

        Reference: Refered FIT2004 Course Notes
        """

        # Exception raised if there is no elements in the queue
        if len(self.queue) == 1:
            raise Exception
        
        # removes the element in queue and return it if only 1 element in queue
        elif len(self.queue) == 2:
            min = self.queue[1][0]
            self.queue.pop(len(self.queue) - 1)
            return min
        
        # removes the element in queue with min key value and returns it
        else:
            min = self.queue[1][0]
            last = self.queue.pop(len(self.queue) - 1)

            # assigns last element to 1st position where element with min key value is (min heap property)
            self.queue[1] = last
            self.queue_key_index[last[0]] = 1

            # moves down the last element assigned till it satisfies min-heap property based on the key
            self.fall(1)
            return min

    def is_empty(self) -> bool:
        """
        Function description: Checks if the queue is empty

        :Input:
            No input
        :Output:
            An boolean that is True if there are no elements stored in the queue else returns False
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # the length of queue is minus 1 so that None in 1st index is not included
        return len(self.queue) - 1 == 0

def optimalRoute(
    start: int, end: int, passengers: list[int], roads: list[(int, int, int, int)]
) -> list:
    """
    Function description: Finds the route to take from the start to end location via the provided roads
                          (roads have shorter time or same time to travel if carpooling)
                          in the city with optional picking up passengers to carpool at locations with 
                          passengers that takes the shortest amount of time.

    Approach description: 
    Since the city is made of roads and locations connected by the routes they seem to form a network so
    the problem is modelled as a graph. For every road there will be two different time taken x, y where
    x is if the user is carpooling and y if the user is not carpooling. Some locations in the city have
    passengers that can be picked up optionally to carpool. This problem situation can be modelled by a
    directed weighted graph. In the graph every location is reprsented as a vertex and every road connecting
    the locations is an edge. So if L locations and R roads the graph has L vertices and R edges. However, 
    each road have diffrent times based on carpooling so we need different weights so we add another 
    L vertices representing locations and another R edges representing roads to connect the L vertexes that
    we just added. And the weights of the R edges we just added to new L locations is the integer reprsented
    by time taken if user is carpooling on the road. The weight of the edges of the first R edges added earlier
    will be the time taken if user is not carpooling on road. Now, we have two connected graphs that are not 
    connected to each other. Each connected graph has the L locations and R roads so we have two copies of each
    location and road in two separate connected graphs, G1 for not carpooling and G2 for carpooling. 
    To give the user the option to pick up passengers at locations with passengers, we need to connect G1 and G2.
    If at location x there is a passenger, the vertex representing x in G1 is connected to vertex representing x
    in G2 with an edge from ([x, G1], [x, G2]) with weight 0. Weight 0 is used because picking up passenger does
    not take extra time. Doing so for all locations where passengers can be picked up will connect G1 and G2 
    forming a single graph G. So with these graph the user can only access edges with carpooling time weigths only
    if they pick up passenger and pass by an edge with weight 0.

    Now with the graph, we can run dijkstra algorithm to find the path from start (source vertex) to other vertex
    that takes the shortest time. Once the dijkstra algorithm is run, there will be two options to reach the 
    end location which is by picking up passengers or not, the one with the shortest time is chosen and using the
    predescesors returned by dijkstra algorithm the route can be computed and this route which will be returned
    is the shortest time taken to reach from start to end.

    The time complexity would be O(R*log(L)), where R is the number of tuples (roads) in the input list
    & L is the number of locations. This graph constructed would have (2L + P) vertices (loaction) and
    (2R + P) edges (roads). Here the roads and locations represent a connected graph because for any two locations
    there is a route and the start and end locations have no passengers for pick up in a. In this case if the 
    graph has n vertex it has a minimum of n-1 edge and since start and end have no passengers the maximum extra
    p edge will be n-2 which is lesser than n-1 edge. So we can say that R dominates P. Hence the space used by
    the graph would be (L+R) where L locations and R roads. The graph uses adjacency lists making the space used to be
    (L+R). Dijkstra algorithm has time complexity of O(E*log(V)) where E is edges and V is vertices. 
    So dijkstra will run in O(R*log(L)) for the graph built. From this the time complexity of the function 
    is O(R*log(L)) and it overrides O(R) from count_location, add_routes and O(L) from graph creation & fastest_route. 
    Since the extra space created is for the graph of O(L+R) dominates O(L) for count_locations & dijkstra, 
    the Auxilary space complexity is O(L+R).

    :Input:
        start: An integer representing the starting location
        end: An integer representing the ending location
        passengers: A list of integers where the integers represent locations where a passenger can
                    be picked up
        roads: A list of tuples (a, b, c, d) representing roads where
            a: start, b: end, c: travel time if not carpooling, d: carpooling travel time
    :Output:
        A list of integers representing the roads that could be taken with optionally picking up passengers
        from start to end in the shortest amount of time
    :Time complexity: O(R*log(L)), where R is the number of tuples (roads) in the input list
                                 , where L is the number of locations
    :Aux space complexity: O(R + L), where R is the number of tuples (roads) in the input list
                                   , where L is the number of locations
    """

    
    l = count_location(roads)   # count the number of locations
    g = Graph(l * 2)    # create the graph

    # add edges to graph based on the paths and locations where passenger can be picked up
    # weight of edges based on carpooling or not and location with passengers
    g = add_routes(g, l, roads, passengers)

    # perform djikstra algorithm to find shorted path from start to all other locations
    dist, pred = dijkstra(g, start)

    # find the roads to take from start to end with or without pick up passengers in shortest time
    return fastest_route(start, end, dist, pred, l)

def count_location(roads: list[(int, int, int, int)]) -> int:
    """
    Function description: Returns the number of locations contained in the list of roads provided
                          by finding the maximum integer representing the locations and adding 1 
                          because the locations are 0...n-1 so by adding 1 to the maximum n-1
                          we get n locations which is reprsented by the roads.

    :Input:
        roads: A list of tuples (a, b, c, d) representing roads where
            a: start, b: end, c: travel time if not carpooling, d: carpooling travel time
    :Output:
        An integer representing the number of locations in the list of roads
    :Time complexity: O(R), where R is the number of tuples (roads) in the input list
    :Aux space complexity: O(1)
    """
    locations = -math.inf

    for r in roads:

        # assign start or end of road to locations if they are larger than locations
        if r[0] > locations: 
            locations = r[0]
        if r[1] > locations:
            locations = r[1]

    # add 1 because locations are 0...n-1
    # adding 1 makes locations correctly store n locations
    locations += 1

    return locations


def add_routes(
    g: Graph, l: int, roads: list[(int, int, int, int)], passengers: list[int]
) -> Graph:
    """
    Function description: Creates a graph representing all the roads, locations and locations where
                          passengers can be picked up by adding edges to a graph based on the 
                          roads, the time taken on the roads if carpooling or not and the locations
                          where passengers can be picked up

    :Input:
        g: A Graph that is represented using adjacency list
        l: An integer representing the number of locations
        roads: A list of tuples (a, b, c, d) representing roads where
            a: start, b: end, c: travel time if not carpooling, d: carpooling travel time
        passengers: A list of integers where the integers represent locations where a passenger can
                    be picked up
    :Output:
        A Graph that is represented using adjacency list. The graph is a representation of the roads
        connecting the locations and locations where passengers can be picked up.
    :Time complexity: O(R), where R is the number of tuples (roads) in the input list
    :Aux space complexity: O(R + L), where R is the number of tuples (roads) in the input list
                                   , where L is the number of locations represented by l
    """

    # insert an edge from start to end for every route (represent non carpool road)
    # insert an edge from start + l to end + l for every route (represent carpool road)
    # the weights are time taken for the roads (based on carpooling or not)
    for r in roads:
        g.insert_edge(r[0], r[1], r[2]) # not carpool
        g.insert_edge(r[0] + l, r[1] + l, r[3]) # carpool

    # insert edge (weight 0) for locations with passenger, p from p to p+l 
    # (reprsent passenger is picked up) so that can use edges that reprsent carpool road after
    for p in passengers:
        g.insert_edge(p, p + l, 0) # pick up passengers

    return g    # return graph


def dijkstra(g: Graph, s: int) -> list[list[int], list[int]]:
    """
    Function description: Runs dijkstra algorithm on a non negative weighted directed graph, g to find the 
                          shortest path from source vertex, s to any other vertex in the graph.
                          Dijkstra algorithm works by selecting vertices in a graph based on shortest distances
                          from s and rlaxes all the adjacent vertices. When relaxing teh adjacent vertices, their
                          distances are update if their edge weight + predescsesor vertex has a shorter distance.
                          If the distance was updated when relaxing the predescesor of the vertex and priority 
                          queue used to select the vertices to visit are also updated. By doing so the shortest
                          distance from source to any other vertex is calculated and the paths can be computed
                          using their predescseors. This makes use of the greedy approach by visiting every vertex 
                          based on shortest distance from s to make local optimum descisions.

    :Input:
        g: A Graph that is represented using adjacency list
        s: An integer representing the starting location (source vertex)
    :Output:
        A list with two lists of integers, [a, b],
            where a is a list of shortest distances for each vertex in g from s,
            where b is a list of predescesor vertex for each of the vertex in g to achieve the shortest
            distance from s to the vertex
    :Time complexity: O(R*log(L)), where R is the number of edges (roads) in g
                                 , where L is the number of locations (vertex) in g
    :Aux space complexity: O(L), where L is the number of locations (vertex) in g

    Reference: Refered FIT2004 Course Notes
    """
    n = len(g) # number of vertex in graph
    dist = [math.inf for _ in range(n)]  # distance of vertex from source vertex
    pred = [None for _ in range(n)]  # predescesor of vertex (use shortes path)
    dist[s] = 0 # source vertex has 0 distance

    # create priority queue with distances of the vertices as keys
    queue = PriorityQueue(n, dist)

    # loops all vertices in graph based on priority (smallest distances)
    while not queue.is_empty():

        # remove vertex with minimum distance from source
        p = queue.extract_min()

        # relax all adjacent vertices to vertex p
        for edge in g.get_edges(p):
            dijkstra_relax(p, edge[0], edge[1], dist, pred, queue)

    return (dist, pred) # return dist & pred


def dijkstra_relax(
    p: int, v: int, w: int, dist: list[int], pred: list[int], queue: PriorityQueue
):
    """
    Function description: Updates vertex v with the shortest distance from the source vertex in a graph.
                          This is done by updating the distance of v if the distance from source to predescesor 
                          vertex p from source + the weight of the edge between p and v is lesser than the 
                          distance of v from the source. If an update is made the predesecsor of vertex v is
                          updated to vertex p. If an update to distance v is made the priority queue,
                          queue is also updated with the latest distance of v so that it satisfies min-heap
                          property.

    :Input:
        p: An integer representing the predecessor vertex of an edge
        v: An integer representing the child vertex of the edge
        w: An integer representing the weight of an edge
        dist: A list of integers representing shortest distances for each vertex in graph from source vertex
        pred: A list of integers representing predescesor vertex for each of the vertex in graph to achieve the
              shortest distance from source vertex
        queue: A PriorityQueue used to determine which vertex should be visited next by djikstra algorithm
               based on the smallest distances of the vertex from the source vertex
    :Output:
        No output given
    :Time complexity: O(log(L)), where L is the number of locations in queue
    :Aux space complexity: O(1)

    Reference: Refered FIT2004 Course Notes
    """

    # update distance vertex v if new shorter distance from vertex found
    # execute if distance of predescessor vertex p from source + weight is smaller 
    # than current distance vertex v from source
    if dist[v] > dist[p] + w:
        dist[v] = dist[p] + w
        pred[v] = p

        # update new distance for vertex v in priority queue to update and satisfy min-heap property
        queue.update(v, dist[v])


def fastest_route(
    start: int, end: int, dist: list[int], pred: list[int], l: int
) -> list[int]:
    """
    Function description: Finds the roads to take from start to end in the city optionally picking up 
                          passengers. Using dist finds out if the path with picking up passengers or
                          without picking up passenger is fastest and select the fastest one. Once selected,
                          the predescessors of every vertex, pred is used to compute the path from end to
                          start in reverse then the path is reversed to obtain the actual fastest path and is
                          returned.

    :Input:
        start: An integer representing the starting location
        end: An integer representing the ending location
        dist: A list of integers representing shortest distances for each vertex in graph from source vertex
        pred: A list of integers representing predescesor vertex for each of the vertex in graph to achieve the
              shortest distance from source vertex
        l: An integer representing the number of locations
    :Output:
        A list of integers representing the shortest path that can be taken from start to end
    :Time complexity: O(L), where L is the number of locations
    :Aux space complexity: O(L), where L is the number of locations
    """

    # the current vertex being traversed
    # chooses the fastest route (pick up passenger or not)
    current = end if dist[end] < dist[end + l] else end + l
    route = []   # the fastest route

    # traverse the graph until reach start of route
    while current != start:
        # append the predescesor vertex to route if route is not empty
        if route:
            # retrieve the last vertex in path
            last = route[(len(route) - 1)]

            # the locations with passenger has two vertex with edge weight 0
            # this makes sure such edges are not considered in the final route
            if last != current % l:
                route.append(current % l)

        # append current vertex if route is empty
        else:
            route.append(current % l)

        # move to predescesor of the current vertex
        current = pred[current]

    # start was not added in the loop so it is appended
    route.append(start)

    # the route found was in reverse so the route is reversed to get correct route
    route.reverse()
    return route    # route is returned
