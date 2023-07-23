optimalRoute(start, end, passengers, roads) is a function that finds the route to take from the start to end location via the provided roads (roads have shorter time or same time to travel if carpooling) in the city with optional picking up passengers to carpool at locations with passengers that takes the shortest amount of time.

\# Example\
start = 0\
end = 4\
\# The locations where there are potential passengers\
passengers = [2, 1]\
\# The roads represented as a list of tuple\
\# For each tuple (a, b, c, d) where a is start point, b is end point, c is time taken without carpool & d is time taken with carpool\
roads = [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10), (2, 4, 30, 25), (2, 0, 2, 2), (0, 1, 10, 10), (1, 4, 30, 20)]\
\# function should return the optimal route (which takes 27 minutes).\
\>>> optimalRoute(start, end, passengers, roads)\
[0, 3, 2, 0, 3, 4]

