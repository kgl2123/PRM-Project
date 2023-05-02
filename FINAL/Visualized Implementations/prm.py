# prm.py
# This program runs the vanilla PRM algorithm
#  
# Authors: Kevin Grant Li, Avigayil Helman, Kevin Li, Joseph Han,  Nicole Neil
# 
# Adapted from code from Professor Brian Plancher and Avigayil Helman
# Adapted from code written by Steve LaValle and Rus Tedrake

import sys, random, math, pygame, numpy
from pygame.locals import *
from util import Util
import heapq
from decimal import Decimal, getcontext
import timeit
import time

class prm:
    def __init__(self, obstacles, start_node, goal_node, XDIM = 640, YDIM = 480, NUM_SAMPLES = 500, MAX_ITER = 100000, LINE_WIDTH = 20, TEST_MODE = 0, K = 5):
        
        self.obstacles = obstacles # Obstacles are represented by a list of points defining line segments 
        self.LINE_WIDTH = 20 # width of obstacle lines (projected out)
        self.XDIM = XDIM # board dimmension -> x-dimension 
        self.YDIM = YDIM # board dimmension -> y-dimension
        self.TEST_MODE = TEST_MODE # do not wait for user to exit and return the final result
        self.start_node = start_node # (x,y) position of starting location  
        self.goal_node = goal_node # (x,y) position of goal location 
        self.NUM_SAMPLES = NUM_SAMPLES # total number of samples we will make within the plane 
        self.K = K # number of nearest neighbors for building out graph
        self.U = Util()

    def runGame(self):
        # initialize and prepare screen with 

        print("RUNNING STANDARD PRM")
        
        pygame.init()
        screen = pygame.display.set_mode([self.XDIM, self.YDIM])
        pygame.display.set_caption('PRM Project')
        white = 255, 240, 200
        black = 20, 20, 40
        red = 255, 0, 0
        blue = 0, 0, 255
        green = 0, 255, 0
        hotpink = 255, 153, 255
        babyblue = 204, 229, 255

        screen.fill(black)
        pygame.draw.circle(screen,babyblue, self.start_node, 8)
        pygame.draw.circle(screen,blue,self.goal_node, 8)

        new_obstacles = self.redefineObstacles(self.obstacles, self.LINE_WIDTH)
        for o in new_obstacles:
            pixel = pygame.draw.polygon(screen, red, o)
            pygame.display.update()

        # start the list of nodes --> make sure to include both the start state and the end state within this list! 
        nodes = set()
        parents = {}
        wastes = 0

        # init our util function object
        u = Util()

        # first select a given number of location samples [(x,y) coordinates] within the grid! 
        grid_samples = self.sample_grid({self.start_node, self.goal_node}, self.XDIM, self.YDIM, self.NUM_SAMPLES) 

        # # visualize all samples generated 
        # for sample in grid_samples: 
        #       pygame.draw.circle(screen,green,sample,4)

        non_obstructed_samples = self.remove_obstacle_samples(grid_samples,self.obstacles,self.LINE_WIDTH,u)

        # visualize all non_obstructed_samples generated 
        for sample in non_obstructed_samples: 
              pygame.draw.circle(screen,green,sample,3)
              pygame.display.update()

        nodes = self.buildNodes(non_obstructed_samples,self.K) 

        self.identifyNearestNeighbors(nodes,self.K)

        # this builds all edges -> includes both edges obstructed and unobstructed 
        edges = self.buildEdges(nodes)

        # remove obstructed edgers
        unobstructedEdges = self.removeObstructedEdges(edges, new_obstacles, self.LINE_WIDTH)

        # this the final graph
        for edge in unobstructedEdges: 
            pygame.draw.lines(screen, white, 0, edge.get_tuple(),2)
            pygame.display.update()


        # identify the starting and ending node
        for node in nodes: 
            if node.access_xy() == self.start_node: 
                starting_node = node 
            elif node.access_xy() == self.goal_node: 
                endgoal_node = node
        
        print("Starting @ " + str(starting_node.access_xy()))
        print("Ending @ " + str(endgoal_node.access_xy()) + "\n")

        # find the shortest path 
        shortest_path = self.dijkstra(starting_node,endgoal_node,unobstructedEdges)

        # print out the shortest path
        self.print_shortest_path(shortest_path)

        # if there is a shortest path -> draw the result
        if shortest_path != None: 
            for i in range(1,len(shortest_path)): 

                nodeprev_location = shortest_path[i-1].access_xy()
                nodenext_location = shortest_path[i].access_xy()
                
                pygame.draw.line(screen,babyblue,nodeprev_location,nodenext_location,5)
                pygame.display.update()

                #pygame.time.wait(1000)

                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")

        if (self.TEST_MODE):
            path = [nodes[-1]]
            node = nodes[-1]
            while node != nodes[0]:
                node = parents[node]
                path.append(node)
            return path
        # Else wait for the user to see the solution to exit
        else:
            while(1):
                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")

    def runGame_benchmark(self):
        # becnhmark __runGame_test

        #random.seed(1)

        n = 100 # num runs

        total_time = 0
        sample_total_time = 0
        knn_total_time = 0
        build_edge_total_time = 0
        check_edge_total_time = 0
        search_total_time = 0


        print("starting test")
        for i in range(n):
            ########################
            ## INNER runGame Func ##
            ########################

            # start the list of nodes --> make sure to include both the start state and the end state within this list!
            nodes = set()
            parents = {}
            wastes = 0

            # init our util function object
            u = self.U
            new_obstacles = self.redefineObstacles(self.obstacles, self.LINE_WIDTH)

            start_time = timeit.default_timer()
            grid_samples = self.sample_grid({self.start_node, self.goal_node}, self.XDIM, self.YDIM, self.NUM_SAMPLES)
            non_obstructed_samples = self.remove_obstacle_samples(grid_samples, self.obstacles, self.LINE_WIDTH, u)
            nodes = self.buildNodes(non_obstructed_samples, self.K)
            sample_total_time += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            self.identifyNearestNeighbors(nodes, self.K)
            knn_total_time += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            edges = self.buildEdges(nodes)
            build_edge_total_time += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            unobstructedEdges = self.removeObstructedEdges(edges, new_obstacles, self.LINE_WIDTH)
            check_edge_total_time += timeit.default_timer() - start_time

            for node in nodes:
                if node.access_xy() == self.start_node:
                    starting_node = node
                elif node.access_xy() == self.goal_node:
                    endgoal_node = node

            start_time = timeit.default_timer()
            shortest_path = self.dijkstra(starting_node, endgoal_node, unobstructedEdges)
            search_total_time += timeit.default_timer() - start_time

            ########################
            ########  END  #########
            ########################

        print("total avg time: ", (sample_total_time + knn_total_time + build_edge_total_time + check_edge_total_time + search_total_time)/n)
        print("sampling avg time: ", sample_total_time / n)
        print("knn avg time: ", knn_total_time / n)
        print("edge building avg time: ", build_edge_total_time / n)
        print("check edge collision avg time: ", check_edge_total_time / n)
        print("djikstra avg time: ", search_total_time / n)

    # samples the grid for (x,y) locations (these are potential nodes)
    def sample_grid(self, nodes, XDIM, YDIM, NUM_SAMPLES): 

        # random.seed(1234)

        samples = nodes.copy()

        i = 2 
        while (i < NUM_SAMPLES): 
            sample = (random.random()*XDIM), (random.random()*YDIM)
            if sample not in samples: 
                samples.add(sample)
                i+=1 
            else: 
                continue 
        
        return samples

    # removes samples that are obstructed (within the obstacles)
    def remove_obstacle_samples(self,grid_samples,obstacles,obstacle_width, u): 

        non_obstructed_samples = set()

        for sample in grid_samples: 
            if u.isCollisionFree(obstacles,sample,obstacle_width):
                non_obstructed_samples.add(sample)
            else: 
                continue 

        return non_obstructed_samples  

    # builds all nodes from a list of (x,y) tuples
    def buildNodes(self,samples,K):
        nodes = [] 
        for sample in samples: 
            nodes.append(Node(sample,K))
        return nodes 

    # updates each node with its k-nearest neighbors
    def identifyNearestNeighbors(self, nodes, K): 
        for node in nodes: 
            node.find_nearest_neighbors(nodes, K)

    # builds all edges -> edges are connections from a node to its k-nearest neighbors
    def buildEdges(self,nodes): 
        edges = list()

        for node1 in nodes: 
            for node2 in node1.k_nearest_neighbors: 
                potential_edge = Edge(node1,node2)
                if self.checkExistingEdges(edges,potential_edge): 
                    edges.append(potential_edge)

        return edges
    
    # check if an edge already exist
    def checkExistingEdges(self,edges,potential_edge): 
        if len(edges) == 0: 
            return True
        
        for edge in edges: 
            if edge.check_equivalent(potential_edge): 
                return False

        return True 

    # redefine obstacles as a list of rects
    def redefineObstacles(self, obstacles, obstacle_width):
        new_obstacles = []
        for key in obstacles.keys():
            obstacle = obstacles[key]
            for i in range(len(obstacle)-1):
                j = i + 1

                rect_points = [None] * 4

                if (obstacle[i][0] == obstacle[j][0]):
                    x1 = obstacle[i][0] - obstacle_width / 2
                    x2 = obstacle[i][0] + obstacle_width / 2
                    y1 = obstacle[i][1]
                    y2 = obstacle[j][1]

                    if y1 > y2:
                        y1, y2 = y2, y1

                    rect_points[0] = (x1, y1)
                    rect_points[1] = (x2, y1)
                    rect_points[2] = (x2, y2)
                    rect_points[3] = (x1, y2)

                elif (obstacle[i][1] == obstacle[j][1]):
                    x1 = obstacle[i][0]
                    x2 = obstacle[j][0]
                    y1 = obstacle[i][1] - obstacle_width / 2
                    y2 = obstacle[i][1] + obstacle_width / 2

                    if x1 > x2:
                        x1, x2 = x2, x1

                    rect_points[0] = (x1, y1)
                    rect_points[1] = (x2, y1)
                    rect_points[2] = (x2, y2)
                    rect_points[3] = (x1, y2)
                else:
                    # https://stackoverflow.com/questions/1250419/finding-points-on-a-line-with-a-given-distance#:~:text=Slope%20m%20is%20just%20the,*%20(m%20*%20s)).
                    m = -1 / ((obstacle[i][1] - obstacle[j][1]) / (obstacle[i][0] - obstacle[j][0]))
                    s = (obstacle_width / 2) / math.sqrt(1 + m * m)

                    rect_points[0] = (obstacle[i][0] - s, obstacle[i][1] - m * s)
                    rect_points[1] = (obstacle[i][0] + s, obstacle[i][1] + m * s)
                    rect_points[2] = (obstacle[j][0] + s, obstacle[j][1] + m * s)
                    rect_points[3] = (obstacle[j][0] - s, obstacle[j][1] - m * s)

                new_obstacles.append(rect_points)

        return new_obstacles

    # iterates through all edges and checks for each edge, if they collide with an obstacle
    # returns only unobstructed edges
    def removeObstructedEdges(self,edges,obstacles,obstacle_width): 

        unobstructedEdges = list()

        for edge in edges: 

            obstructed = False

            for obstacle in obstacles:

                line_segment_point1 = edge.node1.access_xy()
                line_segment_point2 = edge.node2.access_xy()

                if self.line_segment_intersects_rectangle(line_segment_point1, line_segment_point2, obstacle):
                    obstructed = True
                    continue

            if obstructed == False:
                unobstructedEdges.append(edge)
        
        return unobstructedEdges

    # alternative method of checking edge collision
    def line_intersection(self,line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if denominator == 0:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
        
        return False


    def line_segment_intersects_rectangle(self,U, V, rectangle):
        A1, A2, A3, A4 = rectangle
        
        sides = [(A1, A2), (A2, A3), (A3, A4), (A4, A1)]
        UV = (U[0], U[1], V[0], V[1])
        
        for side in sides:
            XY = (side[0][0], side[0][1], side[1][0], side[1][1])
            if self.line_intersection(UV, XY):
                return True
        
        return False


    # perform dijkstra's algorithm from start_node to end_node to find shortest path
    def dijkstra(self, start_node, end_node, edges):
        # Set up data structures
        node_to_distance = {start_node: 0}
        node_to_previous = {start_node: None}
        heap = [(0, start_node)]

        # Perform Dijkstra's algorithm
        while heap:
            (distance, node) = heapq.heappop(heap)
            if node == end_node:
                # We've found the shortest path; build the list of nodes and return it
                path = []
                while node:
                    path.append(node)
                    node = node_to_previous[node]
                return path[::-1]

            for edge in edges:
                if edge.node1 == node and edge.engaged:
                    neighbor = edge.node2
                elif edge.node2 == node and edge.engaged:
                    neighbor = edge.node1
                else:
                    continue
                
                neighbor_distance = node_to_distance[node] + edge.weight
                if neighbor not in node_to_distance or neighbor_distance < node_to_distance[neighbor]:
                    node_to_distance[neighbor] = neighbor_distance
                    node_to_previous[neighbor] = node
                    heapq.heappush(heap, (neighbor_distance, neighbor))

        # There is no path from start_node to end_node
        return None

    def print_shortest_path(self, shortest_path): 
        if shortest_path:
            print("Shortest path: ")
            for node in shortest_path:
                print(node.access_xy())
        else:
            print("There is no path from start_node to end_node")

# definition of the node class
class Node: 
    def __init__(self, xy, K): 
        self.xy = xy 
        self.k_nearest_neighbors = [None] * K 
        self.u = Util() 

    def access_xy(self):
        return self.xy 

    def __lt__(self, other):
        # Compare two nodes based on their xy coordinates
        return self.xy < other.xy

    def find_nearest_neighbors(self, nodes, K):
        distances = list() 
        for node in nodes: 
            dist = self.u.distance(self.xy,node.xy)
            distances.append((node, dist))
        distances.sort(key=lambda tup: tup[1])

        # ignoring the smallest distance (0) b/c that one is just itself 
        for i in range(1,K+1): 
            self.k_nearest_neighbors[i-1] = distances[i][0]


    def display_nearest_neighbors(self):
        print(self.xy)

        for i in range(0,len(self.k_nearest_neighbors)): 
            output_string = "{} near neighbor, location {}, euclidean distance {}".format(i+1, self.k_nearest_neighbors[i].access_xy(),self.u.distance(self.xy, self.k_nearest_neighbors[i].xy))
            print(output_string)
            
# definition of the edge class
class Edge: 
    def __init__(self, node1, node2): 
        self.node1 = node1 
        self.node2 = node2
        self.engaged = True 
        self.u = Util()
        self.weight = self.u.distance(self.node1.access_xy(), self.node2.access_xy())

    def check_equivalent(self, potential_edge): 
        if ((self.node1.access_xy() == potential_edge.node1.access_xy()) and (self.node2.access_xy() == potential_edge.node2.access_xy())) or ((self.node2.access_xy() == potential_edge.node1.access_xy()) and (self.node1.access_xy() == potential_edge.node2.access_xy())):
            return True; 

    def display_edge_info(self): 
        output_string = "node1 @ {}, node 2 @ {}, euclidean distance = {}.".format(self.node1.access_xy(),self.node2.access_xy(),self.u.distance(self.node1.xy, self.node2.xy))        
        print(output_string)

    def get_tuple(self): 
        return (self.node1.xy, self.node2.xy)


# if python says run, then we should run
if __name__ == '__main__':
    main()


