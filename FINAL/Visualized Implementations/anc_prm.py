# anc_prm.py
# This program runs ANC PRM
#  
# Authors: Kevin Grant Li, Avigayil Helman, Kevin Li, Joseph Han,  Nicole Neil
# 
# Adapted from code from Professor Brian Plancher and Avigayil Helman
# Adapted from code written by Steve LaValle and Rus Tedrake

import sys, random, math, pygame
from pygame.locals import *
from util import Util
import heapq
from decimal import Decimal, getcontext

import threading

import numpy as np

# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

class anc_prm:
    def __init__(self, obstacles, start_node, goal_node, XDIM = 640, YDIM = 480, NUM_SAMPLES = 500, LINE_WIDTH = 20, TEST_MODE = 0, K = 5, r = 3):
        
        self.obstacles = obstacles # Obstacles are represented by a list of points defining line segments 
        self.LINE_WIDTH = 20 # width of obstacle lines (projected out)
        self.XDIM = XDIM # board dimmension -> x-dimension 
        self.YDIM = YDIM # board dimmension -> y-dimension
        self.TEST_MODE = TEST_MODE # do not wait for user to exit and return the final result
        self.start_node = start_node # (x,y) position of starting location  
        self.goal_node = goal_node # (x,y) position of goal location 
        self.NUM_SAMPLES = NUM_SAMPLES # total number of samples we will make within the plane 
        self.K = K # number of nearest neighbors for building out graph
        self.r = r # number of radial closest for building out graph
    
    def runGame(self):
        '''
        Main method to run serial ANC PRM
        '''
        # initialize and prepare screen with pygame interface

        print("RUNNING ANC PRM")
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

        # init our util function object
        u = Util()

        # first select a given number of location samples [(x,y) coordinates] within the grid! 
        grid_samples = self.sample_grid({self.start_node, self.goal_node}, self.XDIM, self.YDIM, self.NUM_SAMPLES)

        # remove samples that are in obstacles --> these cannot be reached!
        non_obstructed_samples = self.remove_obstacle_samples(grid_samples,self.obstacles,self.LINE_WIDTH,u)
        
        # visualize all non_obstructed_samples generated 
        for sample in non_obstructed_samples: 
              pygame.draw.circle(screen,green,sample,3)
              pygame.display.update()

        nodes = self.buildNodes(non_obstructed_samples,self.K) # Construct all nodes

        #----------------------------------------------------------------------
        # Run ANC here                                                        |
        #----------------------------------------------------------------------
        G = ANC_Graph(nodes,[],new_obstacles,self.LINE_WIDTH,self.K,self.r)  #|
        G.anc_algorithm() # Running anc without parallelization              #|            
        # ---------------------------------------------------------------------

        # this the final graph
        for edge in G.edges: 
            # print(edge.get_tuple())
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
        shortest_path = self.dijkstra(starting_node,endgoal_node,G.edges)

        # print out the shortest path
        self.print_shortest_path(shortest_path)

        # if there is a shortest path -> draw the result
        
        if shortest_path != None: 
            for i in range(1,len(shortest_path)): 

                nodeprev_location = shortest_path[i-1].access_xy()
                nodenext_location = shortest_path[i].access_xy()
                
                pygame.draw.line(screen,babyblue,nodeprev_location,nodenext_location,5)
                pygame.display.update()

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
                        
    def runGameParallel(self):
        '''
        Main method to run ANC PRM in parallel.
        Parallelized methods include:
        1. Point sampling + point-obstacle collision detection (CUDA GPU)
        2. ANC algorithm (CPU Multi-threading)
        '''
        # initialize and prepare screen with pygame interface
        #pygame.init()
        #screen = pygame.display.set_mode([self.XDIM, self.YDIM])
        #pygame.display.set_caption('PRM Project')
        #white = 255, 240, 200
        #black = 20, 20, 40
        #red = 255, 0, 0
        #blue = 0, 0, 255
        #green = 0, 255, 0
        #hotpink = 255, 153, 255
        #babyblue = 204, 229, 255

        #screen.fill(black)
        #pygame.draw.circle(screen,babyblue, self.start_node, 8)
        #pygame.draw.circle(screen,blue,self.goal_node, 8)

        new_obstacles = self.redefineObstacles(self.obstacles, self.LINE_WIDTH)
        #for o in new_obstacles:
         #   pixel = pygame.draw.polygon(screen, red, o)
          #  pygame.display.update()

        # start the list of nodes --> make sure to include both the start state and the end state within this list! 
        nodes = set()
        parents = {}

        u = Util() # Init our util function object

        # first select a given number of location samples [(x,y) coordinates] within the grid! 

        # Two lines below are in parallel
        grid_samples = self.sample_grid_parallel({self.start_node, self.goal_node}, self.XDIM, self.YDIM, self.NUM_SAMPLES) 

        # remove samples that are in obstacles --> these cannot be reached!
        non_obstructed_samples = self.remove_obstacle_samples_parallel(grid_samples,self.obstacles,self.LINE_WIDTH)

        # visualize all non_obstructed_samples generated 
        #for sample in non_obstructed_samples: 
         #     sample = tuple((int(sample[0]),int(sample[1])))
          #    pygame.draw.circle(screen,green,sample,3)
           #   pygame.display.update()

        nodes = self.buildNodes(non_obstructed_samples,self.K) # Construct all nodes

        #----------------------------------------------------------------------
        # Run ANC here                                                        |
        #----------------------------------------------------------------------
        G = ANC_Graph(nodes,[],new_obstacles,self.LINE_WIDTH,self.K,self.r)  #|   
        G.run_parallel_cpu_anc()                                             #|
        # ---------------------------------------------------------------------

        # this the final graph
        #for edge in G.edges: 
        #    pygame.draw.lines(screen, white, 0, edge.get_tuple(),2)
        #    pygame.display.update()

        # identify the starting and ending node
        for node in nodes: 
            if node.access_xy() == self.start_node: 
                starting_node = node 
            elif node.access_xy() == self.goal_node: 
                endgoal_node = node

        print(self.start_node)
        print(self.goal_node)

        print("Starting @ " + str(starting_node.access_xy()))
        print("Ending @ " + str(endgoal_node.access_xy()) + "\n")

        # find the shortest path 
        shortest_path = self.dijkstra(starting_node,endgoal_node,G.edges)

        # print out the shortest path
        self.print_shortest_path(shortest_path)

        # if there is a shortest path -> draw the result
        
        #if shortest_path != None: 
         #   for i in range(1,len(shortest_path)): 

          #      nodeprev_location = shortest_path[i-1].access_xy()
           #     nodenext_location = shortest_path[i].access_xy()
                
            #    pygame.draw.line(screen,babyblue,nodeprev_location,nodenext_location,5)
             #   pygame.display.update()

              #  for e in pygame.event.get():
               #     if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                #        sys.exit("Leaving because you requested it.")

      #  if (self.TEST_MODE):
       #     path = [nodes[-1]]
        #    node = nodes[-1]
         #   while node != nodes[0]:
          #      node = parents[node]
           #     path.append(node)
           # return path
        # Else wait for the user to see the solution to exit
        #else:
         #   while(1):
          #      for e in pygame.event.get():
           #         if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
            #            sys.exit("Leaving because you requested it.")
                        

    # samples the grid for (x,y) locations (these are potential nodes)
    def sample_grid(self, nodes, XDIM, YDIM, NUM_SAMPLES): 

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
            for i in range(len(obstacle) - 1):
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
    
    def sample_grid_parallel(self,nodes, XDIM, YDIM, NUM_SAMPLES):
        # CUDA kernel to generate random samples
        mod = SourceModule("""
        __device__ unsigned int xorshift32(unsigned int *state) {
            /* Implementation of the XOR shift generator */
            unsigned int x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            return x;
        }

        __device__ float randFloat(int state) {
            unsigned int seed = state;

            // Initialize the random number generator
            unsigned int m_w = seed;
            unsigned int m_z = 0xDEADBEEF;

            // Generate a random float between 0 and 1
            unsigned int u = m_z * 36969 + (m_z >> 16);
            unsigned int v = m_w * 18000 + (m_w >> 16);
            unsigned int x = (u << 16) + v;
            float randomFloat = static_cast<float>(x) / 4294967295;

            return randomFloat;
        }

        __global__ void generate_samples(float *samples, int num_samples, int xdim, int ydim, unsigned int seed) {
            unsigned int state = seed + blockIdx.x * blockDim.x + threadIdx.x;
            if (threadIdx.x < num_samples) {
                samples[threadIdx.x * 2] = randFloat((int)(xorshift32(&state))) * xdim;
                samples[threadIdx.x * 2 + 1] = randFloat((int)(xorshift32(&state))) * ydim;
            }
        }
        """)

        # Convert the nodes to a numpy array
        nodes = np.array(list(nodes), dtype=np.float32)

        # Create a set to store the samples
        samples = set(tuple(row) for row in nodes)

        # Generate random samples in parallel until we have NUM_SAMPLES
        block_size = 1024
        grid_size = (NUM_SAMPLES + block_size - 1) // block_size
        num_threads = grid_size * block_size
        samples_gpu = cuda.mem_alloc(num_threads * 2 * np.dtype(np.float32).itemsize)

        # Declare samples as a numpy array
        samples_np = np.zeros((num_threads * 2,), dtype=np.float32)

        seed = np.random.randint(1, 2**32, dtype=np.uint32)
        func = mod.get_function("generate_samples")
        func(samples_gpu, np.int32(NUM_SAMPLES), np.int32(XDIM), np.int32(YDIM), np.uint32(seed), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        cuda.memcpy_dtoh(samples_np, samples_gpu)

        # Add the samples to the set
        samples_np = np.reshape(samples_np, (-1, 2))[:NUM_SAMPLES - 2]
        samples_np = np.concatenate((nodes, samples_np), axis=0)
        samples_np = set(tuple(row) for row in samples_np)

        return samples_np

    # removes samples that are obstructed (within the obstacles)
    def remove_obstacle_samples_parallel(self,grid_samples,obstacles,obstacle_width): 

        # Define the CUDA Kernel 
        mod = SourceModule("""

        #include <cmath>

        __device__ int distance(int ax, int ay, int bx, int by)
        {
            float dx = ax - bx; 
            float dy = ay - by; 
            float d = std::sqrt(dx * dx + dy * dy); 
            return d; 
        }


        __global__ void remove_obstructed_samples(int* grid_samples, int* obstacles, int* no_obstacles_a, int* flags, int* obstacle_width_a,int* sample_num)
        { 
            int no_obstacles = no_obstacles_a[0]; 
            int obstacle_width = obstacle_width_a[0]; 

            int sample_no = threadIdx.x + blockIdx.x * blockDim.x; 

            if (sample_no>sample_num[0])return;

            int pointx = grid_samples[sample_no * 2]; 
            int pointy = grid_samples[sample_no * 2 + 1];   

            float half_width = (obstacle_width/2); 
            float sq_w = half_width * half_width; 

            for(int i = 0; i < no_obstacles; i++){
            
                int obstacle_x1 = obstacles[i * 4]; 
                int obstacle_y1 = obstacles[i * 4 + 1]; 
                int obstacle_x2 = obstacles[i * 4 + 2]; 
                int obstacle_y2 = obstacles[i * 4 + 3]; 

                float c = distance(obstacle_x1, obstacle_y1, obstacle_x2, obstacle_y2); 
                float a = distance(obstacle_x1, obstacle_y1, pointx, pointy); 
                float b = distance(obstacle_x2, obstacle_y2, pointx, pointy); 

                if (half_width >= a || half_width >= b){
                
                    if (a < b){
                        sq_w = a * a; 
                    } 
                    else {
                        sq_w = b * b; 
                    }
                }  

                float c1 = std::sqrt(a * a - sq_w); 
                float c2 = std::sqrt(b * b - sq_w); 

                if (c1 + c2 <= c){
                    flags[sample_no] = 1;  
                }
            }      
        }
        """)
        
        # create a grid samples array -> [x1, y1, x2, y2, x3, y3 .... ]
        host_grid_samples = np.array(list(grid_samples)).flatten().astype(np.int32)
        device_grid_samples_array = cuda.mem_alloc(host_grid_samples.nbytes)
        cuda.memcpy_htod(device_grid_samples_array, host_grid_samples)

        # create a obstacles array -> [xa1, ya1, xa2, ya2, xb1, yb1, xb2, yb2 ... ]

        host_obstacles = np.array(list(obstacles.values())).flatten().astype(np.int32)
        device_obstacles_array = cuda.mem_alloc(host_obstacles.nbytes)
        cuda.memcpy_htod(device_obstacles_array, host_obstacles)

        # flag for each grid samples, 0 if not obstructed, 1 if obstructed
        host_obstructed_flag = np.zeros(len(grid_samples),dtype=np.int32)
        device_obstructed_flag = cuda.mem_alloc(host_obstructed_flag.nbytes)
        cuda.memcpy_htod(device_obstructed_flag, host_obstructed_flag)

        # to pass in the other information need to construct the relevant array 
        no_obstacles = np.array([len(obstacles)], dtype=np.int32)
        no_obstacles_a = cuda.to_device(no_obstacles)

        obstacle_width_a = cuda.mem_alloc(4)
        cuda.memcpy_htod(obstacle_width_a, np.array([obstacle_width], dtype=np.int32))    

        sample_num = cuda.mem_alloc(4)
        cuda.memcpy_htod(sample_num, np.array([len(grid_samples)], dtype=np.int32))      

        # call the CUDA kernel
        func = mod.get_function("remove_obstructed_samples")

        # UPDATED
        block_size = 1024
        grid_size = (len(grid_samples) + block_size - 1) // block_size
        func(device_grid_samples_array, device_obstacles_array, no_obstacles_a, device_obstructed_flag, obstacle_width_a, sample_num,block=(block_size, 1, 1), grid=(grid_size, 1, 1))

        # copy back the device memory to the host memory
        cuda.memcpy_dtoh(host_obstructed_flag, device_obstructed_flag) 

        # select out only the samples where flag is 0 (ie not colliding)
        unobstructed_samples = [tup for i, tup in enumerate(list(grid_samples)) if host_obstructed_flag[i] == 0]

        return unobstructed_samples

# definition of the node class
class Node: 
    def __init__(self, xy, K): 
        self.xy = xy 
        self.nearest_neighbors = []
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

# ---- Graph object specficially for ANC ----
class ANC_Graph: 

    def __init__(self,nodes,edges,obstacles,obstacle_width,K,r): 
        self.nodes = nodes
        self.edges = edges 
        self.neighbor_finders = [self.k_nearest_neighbors,self.r_nearest_neighbors,self.k_closest_k_rand,self.r_closest_k_rand]
        self.obstacles = obstacles
        self.obstacles_width = obstacle_width
        self.K = K
        self.r = r

        # alternative method of checking edge collision
    def get_edges(self):
        return_list = []
        for edge in self.edges:
            return_list.append(edge.get_tuple())
        return return_list

    def line_intersection(self, line1, line2):
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

    # check all four sides of an obstacle to check for collision within a line segment (defined as two points)
    def line_segment_intersects_rectangle(self,U, V, rectangle):

        A1, A2, A3, A4 = rectangle

        sides = [(A1, A2), (A2, A3), (A3, A4), (A4, A1)]
        UV = (U[0], U[1], V[0], V[1])

        for side in sides:
            XY = (side[0][0], side[0][1], side[1][0], side[1][1])
            if self.line_intersection(UV, XY):
                return True

        return False

        #-----------------------find_nearest_neighbors variants--------------------

    def distance(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        d = math.sqrt(dx * dx + dy * dy)
        return d

    def k_nearest_neighbors(self, q_node, nodes, K): # k-nearest neighbor - renamed function
        distances = list() 
        for node in nodes: 
            if node != q_node: # Ignore the node that is equal to itself
                dist = self.distance(q_node.xy,node.xy)
                distances.append((node, dist))
        distances.sort(key=lambda tup: tup[1])
        
        neighbors = [distances[i][0] for i in range(0,K)]

        return neighbors

    def r_nearest_neighbors(self,q_node,nodes,r):
        '''Nodes <= r in the affinity of a given node are considered neighbors to construct edges with'''
        distances = list()
        for node in nodes:
            dist=self.distance(q_node.xy,node.xy)
            distances.append((node,dist))
        distances.sort(key=lambda tup:tup[1])

        r_neighbors = [node for node, dist in distances if dist <= r and node != q_node]

        return r_neighbors
    
    def k_closest_k_rand(self,q_node,nodes,k,k2):
        assert k2>= k, "k2 should be greater than or equal to k, typically k2 = 3k"

        distances = [(node, self.distance(q_node.xy, node.xy)) for node in nodes if node != q_node]
        distances.sort(key=lambda tup: tup[1])

        k2_closest = distances[:k2]
        k_rand_neighbors = random.sample(k2_closest, k)

        neighbors = [neighbor for neighbor, _ in k_rand_neighbors]

        return neighbors

    def r_closest_k_rand(self, q_node,nodes, k, r):
        distances = [(q_node, self.distance(q_node.xy, node.xy)) for node in nodes if node != q_node]
        distances_within_r = [(node, dist) for node, dist in distances if dist <= r]

        if len(distances_within_r) >= k:
            k_rand_neighbors = random.sample(distances_within_r, k)
        else:
            k_rand_neighbors = distances_within_r

        neighbors = [neighbor for neighbor, _ in k_rand_neighbors]
        return neighbors
    
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
    

    def update_probabilities(self, P, r, c,i):
        '''P = probabilities, r = success rate, c = cost, i = indexing for P'''
        alpha = 0.05
        beta = 0.00

        delta_p = alpha*r - beta*c
        new_p = max(P[i] * (1 + delta_p), 0)  # Ensure the new probability is non-negative
        P[i] = new_p

        # Normalize the probabilities
        total = sum(P)
        for i in range(len(P)):
            P[i] /= total
        return P
    
    def anc_algorithm(self):
        '''Serial anc implementation'''

        # ANC requires a graph input - self. Should have all nodes instantiated.
        count = [0,0,0,0] # comment out late -> just for testing the number of times each finder is used
        
        P = [1 / len(self.neighbor_finders) for _ in self.neighbor_finders] # Probabilities of each neighbor finding method

        # Define k and r for k and r nearest neighbors
        for node in self.nodes: # For each node in graph

            # Randomly pick a neighbor_finder according to P
            nfi = random.choices(self.neighbor_finders, weights=P)[0]
            # Find the neighbors of the current node
            if nfi.__name__ == "k_nearest_neighbors":
                neighbors = nfi(node, self.nodes, self.K) 
            elif nfi.__name__ == "r_nearest_neighbors":
                neighbors = nfi(node,self.nodes,self.r)
            elif nfi.__name__== "k_closest_k_rand":
                neighbors = nfi(node, self.nodes,self.K,3*self.K) # k2-closest-k-random
            else:
                neighbors = nfi(node,self.nodes,self.K,self.r*5) # r-closest,k

                # NOTE = r*5 above for r-closest,k

            # Connect the current node to the neighbors if they are connectable
            n_count = len(neighbors)

            edges = []
            for neighbor in neighbors:
                if neighbor != node:
                    # modified version of ANC
                    edges.append(Edge(node,neighbor))

            unobstructed = self.removeObstructedEdges(edges,self.obstacles,self.obstacles_width)
            self.edges.extend(unobstructed)

            r = 0 if n_count == 0 else len(unobstructed)/n_count # success rate - how many connections are successful
            i = 0 # indexing

            for index, n_finder in enumerate(self.neighbor_finders):
                if nfi.__name__ == n_finder.__name__:
                    i = index
                    count[i] += 1 # comment out later -> just used for testing
                    break

            # compute the cost by considering largest edge distance
            # cost function prioritizes shorter distance paths between nodes
            largest_edge_weight = 0
            # print(unobstructed)
            if len(unobstructed)>0:
                unobstructed.sort(key=lambda edge: edge.weight)
            # print(unobstructed[-1].weight)
                largest_edge_weight = unobstructed[-1].weight

            # print(largest_edge_weight)

            P = self.update_probabilities(P,r,largest_edge_weight,i)
            # print(P)

        # print(P)
        # print(count)
    
# Code for CPU parallelization

    def run_parallel_cpu_anc(self, num_threads=5):
        '''
        Main function to run parallel CPU ANC
        Creates num_threads threads to run anc in parallel
        '''
        num_nodes = len(self.nodes)
        num_threads = min(num_threads,num_nodes) # in case there are more more proceses than nodes
        nodes_per_thread = num_nodes//num_threads

        thread_args = [
            (i * nodes_per_thread, (i + 1) * nodes_per_thread) for i in range(num_threads)
        ]

        thread_args[-1] = (thread_args[-1][0],num_nodes)

        threads = []
        results = [None] * num_threads

        def worker(start,end,idx):
            results[idx] = self.anc_parallel(start,end)

        # Parallelization below
        for i, args in enumerate(thread_args):
            t = threading.Thread(target=worker,args=(*args,i))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        # Recombine edges and counts for analysis

        edges = []
        combined_P = [0 for _ in range(len(self.neighbor_finders))]
        combined_count = [0 for _ in range(len(self.neighbor_finders))]
        for edge_set, P, count in results:
            edges.extend(edge_set)
            combined_P = [p1 + p2 for p1, p2 in zip(combined_P, P)]
            combined_count = [c1 + c2 for c1, c2 in zip(combined_count, count)]

        # Normalize the combined_P
        total = sum(combined_P)
        for i in range(len(combined_P)):
            combined_P[i] /= total

        self.edges = edges

        print(combined_P)
        print(combined_count)

    def anc_parallel(self, start_index,end_index):
        '''
        anc for each thread for parallel implementation
        '''

        count = [0,0,0,0]

        P = [1 / len(self.neighbor_finders) for _ in self.neighbor_finders]
        # P = [1,0,0,0]

        chunk_edges = []

        for node in self.nodes[start_index:end_index]:
           # Randomly pick a neighbor_finder according to P
            nfi = random.choices(self.neighbor_finders, weights=P)[0]
            # Find the neighbors of the current node
            if nfi.__name__ == "k_nearest_neighbors":
                neighbors = nfi(node, self.nodes, self.K) 
            elif nfi.__name__ == "r_nearest_neighbors":
                neighbors = nfi(node,self.nodes,self.r)
            elif nfi.__name__== "k_closest_k_rand":
                neighbors = nfi(node, self.nodes,self.K,3*self.K)
            else:
                neighbors = nfi(node,self.nodes,self.K,self.r*5) # r-closest,k
                # NOTE = r*5 above for r-closest,k

            # Connect the current node to the neighbors if they are connectable
            n_count = len(neighbors)

            edges = []
            for neighbor in neighbors:
                if neighbor != node:
                    # modified version of ANC
                    edges.append(Edge(node, neighbor))

            unobstructed = self.removeObstructedEdges(edges, self.obstacles, self.obstacles_width)
            chunk_edges.extend(unobstructed)

            r = 0 if n_count == 0 else len(unobstructed) / n_count
            i = 0

            for index, n_finder in enumerate(self.neighbor_finders):
                if nfi.__name__ == n_finder.__name__:
                    i = index
                    count[i] += 1
                    break

            largest_edge_weight = 0
            if len(unobstructed) > 0:
                unobstructed.sort(key=lambda edge: edge.weight)
                largest_edge_weight = unobstructed[-1].weight

            P = self.update_probabilities(P, r, largest_edge_weight, i)

        # print(P)
        # print(count)
        return chunk_edges,P, count
