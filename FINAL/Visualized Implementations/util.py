import math
import random
from math import sqrt,cos,sin,atan2

class Util:
    ########################################
    #   Mandatory functions for the rrt    #
    ########################################

    # Tests if the new_node is close enough to the goal to consider it a goal
    def winCondition(self,new_node,goal_node,WIN_RADIUS):
        """
        new_node - newly generated node we are checking
        goal_node - goal node
        WIN_RADIUS - constant representing how close we have to be to the goal to
            consider the new_node a 'win'
        """
        dx = new_node[0] - goal_node[0]
        dy = new_node[1] - goal_node[1]
        d = sqrt(dx*dx + dy*dy)
        if d <= WIN_RADIUS: return True
        return False

    # Find the nearest node in our list of nodes that is closest to the new_node
    # Hint: If your solution appears to be drawing squiggles instead of the fractal like pattern 
    #       of striaght lines you are probably extending from the last point not the closest point!
    def nearestNode(self,nodes,new_node):
        """
        nodes - a list of nodes in the RRT
        new_node - a node generated from getNewPoint
        """
        closest_node = nodes[0]
        closest_node_dist = math.inf
        for node in nodes:
            dx = new_node[0] - node[0]
            dy = new_node[1] - node[1]
            d = sqrt(dx * dx + dy * dy)
            if d < closest_node_dist:
                closest_node_dist = d
                closest_node = node
        return closest_node

    # Find a new point in space to move towards uniformally randomly but with
    # probability 0.05, sample the goal. This promotes movement to the goal.
    # For the autograder to work you MUST use the already imported
    # random.random() as your random number generator.
    def getNewPoint(self,XDIM,YDIM,XY_GOAL):
        """
        XDIM - constant representing the width of the game aka grid of (0,XDIM)
        YDIM - constant representing the height of the game aka grid of (0,YDIM)
        XY_GOAL - node (tuple of integers) representing the location of the goal
        """
        p = random.random()
        if p <= 0.05: return XY_GOAL
        else: return ((int)(random.random()*XDIM), (int)(random.random()*YDIM))

    # Extend (by at most distance delta) in the direction of the new_point and place
    # a new node there
    def extend(self,current_node,new_point,delta):
        """
        current_node - node from which we extend
        new_point - point in space which we are extending toward
        delta - maximum distance we extend by
        """
        dx = new_point[0] - current_node[0]
        dy = new_point[1] - current_node[1]
        d = sqrt(dx * dx + dy * dy)
        if d < delta: return new_point

        # find angle,
        # then find smaller triangle of h=delta to find new dx, dy
        theta = atan2(dy, dx)
        dx_new = cos(theta) * delta
        dy_new = sin(theta) * delta

        return (current_node[0] + dx_new, current_node[1] + dy_new)

    # iterate throught the obstacles and check that our point is not in any of them
    def isCollisionFree(self,obstacles,point,obs_line_width, pad=0):
        """
        obstacles - a dictionary with multiple entries, where each entry is a list of
            points which define line segments of with obs_line_width
        point - the location in space that we are checking is not in the obstacles
        obs_line_width - the length of the line segments that define each obstacle's
            boundary
        """

        half_width = (obs_line_width/2 + pad)
        sq_w = half_width**2

        for val in obstacles.values():

            for i in range(len(val)-1):
                j = i + 1

                # Form a triangle between the two
                # obstacle points and point
                # Let c represent the segment bw val[i], val[j]
                # and a, b their segments made with point
                # If we assume a line segment drawn from point to
                # the closest point on c has a height of obs_line_width,
                # then c1 + c2 will be > than c if
                # the height is an underestimate, <= if hieght is less
                # since any height > obs_line_width falls outside
                # this obstacle, reject
                c = self.distance(val[i], val[j]) + pad*2
                a = self.distance(val[i], point)
                b = self.distance(val[j], point)

                if half_width >= a or half_width >= b:
                    sq_w = min(a, b) ** 2

                c1 = sqrt(a**2 - sq_w)
                c2 = sqrt(b**2 - sq_w)

                if c1 + c2 <= c: return False

        return True

    ################################################
    #  Any other helper functions you need go here #
    ################################################
    def distance(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        d = sqrt(dx * dx + dy * dy)
        return d