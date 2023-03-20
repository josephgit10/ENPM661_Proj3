#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install moviepy


# In[2]:


# Importing necessary libraries
import heapq 
import cv2
import numpy as np
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from math import dist
from moviepy.editor import *


# In[3]:


# Storing nodes as objects
class Node:

    def __init__(self, x, y, theta, cost, parent_id, c2g = 0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = cost
        self.parent_id = parent_id
        self.c2g = c2g 
        
        
    def __lt__(self,other):
        return self.cost + self.c2g < other.cost + other.c2g
    
# Defining actions
def move2Up(x, y, theta, step, cost):
    theta = theta+60
    x = x + (step*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def moveUp(x, y, theta, step, cost):
    theta = theta+30
    x = x + (step*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def moveForward(x, y, theta, step, cost):
    theta = theta+0
    x = x + (step*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def moveDown(x, y, theta, step, cost):
    theta = theta-30
    x = x + (step*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def move2Down(x, y, theta, step, cost):
    theta = theta-60
    x = x + (step*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

# Defining the Action set
def Action_Set (move, x, y, theta, step, cost):
    if move == '2up':
        return move2Up(x, y, theta, step, cost)
    elif move == 'up':
        return moveUp(x, y, theta, step, cost)
    elif move == 'forward':
        return moveForward(x, y, theta, step, cost)
    elif move == 'down':
        return moveDown(x, y, theta, step, cost)
    elif move == '2down':
        return move2Down(x, y, theta, step, cost)
    else:
        return None
    
# Defining the obstacle space:
def obs_Space(clearance, robot_radius):
    height = 600
    width = 250
    tot = robot_radius + clearance
    img = np.zeros((width, height, 3), dtype="uint8")

    # Drawing obstacles
    # Rectangles
    img = cv2.rectangle(img, (100-tot, 0), (150+tot, 100+tot), (128, 128, 128), -1)
    img = cv2.rectangle(img, (100-tot, 150-tot), (150+tot, 250), (128, 128, 128), -1)

    # Drawing the triangle by connecting the vertices with lines of thickness 2 and blue color
    triangle_vertices = [(460-tot, 25-tot), (460-tot, 225+tot), (510+tot, 125)]
    img = cv2.line(img, triangle_vertices[0], triangle_vertices[1], (128, 128, 128), 5)
    img = cv2.line(img, triangle_vertices[1], triangle_vertices[2], (128, 128, 128), 5)
    img = cv2.line(img, triangle_vertices[2], triangle_vertices[0], (128, 128, 128), 5)

    pts = np.array(triangle_vertices, np.int32)
    cv2.fillPoly(img, [pts], (128, 128, 128))

    center = (300, 125)
    side = 75+tot
    
    # Computing the vertices of the hexagon by using trigonometry
    angle = np.radians(30)
    hexagon_vertices = []
    for i in range(6):
        x = int(center[0] + side * np.cos(angle))
        y = int(center[1] + side * np.sin(angle))
        hexagon_vertices.append((x, y))
        angle += np.radians(60)
        
    # Drawing the hexagon by connecting the vertices with lines of thickness 2 and red color
    pts = np.array(hexagon_vertices, np.int32)

    # Using cv2.fillConvexPoly() to fill the hexagon with blue color
    cv2.fillConvexPoly(img, pts, (128, 128, 128))

    # Adding obstacle clearance to the canvas by eroding the image with a circular kernel of radius `tot`
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tot, tot))
    img = cv2.erode(img, kernel)

    # Converting the canvas to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Defining boundaries
    for i in range(600):
        img[0][i] = 128

    for i in range(600):
        img[249][i] = 128

    for i in range(250):
        img[i][1] = 128

    for i in range(250):
        img[i][599] = 128

    return img
    
# Checking for validity of the move
def moveValid(x, y, obs_space):
    shape = obs_space.shape

    if( x > shape[1] or x < 0 or y > shape[0] or y < 0 ):
        return False
    
    else:
        try:
            if(obs_space[y][x] == 128):
                return False
        except:
            pass
    return True

# Checking for validity of orientation
def orientValid(theta):
    if theta%30 == 0:
        return theta
    else:
        return False

# Checking if goal node is current node
def  Check_goal(curr, goal):
    distance = dist((curr.x, curr.y), (goal.x, goal.y))
    if distance<1.5:
        return True
    else:
        return False
    
# Generating a unique key
def key(node):
    key = 211*node.x + 111*node.y
    return key

# A-star algorithm
def a_star(start, goal, obs_space, step):
    
    if Check_goal(start, goal):
        return None,1
    goal_node = goal
    start_node = start
    
    moves = ['2up','up', 'forward', 'down', '2down']   
    unexplored = {}
    
    # Generating a unique key for identifying the node
    start_key = key(start_node) 
    unexplored[(start_key)] = start_node
    
    explored = {} 
    priority_list = [] 
    heapq.heappush(priority_list, [start_node.cost, start_node])
    
    all_nodes = []
    

    while (len(priority_list) != 0):

        curr_node = (heapq.heappop(priority_list))[1]
        all_nodes.append([curr_node.x, curr_node.y, curr_node.theta])          
        curr_id = key(curr_node)
        if Check_goal(curr_node, goal_node):
            goal_node.parent_id = curr_node.parent_id
            goal_node.cost = curr_node.cost
            print("Goal Node found")
            return all_nodes,1

        if curr_id in explored:  
            continue
        else:
            explored[curr_id] = curr_node

        del unexplored[curr_id]

        for move in moves:
            x,y,theta,cost = Action_Set(move,curr_node.x,curr_node.y,curr_node.theta, step_size, curr_node.cost)  ##newaddd
            
            c2g = dist((x, y), (goal.x, goal.y))  
   
            new_node = Node(x,y,theta, cost,curr_node, c2g)   
   
            new_node_id = key(new_node) 
   
            if not moveValid(new_node.x, new_node.y, obs_space):
                continue
            elif new_node_id in explored:
                continue
   
            if new_node_id in unexplored:
                if new_node.cost < unexplored[new_node_id].cost: 
                    unexplored[new_node_id].cost = new_node.cost
                    unexplored[new_node_id].parent_id = new_node.parent_id
            else:
                unexplored[new_node_id] = new_node

            heapq.heappush(priority_list, [(new_node.cost + new_node.c2g), new_node]) 
   
    return  all_nodes,0

# Backtracking
def Backtrack(goal_node):  
    x_path = []
    y_path = []
    x_path.append(goal_node.x)
    y_path.append(goal_node.y)

    parent_node = goal_node.parent_id
    while parent_node != -1:
        x_path.append(parent_node.x)
        y_path.append(parent_node.y)
        parent_node = parent_node.parent_id
        
    x_path.reverse()
    y_path.reverse()
    
    x = np.asarray(x_path)
    y = np.asanyarray(y_path)
    
    return x,y


# Creating Animation Video
def animate(start_node, goal_node, x_path, y_path, all_nodes, obs_space, interval=50):
    fig = plt.figure()
    im = plt.imshow(obs_space, "GnBu")
    plt.plot(start_node.x, start_node.y, "Dw")
    plt.plot(goal_node.x, goal_node.y, "Dg")
    ax = plt.gca()
    ax.invert_yaxis()

    def update(i):
        if i >= len(all_nodes):
            return
        node = all_nodes[i]
        plt.plot(node[0], node[1], "2g-")
        if i == len(all_nodes) - 1:
            plt.plot(x_path, y_path, ':r')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(all_nodes), interval=interval, blit=True)
    ani.save('animation_video.gif')
    clip = (VideoFileClip("animation_video.gif")
        .resize(height=480)
        .write_videofile("animation_video.mp4", fps=24))
    plt.show()
    print("Animation Video saved\n")


# Main Body


if __name__ == '__main__':
    
    # Obstacle Clearance
    obs_clearance = input("Assign Clearance to the Obstacles: ")
    obs_clearance = int(obs_clearance)

        
    # Robot Radius
    robot_radius = input("Enter the Radius of the Robot: ") 
    robot_radius = int(robot_radius)
    
    # Step Size of the Robot
    robot_step_size = input("Enter Step size of the Robot: ")
    step_size = int(robot_step_size)
    obs_space = obs_Space(obs_clearance, robot_radius)
    c2g = 0
    
    # Taking start node coordinates as input from user
    start_coordinates = input("Enter coordinates for Start Node: ")
    start_x, start_y = start_coordinates.split()
    start_x = int(start_x)
    start_y = int(start_y)
        
    # Taking Orientation for the robot
    start_theta = input("Enter Orientation of the robot at start node: ")
    start_t = int(start_theta)
    
    # Checking if the user input is valid #####
    if not moveValid(start_x, start_y, obs_space):
        print("Start node is out of bounds")
        exit(-1)
        
    if not orientValid(start_t):
        print("Orientation has to be a multiple of 30")
        exit(-1)
   
    # Taking Goal node coordinates as input from user
    goal_coordinates = input("Enter coordinates for Goal Node: ")
    goal_x, goal_y = goal_coordinates.split()
    goal_x = int(goal_x)
    goal_y = int(goal_y)
        
    # Taking Orientation for the robot
    goal_theta = input("Enter Orientation of the robot at goal node: ")
    goal_t = int(goal_theta)
    
    
    # Checking if the user input is valid
    if not moveValid(goal_x, goal_y, obs_space):
        print("Goal node is out of bounds")
        exit(-1)
        
    if not orientValid(goal_t):
        print("Orientation has to be a multiple of 30")
        exit(-1)

    # Timer
    timer_start = time.time()
    
    # Creating start_node and goal_node objects 
    start_node = Node(start_x, start_y,start_t, 0.0, -1,c2g)
    goal_node = Node(goal_x, goal_y,goal_t, 0.0, -1, c2g)
    all_nodes, flag = a_star(start_node, goal_node, obs_space, robot_step_size)
    if (flag)==1:
        
        x_path,y_path = Backtrack(goal_node)
    
    else:
        print("No path was found")

    animate(start_node, goal_node, x_path, y_path, all_nodes, obs_space, interval=50)
    timer_stop = time.time()
    
    C_time = timer_stop - timer_start
    print("The Total Runtime is:  ", C_time)
    
    
       

