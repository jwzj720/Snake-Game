import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import heapq
import pygame
import numpy as np
import networkx as nx
import math

import sys

from heapq import *
 
pygame.init()

# Defining the colors that the snake and food will use.  
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
 
# Width of the game board (in tiles). 
WIDTH  = 20
# Height of the game board (in tiles).
HEIGHT = 20

# Size of each tile (in pixels).
STEPSIZE = 50

# How fast the game runs. Higher values are faster. 
CLOCK_SPEED = 1000
 
# Making a pygame display. 
dis = pygame.display.set_mode((WIDTH*STEPSIZE,HEIGHT*STEPSIZE))
pygame.display.set_caption('Snake!')

# Initial variables to store the starting x and y position,
# and whether the game has ended. 
game_over = False 
x1 = 5
y1 = 5
snake_list = [(x1,y1)]
snake_len = 1 
x1_change = old_x1_change = 0       
y1_change = old_y1_change = 0

# PyGame clock object.  
clock = pygame.time.Clock()

food_eaten = True

# Random obstacles, if desired. 
obstacles = [(np.random.randint(low=0, high=WIDTH),np.random.randint(low=0, high=HEIGHT)) for i in range(int(sys.argv[2]))]

# Bstate is a matrix representing the game board:
### Array cells with a 0 are empty locations. 
### Array cells with a -1 are the body of the snake.
### The cell marked with a -2 is the head of the snake.
### The cell marked with a 1 is the food.
def get_AI_moves(ai_mode, bstate):
    if ai_mode == 'rand':
        return random_AI(bstate)
    elif ai_mode == 'greedy':
        return greedy_AI(bstate)
    elif ai_mode == 'astar':
        return astar_AI(bstate)  
    elif ai_mode == 'dijkstra':
        return dijkstra_AI(bstate)  
    elif ai_mode == 'backt':
        return backt_AI(bstate)    
    else:
        raise NotImplementedError("Not a valid AI mode!\nValid modes are rand, greedy, astar, dijkstra, and backt.")    

def neighbors(array, x, y):
    rows = len(array)
    cols = len(array[0])
    neigh = []
    if x > 0:
        if array[x-1][y] == 0 or array[x-1][y] == 1:
            neigh.append((x-1, y))
    else:
        if array[cols-1][y] == 0 or array[cols-1][y] == 1:
            neigh.append((cols-1, y))
    
    if x < cols -1:
        if array[x+1][y] == 0 or array[x+1][y] == 1:
            neigh.append((x+1, y))
    else:
        if array[0][y] == 0 or array[0][y] == 1:
            neigh.append((0, y))

    if y > 0:
        if array[x][y-1] == 0 or array[x][y-1] == 1:
            neigh.append((x, y-1))
    else:
        if array[x][rows-1] == 0 or array[x][rows-1] == 1:
            neigh.append((x, rows-1))
    
    if y < rows-1:
        if array[x][y+1] == 0 or array[x][y+1] == 1:
            neigh.append((x, y+1))
    else:
        if array[x][0] == 0 or array[x][0] == 1:
            neigh.append((x, 0))

    return neigh
        


# Each method takes in a game board (as described above), and
# should output a series of moves. Valid moves are: 
# (0,1),(0,-1),(1,0), and (-1,0). This means if you want to
# move in any more complicated way, you need to convert the move
# you want to make into a sequence like this one.
# For example, if I wanted my snake to move +5 in the x direction and +3
# in the y direction, I could return 
# [(0,1),(0,1),(0,1),(0,1),(0,1),(1,0),(1,0),(1,0)].

def dijkstra(graph, s, t):
    dist = {}
    pred = {}
    pqueue = []
    
    for row in range(len(graph)):
        for col in range(len(graph[row])):
            if graph[row][col] == -2:
                dist[(row,col)] = 0
            else:
                dist[(row,col)] = 1e12
            pred[(row,col)] = None    
            heapq.heappush(pqueue, (dist[(row,col)], (row,col)) )
    
    #print(pqueue)
    while len(pqueue) > 0:
        
        dv, v = heapq.heappop(pqueue) 
        if dist[v] == dv:
            for N in neighbors(graph, v[0], v[1]):
                
                new_dist = dist[v] + 1
                #print(new_dist)
                if new_dist < dist[N]:
                    pred[N] = v
                    dist[N] = new_dist
                    heapq.heappush(pqueue, (new_dist, N))
        
    shortest_path = []
    shortest_path.append(t)
    #print(shortest_path)
    
    #print(pred)
    while shortest_path[0] != s:
        current_node = shortest_path[0]
        #print(current_node)
        if current_node is None:
            break
        new_node = pred[current_node]
        shortest_path.insert(0, new_node)
    return shortest_path

def list_to_moves(path, start):
    moves = []
    for move in path:
        if move is None:
            return [(0,0)]
        m = (move[0]-start[0], move[1]-start[1])
        if m != (0,0):
            moves.append(m)
        start = move
    return moves

def a_star_alg(graph, s, t):
    dist = {}
    pred = {}
    pqueue = []
    
    for row in range(len(graph)):
        for col in range(len(graph[row])):
            if graph[row][col] == -2:
                dist[(row,col)] = 0
            else:
                dist[(row,col)] = 1e12
            pred[(row,col)] = None    
            heapq.heappush(pqueue, (dist[(row,col)], (row,col)) )
    
    #print(pqueue)
    while len(pqueue) > 0:
        
        dv, v = heapq.heappop(pqueue) 
        if dist[v] == dv:
            for N in neighbors(graph, v[0], v[1]):
                
                new_dist = dist[v] + distance(s, t) # this is the hueristic
                #print(new_dist)
                if new_dist < dist[N]:
                    pred[N] = v
                    dist[N] = new_dist
                    heapq.heappush(pqueue, (new_dist, N))
        
    shortest_path = []
    shortest_path.append(t)
    #print(shortest_path)
    
    #print(pred)
    while shortest_path[0] != s:
        current_node = shortest_path[0]
        #print(current_node)
        if current_node is None:
            break
        new_node = pred[current_node]
        shortest_path.insert(0, new_node)
    return shortest_path

def distance(one, two):
    x = two[0]-one[0]
    y = two[1]- one[1]
    return math.sqrt(x*x + y*y)         
            
def astar_AI(bstate):
    source = np.array(np.where(bstate == -2))
    target = np.array(np.where(bstate == 1))

    list_moves = a_star_alg(bstate, (source[0].item(), source[1].item()), (target[0].item(), target[1].item()))
    moves = list_to_moves(list_moves, (source[0].item(), source[1].item()))
    
    return moves
    
def backt_AI(bstate):
    source = np.array(np.where(bstate == -2))
    target = np.array(np.where(bstate == 1))
    return random_AI(bstate)
    
def dijkstra_AI(bstate):
    source = np.array(np.where(bstate == -2))
    target = np.array(np.where(bstate == 1))

    list_moves = dijkstra(bstate, (source[0].item(), source[1].item()), (target[0].item(), target[1].item()))
    moves = list_to_moves(list_moves, (source[0].item(), source[1].item()))
    
    return moves

def get_greedy(graph, source, target, choosen_neighbor=[]):
    source_neighbors = neighbors(graph, source[0], source[1])
    dis = 1000
    smallest_neighbor = None
    if source[0] == target[0] and source[1] == target[1]:
        return choosen_neighbor
    for n in source_neighbors:
        d = distance(n, target)
        if d < dis:
            dis = d
            smallest_neighbor = n
    choosen_neighbor.append(smallest_neighbor)
    return get_greedy(graph, smallest_neighbor, target, choosen_neighbor)
    


        
def greedy_AI(bstate):
    source = np.array(np.where(bstate == -2))
    target = np.array(np.where(bstate == 1))
    coords = get_greedy(bstate, (source[0].item(), source[1].item()), (target[0].item(), target[1].item()))
    moves = list_to_moves(coords, (source[0].item(), source[1].item()))

    return moves
    
def random_AI(bstate):
    return [[(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(low=0,high=4)]]
    
mode = str(sys.argv[1])

AI_moves = []

# Code below this point was mostly adapted from @Cory_Scott's code, given to me in class. 

while not game_over:


    if food_eaten:   
        fx = np.random.randint(low=0,high=WIDTH)
        fy = np.random.randint(low=0,high=HEIGHT)
        while (fx,fy) in snake_list or (fx,fy) in obstacles:
            fx = np.random.randint(low=0,high=WIDTH)
            fy = np.random.randint(low=0,high=HEIGHT)
        food_eaten = False
        
    dis.fill(white)
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if mode == 'human':    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -1
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = 1
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -1
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = 1
                    x1_change = 0
    if mode != 'human':
        if len(AI_moves) == 0:
            bstate = np.zeros((WIDTH,HEIGHT))
            for xx,yy in snake_list:
                bstate[xx,yy] = -1
            for xx,yy in obstacles:
                bstate[xx,yy] = -1
            bstate[snake_list[-1][0], snake_list[-1][1]] = -2
            bstate[fx,fy] = 1    
            AI_moves = get_AI_moves(mode, bstate)     
        x1_change, y1_change = AI_moves.pop(0)               
    if len(snake_list) > 1 :
        if ((snake_list[-1][0] + x1_change) % WIDTH) == snake_list[-2][0] and ((snake_list[-1][1] + y1_change)% HEIGHT) == snake_list[-2][1]:
            x1_change = old_x1_change
            y1_change = old_y1_change
    x1 += x1_change
    y1 += y1_change          
    
    x1 = x1 % WIDTH
    y1 = y1 % HEIGHT
    
    if x1 == fx and y1 == fy:
        snake_len += 1
        food_eaten = True
    
    snake_list.append((x1,y1))
    snake_list = snake_list[-snake_len:]
    
    if len(list(set(snake_list))) < len(snake_list) or len(set(snake_list).intersection(set(obstacles))) > 0:
        print("You lose! Score: %d" % snake_len)
        game_over = True
    else:
        sncols = np.linspace(.5,1.0, len(snake_list))
        for jj, (xx, yy) in enumerate(snake_list):
            pygame.draw.rect(dis, (0, 255*sncols[jj], 32*sncols[jj]), [xx*STEPSIZE, yy*STEPSIZE, STEPSIZE, STEPSIZE])

        for (xx, yy) in np.cumsum(np.array([[.5,.5],snake_list[-1]] + AI_moves), axis=0)[2:]:
            pygame.draw.circle(dis, red, (xx*STEPSIZE,yy*STEPSIZE), STEPSIZE/4)            
        
        if not food_eaten:
            pygame.draw.rect(dis, red, [fx*STEPSIZE, fy*STEPSIZE, STEPSIZE, STEPSIZE])
        
        for xx, yy in obstacles:
            pygame.draw.rect(dis, blue, [xx*STEPSIZE, yy*STEPSIZE, STEPSIZE, STEPSIZE])
        pygame.display.update()
     
        clock.tick(CLOCK_SPEED)
        
        old_x1_change = x1_change
        old_y1_change = y1_change
 
pygame.quit()
quit()
