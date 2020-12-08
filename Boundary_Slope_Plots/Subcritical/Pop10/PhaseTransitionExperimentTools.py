# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:44:35 2020

@author: lnajt
"""

import os
import random
import json
import geopandas as gpd
import functools
import datetime
import time
import matplotlib
#matplotlib.use('Agg')

matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
#import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np
import pickle

from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election,Tally,cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap

########### Linked list class for storing boundary angles efficiently

def polar_interpolation(angles, radii):
    #This needs to return a linked list... so its better to build it into the dat structure
    new_angles= []
    new_radii = []
    granularity = 30
    target_granularity = np.pi/100

    for t in range(1,len(radii)):
        last_angle = angles[t - 1]
        last_radii = radii[t - 1]
        current_angle = angles[t]
        granularity = int( angle_distance(last_angle, current_angle) / target_granularity ) + 2

        current_radii = radii[t]

        interpolating_angles = np.linspace( last_angle, current_angle, granularity)[:-1]
        interpolating_radii = np.linspace( last_radii, current_radii, granularity)[:-1]
        new_angles += list(interpolating_angles)
        new_radii += list(interpolating_radii)



    return [new_angles, new_radii]

class Node:
    def __init__(self, data):
        self.data = data
        self.repeats = 1
        self.next = None

        self.start_time = 0
        self.end_time = 1
        ## Amount of time spent there -- used for interpolation.
        ##Actual steps in the markov chain take integer times, interpolants fractional.
    def __repr__(self):
        return str(self.data) + "   " + str(self.repeats)


class LinkedList:
    #Modified linked list optimized for storing the boundary slope plots

    def __init__(self):
        self.head = None
        self.last = None
        self.len = 0
        self.last_non_zero = None
        self.last_vector = [0,0]
    def __len__(self):
        return self.len
    def __repr__(self):

        print("are you sure you want to print?")
        time.sleep(3)
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node)
            #nodes.append(str(node.repeats))
            node = node.next
        #nodes.append("None")
        return str(nodes)


    def __iter__(self):
        #we are going to iterature through skipping repeats

        node = self.head
        #counter = 0
        while node is not None:
            yield node
            node = node.next

            #if counter <= node.repeats:
            #    counter += 1

            #if counter == node.repeats:
            #    counter = 0
            #    node = node.next



    def append(self, data):
        self.len += 1
        if not self.head:
            #case of llist being empty
            new_node = Node(data)
            self.head = new_node
            self.last = new_node


        else:
            if (self.last.data == data):
                self.last.repeats += 1
                self.last.end_time += 1
            else:
                ##Add interpolations here...
                #careful with time variable

                new_node = Node(data)
                new_node.start_time = self.last.end_time + 1
                new_node.end_time = new_node.start_time + 1

                if data == False or self.last.data == False:
                    #the non-simply connected case
                    # either adding a non-simply connected partition,
                    # or picking up from one
                    self.last.next = new_node
                    self.last = new_node


                else:
                    angle_old = self.last.data
                    angle_new = new_node.data

                    #angle_old = np.arctan2( self.last.data[1], self.last.data[0])
                    #angle_new = np.arctan2( new_node.data[1], new_node.data[0])



                    target_granularity = np.pi/200
                    granularity = int( angle_distance(angle_old, angle_new) / target_granularity ) + 2
                    interpolating_angles = np.linspace( angle_old, angle_new, granularity)[:-1]
                    interpolating_times = np.linspace(self.last.end_time, self.last.end_time + 1, granularity)[:-1]


                    try:
                        time_step = interpolating_times[1] - interpolating_times[0]
                        #this could throw an error if the interpolating_times list ends up being length 0 because the granularity is already small...
                    except:
                        time_step = 0
                    interpolants = zip (interpolating_angles, interpolating_times)

                    current_node = self.last
                    for t in interpolants:
                        interpolating_node = Node(t[0])
                        interpolating_node.start_time = t[1]
                        interpolating_node.end_time = t[1] + time_step
                        current_node.next = interpolating_node
                        current_node = interpolating_node


                    current_node.next = new_node
                    self.last = new_node
        if type(self.last.data) != bool:
            self.last_non_zero = self.last

    def last_value(self):
        return self.last_non_zero.data

def previous_point(llist, node):
    for x in llist:
        if x.next == node:
            return x



'''
test = LinkedList()
for i in [False, 0, np.pi/4, False]: # np.pi/2, False, 3* np.pi/2]:
    for t in range(1):
        print(np.array(i))
        test.append(np.array(i))
test
for x in test:
    print(x.data)
test.last


test2 = LinkedList()
for i in [False, 0, np.pi/4, np.pi/2, False, 3* np.pi/2]:
    for t in range(1):
        print(np.array(i))
        test2.append(np.array(i))
test2
for x in test2:
    print(x.data)
test2.last
'''
#########

def annealing_cut_accept_backwards(partition):
    boundaries1  = {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})
    boundaries2  = {x[0] for x in partition.parent["cut_edges"]}.union({x[1] for x in partition.parent["cut_edges"]})

    t = partition["step_num"]


    #if t <100000:
    #    beta = 0
    #elif t<400000:
    #    beta = (t-100000)/100000 #was 50000)/50000
    #else:
    #    beta = 3
    base = .1
    beta = 5

    bound = 1
    if partition.parent is not None:
        #partition.parent["geom"] = geom_wait(partition.parent)

        bound = (base**(beta*(-len(partition["cut_edges"])+len(partition.parent["cut_edges"]))))*(len(boundaries1)/len(boundaries2))

        if not popbound(partition):
            bound = 0
        if not single_flip_contiguous(partition):
            bound = 0
        #bound = min(1, (how_many_seats_value(partition, col1="G17RATG",
         #col2="G17DATG")/how_many_seats_value(partition.parent, col1="G17RATG",
         #col2="G17DATG"))**2  ) #for some states/elections probably want to add 1 to denominator so you don't divide by zero


    return random.random() < bound


def go_nowhere(partition):
    return partition.flip(dict())


def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    flip = random.choice(list(partition["b_nodes"]))

    return partition.flip({flip[0]: flip[1]})

def slow_reversible_propose_bi(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    fnode = random.choice(list(partition["b_nodes"]))

    return partition.flip({fnode: -1*partition.assignment[fnode]})

def geom_wait(partition):
    return int(np.random.geometric(len(list(partition["b_nodes"]))/(len(partition.graph.nodes)**(len(partition.parts))-1) ,1))-1


def b_nodes(partition):
    return {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
               }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})


def uniform_accept(partition):

    bound = 0
    if popbound(partition) and single_flip_contiguous(partition) and boundary_condition(partition):
        bound = 1

    return random.random() < bound




def cut_accept(partition):

    bound = 1
    if partition.parent is not None:
        #print("new proposal")
        #print(partition.parent.geom)
        partition.parent.geom = geom_wait(partition.parent)
        #print(geom_wait(partition.parent))
        #print(partition.parent.geom)
        bound = (partition["base"]**(-len(partition["cut_edges"])+len(partition.parent["cut_edges"])))#*(len(boundaries1)/len(boundaries2))

    c = random.random()
    #print( c < bound)

    return c < bound



def fixed_endpoints(partition):
    return partition.assignment[(19,0)] != partition.assignment[(20,0)] and partition.assignment[(19,39)] != partition.assignment[(20,39)]


def boundary_condition(partition):

    blist = partition["boundary"]
    o_part = partition.assignment[blist[0]]

    for x in blist:
        if partition.assignment[x] != o_part:
            return True

    return False


def boundary_slope(partition):

    a=[]
    b=[]
    c=[]
    d=[]
    e=[]

    for x in partition["cut_edges"]:
        if x[0][0] == 0 and x[1][0] == 0:
            a.append(x)
        elif x[0][1] == 0 and x[1][1] == 0:
            b.append(x)
        elif x[0][0] == 39 and x[1][0] == 39:
            c.append(x)
        elif x[0][1] == 39 and x[1][1] == 39:
            d.append(x)
        #elif x in [((0,1),(1,0)), ((0,38),(1,39)), ((38,0),(39,1)), ((38,39),(39,38))]:
        #    e.append(x)
        #elif x in [((1,0),(0,1)), ((1,39),(0,38)), ((39,1),(38,0)), ((39,38),(38,39))]:
        #    e.append(x)


    return list(set(a+b+c+d+e))

def boundary_ends(partition):
    #Writing a version that works more generally. We wont run it every step of the chain.
    #SInce it's a grid, we'll rotate each boundary edge by 90 degrees, and extract the 2 positions we only see once this way.
    boundary_points = []

    for x in partition["cut_edges"]:
        a = np.asarray(x[0])
        b = np.asarray(x[1])

        #print("ab: ",a,b)
        center = (a + b)/2
        #print(center)
        l = a - center
        r = b - center
        #print("lr: ", l,r)
        l_rotated = np.asarray( [-1* l[1],  l[0]] )
        r_rotated = np.asarray( [-1* r[1], r[0]] )
        #print("rot: ", l_rotated, r_rotated)
        p = l_rotated + center
        q = r_rotated + center
        #print(p,q)
        boundary_points.append(p)
        boundary_points.append(q)

    ##Set ends by removing elements bof boundary points appearing twice

    ends = [ x for x in boundary_points if len ( [y for y in boundary_points if y[0] == x[0] and y[1] == x[1]]) == 1 ]
    #probably there's a better way

    #Note: IN the case that ends is empty, that means that the boundary was a circle. One of the districts wasn't simply connected.
    ## The case that ends consists of 1 element is also exceptional.

    #There's no natural order here, so make sure to mod out by that in the graphing.


    return ends

def angle_distance(angle_1, angle_2):
    #these angles come from arctan2, so are radians in [-pi, pi]
    #return the angle between them.

    d = abs(angle_1 - angle_2)
    if d > np.pi:
        return 2 * np.pi - d
    return d



def draw_other_plots(balances, graph, alignment, identifier_string, base, pop1, part, ns):
    # Put here to avoid interfering with the ongoing
    # plt environment -- probably not the right way
    # to do things

    plt.figure()
    plt.title("Balances")
    plt.bar(balances.keys(), balances.values(), .01, color='g')
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "balances.svg")
    plt.close()



    plt.figure()
    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [0 for x in graph.nodes()] ,node_size = 10, edge_color = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape ='s',cmap = 'jet',width =5)
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "edges.svg")
    plt.close()



    plt.figure()
    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "end.svg")
    plt.close()

    '''
    A2 = np.zeros([40,40])

    for n in graph.nodes():
        A2[n[0],n[1]] = dict(part.assignment)[n]

    plt.figure()
    plt.imshow(A2,cmap='jet')
    plt.colorbar()
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "end2.svg")
    plt.close()






    plt.figure()
    plt.title("Flips")
    nx.draw(graph,pos= {x:x for x in graph.nodes()},node_color=[graph.nodes[x]["num_flips"] for x in graph.nodes()],node_size=ns,node_shape='s',cmap="jet")
    plt.title("Flips")
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "flip.svg")
    plt.close()


    A2 = np.zeros([40,40])

    for n in graph.nodes():
        A2[n[0],n[1]] = graph.nodes[n]["num_flips"]


    plt.figure()
    plt.imshow(A2,cmap='jet')
    plt.colorbar()
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "flip2.svg")
    plt.close()


    plt.figure()
    plt.title("Flips")
    nx.draw(graph,pos= {x:x for x in graph.nodes()},node_color=[graph.nodes[x]["lognum_flips"] for x in graph.nodes()],node_size=ns,node_shape='s',cmap="jet")
    plt.title("Flips")
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "logflip.svg")
    plt.close()


    A2 = np.zeros([40,40])

    for n in graph.nodes():
        A2[n[0],n[1]] = graph.nodes[n]["lognum_flips"]


    plt.figure()
    plt.imshow(A2,cmap='jet')
    plt.colorbar()
    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "logflip2.svg")
    plt.close()

'''
