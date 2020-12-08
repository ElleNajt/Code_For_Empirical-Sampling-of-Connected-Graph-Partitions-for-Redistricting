

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
import gc

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
from PhaseTransitionExperimentTools import *


def run_experiment(bases = [2*  2.63815853], pops = [.1],     time_between_outputs = 10000,     total_run_length = 100000000000000  ):

    mu = 2.63815853
    subsequence_step_size = 10000
    balances_burn_in = 1000000 #ignore the first 10000 balances

    # creating the boundary figure plot
    plt.figure()
    fig = plt.figure()
    #fig_intervals = plt.figure()
    #ax2=fig.add_axes([0,0,1,1])
    ax = plt.subplot(111, projection='polar')
    #ax.set_axis_off()
    #ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])


    for pop1 in pops:
        for base in bases:
            for alignment in [1]:

                gn=20
                k=2
                ns=120
                p=.5

                graph=nx.grid_graph([k*gn,k*gn])

                ########## BUILD ASSIGNMENT
                #cddict = {x: int(x[0]/gn)  for x in graph.nodes()}
                cddict = {x: 1-2*int(x[0]/gn)  for x in graph.nodes()}

                for n in graph.nodes():
                    if alignment == 0:
                        if n[0] > 19:
                            cddict[n] =1
                        else:
                            cddict[n]=-1
                    elif alignment == 1:
                        if n[1] > 19:
                            cddict[n] =1
                        else:
                            cddict[n]=-1
                    elif alignment == 2:
                        if n[0]>n[1]:
                            cddict[n] = 1
                        elif n[0] == n[1] and n[0]>19:
                            cddict[n] = 1
                        else:
                            cddict[n] = -1
                    elif alignment == 10:
                        #This is for debugging the case of reaching trivial partitions.
                        if n[0] == 10 and n[1] == 10:
                            cddict[n] = 1
                        else:
                            cddict[n] = -1

                for n in graph.nodes():
                    graph.nodes[n]["population"]=1
                    graph.nodes[n]["part_sum"]=cddict[n]
                    graph.nodes[n]["last_flipped"]=0
                    graph.nodes[n]["num_flips"]=0

                    if random.random()<p:
                        graph.nodes[n]["pink"]=1
                        graph.nodes[n]["purple"]=0
                    else:
                        graph.nodes[n]["pink"]=0
                        graph.nodes[n]["purple"]=1
                    if 0 in n or k*gn-1 in n:
                        graph.nodes[n]["boundary_node"]=True
                        graph.nodes[n]["boundary_perim"]=1

                    else:
                        graph.nodes[n]["boundary_node"]=False

                #graph.add_edges_from([((0,1),(1,0)), ((0,38),(1,39)), ((38,0),(39,1)), ((38,39),(39,38))])

                for edge in graph.edges():
                    graph[edge[0]][edge[1]]['cut_times'] = 0

                #this part adds queen adjacency
                #for i in range(k*gn-1):
                #    for j in range(k*gn):
                #        if j<(k*gn-1):
                #            graph.add_edge((i,j),(i+1,j+1))
                #            graph[(i,j)][(i+1,j+1)]["shared_perim"]=0
                #        if j >0:
                #            graph.add_edge((i,j),(i+1,j-1))
                #            graph[(i,j)][(i+1,j-1)]["shared_perim"]=0


                #graph.remove_nodes_from([(0,0),(0,39),(39,0),(39,39)])

                #del cddict[(0,0)]

                #del cddict[(0,39)]

                # cddict[(39,0)]

                #del cddict[(39,39)]
                ######PLOT GRIDS
                """
                plt.figure()
                nx.draw(graph, pos = {x:x for x in graph.nodes()} ,node_size = ns, node_shape ='s')
                plt.show()

                cdict = {1:'pink',0:'purple'}

                plt.figure()
                nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [cdict[graph.nodes[x]["pink"]] for x in graph.nodes()],node_size = ns, node_shape ='s' )
                plt.show()

                plt.figure()
                nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [cddict[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
                plt.show()
                """
                ####CONFIGURE UPDATERS

                def new_base(partition):
                    return base

                def step_num(partition):
                    parent = partition.parent

                    if not parent:
                        return 0


                    return parent["step_num"] + 1


                bnodes = [x for x in graph.nodes() if graph.nodes[x]["boundary_node"] ==1]

                def bnodes_p(partition):


                    return [x for x in graph.nodes() if graph.nodes[x]["boundary_node"] ==1]

                updaters = {'population': Tally('population'),
                                    "boundary":bnodes_p,
                                    #"slope": boundary_slope,
                                    'cut_edges': cut_edges,
                                    'step_num': step_num,
                                    'b_nodes':b_nodes_bi,
                                    'base':new_base,
                                    'geom':geom_wait,
                                    #"Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                        }



                balances = []

                #########BUILD PARTITION

                grid_partition = Partition(graph,assignment=cddict,updaters=updaters)

                #ADD CONSTRAINTS
                popbound=within_percent_of_ideal_population(grid_partition,pop1)

                #plt.figure()
                #nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(grid_partition.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
                #plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"start.png")
                #plt.close()


                #########Setup Proposal
                ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)

                tree_proposal = partial(recom,
                                       pop_col="population",
                                       pop_target=ideal_population,
                                       epsilon=pop1,
                                       node_repeats=1
                                      )

                #######BUILD MARKOV CHAINS


                exp_chain = MarkovChain(slow_reversible_propose_bi ,Validator([single_flip_contiguous,popbound#,boundary_condition
                ]), accept = cut_accept, initial_state=grid_partition,
                total_steps = total_run_length)





                #########Run MARKOV CHAINS

                rsw = []
                rmm = []
                reg = []
                rce = []
                rbn=[]
                waits= []

                slopes = []
                angles = []
                angles_safe = []
                ends_vectors_normalized = LinkedList()
                ends_vectors_normalized_bloated = LinkedList()
                import time

                st = time.time()

                total_waits = 0
                last_total_waits = 0
                t=0

                subsequence_timer = 0


                balances = {}
                for b in np.linspace(0,2,100001):
                    balances[int(b*100)/100] = 0

                #first_partition = True
                for part in exp_chain:
                    rce.append(len(part["cut_edges"]))
                    wait_time_rv = part.geom
                    waits.append(wait_time_rv)
                    total_waits += wait_time_rv
                    rbn.append(len(list(part["b_nodes"])))




                    if total_waits > subsequence_timer + subsequence_step_size:

                        last_total_waits = total_waits

                        ends = boundary_ends(part)
                        if len(ends) == 2:
                            ends_vector = np.asarray(ends[1]) - np.asarray(ends[0])
                            ends_vector_normalized = ends_vector / np.linalg.norm(ends_vector)

                            #if first_partition == True:
                            #    ends_vectors_normalized.last_vector = ends_vector_normalized
                            #    first_partition = False

                            if ends_vectors_normalized.last:
                                # We choose the vector that preserves continuity
                                # previous_angle = ends_vectors_normalized.last_value()
                                previous = ends_vectors_normalized.last_vector

                                d_previous = np.linalg.norm( previous - ends_vector_normalized )
                                d_previous_neg = np.linalg.norm( previous + ends_vector_normalized )
                                if d_previous < d_previous_neg:
                                    continuous_lift = ends_vector_normalized
                                else:
                                    continuous_lift = -1* ends_vector_normalized
                                    #print(previous, ends_vector_normalized)

                            else:
                                continuous_lift = ends_vector_normalized # *random.choice([-1,1])
                                # just to debias it, in the regime of very unbalanced partitions
                                # that touch the empty partition frequently

                        else:
                            continuous_lift = [0,0]



                        ##############For Debugging#############
                        '''
                        if total_waits > subsequence_timer + subsequence_step_size:

                            last_total_waits = total_waits

                            ends = boundary_ends(part)
                            if ends:
                                ends_vector = np.asarray(ends[1]) - np.asarray(ends[0])
                                ends_vector_normalized = ends_vector / np.linalg.norm(ends_vector)

                                if ends_vectors_normalized_bloated.last:
                                    # We choose the vector that preserves continuity
                                    previous = ends_vectors_normalized_bloated.last_value()
                                    d_previous = np.linalg.norm( ends_vector_normalized - previous)
                                    d_previous_neg = np.linalg.norm( ends_vector_normalized + previous )
                                    if d_previous < d_previous_neg:
                                        continuous_lift_bloated = ends_vector_normalized
                                    else:
                                        continuous_lift_bloated = -1* ends_vector_normalized

                                else:
                                    continuous_lift_bloated = ends_vector_normalized # *random.choice([-1,1])
                                    # just to debias it, in the regime of very unbalanced partitions
                                    # that touch the empty partition frequently

                            else:
                                continuous_lift_bloated = [0,0]
                                '''
                        ################

                        # Pop balance stuff:
                        left_pop, right_pop = part["population"].values()
                        ideal_population = (left_pop + right_pop)/2
                        left_bal = (left_pop/ideal_population)
                        right_bal = (right_pop/ideal_population)



                        while subsequence_timer < total_waits:
                            subsequence_timer += subsequence_step_size
                            if (continuous_lift == np.asarray([0,0])).all():
                                lifted_angle = False
                                print("false")
                                draw_other_plots(balances, graph, alignment, "NonSimplyConnected", base, pop1, part, ns)
                                #Flag to hold the exceptional case of the boundary vanishing
                            else:
                                lifted_angle = np.arctan2( continuous_lift[1], continuous_lift[0])
                                #+ np.pi
                                ends_vectors_normalized.last_vector = continuous_lift
                            ends_vectors_normalized.append(lifted_angle)


                            if subsequence_timer > balances_burn_in:
                                left_bal_rounded = int( left_bal * 100)/100
                                right_bal_rounded = int( right_bal * 100)/100
                                balances[left_bal_rounded] += 1 #left_bal_rounded
                                balances[right_bal_rounded] += 1 # right_bal_rounded
                                #NB wait times are accounted for by the while loops

                    for edge in part["cut_edges"]:
                        graph[edge[0]][edge[1]]["cut_times"] += wait_time_rv
                        #print(graph[edge[0]][edge[1]]["cut_times"])


                    if part.flips is not None:
                        f = list(part.flips.keys())[0]

                        graph.nodes[f]["part_sum"]=graph.nodes[f]["part_sum"]-part.assignment[f]*(total_waits-graph.nodes[f]["last_flipped"])
                        graph.nodes[f]["last_flipped"]=total_waits
                        graph.nodes[f]["num_flips"]=graph.nodes[f]["num_flips"]+wait_time_rv

                    t+=1






                    if t % time_between_outputs == 0:

                        #ends_vectors_normalized[1:] #Remove the first one because it will overlap with last one of previous dump

                        identifier_string = "state_after_num_steps" + str(t) + "and_time" + str(st-time.time())

                        #print("finished no", st-time.time())
                        with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + "wait.txt",'w') as wfile:
                            wfile.write(str(sum(waits)))

                        #with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + "ends_vectors.txt",'w') as wfile:
                        #    wfile.write(str(ends_vectors_normalized))



                        #with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + "ends_vectors.pkl",'wb') as wfile:
                        #    pickle.dump(ends_vectors_normalized, wfile)


                        with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + "balances.txt",'w') as wfile:
                            wfile.write(str(balances))
                        with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + "balances.pkl",'wb') as wfile:
                            pickle.dump(balances, wfile)



                        for n in graph.nodes():
                            if graph.nodes[n]["last_flipped"] == 0:
                                graph.nodes[n]["part_sum"]=total_waits*part.assignment[n]
                            graph.nodes[n]["lognum_flips"] = math.log(graph.nodes[n]["num_flips"] + 1)



                        total_part_sum = 0
                        for n in graph.nodes():
                            total_part_sum += graph.nodes[n]["part_sum"]



                        for n in graph.nodes():
                            if total_part_sum != 0:
                                graph.nodes[n]["normalized_part_sum"] = graph.nodes[n]["part_sum"] / total_part_sum
                            else:
                                graph.nodes[n]["normalized_part_sum"] = 0

                        #print(len(rsw[-1]))
                        #print(graph[(1,0)][(0,1)]["cut_times"])


                        print("creating boundary plot, ",  time.time())

                        max_time = ends_vectors_normalized.last.end_time
                        non_simply_connected_intervals = [ [x.start_time , x.end_time ] for x in ends_vectors_normalized if type(x.data) == bool ]

                        for x in ends_vectors_normalized:
                            if type(x.data) != bool:

                                #times = np.linspace(x.start_time, x.end_time, 100)
                                #times.append(x.end_time)

                                times = [x.start_time, x.end_time]
                                angles = [x.data] * len( times)
                                plt.polar ( angles,times, lw = .1, color = 'b')


                                next_point = x.next
                                #'''
                                if next_point != None:
                                    if type(next_point.data) != bool:
                                        if np.abs( (x.data - next_point.data)) % (2 * np.pi)   < .1:
                                            # added that last if to avoid
                                            # the big jumps that happen with
                                            # small size subcritical
                                            plt.polar ( [x.data, next_point.data],[x.end_time, next_point.start_time], lw = .1, color = 'b')
                                #'''


                        # Create the regular segments corresponding to time
                        '''
                        for k in range(11):
                            plt.polar ( np.arange(0, (2 * np.pi), 0.01), [int(max_time/10) * k ] * len( np.arange(0, (2 * np.pi), 0.01)), lw = .2, color = 'g' )
                        '''
                        # Create the intervals representing when the partition is null.
                        # Removing these might be just as good, and a little cleaner.
                        #'''
                        for interval in non_simply_connected_intervals:
                            start = interval[0]
                            end = interval[1]
                            for s in np.linspace(start,end,10):
                                plt.polar ( np.arange(0, (2 * np.pi), 0.01), s * np.ones(len( np.arange(0, (2 * np.pi), 0.01))), lw = .3, color = 'r' )
                        #'''


                        #plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + str("proposals_") + str( max_time * subsequence_step_size ) + "boundary_slope.svg")
                        plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1)) + str("proposals_") + identifier_string + "boundary_slope.png", dpi=500)

                        # now clear the ends vectors list
                        last = ends_vectors_normalized.last
                        last_non_zero = ends_vectors_normalized.last_non_zero
                        last_vector = ends_vectors_normalized.last_vector
                        # Explicit Garbage collection https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python

                        print("finished boundary plot, doing garbage collection", time.time())
                        del ends_vectors_normalized
                        gc.collect()



                        ends_vectors_normalized = LinkedList()
                        ends_vectors_normalized.head = last
                        ends_vectors_normalized.last = last
                        ends_vectors_normalized.last_non_zero = last_non_zero # can be ahead of head...
                        ends_vectors_normalized.last_vector = last_vector
                        #print(last)

                        print("drawing other plots",  time.time())
                        draw_other_plots(balances, graph, alignment, identifier_string, base, pop1, part, ns)
                        print("finished drawing other plots, " ,  time.time())
