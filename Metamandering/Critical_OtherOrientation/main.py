import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np


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


mu = 2.63815853
bases = [mu]
base=.1
pops=[.05,.1,.5]


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
        elif x in [((0,1),(1,0)), ((0,38),(1,39)), ((38,0),(39,1)), ((38,39),(39,38))]:
            e.append(x)
        elif x in [((1,0),(0,1)), ((1,39),(0,38)), ((39,1),(38,0)), ((39,38),(38,39))]:
            e.append(x)


    return list(set(a+b+c+d+e))


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
        bound = (base**(beta*(-len(partition["cut_edges"])+len(partition.parent["cut_edges"]))))*(len(boundaries1)/len(boundaries2))

        #if not popbound(partition):
        #    bound = 0
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
    if single_flip_contiguous(partition) and boundary_condition(partition):
        bound = 1

    return random.random() < bound

#BUILD GRAPH



def cut_accept(partition):

    bound = 1
    if partition.parent is not None:
        bound = (partition["base"]**(-len(partition["cut_edges"])+len(partition.parent["cut_edges"])))#*(len(boundaries1)/len(boundaries2))



    return random.random() < bound



def biased_diagonals(m):
    G = nx.grid_graph([4 * m, 4 * m])

    for n in G.nodes():
        if ((4 - width) * m - 1 <= n[1] <= 4 * m - 2 or 0 <= n[1] <= width * m - 1) and n[0] <= 4* m - 2:
            G.add_edge(n, (n[0] + 1, n[1] + 1))
        G.nodes[n]['pos'] = (n[0], n[1])
    return G

def debiased_diagonals(m):
    G = nx.grid_graph([4 * m, 4 * m])

    for n in G.nodes():
        if n[0] % 2 == 0:
            if ((4 - width) * m - 1 <= n[0] <= 4 * m - 2 or 0 <= n[0] <= width * m - 1) and n[1] <= 4* m - 2:
                G.add_edge(n, (n[0] + 1, n[1] + 1))
        if n[0] % 2 == 1:
            if ((4 - width) * m - 1 <= n[0] <= 4 * m - 2 or 0 <= n[0] <= width * m - 1) and n[1] <= 4* m - 2:
                G.add_edge(n, (n[0] + 1, n[1] - 1))
        G.nodes[n]['pos'] = (n[0], n[1])
    return G

width = 1.5

for pop1 in pops:
    for base in bases:
        for alignment in [0]:

            gn=20
            k=2
            ns=120
            p=.5

            #graph=nx.grid_graph([k*gn,k*gn])
            graph = biased_diagonals(10)
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

            graph.add_edges_from([((0,1),(1,0)), ((0,38),(1,39)), ((38,0),(39,1)), ((38,39),(39,38))])

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


            graph.remove_nodes_from([(0,0),(0,39),(39,0),(39,39)])

            del cddict[(0,0)]

            del cddict[(0,39)]

            del cddict[(39,0)]

            del cddict[(39,39)]
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
                                'cut_edges': cut_edges,
                                'step_num': step_num,
                                'b_nodes':b_nodes_bi,
                                'base':new_base,
                                'geom':geom_wait,
                                #"Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                    }





            #########BUILD PARTITION

            grid_partition = Partition(graph,assignment=cddict,updaters=updaters)

            #ADD CONSTRAINTS
            popbound=within_percent_of_ideal_population(grid_partition,pop1)

            plt.figure()
            nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(grid_partition.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
            plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"start.png")
            plt.close()


            #########Setup Proposal
            ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)

            tree_proposal = partial(recom,
                                   pop_col="population",
                                   pop_target=ideal_population,
                                   epsilon=0.05,
                                   node_repeats=1
                                  )

            #######BUILD MARKOV CHAINS


            exp_chain = MarkovChain(slow_reversible_propose_bi ,Validator([single_flip_contiguous, popbound#,boundary_condition
            ]), accept = cut_accept, initial_state=grid_partition,
            total_steps =  1000000)





            #########Run MARKOV CHAINS

            rsw = []
            rmm = []
            reg = []
            rce = []
            rbn=[]
            waits= []

            slopes = []
            angles = []

            import time

            st = time.time()


            t=0
            for part in exp_chain:
                rce.append(len(part["cut_edges"]))
                waits.append(part["geom"])
                rbn.append(len(list(part["b_nodes"])))
                #print(part["slope"])
                #temp = part["slope"]

                #enda = ((temp[0][0][0]+temp[0][1][0])/2,(temp[0][0][1]+temp[0][1][1])/2)
                #endb = ((temp[1][0][0]+temp[1][1][0])/2,(temp[1][0][1]+temp[1][1][1])/2)

                #if endb[0]!= enda[0]:

                #    slope = (endb[1]-enda[1])/(endb[0]-enda[0])
                #else:
                #    slope = np.Inf

                #slopes.append(slope)
                for edge in part["cut_edges"]:
                    graph[edge[0]][edge[1]]["cut_times"] += 1
                    #print(graph[edge[0]][edge[1]]["cut_times"])



                #anga = (temp[0][0]-20,temp[0][1]-20)
                #angb = (temp[1][0]-20,temp[1][1]-20)
                #anga = (enda[0]-20,enda[1]-20)
                #angb = (endb[0]-20,endb[1]-20)

                #angles.append(np.arccos(np.clip(np.dot(anga / np.linalg.norm(anga),angb / np.linalg.norm(angb)),-1,1)))

                if part.flips is not None:
                    f = list(part.flips.keys())[0]
                    graph.nodes[f]["part_sum"]=graph.nodes[f]["part_sum"]-part.assignment[f]*(t-graph.nodes[f]["last_flipped"])
                    graph.nodes[f]["last_flipped"]=t
                    graph.nodes[f]["num_flips"]=graph.nodes[f]["num_flips"]+1

                t+=1
                """
                plt.figure()
                nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
                plt.savefig(f"./Figures/recom_{part['step_num']:02d}.png")
                plt.close()
                """
                if t % 10000 == 0:

                    identifier_string = "state_after_num_steps" + str(t) + "and_time" + str(st-time.time())
                    #print("finished no", st-time.time())
                    with open("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+  identifier_string + "wait.txt",'w') as wfile:
                        wfile.write(str(sum(waits)))




                    for n in graph.nodes():
                        if graph.nodes[n]["last_flipped"] == 0:
                            graph.nodes[n]["part_sum"]=t*part.assignment[n]
                        graph.nodes[n]["lognum_flips"] = math.log(graph.nodes[n]["num_flips"] + 1)




                    #print(len(rsw[-1]))
                    print(graph[(1,0)][(0,1)]["cut_times"])

                    plt.figure()
                    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [0 for x in graph.nodes()] ,node_size = 10, edge_color = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape ='s',cmap = 'jet',width =5)
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "edges.png")
                    plt.close()



                    plt.figure()
                    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "end.png")
                    plt.close()


                    A2 = np.zeros([40,40])

                    for n in graph.nodes():
                        A2[n[0],n[1]] = dict(part.assignment)[n]


                    plt.figure()
                    plt.imshow(A2,cmap='jet')
                    plt.colorbar()
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "end2.png")
                    plt.close()


                    plt.figure()
                    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [graph.nodes[x]["part_sum"] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'jet')
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "wca.png")
                    plt.close()


                    A2 = np.zeros([40,40])

                    for n in graph.nodes():
                        A2[n[0],n[1]] = graph.nodes[n]["part_sum"]


                    plt.figure()
                    plt.imshow(A2,cmap='jet')
                    plt.colorbar()

                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "wca2.png")
                    plt.close()



                    #plt.figure()
                    #plt.title("Slopes")
                    #plt.plot(slopes)
                    #plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "slope.png")
                    #plt.close()

                    #plt.figure()
                    #plt.title("Angle")
                    #plt.plot(angles)
                    #plt.ylim([0,6.3])
                    #plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "angle.png")
                    #plt.close()



                    plt.figure()
                    plt.title("Flips")
                    nx.draw(graph,pos= {x:x for x in graph.nodes()},node_color=[graph.nodes[x]["num_flips"] for x in graph.nodes()],node_size=ns,node_shape='s',cmap="jet")
                    plt.title("Flips")
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "flip.png")
                    plt.close()


                    A2 = np.zeros([40,40])

                    for n in graph.nodes():
                        A2[n[0],n[1]] = graph.nodes[n]["num_flips"]


                    plt.figure()
                    plt.imshow(A2,cmap='jet')
                    plt.colorbar()
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "flip2.png")
                    plt.close()


                    plt.figure()
                    plt.title("Flips")
                    nx.draw(graph,pos= {x:x for x in graph.nodes()},node_color=[graph.nodes[x]["lognum_flips"] for x in graph.nodes()],node_size=ns,node_shape='s',cmap="jet")
                    plt.title("Flips")
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "logflip.png")
                    plt.close()


                    A2 = np.zeros([40,40])

                    for n in graph.nodes():
                        A2[n[0],n[1]] = graph.nodes[n]["lognum_flips"]


                    plt.figure()
                    plt.imshow(A2,cmap='jet')
                    plt.colorbar()
                    plt.savefig("./plots/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+identifier_string + "logflip2.png")
                    plt.close()
