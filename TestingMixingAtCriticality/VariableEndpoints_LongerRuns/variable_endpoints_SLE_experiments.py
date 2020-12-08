# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:41:17 2020

@author: lnajt
"""


import gerrychain


from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, contract_leaves_until_balanced_or_none, find_balanced_edge_cuts

import pickle
import copy
import Facefinder
import SLEExperiments
from auxiliary_functions_for_variable_endpoint_SLE import *
import numpy as np

'''
This will run gerrychain on instances of the integral disc graph.
Then, for observations ofpartitions, will construct the boundary curve
figure out the nearest points on the disc, and apply a conformal transofmration
taking those endpoints to -1 and 1. Then, it will map up the upper half plane and
check whether it intersects the given balls.

Much of this is covered by code in SLEExperiments. We can use Facefinder to construct
the boundary curve.

'''


def initial_partition(disc):


    updaters = {'population': Tally('population'),'base':new_base, 'b_nodes':b_nodes_bi,
                        'cut_edges': cut_edges, "boundary":bnodes_p,
                        'step_num': step_num, 'geom' : geom_wait,
                        }

    assignment = {}
    for x in disc.nodes():

        if x[0] > 0:
            assignment[x] = 1
        else:
            assignment[x] = -1

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]}.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})


    partition = Partition(disc, assignment, updaters = updaters)

    return partition


def make_sample(data_vector):
    disc = data_vector[0]
    num_steps = data_vector[1]
    x = data_vector[2]
    disc_partition = initial_partition(disc)
    observed_partition = run_chain(disc, disc_partition, num_steps)
    boundary_path = convert_partition_to_boundary(disc, observed_partition)
    boundary_path = convert_path_graph_to_sequence_of_nodes(boundary_path)
    aligned_boundary = conformally_align(boundary_path)
    #Applies a conformal map moving endpoints to -1 and 1.

    observed_trajectory = []
    return [aligned_boundary, observed_trajectory, x]

def debug():

    num_steps = 100000
    disc = integral_disc(10)
    disc_partition = initial_partition(disc)
    exp_chain = MarkovChain(slow_reversible_propose_bi ,Validator([single_flip_contiguous#,boundary_condition
            ]), accept = cut_accept, initial_state=disc_partition,
            total_steps =  num_steps)

    t = 0
    for observation in exp_chain:
        t += 1


    bd = convert_partition_to_boundary(disc, observation)

    Facefinder.draw_with_location(disc)
    Facefinder.draw_with_location(bd, "red")

    path = convert_path_graph_to_sequence_of_nodes(bd)

    aligned_boundary = conformally_align(path)

    return aligned_boundary




def run_chain(disc, disc_partition, num_steps):



    exp_chain = MarkovChain(slow_reversible_propose_bi ,Validator([single_flip_contiguous#,boundary_condition
            ]), accept = cut_accept, initial_state=disc_partition,
            total_steps =  num_steps)

    t = 0
    for observation in exp_chain:
        t += 1



    return observation

def convert_partition_to_boundary(disc, partition):
    dual = disc.graph["dual"]
    dual_cycle = Facefinder.cut_set_to_dual(dual, partition["cut_edges"])
    dual_copy = copy.deepcopy(dual)
    path_subgraph = nx.edge_subgraph(dual_copy, dual_cycle)
    return path_subgraph


def variable_endpoints_estimate_probabilities(disc, radius, num_samples, num_steps):


    count = 0

    #pool = mp.Pool(processes=1)
    #results = pool.map(make_sample, [[disc, num_steps, x] for x in range(num_samples)])

    results = [make_sample([disc, num_steps,x]) for x in range(num_samples)]

    num_samples = len(results)

    for sample_path in results:
        if in_disc([1,0], radius, disc, sample_path[0]):
            count += 1

    right_prob = count / num_samples

    count = 0
    for sample_path in results:
        if in_disc([-1,0], radius, disc, sample_path[0]):
            count += 1

    left_prob = count / num_samples

    return [right_prob, left_prob], results


def test():
    #bases = [mu]
    print("running tests")
    experimental_results = []

    small = 30
    large = 41
    base_num_steps = 100000
    desired_error = .2
    for num_steps in [(10**r) * base_num_steps for r in range(1,3)]:
        for r in range(small,large,10):
            print("num_steps:", num_steps)
            print("r:", r)
            radius = .818610421572298  # (The radius for the half disc ... this value makes the RV Bernoulii(1/2)
            true_mean = 1 - (1 - radius ** 2) ** (5 / 8)
            true_mean = 1 - (1 - radius ** 2) ** (5 / 8)
            true_variance = true_mean*(1 - true_mean)
            var_times_accuracy_sq = true_variance * ( 1 / desired_error)**2
            desired_confidence = .05
            # Want
            num_samples = round(var_times_accuracy_sq / desired_confidence) + 1
            print(num_samples)


            disc = integral_disc(r)

            prob, samples = variable_endpoints_estimate_probabilities(disc, radius, num_samples, num_steps)
            print("for r", r)
            print("correct value",  1-(1-radius**2)**(5/8))
            # https://arxiv.org/pdf/math/0112246.pdf
            print("estimated probabiltiy", prob)
            result_vector = [r, prob, samples, disc]


            name = str(r) + "_samples" + str(num_samples) + "_steps" + str(num_steps)
            #experimental_results.append(result_vector)
            create_plots([result_vector], name)
            with open(str(r) + '_steps:' + str(num_steps) + 'data.data', 'wb') as outfile:
                pickle.dump(result_vector, outfile)
            result_vector = []

    test = []

test()
