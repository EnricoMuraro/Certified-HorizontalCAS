import math
import numpy as np
import gurobipy
from gurobipy import Model, GRB, quicksum
from maraboupy import Marabou, MarabouCore

import utils

"""
For a far away intruder, the network advises COC even if in front.

Input ranges: 44000 ≤ ρ ≤ 50000, -1 ≤ θ ≤ 1, −3.141592 ≤ ψ ≤ −3.141592 + 0.01

Desired output: the score for COC is maximal.

{COC, weak left, weak right, strong left, strong right}

- opposite output looking for a counterexample (y0 is COC)
y0 ≤ y1 or
y0 ≤ y2 or
y0 ≤ y3 or
y0 ≤ y4

"""

def add_constraints(network):
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    # 44000 ≤ ρ ≤ 50000
    network.setLowerBound(inputVars[0], utils.normalize_distance(44000))
    network.setUpperBound(inputVars[0], utils.normalize_distance(50000))
    # 1 ≤ θ ≤ 1
    network.setLowerBound(inputVars[1], utils.normalize_angle(-1))
    network.setUpperBound(inputVars[1], utils.normalize_angle(1))
    # −3.141592 ≤ ψ ≤ −3.141592 + 0.01
    network.setLowerBound(inputVars[2], utils.normalize_angle(-math.pi))
    network.setUpperBound(inputVars[2], utils.normalize_angle(-math.pi + 0.01))

    disjunction = []
    for i in range(4):
        # y0 - yi <= 0
        eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        eq.addAddend(1, outputVars[0])
        eq.addAddend(-1, outputVars[i+1])
        eq.setScalar(0)
        disjunction.append([eq])

    network.addDisjunctionConstraint(disjunction)
    return network


def sample(cx, cy, N_samples = 10):
    Inputs = []
    Outputs = []

    env = gurobipy.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model("repair", env=env)
    s = []
    for i in range(len(cy)):
        s.append(m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s{}".format(i)))

    m.update()
    for i in range(len(cy)):
        # yi + si <= y0 + s0
        m.addConstr(cy[i] + s[i] <= cy[0] + s[0])

    m.setObjective(quicksum(s[i] * s[i] for i in range(len(cy))), GRB.MINIMIZE)
    m.update()
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    for i in range(len(s)):
        if s != 0:
            break
        print("Property 5 satisfied")
        return Inputs, Outputs

    print("Property 5 unsatisfied")
    for i in range(len(cy)):
        cy[i] = cy[i] + s[i].X

    Inputs.append(cx)
    Outputs.append(cy)

    for i in range(N_samples-1):
        # if y0 is not the maximum anymore keep generating
        while True:
            cy_sample = cy + 0.05 * np.random.rand(len(cy))
            if cy_sample[0] >= max(cy_sample):
                Inputs.append(cx)
                Outputs.append(list(cy_sample))
                break

    return Inputs, Outputs
