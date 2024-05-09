import math
import numpy as np
import gurobipy
from gurobipy import Model, GRB, quicksum
from maraboupy import Marabou, MarabouCore

import utils

"""
If the intruder is directly ahead and is moving
towards the ownship, the score for COC will not be maximal.

Input ranges: 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, 3.10 ≤ ψ ≤ 3.14

Desired output: the score for COC is not the maximal score.

{COC, weak left, weak right, strong left, strong right}

- opposite output looking for a counterexample (y0 is COC)
y0 ≥ y1 and
y0 ≥ y2 and
y0 ≥ y3 and
y0 ≥ y4

"""

def add_constraints(network):
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    # 1500 ≤ ρ ≤ 1800
    network.setLowerBound(inputVars[0], utils.normalize_distance(1500))
    network.setUpperBound(inputVars[0], utils.normalize_distance(1800))

    # −0.06 ≤ θ ≤ 0.06
    network.setLowerBound(inputVars[1], utils.normalize_angle(-0.06))
    network.setUpperBound(inputVars[1], utils.normalize_angle(0.06))

    # 3.10 ≤ ψ ≤ 3.14
    network.setLowerBound(inputVars[2], utils.normalize_angle(3.10))
    network.setUpperBound(inputVars[2], utils.normalize_angle(math.pi))

    for i in range(4):
        # y0 - yi >= 0
        # -y0 + yi <= 0
        network.addInequality([outputVars[0], outputVars[i+1]], [-1, 1], 0)

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
        # y0 + s0 <= yi + si
        m.addConstr(cy[0] + s[0] <= cy[i] + s[i])

    m.setObjective(quicksum(s[i] * s[i] for i in range(len(cy))), GRB.MINIMIZE)
    m.update()
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    for i in range(len(s)):
        if s != 0:
            break
        print("Property 2 satisfied")
        return Inputs, Outputs

    print("Property 2 unsatisfied")
    for i in range(len(cy)):
        cy[i] = cy[i] + s[i].X

    Inputs.append(cx)
    Outputs.append(cy)

    for i in range(N_samples-1):
        # if y0 becomes the maximum keep generating
        while True:
            cy_sample = cy + 0.05 * np.random.rand(len(cy))
            if cy_sample[0] != max(cy_sample):
                Inputs.append(cx)
                Outputs.append(list(cy_sample))
                break

    return Inputs, Outputs
