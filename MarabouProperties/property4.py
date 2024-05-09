import math
import numpy as np
import gurobipy
from gurobipy import Model, GRB, quicksum
from maraboupy import Marabou, MarabouCore

import utils

"""
Even if the previous advisory was “weak right,”
the presence of a nearby intruder will cause the network to
output a “strong left” advisory instead.
+
Input ranges: 2000 ≤ ρ ≤ 5000, 0.7 ≤ θ ≤ 3.141592,
−3.141592 ≤ ψ ≤ −3.141592 + 0.01

Desired output: the score for “strong left” is maximal.

{COC, weak left, weak right, strong left, strong right}

- opposite output looking for a counterexample (y3 is SL)
y3 ≤ y0 or
y3 ≤ y1 or
y3 ≤ y2 or
y3 ≤ y4
"""

def add_constraints(network):
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    # 2000 ≤ ρ ≤ 5000
    network.setLowerBound(inputVars[0], utils.normalize_distance(2000))
    network.setUpperBound(inputVars[0], utils.normalize_distance(5000))
    # 0.7 ≤ θ ≤ 3.141592
    network.setLowerBound(inputVars[1], utils.normalize_angle(0.7))
    network.setUpperBound(inputVars[1], utils.normalize_angle(math.pi))
    # −3.141592 ≤ ψ ≤ −3.141592 + 0.01
    network.setLowerBound(inputVars[2], utils.normalize_angle(-math.pi))
    network.setUpperBound(inputVars[2], utils.normalize_angle(-math.pi + 0.01))

    disjunction = []
    for i in range(5):
        if i != 3:
            # y3 - yi <= 0
            eq = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq.addAddend(1, outputVars[3])
            eq.addAddend(-1, outputVars[i])
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
        # yi + si <= y3 + s4
        m.addConstr(cy[i] + s[i] <= cy[3] + s[3])

    m.setObjective(quicksum(s[i] * s[i] for i in range(len(cy))), GRB.MINIMIZE)
    m.update()
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    for i in range(len(s)):
        if s != 0:
            break
        print("Property 4 satisfied")
        return Inputs, Outputs

    print("Property 4 unsatisfied")
    for i in range(len(cy)):
        cy[i] = cy[i] + s[i].X

    Inputs.append(cx)
    Outputs.append(cy)

    for i in range(N_samples-1):
        # if y3 is not the maximum anymore keep generating
        while True:
            cy_sample = cy + 0.05 * np.random.rand(len(cy))
            if cy_sample[3] >= max(cy_sample):
                Inputs.append(cx)
                Outputs.append(list(cy_sample))
                break

    return Inputs, Outputs
