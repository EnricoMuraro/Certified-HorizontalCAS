import math
import numpy as np
import gurobipy
from gurobipy import Model, GRB, quicksum
from maraboupy import Marabou, MarabouCore

import utils

"""
– Description: If the intruder is near and approaching from the right, the network
advises “strong left”.
– Input constraints: 250 ≤ ρ ≤ 400, -0.4 ≤ θ ≤ -0.2, −3.141592 ≤ ψ ≤
−3.141592 + 0.005, 100 ≤ v own ≤ 400, 0 ≤ v int ≤ 400.
– Desired output property: the score for “strong left” is the maximal score.


{COC, weak left, weak right, strong left, strong right}
- opposite output looking for a counterexample (y3 is strong left)
y3 ≤ y0 or
y3 ≤ y1 or
y3 ≤ y2 or
y3 ≤ y4

"""


def add_constraints(network):
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    network.setLowerBound(inputVars[0], utils.normalize_distance(250))
    network.setUpperBound(inputVars[0], utils.normalize_distance(400))

    network.setLowerBound(inputVars[1], utils.normalize_angle(-0.4))
    network.setUpperBound(inputVars[1], utils.normalize_angle(-0.2))

    network.setLowerBound(inputVars[2], utils.normalize_angle(-math.pi))
    network.setUpperBound(inputVars[2], utils.normalize_angle(-math.pi + 0.005))

    disjunction = []
    for i in range(4):
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
        # yi + si <= y3 + s3
        m.addConstr(cy[i] + s[i] <= cy[3] + s[3])

    m.setObjective(quicksum(s[i] * s[i] for i in range(len(cy))), GRB.MINIMIZE)
    m.update()
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    for i in range(len(s)):
        if s != 0:
            break
        print("Property 6 satisfied")
        return Inputs, Outputs

    print("Property 6 unsatisfied")
    for i in range(len(cy)):
        cy[i] = cy[i] + s[i].X

    Inputs.append(cx)
    Outputs.append(cy)

    for i in range(N_samples-1):
        # if y4 is not the maximum anymore keep generating
        while True:
            cy_sample = cy + 0.05 * np.random.rand(len(cy))
            if cy_sample[3] >= max(cy_sample):
                Inputs.append(cx)
                Outputs.append(list(cy_sample))
                break

    return Inputs, Outputs
