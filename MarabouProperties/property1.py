import math
import numpy as np
import gurobipy
from gurobipy import Model, GRB, quicksum
from maraboupy import Marabou, MarabouCore

import utils

"""
– Description: If the intruder is near and approaching from the left, the network
advises “strong right”.
– Input constraints: 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, −3.141592 ≤ ψ ≤
−3.141592 + 0.005, 100 ≤ v own ≤ 400, 0 ≤ v int ≤ 400.
– Desired output property: the score for “strong right” is the maximal score.

normalized:
ρ range: 56000  mean: 11424

- inputs
-0.1995 ≤ x0 ≤ −0.1969 (ρ)
0.0318 ≤ x1 ≤ 0.0637 (θ)
−0.5 ≤ x2 ≤ -0.4995 (ψ)

{COC, weak left, weak right, strong left, strong right}
- opposite output looking for a counterexample (y4 is strong right)
y4 ≤ y0 or
y4 ≤ y1 or
y4 ≤ y2 or
y4 ≤ y3

"""


def add_constraints(network):
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    # -0.1995 ≤ x0 ≤ −0.1969 (ρ)
    print(f"{utils.normalize_distance(250)} {utils.normalize_distance(400)}")
    network.setLowerBound(inputVars[0], utils.normalize_distance(250))
    network.setUpperBound(inputVars[0], utils.normalize_distance(400))
    # 0.0318 ≤ x1 ≤ 0.0637 (θ)
    print(f"{utils.normalize_angle(0.2)} {utils.normalize_angle(0.4)}")
    network.setLowerBound(inputVars[1], utils.normalize_angle(0.2))
    network.setUpperBound(inputVars[1], utils.normalize_angle(0.4))
    # −0.5 ≤ x2 ≤ -0.4995(ψ)
    print(f"{utils.normalize_angle(-math.pi)} {utils.normalize_angle(-math.pi + 0.005)}")
    network.setLowerBound(inputVars[2], utils.normalize_angle(-math.pi))
    network.setUpperBound(inputVars[2], utils.normalize_angle(-math.pi + 0.005))

    disjunction = []
    for i in range(4):
        # y4 - yi <= 0
        eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        eq.addAddend(1, outputVars[4])
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
        # yi + si <= y4 + s4
        m.addConstr(cy[i] + s[i] <= cy[4] + s[4])

    m.setObjective(quicksum(s[i] * s[i] for i in range(len(cy))), GRB.MINIMIZE)
    m.update()
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    for i in range(len(cy)):
        cy[i] = cy[i] + s[i].X

    Inputs.append(cx)
    Outputs.append(cy)

    for i in range(N_samples-1):
        # if y4 is not the maximum anymore keep generating
        while True:
            cy_sample = cy + 0.05 * np.random.rand(len(cy))
            if cy_sample[4] >= max(cy_sample):
                Inputs.append(cx)
                Outputs.append(list(cy_sample))
                break

    return Inputs, Outputs
