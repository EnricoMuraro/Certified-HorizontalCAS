from MarabouProperties import property1
from MarabouProperties import property2
from MarabouProperties import property3
from MarabouProperties import property4
from MarabouProperties import property5

def add_constraints(network, property_number):
    if property_number == 1:
        network = property1.add_constraints(network)
    if property_number == 2:
        network = property2.add_constraints(network)
    if property_number == 3:
        network = property3.add_constraints(network)
    if property_number == 4:
        network = property4.add_constraints(network)
    if property_number == 5:
        network = property5.add_constraints(network)
    return network


def generate_samples(counterexample_x, counterexample_y, property_number):
    if property_number == 1:
        return property1.sample(counterexample_x, counterexample_y)
    if property_number == 2:
        return property2.sample(counterexample_x, counterexample_y)
    if property_number == 3:
        return property3.sample(counterexample_x, counterexample_y)
    if property_number == 4:
        return property4.sample(counterexample_x, counterexample_y)
    if property_number == 5:
        return property5.sample(counterexample_x, counterexample_y)