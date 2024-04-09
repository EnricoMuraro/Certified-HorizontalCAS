from MarabouProperties import property1


def add_constraints(network, property_number):
    if property_number == 1:
        return property1.add_constraints(network)


def generate_samples(counterexample_x, counterexample_y, property_number):
    if property_number == 1:
        return property1.sample(counterexample_x, counterexample_y)
