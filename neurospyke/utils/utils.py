import math

def get_in_samples(value, sampling_time):
    value = math.floor(value/sampling_time)

    return value