import numpy
#pole na którym żyje bakteria
def field(x,y):
    res = 2*(x-y) + 3*x*x - y*2
    return res
#temperatura i ciśnienie
def evol()

    return degree

def f(x):
    return (x+1)*(x+1)-2
function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44
def fitness_func(solution, solution_idx):
    #output = numpy.sum(solution*function_inputs)
    #fitness = 1.0 / numpy.abs(output - desired_output)
    output= f(solution)
    fitness = output
    return fitness