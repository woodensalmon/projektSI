import numpy
import pygad
import pygad.nn
import pygad.gann
import pandas as pd

nazwa_pliku = "plik_z_danymi.csv"
nazwa_pliku2 = "plik_z_danymi_liczbowo.csv"
nazwa_pliku3 = "plik_z_danymi_liczbowo_wieksze.csv"


def load_panda():
    df_in = pd.read_csv(nazwa_pliku3, sep=',', header=0, usecols=[0, 1, 2, 3])
    df_out = pd.read_csv(nazwa_pliku3, sep=',', header=0, usecols=[4])
    out = numpy.concatenate(df_out.values)
    print(df_in)
    print(type(df_in.values))
    return df_in.values, out


inp, out = load_panda()
print("łodpalony")
print(inp)
print(out)

# nieużywane
"""
def load_file():
    with open('plik_z_danymi.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        inputs = numpy.empty((1,4))
        output = numpy.empty((1,1))
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are \n\t {", ".join(row)}')
                line_count += 1
            else:
                print(f'\t{row[0]}  {row[1]}  {row[2]} {row[3]} {row[4]} ')
                line_count += 1
                myrow_in=numpy.array([row[0], row[1], row[2], row[3]])
                numpy.concatenate((inputs[line_count-1], myrow_in))
                #numpy.concatenate((output, row[4]))

        print(f'Processed {line_count} lines.')
    return inputs

inp,out=load_file()
print(inp)
print(out)
"""

# nieużywane
"""
def sigm():

    x = numpy.linspace(-10, 10, 100)
    z = 1 / (1 + numpy.exp(-x))

    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("Sigmoid(X)")

    plt.show()
    return
"""


def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generacja = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Zmiana     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))

    last_fitness = ga_instance.best_solution()[1].copy()


def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions / data_outputs.size) * 100

    return solution_fitness


data_inputs = inp
data_outputs = out
last_fitness = 0
num_inputs = data_inputs.shape[1]


num_classes = 4
num_solutions = 20
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[150, 100, 45],
                                num_neurons_output=num_classes,
                                hidden_activations=["relu","relu","relu"],
                                output_activation="softmax")


#pygad.gann.validate_network_parameters()


population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
population_initialize = population_vectors.copy()

type_of_crossover = "single_point"
type_of_mutation = "random"
parent_type_selection = "sss"

num_parents_mating = 4
num_generations = 40
mutation_percent_genes = 5


keep_parents = -1


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=population_initialize,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       parent_selection_type=parent_type_selection,
                       crossover_type=type_of_crossover,
                       mutation_type=type_of_mutation,
                       keep_parents=keep_parents,
                       callback_generation=callback_generation)
ga_instance.run()

ga_instance.plot_result()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parametry najlepszego rozwiązania : {solution}".format(solution=solution))
print("Wartość fitness najlepszego rozwiązania = {solution_fitness}".format(solution_fitness=solution_fitness))

if ga_instance.best_solution_generation != -1:
    print("Najlepsza wartosc fitness osiagnieta po {best_solution_generation} generacjach.".format(
        best_solution_generation=ga_instance.best_solution_generation))


predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                               data_inputs=data_inputs)
print("Prognozy wyszkolonej sieci : {predictions}".format(predictions=predictions))

        # STATYSTYKI
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct / data_outputs.size)
tmp = round(accuracy,2)
print("Dokładnosc klasyfikacji : {accuracy}%.".format(accuracy=tmp))
print("Liczba poprawnie rozpoznanych obiektow  : {num_correct}.".format(num_correct=num_correct))
print("Liczba blednie rozpoznanych obiektow : {num_wrong}.".format(num_wrong=num_wrong.size))
