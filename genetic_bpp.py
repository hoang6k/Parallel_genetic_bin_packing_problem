import numpy as np

np.random.seed(0)


class Item:
    def __init__(self, label: int = None, size: int = None):
        self._label = label
        self._size = size


class Bin:
    def __init__(self, items: list = [], capacity: int = 0):
        self._items = items
        self._capacity = capacity

    def add_item(self, item: Item):
        self._items.append(item)
        self._capacity += item._size

    def get_labels(self):
        return [item._label for item in self._items]


class Chromosome:
    def __init__(self, bins: list = [], max_capacity: int = 0):
        self._bins = bins
        self._max_capacity = max_capacity
        self._fitness = None

    def calculate_fitness(self):
        num_bins = len(self._bins)
        bins_capacity = np.asarray([_bin._capacity for _bin in self._bins], dtype=np.int)
        self._fitness = np.sum(np.power(bins_capacity / self._max_capacity, 2)) / num_bins
        return self._fitness


class Population:
    def __init__(self, first_population: list = [], result: int = None):
        self._population_size = len(first_population)
        self._first_population = first_population
        self._generations = [self._first_population]
        self._generations_fitness = [
            np.max(np.asarray([chromo.calculate_fitness() for chromo in self._first_population]))]
        self._result = result

    def roulette_wheel_selection(self, array, k: int = 2):
        max_value = np.sum(array)
        points = [0]
        sum = 0
        for fitness in array:
            sum += fitness
            points.append(sum)
        k_chosen = max_value * np.random.random_sample(3)
        chosen_indexes = []
        for var in k_chosen:
            for i in range(len(array)):
                if points[i] < var < points[i + 1]:
                    chosen_indexes.append(i)
        chosen_index = chosen_indexes[int(np.argmax(array[chosen_indexes]))]
        return chosen_index

    def first_fit_descending(self, chromosome: Chromosome, items: list):
        items = sorted(items, key=lambda item: item._size, reverse=True)
        max_capacity = chromosome._max_capacity
        current_bin = Bin()
        _flag = False
        for item in items:
            for i in range(len(chromosome._bins)):
                if chromosome._bins[i]._capacity + item._size <= max_capacity:
                    chromosome._bins[i].add_item(item)
                    _flag = True
                    break
            if not _flag:
                if current_bin._capacity + item._size <= max_capacity:
                    current_bin.add_item(item)
                else:
                    chromosome._bins.append(current_bin)
                    current_bin = Bin()
                    current_bin.add_item(item)
        return chromosome

    def replacement(self, chromosome: Chromosome, items: list):
        pass

    def crossover_one_chromosome(self, chromosome: Chromosome, points_insert: int, genes: list):
        child = Chromosome(genes, max_capacity=chromosome._max_capacity)
        take_labels = [_bin.get_labels() for _bin in genes]
        take_labels = [label for bin_labels in take_labels for label in bin_labels]
        left_bins = []
        right_bins = []
        free_items = []
        for i in range(len(chromosome._bins)):
            intersection = [label for label in chromosome._bins[i].get_labels() if label in take_labels]
            if len(intersection) == 0:
                if i < points_insert:
                    left_bins.append(chromosome._bins[i])
                else:
                    right_bins.append(chromosome._bins[i])
            else:
                for item in chromosome._bins[i]._items:
                    if item._label not in intersection:
                        free_items.append(item)
        child._bins = left_bins + child._bins + right_bins
        # new_chromosome, free_items = self.replacement(new_chromosome, free_items)
        return self.first_fit_descending(child, free_items)

    def crossover(self, father: Chromosome, mother: Chromosome):
        # print('\t\t\tCreating childs')
        while True:
            two_points_father = np.random.randint(len(father._bins) + 1, size=2)
            two_points_mother = np.random.randint(len(mother._bins) + 1, size=2)
            if len(set(two_points_father)) == 2 and len(set(two_points_mother)) == 2:
                break
        two_points_father.sort()
        two_points_mother.sort()
        # print('\t\t\t\t1st child')
        first_child = self.crossover_one_chromosome(chromosome=father, points_insert=two_points_father[1],
                                                    genes=mother._bins[two_points_mother[0]:two_points_mother[1]])
        # print('\t\t\t\t2nd child')
        second_child = self.crossover_one_chromosome(chromosome=mother, points_insert=two_points_mother[0],
                                                     genes=father._bins[two_points_father[0]:two_points_father[1]])
        return first_child, second_child

    def mutation(self, parent: Chromosome):
        while True:
            bins_mutation_indexes = np.random.randint(len(parent._bins), size=self._mutation_size)
            if len(set(bins_mutation_indexes)) == self._mutation_size:
                break
        bins_mutation_indexes.sort()
        bins_mutation_indexes = np.flip(bins_mutation_indexes)
        free_bins = [parent._bins.pop(idx) for idx in bins_mutation_indexes]
        free_items = [item for _bin in free_bins for item in _bin._items]
        return self.first_fit_descending(parent, free_items)

    def generate_next(self):
        print('\n\nIteration {}'.format(len(self._generations_fitness)))
        current_generation = self._generations[-1]
        current_generation_fitness = self._generations_fitness[-1]
        fitness = np.asarray([chromo._fitness for chromo in current_generation])
        children = []

        # cross-over phase
        print('\tCROSS-OVER PHASE')
        for i in range(int(self._offspring_number / 2)):
            print('\t\tStarting cross-over {}th 2 childs'.format(i + 1))
            _crossover = bool(np.random.rand(1) <= self._crossover_probability)
            if _crossover:
                father = current_generation[self.roulette_wheel_selection(array=fitness)]
                mother = current_generation[self.roulette_wheel_selection(array=fitness)]
                child_1, child_2 = self.crossover(father, mother)
                children.append(child_1)
                children.append(child_2)

        # mutation phase
        print('\tMUTATION PHASE')
        for i in range(self._offspring_number):
            print('\t\tStarting mutation {}th child'.format(i + 1))
            _mutation = bool(np.random.rand(1) <= self._mutation_probability)
            if _mutation:
                new_child = self.mutation(children[i])
                children[i] = new_child

        # replace worst chromosomes
        sorted_indexes = np.argsort(fitness)
        worst_indexes = sorted_indexes[:self._chromosomes_replace]
        worst_indexes.sort()
        worst_indexes = np.flip(worst_indexes)
        for idx in worst_indexes:
            current_generation.pop(idx)
        next_generation = current_generation + children
        children_fitness = np.max(np.asarray([chromo.calculate_fitness() for chromo in children]))
        self._generations.append(next_generation)
        self._generations_fitness.append(np.max([current_generation_fitness, children_fitness]))
        return self._generations_fitness[-1]

    def generate_populations(self, generate_config: dict):
        self._generations_number = generate_config['generations_number']
        self._crossover_probability = generate_config['crossover_probability']
        self._mutation_probability = generate_config['mutation_probability']
        self._mutation_size = generate_config['mutation_size']
        self._offspring_number = generate_config['offspring_number']
        self._chromosomes_replace = generate_config['chromosomes_replace']
        self._stop_criterion_depth = generate_config['stop_criterion_depth']

        stop_depth = 0
        best_fitness = self._generations_fitness[-1]
        for i in range(self._generations_number):
            new_generation_fitness = self.generate_next()
            print('Generation {}: fitness {}'.format(i + 1, new_generation_fitness))
            if new_generation_fitness <= best_fitness:
                stop_depth += 1
                print('\tFitness not increase for {} generations'.format(stop_depth))
                if stop_depth > self._stop_criterion_depth:
                    print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                print('\tFitness increased')
                stop_depth = 0
                best_fitness = new_generation_fitness

    @staticmethod
    def population_initialization(D: int = None, N: int = None, B: int = None, d_list: list = None,
                                  population_size: int = 100):
        chromos = []
        for i in range(population_size):
            bins = []
            current_bin = Bin()
            indexes = np.arange(N)
            np.random.shuffle(indexes)
            for idx in indexes:
                if current_bin._capacity + d_list[idx] <= D:
                    current_bin.add_item(Item(label=idx, size=d_list[idx]))
                else:
                    bins.append(current_bin)
                    current_bin = Bin()
                    current_bin.add_item(Item(label=idx, size=d_list[idx]))
            bins.append(current_bin)
            chromos.append(Chromosome(bins, max_capacity=D))
        return Population(chromos, result=B)


def load_data(data_path):
    f = open(data_path, 'r')
    lines = f.readlines()
    num_set = int(lines[0])
    length = int((len(lines) - 1) / num_set)
    sets = [lines[i * length + 1: (i + 1) * length + 1] for i in range(num_set)]
    test_sets = []
    for _set in sets:
        D, N, B = [int(num) for num in _set[1].split()]
        d_list = [int(num) for num in _set[2:]]
        test_sets.append({'D': D, 'N': N, 'B': B, 'd_list': d_list})
    return test_sets


if __name__ == '__main__':
    generate_config = {'population_size': 60, 'offspring_number': 50, 'chromosomes_replace': 50,
                       'crossover_probability': 1.0, 'mutation_probability': 0.66, 'mutation_size': 2,
                       'generations_number': 500, 'stop_criterion_depth': 20}
    generate_config['offspring_number'] = int(generate_config['population_size'] / 2)
    generate_config['chromosomes_replace'] = int(generate_config['population_size'] / 2)
    # path = 'data/binpack_test.txt'
    path = 'data/binpack1.txt'
    test_sets = load_data(path)
    for _set in test_sets:
        population = Population.population_initialization(D=_set['D'], N=_set['N'], B=_set['B'],
                                                          d_list=_set['d_list'],
                                                          population_size=generate_config['population_size'])
        population.generate_populations(generate_config)

        break

    pass
