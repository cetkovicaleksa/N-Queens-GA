import numpy as np
from typing import List, Sequence, Callable






def swap_mutation(chromosome: List[int]) -> None:
    """
    Swap two randomly selected numbers from the given chromosome.
    """

    first_idx, second_idx = np.random.choice(len(chromosome), size=2, replace=False)
    chromosome[first_idx], chromosome[second_idx] = chromosome[second_idx], chromosome[first_idx]




def mutation(
        population: Sequence[List[int]], 
        mutation_rate: float, 
        operator: Callable[[List[int]], None] = swap_mutation
    ):
    """
    Preform in place mutation on the given population with the given mutation_rate.
    """

    for i in np.arange(len(population)):
        if np.random.random() < mutation_rate:
            operator(population[i])




if __name__ == "__main__":

    population = np.array([np.random.permutation(i) for i in 10*[3]])
    copy = np.array(population)
    mutation(population, 0.3)
    # print(copy, population, sep='\n----\n')
    print("Mutated chromosomes:\n", population[np.unique(np.where(population != copy)[0])])
