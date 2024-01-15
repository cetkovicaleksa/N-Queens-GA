import numpy as np
from typing import Tuple, Sequence, Collection
from representation import ChromosomeWithPreCalcFitness


Chromosome = ChromosomeWithPreCalcFitness # ah don't know what to do with this



def _finalize_chromosome(chromosome: np.ndarray) -> Chromosome:
    ...


def mate(parent1: Sequence[int], parent2: Sequence[int]) -> Tuple[Chromosome, Chromosome]:
    """Single point crossover."""

    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)

    index = np.random.choice(len(parent1)-1)
    
    child1 = np.concatenate(parent1[:index], parent2[index+1:])
    child2 = np.concatenate(parent2[:index], parent1[index+1:])

    return _finalize_chromosome(child1), _finalize_chromosome(child2)

    


def crossover(population: Sequence[int], pair_indices: Collection[Tuple[int, int]]):

    return [child for (i, j) in pair_indices for child in mate(population[i], population[j])]