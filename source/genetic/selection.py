import numpy as np
import numpy.typing as npt
from typing import Protocol, Tuple, Sequence, Literal, Iterable



class HasFitness(Protocol):
    @property
    def fitness(self) -> float: 
        """Get the fitness value. Higher values are better."""
        ...


Selection = np.ndarray[Tuple[int, int]]




def natural_selection(population: Sequence[HasFitness], n_keep: int) -> np.ndarray[int]:
    """
    Get the indices of the fittest individuals from the population.
    
    Parameters
    ----------
    population : Sequence[HasFitness]
        A sequence of individuals that have fitness property.

    n_keep : int
        Number of fittest inidividuals to select (n_keep > 0).

    Returns
    -------
    out : np.ndarray
        Array of indices of n_keep fittest individuals from the population.
    """
    
    return np.argsort(np.fromiter(c.fitness for c in population))[-n_keep:]





def roulette_selection(population: Sequence[HasFitness]) -> Tuple[int, int]:

    fitnesses = np.fromiter(c.fitness for c in population)
    sum_fitness = np.sum(fitnesses)

    probabilities = fitnesses / sum_fitness

    

    ...


def roulette() -> int:
    ...





def tournament_selection(population: Sequence[HasFitness], gladiator_count: int) -> Tuple[int, int]:
    """
    """

    population = np.asarray(population)

    first_winner_idx = tournament(population, gladiator_count)
    second_winner_idx = tournament(population[np.arange(population.size) != first_winner_idx], gladiator_count) # no hermaphrodite

    return first_winner_idx, second_winner_idx


def tournament(population: Sequence[HasFitness], gladiator_count: int) -> int:
    """
    Returns the index in the given population of the winner of the gladiator_count-way tournament.
    
    ...
    """
    # draw random gladiators from the population
    gladiator_indices = np.random.choice(len(population), gladiator_count, replace=False)
    
    # winner is the fittest gladiator
    winner_index = np.max(gladiator_indices, key=lambda i: population[i].fitness)
    return winner_index



def selection(
    population: Sequence[HasFitness], 
    selection_rate: float = 1, 
    method: Literal['roulette', 'tournament'] = ...
) -> Selection:
    
    n = np.ceil(len(population) * selection_rate)
    selection_method = ...

    return [selection_method(population) for _ in range(n)]