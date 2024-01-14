"""
...
"""

__author__ = 'Алекса Ћетковић'
__date__ = '13. I 2024.'



from itertools import count
import numpy as np
from typing import Any, Sequence, Iterable






def count_diagonal_conflicts(chromosome: Sequence[int]) -> int:  # TODO: Iterable[int] may be more appropriate?
    """
    Returns the total number of diagonal conflicts for a given
    permutation representation of an N-Queens problem. 

    Diagonal conflicts occur when two or more queens share the same diagonal.
    The number of conflicts for a diagonal is zero if there is a single queen or no queens
    on the diagonal. If there are n queens (n > 1) on the diagonal, the number of conflicts
    is n - 1.

    This function works for both row-implicit and column-implicit representations.

    It counts conflicts by assigning counters for the number of queens on each
    diagonal and then evaluates the total conflicts.
    
    Parameters
    ----------
    chromosome : Sequence
        A sequence of nonnegative integers smaller than the length
        of the sequence. The sequence may or may not contain repetitions, 
        only diagonal conflicts are counted.

    Returns
    -------
    out : int
        The total number of diagonal conflicts.

    Examples
    --------
    >>> conflicts = count_diagonal_conflicts(range(4)) # same as passing [0, 1, 2, 3]
    >>> conflicts
    3
    >>> conflicts = count_diagonal_conflicts([0, 0, 0])
    >>> conflicts
    0
    """

    rows = np.asarray(chromosome, dtype=int)
    n = rows.size
    cols = np.arange(0, n, dtype=int)

    # array containing number of queens on each left diagonal
    left_diagonal  = np.bincount(n - 1 - rows + cols, minlength=(2*n - 1))
    # array containing number of queens on each right diagonal
    right_diagonal = np.bincount(rows + cols, minlength=(2*n - 1))

    conflicts = (
        np.where(left_diagonal > 1, left_diagonal - 1, 0),
        np.where(right_diagonal > 1, right_diagonal - 1, 0)
    )

    return np.sum(conflicts)








def main():
    from representation import chromosome_to_mat, ChromosomeWithPreCalcFitness
    
    # solving n queens brute force :)
    for i in count():
        c = ChromosomeWithPreCalcFitness(np.random.permutation(8))

        if c.fitness == 0:
            print(f"Num iterations: {i}", c, chromosome_to_mat(c), sep='\n')
            break

    print("{:,}".format(count_diagonal_conflicts(np.random.permutation(10_000_000))))
        


if __name__ == "__main__":
    main()