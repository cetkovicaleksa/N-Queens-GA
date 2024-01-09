"""
TODO: ...
"""








import numpy as np
import numpy.typing as npt
from typing import Literal, Sequence, Any




# Chromosome = Sequence[int] | np.ndarray[Any, np.dtype[np.uint]] | np.ndarray



def chromosome_to_mat(
        chromosome: Sequence[int], 
        *,
        dtype: npt.DTypeLike = int, 
        repr_method: Literal['implicit_rows', 'implicit_columns'] = ...
    ):
    """
    Convert a permutation representation of an N-Queens board to matrix representation.

    Parameters
    ----------
    chromosome : Sequence[int]
        A sequence of row/column indices. The chromosome must not contain negative numbers
        nor a number greater than or equal to its length.

    dtype : DTypeLike, optional
        Type of values of the output matrix. Defaults to int.

    repr_method : {'implicit_rows', 'implicit_columns'}, default 'implicit_columns'
        Specifies the interpretation of the chromosome:

        - 'implicit_rows': The chromosome represents column indices.
        The index of each element denotes the actual row index of the queen in the respective column.

        - 'implicit_columns' The chromosome represents row indices.
        The index of each element denotes the actual column index of the queen in the respective row.

    Returns
    -------
    np.ndarray :
        A square matrix representing the N-Queens board with the placement of the queens.

    Examples
    --------
    >>> chromosome_to_mat([1, 2, 0], repr_method='implicit_rows')
    array([[0, 1, 0],
           [0, 0, 1],
           [1, 0, 0]])

    >>> chromosome_to_mat([1, 2, 0], repr_method='implicit_columns')
    array([[0, 0, 1],
           [1, 0, 0],
           [0, 1, 0]])        
    
    >>> chromosome_to_mat([0, 0], repr_method='implicit_rows')
    array([[1, 0],
           [1, 0]])

    >>> chromosome_to_mat([0, 0], repr_method='implicit_columns')
    array([[1, 1],
           [0, 0]])
    """
    
    identity_matrix = np.identity(len(chromosome), dtype)

    if repr_method == 'implicit_rows':
        return identity_matrix[np.asarray(chromosome)]
    
    return identity_matrix[::, np.asarray(chromosome)]



def mat_to_chromosome(
        matrix: npt.ArrayLike, 
        *,
        dtype: npt.DTypeLike = int,
        repr_method: Literal['implicit_rows', 'implicit_columns'] = ...
    ):
    """
    Convert a matrix representation of the N-Queens board to a permutation representation.

    Parameters
    ----------
    matrix : ArrayLike
        A 2d matrix that can be represented ...

    dtype : DTypeLike, optional
        Type that the permutation will hold. Defaults to int.

    repr_method : {'implicit_rows', 'implicit_columns'}, default 'implicit_columns'
        ...
    
    Returns
    -------
        ...

    Examples
    --------
        ...
    ...
    """
    
    # row and corresponding column indices where a queen exists
    row_indices, col_indices = np.where(matrix)

    implicit, explicit = (row_indices, col_indices) if repr_method == 'implicit_rows' \
                    else (col_indices, row_indices)
    
    sorted_unique_implicit, index = np.unique(implicit, return_index="Да, хоћу индекс. :)")

    return np.asarray(explicit[index], dtype=dtype)








if __name__ == "__main__":
    # Tests are not completed
    
    # these two should represent the same arrangement of 8 queens, or any other n queens
    impl_rows = np.array([0, 4, 7, 5, 2, 6, 1, 3]) 
    impl_cols = np.array([0, 6, 4, 7, 1, 3, 5, 2])
    
    mat_cols = chromosome_to_mat(impl_cols, repr_method='implicit_columns')
    mat_rows = chromosome_to_mat(impl_rows, repr_method='implicit_rows')

    revived_rows = mat_to_chromosome(mat_rows, repr_method='implicit_rows')
    revived_cols = mat_to_chromosome(mat_cols, repr_method='implicit_columns')

    print(
        "---",
        "Implicit rows:",
        "---",
        f"original coordinates: {impl_rows}", 
        mat_rows,
        f"back to coordinates: {revived_rows}", 
        "---",
        sep='\n'
    )

    print(
        "---",
        "Implicit columns:",
        "---",
        f"original coordinates: {impl_cols}", 
        mat_cols,
        f"back to coordinates: {revived_cols}", 
        "---",
        sep='\n'
    )

    # the original and revived coordinates should be the same and matrix representation of both should be equal
    print("OK? : ", np.all(impl_rows == revived_rows) and np.all(impl_cols == revived_cols) and np.all(mat_cols == mat_rows))

    # try:
    #     mat_cols_copy = np.copy(mat_cols)
    #     mat_cols_copy[0, 0] = 0 
    #     mat_to_chromosome(mat_cols_copy, repr_method='implicit_columns')
    #     print(f"Missing queen on column mat to chromosome column repr OK? : {False}")
    # except ValueError:
    #     print(f"Invalid mat to chromosome column repr OK? : {True}")

    