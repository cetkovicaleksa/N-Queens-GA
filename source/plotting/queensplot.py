"""
QueensPlot
==========

A module for visualizing solutions to the N-Queens problem on a chessboard.
Utilizes NumPy and Matplotlib for plotting.

"""

__all__ = ['checkered_with_shape', 'chess_column_labels', 'chess_row_labels', ...] # TODO
__author__ = 'Алекса Ћетковић'
__date__ = '5. I 2024.'
__version__ = '1.0'




import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.axis import Axis
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
from typing import Generator, Sequence, Iterable, Any






def checkered_with_shape(shape: Sequence[int] = (8, 8)) -> npt.NDArray[np.int_]: # TODO: recheck type hints
    """
    Generate a checkered board pattern of ones and zeroes.

    Parameters
    ----------
    shape : sequence of ints (default: (8, 8) the shape of standard chess board)
        The shape of the grid.

    Returns
    -------
    board : A NumPy array representing the checkered board pattern of the given shape.

    See also
    --------
    https://stackoverflow.com/a/51715491

    Examples
    --------
    >>> board = checkered_with_shape([3, 3])
    >>> board
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    >>> board = checkered_with_shape([8])
    >>> board
    array([0, 1, 0, 1, 0, 1, 0, 1])
    """

    return np.indices(shape).sum(axis=0) % 2



def chess_row_labels(n: int) -> Generator[str, None, None]:
    """
    Generate labels for rows in a chessboard.

    Parameters
    ----------
    n : int
        The number of rows.

    Yields
    ------
    str :
        String labels representing rows.
    
    Examples
    --------
    >>> labels = list(chess_row_labels(8))
    >>> labels
    ['1', '2', '3', '4', '5', '6', '7', '8']
    """
    
    yield from map(str, range(1, n + 1))



def chess_column_labels(n: int) -> Generator[str, None, None]:
    """
    Generate labels for columns in a chessboard.
    
    Parameters
    ----------
    n : int
        The number of columns.
        
    Yields
    ------
    str : 
        String labels representing columns.

    Notes
    -----
    This function yields characters in the uppercase range 'A' to 'Z'.
    Behaviour for column labels after this range may vary and is undefined.

    Examples
    --------
    >>> labels = list(chess_column_labels(8))
    >>> labels
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I']
    """

    yield from map(lambda i: chr(ord('A') + i), range(n))


queen_coords = [
    (0, 0), (1, 0), (1, 3), (0.5, 4), (0, 3),  # Crown
    (0, 7), (1, 7), (1, 4), (0.5, 3), (0, 4),  # Body
    (-1, 4), (2, 4), (2, 3.5), (-1, 3.5)  # Base
]



def plot_n_queens(
          solution: npt.ArrayLike, 
          separate_figure: bool = False,
          black: str = '#779954', 
          white: str = '#E9EDCC',
          label_color: str = '#8B4513',
          label_font_family: str = 'Times New Roman',
          show_row_labels: bool = True, 
          show_col_labels: bool = True, 
        ) -> None:
        """
        ...
        """
        nrows, ncols = solution.shape

        fig: Figure; ax: Axis
        fig, ax = plt.subplots()      

        image = plt.matshow(checkered_with_shape(solution.shape), cmap=ListedColormap([black, white]), fignum=0)
        
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(chess_column_labels(ncols))
        # ax.tick_params(labelcolor=label_color, labelfontfamily=label_font_family)

        ax.set_yticks(range(nrows))
        ax.set_yticklabels(chess_row_labels(nrows))
        ax.invert_yaxis()

        # coord is coordinates where solution matrix evaluates to True
        for coord in zip(*np.where(solution)): 
            ax.add_patch(patches.Circle(coord, 0.2, color='purple'))


        plt.tight_layout()
        plt.show()

        if separate_figure: 
            plt.close(fig)


def _plot_board(n: int, m: int) -> None:
    
    plt.matshow(checkered_with_shape([n, m]), fignum=0, cmap='copper')
    plt.xticks(range(m), chess_column_labels(m))
    plt.yticks(range(n), chess_row_labels(n))
    plt.gca().invert_yaxis() # because mathshow indexes matrix from top left and we are indexing from bottom left
    
     

    






def queen_patch(row, col) -> any:
    queen_icon = '♕'
    return AnnotationBbox(OffsetImage(queen_icon), (row, col), fameon=False)







# just needs some adjustment and can already be used in the project :)
def plot_board(queens, nrows: int = 8, ncols: int = 8):
    image = checkered_with_shape([nrows, ncols])
    

    plt.matshow(image, cmap="copper")
    plt.xticks(range(ncols), chess_column_labels(ncols))
    plt.yticks(range(nrows), chess_row_labels(nrows))

    queens = [plt.Circle((col, row), 0.4, color='purple') for row, col in enumerate(queens)]
    for queen in queens:
        plt.gca().add_patch(queen)
    plt.gcf().tight_layout()
    # for row, col in enumerate(queens):
    #     plot_queen(row, col)
    plt.show()





if __name__ == "__main__":
    rows, cols = 8, 8
    print(''.center(80, '='))

    print("Row labels demo for %d rows:\n"%(rows), list(chess_row_labels(rows)))
    print("Column labels demo for %d columns:\n"%(cols), list(chess_column_labels(cols)))
    
    print("\nTODO: n queens plot demo")
    
    print(''.center(80, '='))

    plot_n_queens(np.eye(8))

    # print(type(checkered_with_shape([1, 2])))

    

    # plot_queen(1, 1)