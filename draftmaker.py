"""
by Elizabeth Pankratz, 2022
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import util



def inv(img_array):
    """ 
    This encoding represents a white cell as 0 and a black cell as 1, but by default, 
    skimage displays 0 as black and 1 as white. This functions inverts those values for display.

    Arg:
        img_array: np array representing image 
    Returns:
        np array of the same dimensions as img_array with colours inverted
    """
    return util.invert(img_array)


def display_fabric(fabric_array):
    """
    Uses matplotlib to display image.

    Arg:
        fabric_array: np array representing the generated fabric (0s and 1s)
    Returns:
        Nothing, displays image.
    """
    plt.gray()
    plt.axis('off')
    plt.imshow(inv(fabric_array))
    plt.show()


def gen_bool_array(numbered_array):
    """
    Transforms a provided array containing numbered treadles or harnesses into a two-dimensional boolean array
    (useful for displaying).

    Arg:
        numbered_array: numpy array, either of arrays (for treadling) or integers (for threading)
    Returns:
        Array of arrays; for every weft row, one inner array, each containing 0 (not active)/1 (active) for every treadle.
    """
    # Find max value and add 1; this is how many treadles there are
    n_treadles = numbered_array.max() + 1

    # Init list that will contain the 0/1 values for each row.
    bool_treadle = []

    # Iterate through treadle_array.
    for curr_treadles in numbered_array:
        
        # Create zero array of length n_treadles.
        bool_row = np.zeros(n_treadles)

        # Replace the zeroes with ones at indices given in curr_treadles and append to accumulator list.
        np.put(bool_row, curr_treadles, 1)
        bool_treadle.append(bool_row)
    
    return np.array(bool_treadle)


def display_threading(thread_spec):
    """
    Arg:
        thread_spec: np array; integer values represent harness of each thread.
    Returns:
        Nothing, displays image.
    """
    plt.axis('off')
    plt.gray()
    plt.imshow(inv(np.rot90(gen_bool_array(thread_spec))))
    plt.show()


def display_treadling(treadle_spec):
    """ 
    Arg:
        treadle_spec: np array of arrays; outer arrays = weft row, inner arrays = integer values representing active treadles.
    Returns:
        Nothing, displays image.
    """
    plt.gray()
    plt.axis('off')
    plt.imshow(inv(gen_bool_array(treadle_spec)))
    plt.show()


def display_tieup(tieup_spec):
    """
    Arg:
        tieup_spec: np array of arrays; outer arrays = harnesses bottom to top, inner arrays = bool values indicating tie-up to treadles.
    Returns:
        Nothing, displays image.
    """
    plt.gray()
    plt.axis('off')
    plt.imshow(inv(np.flip(tieup_spec, axis=0)))
    plt.show()


def gen_fabric(tieup_spec, thread_spec, treadle_spec):
    """
    Generates fabric based on the provided tie-up, treadling, and threading.

    Args:
        tieup_spec: np array of arrays; outer arrays = harnesses bottom to top, inner arrays = bool values indicating tie-up to treadles.
        thread_spec: np array; integer values represent harness of each thread.
        treadle_spec: np array of arrays; outer arrays = weft row, inner arrays = integer values representing active treadles.
    Returns:
        np array of shape (len(treadle_spec), len(thread_spec)) with values 0 (weft-face) and 1 (warp-face)
    """

    # Check that the dimensions are all compatible: 
    # n rows in tieup = max+1 of thread_spec.
    n_harnesses = thread_spec.max() + 1
    assert n_harnesses == tieup_spec.shape[0], "Number of rows in tie-up don't match number of harnesses"

    # n columns in tieup = max+1 of treadle_spec.
    n_treadles = treadle_spec.max() + 1
    assert n_treadles == tieup_spec.shape[1], "Number of columns in tie-up don't match number of treadles"

    # Get the number of weft picks and warp ends in the fabric.
    n_weft = len(treadle_spec)
    n_warp = len(thread_spec)

    # Init drawdown array filled with zeroes.
    drawdown = np.zeros((n_weft, n_warp))

    # Iterate through elements of outer array (i.e., weft rows), and then through
    # elements of inner array (i.e., warp ends).

    for row_idx in range(n_weft):
        for end_idx in range(n_warp):

            # Get the harness through which the current end is threaded; an int.
            curr_harness = thread_spec[end_idx]

            # Check in the tie-up which treadles are tied to this harness.
            this_harness_tie_up = tieup_spec[curr_harness]
            tied_up_treadles = np.nonzero(this_harness_tie_up)[0]

            # Check whether any of these treadles are contained in current treadling.
            curr_treadling = treadle_spec[row_idx]

            # If the intersection of elements in this array is >0 (i.e., there's overlap between tie-up and current treadling),
            # then the current warp thread is lifted, so we should assign this cell in drawdown a 1.
            treadle_intersect = np.intersect1d(tied_up_treadles, curr_treadling)
            if len(treadle_intersect) != 0:
                drawdown[row_idx, end_idx] = 1

    return drawdown


def display_full_draft(tieup_spec, thread_spec, treadle_spec, n_pixel_sep=1, overlay_grid=True, pdf_filename=None):
    """
    Generates fabric based on provided tie-up, treadling, and threading. 
    Combines these four components into a traditionally-formatted weaving draft.
    If filename not None, saves as pdf in the provided location.

    Args:
        tieup_spec: np array of arrays; outer arrays = harnesses bottom to top, inner arrays = bool values indicating tie-up to treadles.
        thread_spec: np array; integer values represent harness of each thread.
        treadle_spec: np array of arrays; outer arrays = weft row, inner arrays = integer values representing active treadles.
        n_pixel_sep: int, the number of pixels as whitespace padding between each rectangle in draft (default 1)
        overlay_grid: bool, whether or not to include pixel grid (default True)
        pdf_filename: string ending in '.pdf', the desired filename (potentially incl. full path) to save the draft under (default None)
    Returns:
        Nothing, displays image and, if filename not None, saves image in provided path.
    """
    # Generate the upper part of the draft by combining threading, padding, and tie-up.
    padding_upper = np.zeros((tieup_spec.shape[0], n_pixel_sep))
    thread_bool = np.rot90(gen_bool_array(thread_spec))
    tieup_bool = np.flip(tieup_spec, axis=0)
    upper = np.column_stack([thread_bool, padding_upper, tieup_bool])

    # Generate the second "row" by generating fabric drawdown and combining it with tall padding and treadling.
    padding_lower = np.zeros((len(treadle_spec), n_pixel_sep))
    fabric_bool = gen_fabric(tieup_spec, thread_spec, treadle_spec)
    treadle_bool = gen_bool_array(treadle_spec)
    lower = np.column_stack([fabric_bool, padding_lower, treadle_bool])

    # Ensure that both rows have same width.
    assert upper.shape[1] == lower.shape[1], 'Upper and lower rows do not have same width'

    # Stack the two rows and put wide padding in between.
    padding_wide = np.zeros((n_pixel_sep, upper.shape[1]))
    draft = np.row_stack([upper, padding_wide, lower])

    # Now the display code.
    plt.gray()
    plt.axis('off')
    plt.imshow(inv(draft))

    # Messy code for plotting gridlines over each of the components individually (that is, not over padding)
    if overlay_grid:

        # Threading grid
        thread_rows, thread_cols = thread_bool.shape
        for row_idx in range(thread_rows+1):
            plt.hlines(y = row_idx - 0.5, xmin = -0.5, xmax = thread_cols - 0.5, color='black', linewidth=0.3)
        for col_idx in range(thread_cols+1):
            plt.vlines(x = col_idx - 0.5, ymin = -0.5, ymax = thread_rows - 0.5, color='black', linewidth=0.3)

        # Tie-up grid
        tieup_rows, tieup_cols = tieup_bool.shape
        tieup_start_x = thread_bool.shape[1] + n_pixel_sep  # offset
        for row_idx in range(tieup_rows+1):
            plt.hlines(y = row_idx - 0.5, xmin = tieup_start_x - 0.5, xmax = tieup_start_x + tieup_cols - 0.5, color='black', linewidth=0.3)
        for col_idx in range(tieup_start_x, tieup_start_x + tieup_cols+1):
            plt.vlines(x = col_idx - 0.5, ymin = -0.5, ymax = tieup_rows - 0.5, color='black', linewidth=0.3)

        # Fabric grid
        fabric_rows, fabric_cols = fabric_bool.shape
        fabric_start_y = thread_bool.shape[0] + n_pixel_sep # offset
        for row_idx in range(fabric_start_y, fabric_start_y + fabric_rows+1):
            plt.hlines(y = row_idx - 0.5, xmin = -0.5, xmax = fabric_cols - 0.5, color='black', linewidth=0.3)
        for col_idx in range(fabric_cols+1):
            plt.vlines(x = col_idx - 0.5, ymin = fabric_start_y-0.5, ymax = fabric_start_y + fabric_rows - 0.5, color='black', linewidth=0.3)

        # Treadling grid
        treadle_rows, treadle_cols = treadle_bool.shape
        treadle_start_x =  fabric_bool.shape[1] + n_pixel_sep  # offset
        treadle_start_y = thread_bool.shape[0] + n_pixel_sep   # offset
        for row_idx in range(treadle_start_y, treadle_start_y + treadle_rows+1):
            plt.hlines(y = row_idx - 0.5, xmin = treadle_start_x - 0.5, xmax =treadle_start_x + treadle_cols - 0.5, color='black', linewidth=0.3)
        for col_idx in range(treadle_start_x, treadle_start_x + treadle_cols+1):
            plt.vlines(x = col_idx - 0.5, ymin = treadle_start_y-0.5, ymax = treadle_start_y + treadle_rows - 0.5, color='black', linewidth=0.3)

    if pdf_filename != None:
        # Saving img as pdf gives best result; rasterised image formats don't align 
        # the hlines and vlines with the cells very nicely.
        assert pdf_filename.endswith('.pdf'), 'Please provide a filename with a .pdf extension, e.g., "myfile.pdf".'
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Displaying draft for 2x2 twill (herringbone threading, straight treadling)")
    twill2x2_tieup = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    straight_treadle_4 = np.array([[0], [1], [2], [3]] * 6)
    herringbone_thread = np.array(np.concatenate([[0, 1, 2, 3] * 2 + [1, 0, 3, 2] * 2] * 3))
    display_full_draft(twill2x2_tieup, herringbone_thread, straight_treadle_4, pdf_filename='sample_drafts/2x2twill_herringbone.pdf')
