"""
Functions to aid with output.
"""
import matplotlib.pyplot as plt
import subprocess
import dateutil.tz
import datetime
import logging
import uuid
import sys
import os


logger = logging.getLogger(__name__)


def format_output_path(path, *paths, **kwargs):
    """
    Create a filesystem path string.

    * First the path is constructed by joining all components.
    * Second, the key word arguments are passed to the str.format function.
    * Finally, the special string '~' is expanded as the users home directory.

    Two special key word arguments are supplied by default.

    ===== =================== ===================
    kwarg description         usage
    ===== =================== ===================
    dt    a datetime object   ``'{dt:%Y_%m_%d}'``
    uuid  a unique identifier ``'{uuid}'``
    ===== =================== ===================

    Args:
        path (str): first path component
        paths (list): remaining path components
        kwargs (dict): format kwargs

    Returns:
        str: filesystem path

    Examples:
        >>> path = format_output_path('~', '{dt:%Y-%m-%d}-{s}.png', s='test')
        '/home/user/2016-09-06-test.png'
    """
    dt = datetime.datetime.now(dateutil.tz.tzlocal())
    _kwargs = dict(uuid=uuid.uuid4(), dt=dt)
    _kwargs.update(**kwargs)
    path = os.path.join(path, *paths).format(**_kwargs)
    path = os.path.expanduser(path)
    path = os.path.realpath(path)
    return path


def cropfig(fname, border=10):
    """
    Crop an image or PDF.

    * uses the command `pdfcrop` for PDF files
    * uses the command `convert` for IMG files

    Args:
        fname (str): filesystem path of figure to crop
        border (int): number of whitespace pixels to remain on border
    """
    # do nothing if the file does not exist
    if not os.path.isfile(fname):
        return

    # do nothing if we are not on Linux
    if not sys.platform.lower() in ['linux', 'linux2']:
        return

    stub, ext = os.path.splitext(fname)
    if ext == '.pdf':
        command = 'pdfcrop -margins %d %s %s' % (border, fname, fname)
    else:
        command = 'convert -trim -border %d -bordercolor white %s %s' % (
            border, fname, fname)

    try:
        subprocess.check_call(command.split())
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def savefig(fname, crop=True, border=10, **kwargs):
    """
    Call `matplotlib.pyplot.savefig` and crop the output using `cropfig`.

    Args:
        fname (str): save filesystem path
        crop (bool): crop the figure?
        border (int): number of whitespace pixels to remain on border
        kwargs (dict): savefig kwargs
    """
    _kwargs = dict(dpi=600)
    _kwargs.update(**kwargs)
    logger.debug('saving: %s', fname)
    plt.savefig(fname, **_kwargs)
    if crop:
        logger.debug('cropping: %s', fname)
        cropfig(fname, border)


def subplots(n_row, n_col, cell_w=5.0, cell_h=5.0, sep_w=2.0, sep_v=2.0,
             ls=1.5, rs=1.5, ts=1.5, bs=1.5):
    """
    Create subplot grid using absolute widths.

    Parameters:
        n_row (int): rows in subplot grid
        n_col (int): columns in subplot grid
        cell_w (float): width of each cell
        cell_h (float): height of each cell
        sep_w (float): horizontal space between each subplot
        sep_v (float): vertical space between each subplot
        ls (float): left margin
        rs (float): right margin
        ts (float): top margin
        bs (float): bottom margin
    """
    tw = rs + ls + n_col * cell_w + (n_col - 1) * sep_w
    th = bs + ts + n_row * cell_h + (n_row - 1) * sep_v

    fl = ls / float(tw)
    fr = (tw - rs) / float(tw)

    fb = bs / float(th)
    ft = (th - ts) / float(th)

    if cell_w > 0.0:
        ws = sep_w / cell_w
    else:
        ws = 0.0

    if cell_h > 0.0:
        hs = sep_v / cell_h
    else:
        hs = 0.0

    fig, axes = plt.subplots(n_row, n_col, figsize=(tw, th))
    plt.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, wspace=ws,
                        hspace=hs)

    return fig, axes
