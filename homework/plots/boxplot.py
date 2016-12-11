import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math


from . import helpers


logger = logging.getLogger(__name__)


def many(bunch, trait, n_row=2, group_into=10):
    """
    Create boxplots(s) of an object trait.

    Parameters:
        bunch (RADARBunch): data bunch
        trait (str): object trait to plot
        n_row (int): number of subplots in a column
        group_into (int): size of the groups to partition objects into
    """
    # figure out how many subplots are needed
    n_lab = len(bunch.object_labels)
    n_plt = int(math.ceil(n_lab / group_into))
    n_col = int(math.ceil(n_plt / n_row))

    # log some info
    logger.debug('creating %s boxplots', trait)
    logger.debug('there are %d total objects', n_lab)
    logger.debug('objects will be partitioned into groups of %d', group_into)
    logger.debug('plotting will occur on a %dx%d subplot grid', n_row, n_col)
    logger.debug('the total number of subplots is %d', n_plt)

    # create figure
    fig, axes = helpers.subplots(n_row, n_col)
    y_min, y_max = 0, 1

    # iterate over groups
    for axis_i, chunk_i in enumerate(
            range(0, n_lab, group_into)):

        # obtain data subset
        labels = bunch.object_labels[chunk_i:chunk_i + group_into]
        subset = bunch.get_subset_for_many_objects(*labels, traits=[trait])

        # keep track of y-limits
        y_max = max(y_max, subset.values.max())
        y_min = max(y_min, subset.values.min())

        # create boxplot
        plt.sca(axes.flat[axis_i])
        sns.boxplot(data=subset)

        # adjust figure aesthetics
        plt.xticks(range(len(labels)), labels)
        plt.xlabel('object label (#)')
        plt.ylabel(trait)
        plt.title('Objects {} to {}'.format(labels[0], labels[-1]))

    # zoom out a little bit
    y_rng = abs(y_max - y_min)
    y_min -= 0.10 * y_rng
    y_max += 0.10 * y_rng

    # adjust y-limits
    for axis in axes.flat:
        plt.sca(axis)
        plt.ylim(y_min, y_max)
        sns.despine(fig, axis)
