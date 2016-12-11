"""
Analyze RADAR data in a CSV file.

The script will compare the source position and velocity
    with an object identified by the --label command.

If the --label command is absent, then only a boxplot
    of object uiLifeCycles will be made.
Examine the boxplots of uiLifeCycles to determine which
    object is of interest.

The columns:
    (required) [us ] TimeStamp
    (optional) [#  ] CycleCount

The source columns:
    (required) [kph] CAN Global.RelSpd_tg1
    (required) [deg] CAN Global.Angle_tg1
    (required) [m  ] CAN Global.Range_tg1
    (optional) [?  ] CAN <trait>

The object columns:
    (required) [#  ] aObject[i].General.uiLifeCycles
    (required) [m  ] aObject[i].Kinematic.fDistX
    (required) [m  ] aObject[i].Kinematic.fDistY
    (required) [mps] aObject[i].Kinematic.fVrelX
    (required) [mps] aObject[i].Kinematic.fVrelY
    (optional) [?  ] aObject[i].<trait>
"""
import matplotlib.pyplot as plt
import collections
import argparse
import logging


from . import bunch
from . import plots
from . import log


import pandas as pd
pd.set_option('display.width', 128)
pd.set_option('display.max_columns', 64)
pd.set_option('display.notebook_repr_html', True)


logger = logging.getLogger(__name__)


class Options(collections.OrderedDict):
    """
    Container for program options.
    """
    __setattr__ = collections.OrderedDict.__setitem__
    __getattr__ = collections.OrderedDict.__getitem__
    __delattr__ = collections.OrderedDict.__delitem__

    def __init__(self):
        super(Options, self).__init__()
        self.input_csv = 'homework.csv'
        self.label = 32
        self.size  = 32
        self.save  = False
        self.show  = False

    def log_options(self, fmt='{}: {}'):
        """
        Write options to the logger.
        """
        for key, val in self.items():
            logger.debug(fmt.format(key, val))

    @classmethod
    def from_command_line(cls, args=None):
        """
        Parse command line for options.
        """
        # create parse
        parser = argparse.ArgumentParser(
            prog='python3 -m homework',
            description=__doc__, argument_default=argparse.SUPPRESS,
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # add command line arguments
        parser.add_argument(dest='input_csv', nargs='?', metavar='input.csv',
                            help='input CSV filesystem path')
        parser.add_argument('--label', type=int, help='object to analyze')
        parser.add_argument('--size', type=int, help='rolling window size')
        parser.add_argument('--show', action='store_true', help='show plots')
        parser.add_argument('--save', action='store_true', help='save plots')

        # parse command line
        namespace = parser.parse_args(args=args)

        # store results in Options instance
        obj = cls()
        for key, val in namespace.__dict__.items():
            if hasattr(obj, key):
                setattr(obj, key, val)
            else:
                raise KeyError('unknown option: %s' % key)
        obj.update(namespace.__dict__)

        return obj


def main(csv, label, size=32, show=False, save=False):
    """
    Main function of script.

    Seealso:
        See the Options class for parameter description.
    """
    data_bunch = bunch.RADARBunch(csv)

    # subset = data_bunch.get_subset_for_many_objects(
    #     *data_bunch.object_labels, traits=['General.uiLifeCycles'])

    # analyze an object
    robj = plots.trajectory.ReferenceObject(data_bunch)
    tobj = plots.trajectory.TargetObject(data_bunch, label)
    dobj = plots.trajectory.DeltaObject(data_bunch, robj, tobj)

    # table output
    logger.debug('target summary\n%s',
                 robj.to_data_frame().describe())

    logger.debug('reference summary\n%s',
                 robj.to_data_frame().describe())

    logger.debug('reference-target summary\n%s',
                 dobj.to_data_frame().describe())

    # create plots
    if show or save:
        # boxplot uiLifeCycles to see meaningful signals
        plots.boxplot.many(data_bunch, trait='General.uiLifeCycles')
        if save:
            plots.helpers.savefig('uiLifeCycles.pdf')

        # trajectory plots
        plots.trajectory.components(objs=[robj, tobj])
        if save:
            plots.helpers.savefig('traj_{}.pdf'.format(label))

        plots.trajectory.components(objs=[dobj],
                                    func=plots.trajectory.attr_histogram)
        if save:
            plots.helpers.savefig('hist_{}.pdf'.format(label))

        for func in ['mean', 'std', 'median', 'cov']:
            plots.trajectory.components(
                objs=[dobj], func=lambda *args, **kwargs:
                plots.trajectory.window(*args, func=func, size=size, **kwargs))
            if save:
                plots.helpers.savefig('rollex_{}.pdf'.format(func))

        if show:
            plt.show()
    else:
        logger.debug('skipping plots (use --show or --save)')


if __name__ == '__main__':
    # setup the logging module
    log.setup_logging(level=logging.DEBUG, filename='homework.log')

    # parse command line arguments
    opts = Options.from_command_line()
    opts.log_options()

    # run main function
    with log.capture(fatal=True):
        main(csv=opts.input_csv, label=opts.label, size=opts.size,
             show=opts.show, save=opts.save)
