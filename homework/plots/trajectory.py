import matplotlib.pyplot as plt
import matplotlib.transforms
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import logging


from . import helpers


logger = logging.getLogger(__name__)


labels = dict(t='t (s)',
              a='$\Theta (^{\circ})$',
              x='x (m)',
              y='y (m)',
              r='r (m)',
              u='u (m s$^{-1}$)',
              v='v (m s$^{-1}$)',
              s='s (m s$^{-1}$)')


class BaseObject:
    """
    Container for position, velocity, angle, and time arrays.
    """
    def __init__(self, bunch):
        """
        Parameters:
            bunch (RADARBunch): data bunch
        """
        self.t = bunch.data_frame['TimeStamp[s]']
        self.a = np.zeros_like(self.t)
        self.x = np.zeros_like(self.t)
        self.y = np.zeros_like(self.t)
        self.r = np.zeros_like(self.t)
        self.u = np.zeros_like(self.t)
        self.v = np.zeros_like(self.t)
        self.s = np.zeros_like(self.t)

    def _compute_gradients(self):
        self.dt = np.array(np.gradient(self.t))
        self.dx = np.array(np.gradient(self.x))
        self.dy = np.array(np.gradient(self.y))
        self.dr = np.array(np.gradient(self.r))
        self.du = np.array(np.gradient(self.x))
        self.dv = np.array(np.gradient(self.y))
        self.ds = np.array(np.gradient(self.r))
        self.dxdt = self.dx / self.dt
        self.dydt = self.dy / self.dt
        self.drdt = self.dr / self.dt
        self.dudt = self.du / self.dt
        self.dvdt = self.dv / self.dt
        self.dsdt = self.ds / self.dt

    def to_data_frame(self):
        data = collections.OrderedDict()
        data['t'] = self.t
        data['x'] = self.x
        data['y'] = self.y
        data['r'] = self.r
        data['u'] = self.u
        data['v'] = self.v
        data['s'] = self.s
        return pd.DataFrame(data, columns=data.keys())


class ReferenceObject(BaseObject):
    """
    Container for reference object.
    """
    def __init__(self, bunch):
        """
        Parameters:
            bunch (RADARBunch): data bunch
        """
        super(ReferenceObject, self).__init__(bunch)

        # angle (degrees)
        self.a = -1.0 * bunch.get_subset_for_source(
            traits=['Global.Angle_tg1'])

        # position (m)
        self.r = bunch.get_subset_for_source(traits=['Global.Range_tg1'])
        self.x = self.r * np.cos(np.deg2rad(self.a))
        self.y = self.r * np.sin(np.deg2rad(self.a))

        # velocity (m/s)
        self.s = bunch.get_subset_for_source(
            traits=['Global.RelSpd_tg1']) * 1000.0 / 60.0 / 60.0
        self.u = self.s * np.cos(np.deg2rad(self.a))
        self.v = self.s * np.sin(np.deg2rad(self.a))

        # plot parameters
        self.label = 'reference'


class TargetObject(BaseObject):
    """
    Container for target object.
    """
    def __init__(self, bunch, label):
        """
        Parameters:
            bunch (RADARBunch): data bunch
            label  (int): object label
        """
        super(TargetObject, self).__init__(bunch)

        # position (m)
        self.x = bunch.get_subset_for_object(
            label, traits=['Kinematic.fDistX'])
        self.y = bunch.get_subset_for_object(
            label, traits=['Kinematic.fDistY'])
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        # angle (degrees)
        self.a = np.rad2deg(np.arctan2(self.y, self.x))

        # velocity (m/s)
        self.u = bunch.get_subset_for_object(
            label, traits=['Kinematic.fVrelX'])
        self.v = bunch.get_subset_for_object(
            label, traits=['Kinematic.fVrelY'])
        self.s = np.sqrt(self.u ** 2 + self.v ** 2)

        # plot parameters
        self.label = 'target {}'.format(label)


class DeltaObject(BaseObject):
    """
    Container for reference-target difference.
    """
    def __init__(self, bunch, obj0, obj1):
        super(DeltaObject, self).__init__(bunch)
        for name in ['a', 'x', 'y', 'r', 'u', 'v', 's']:
            diff = getattr(obj1, name) - getattr(obj0, name)
            setattr(self, name, diff)
        self.label = '$\Delta_{%s-%s}$' % (obj1.label, obj0.label)


def determin_extent(objs, names=None, v_min=None, v_max=None):
    """
    Figure out lowest and highest values across all object arrays.

    Parameters:
        objs (list): objects
        names (list[str]): attribute names
        v_min (float): starting min
        v_max (float): starting max

    Returns:
        list[float]: min and max across all arrays
    """
    if names is None:
        return v_min, v_max

    if v_min is None:
        v_min = np.amin(getattr(objs[0], names[0]))

    if v_max is None:
        v_max = np.amax(getattr(objs[0], names[0]))

    for obj in objs:
        for name in names:
            v_min = min(np.amin(getattr(obj, name)), v_min)
            v_max = max(np.amax(getattr(obj, name)), v_max)

    return v_min, v_max


def t_vs_attr(objs, name='x', axis=None, **kwargs):
    """
    Plot time vs reference and target attribute.

    Parameters:
        objs (list): reference, targets, etc
        name (str): attribute name
        axis: axis instance
        kwargs (dict): plot kwargs
    """
    if axis is None:
        fig, axis = helpers.subplots(1, 1)

    plt.sca(axis)
    for obj in objs:
        y0, x0, = obj.t, getattr(obj, name)
        plt.plot(x0, y0, label=obj.label, **kwargs)

    plt.ylabel('t (s)')

    try:
        plt.xlabel(labels[name])
    except KeyError:
        pass


def attr_histogram(objs, name='x', axis=None, **kwargs):
    """
    Plot histogram of variable.

    Parameters:
        objs (list): reference, targets, etc
        name (str): attribute name
        axis: axis instance
        kwargs (dict): plot kwargs
    """
    if axis is None:
        fig, axis = helpers.subplots(1, 1)

    _kwargs = dict(hist_kws=dict(histtype='step', linewidth=3), kde_kws=dict(linewidth=3))
    _kwargs.update(**kwargs)

    plt.sca(axis)
    transform = matplotlib.transforms.blended_transform_factory(
        axis.transData, axis.transAxes)

    for obj in objs:
        x0 = getattr(obj, name)
        sns.distplot(x0, label=obj.label, **_kwargs)
        avg, std = np.mean(x0), np.std(x0)

        xs = np.linspace(*axis.get_xlim(), 128)
        plt.plot(xs, norm.pdf(xs, avg, std), label='normal', linewidth=3)

        plt.axvline(x=avg, color='k')
        plt.errorbar([avg], [0.5], xerr=std, marker='o', color='k',
                     transform=transform, ecolor='k', capsize=10, mew=2)
        plt.text(1.0, 1.0, '$\mathtt{{\mu = {:+.3e}}}$'.format(avg),
                 transform=axis.transAxes,
                 ha='right', va='top', fontsize='x-small')
        plt.text(1.0, 0.9, '$\mathtt{{\sigma = {:+.3e}}}$'.format(std),
                 transform=axis.transAxes,
                 ha='right', va='top', fontsize='x-small')

    plt.ylabel('count (#)')

    try:
        plt.xlabel(labels[name])
    except KeyError:
        pass


def window(objs, name='x', axis=None,
           size=32, func='mean', data=True, **kwargs):
    """
    Plot histogram of variable.

    Parameters:
        objs (list): reference, targets, etc
        name (str): attribute name
        axis: axis instance
        size (int): window size
        func (str): rolling method (mean, std, etc)
        data (bool): plot original data also?
        kwargs (dict): plot kwargs
    """
    if axis is None:
        fig, axis = helpers.subplots(1, 1)

    plt.sca(axis)

    for obj in objs:
        x0 = obj.t
        y0 = getattr(obj, name)
        y1 = getattr(y0.rolling(window=size), func)()
        y2 = getattr(y0.expanding(), func)()

        if data:
            plt.plot(x0, y0, alpha=0.25, label='data', **kwargs)

        label = 'rolling (window={}) {}'.format(size, func)
        plt.plot(x0, y1, label=label, **kwargs)
        plt.plot(x0, y2, label='expanding {}'.format(func), **kwargs)

        plt.xlabel('t (s)')
        plt.ylabel(labels[name])


def components(objs, func=None, tlim=None, qlim=None, plim=None, **kwargs):
    """
    Plot time vs reference and target attributes.

    Parameters:
        objs (list[BaseObject]): reference, targets, etc
        func (method): plotting function
        tlim (list): t limits (time)
        qlim (list): q limits (position)
        plim (list): p limits (velocity)
        kwargs (dict): plot kwargs
    """
    if func is None:
        func = t_vs_attr

    logger.debug('creating compnents plot via {}'.format(func))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = helpers.subplots(3, 2, ts=2)
    func(objs, name='x', axis=ax1, **kwargs)
    func(objs, name='y', axis=ax2, **kwargs)
    func(objs, name='u', axis=ax3, **kwargs)
    func(objs, name='v', axis=ax4, **kwargs)
    func(objs, name='r', axis=ax5, **kwargs)
    func(objs, name='s', axis=ax6, **kwargs)

    if tlim is not None:
        ax1.set_ylim(*tlim)
        ax2.set_ylim(*tlim)
        ax3.set_ylim(*tlim)
        ax4.set_ylim(*tlim)

    if qlim is not None:
        ax1.set_xlim(*qlim)
        ax2.set_xlim(*qlim)

    if plim is not None:
        ax3.set_xlim(*plim)
        ax4.set_xlim(*plim)

    sns.despine(fig, ax1)
    sns.despine(fig, ax2)
    sns.despine(fig, ax3)
    sns.despine(fig, ax4)

    transform = matplotlib.transforms.blended_transform_factory(
        fig.transFigure, ax1.transAxes
    )
    plt.sca(ax4)
    handles, _ = ax4.get_legend_handles_labels()
    plt.legend(loc='lower center', ncol=len(handles), bbox_transform=transform,
               bbox_to_anchor=(0.5, 1.05), frameon=True)
