import pickle
import numpy as np
import matplotlib
# matplotlib.use('MacOSX')
# matplotlib.use('Qt')
from matplotlib import rcParams
from cycler import cycler
from matplotlib import pyplot as plt
import os, sys

def format_y(y):
    if isinstance(y, list) and len(y) != 0:
        if isinstance(y[0], list):
            lengths = [len(obj) for obj in y]
            minlength = min(lengths)
            y = [obj[:minlength] for obj in y]
    return y

def plot_figures(output_path, desc, y, xlabel, ylabel, x=None, yerr = None, legend=None, legendloc='best',
                 legendncol=1, title=None, xlim=None, ylim=None, show_plot=False, gen_pkl=True, save_pdf=False,
                 plt_only=False):
    plt.clf()
    plt.close()

    if not plt_only:
        rcParams.update({'font.size': 20})
        rcParams['interactive'] = True
        plt.ioff()
        plt.rc('axes', prop_cycle=cycler('color',['black', 'red', 'blue', 'black', 'red', 'blue', 'black','red', 'blue', 'black', 'red', 'blue', 'black']) + cycler('marker', ['*', '+', 'x', 'o', '<', '>', 'v', '^', ',', "_", '.', '|', 'X']) + cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.',':', '-', '--', '-.',':','-']))
        # this ensures that type-3 fonts are not used when generating figures
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        markersize=8
        linewidth=2
        capsize = 6 # not recognized in plt.rc
        elinewidth = 2 # same
        markeredgewidth = 1
        plt.rc('lines', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        # plt.gca().set_prop_cycle(cycler('color',['red', 'green', 'blue', 'red', 'green', 'blue','red']))
        # markers = ['*', '+', 'x', 'o', '<', '>', ',']  # http://matplotlib.org/api/markers_api.html
        # linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # http://matplotlib.org/api/lines_api.html
        fig = plt.figure(1, figsize=(40,15))  # width, height
        # fig = plt.figure(1, figsize=(7.5,7.5))  # width, height
        # fig = plt.figure(1)  # width, height

    y = format_y(y)
    y = np.array(y)

    if yerr is not None:
        yerr = format_y(yerr)
        yerr = np.array(yerr)
        assert np.shape(y) == np.shape(yerr)

    shape = y.shape
    if len(shape) == 1:
        ncols = shape[0]
        nrows = 1
    else:
        nrows, ncols = shape
    if x is None:
        x = range(1,ncols+1)

    if nrows == 1:
        if yerr is None:
            plt.plot(x, y)
        else:
            ax = plt.gca() # use this only if needed
            ax.set_xscale('log')
            # ax.set_yscale('log')
            (_, caps, _) = plt.errorbar(x, y, yerr, capsize=capsize, elinewidth=elinewidth)
            for cap in caps:
                cap.set_markeredgewidth(3)

    else:
        if yerr is None:
            for var_indx in range(nrows):
                plt.plot(x, y[var_indx, :])
        else:
            ax = plt.gca() # use this only if needed
            ax.set_xscale('log')
            for var_indx in range(nrows):
                (_, caps, _) = plt.errorbar(x, y[var_indx, :], yerr[var_indx, :], capsize=capsize, elinewidth=elinewidth)
                for cap in caps:
                    cap.set_markeredgewidth(3)

    # plt.ylim(ymin=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend, loc=legendloc, ncol=legendncol)
    if title is not None:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, which='both')

    if not plt_only:
        # fig.tight_layout()
        filename = 'fig_' + desc
        if save_pdf:
            fig.savefig(os.path.join(output_path, filename + '.pdf'))
        plt.savefig(os.path.join(output_path, filename + '.png'))
        if gen_pkl:
            save_object1(fig, os.path.join(output_path, 'pkl', filename + '.pkl'))
        if show_plot:
            plt.show()
        plt.clf()
        plt.close()

def plot_figures_subplot(output_path, desc, y, xlabels, ylabels, x=None, legends=None, legendlocs=None, legendncols=None, show_plot=False, gen_pkl=True, save_pdf=False, save_eps=False):

    rcParams.update({'font.size': 20})
    plt.ioff()
    plt.rc('axes', prop_cycle=cycler('color',['black', 'red', 'blue', 'black', 'red', 'blue', 'black','red', 'blue', 'black', 'red', 'blue', 'black']) + cycler('marker', ['*', '+', 'x', 'o', '<', '>', 'v', '^', ',', "_", '.', '|', 'X']) + cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.',':', '-', '--', '-.',':','-']))
    markersize=3
    linewidth=3
    plt.rc('lines', linewidth=linewidth, markersize=markersize)

    # plt.gca().set_prop_cycle(cycler('color',['red', 'green', 'blue', 'red', 'green', 'blue','red']))
    # markers = ['*', '+', 'x', 'o', '<', '>', ',']  # http://matplotlib.org/api/markers_api.html
    # linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # http://matplotlib.org/api/lines_api.html
    if isinstance(y, list) and len(y) != 0:
        if isinstance(y[0], list):
            lengths = [len(obj) for obj in y]
            minlength = min(lengths)
            y = [obj[:minlength] for obj in y]
    y = np.array(y)
    shape = y.shape
    if len(shape) == 1:
        ncols = shape[0]
        nrows = 1
    else:
        nrows, ncols = shape
    if x is None:
        x = range(1,ncols+1)

    fig,_ = plt.subplots(nrows, 1, figsize=(11.25,7.5))  # width, height

    if legends is None:
        legends = legendlocs = legendncols = [None]*nrows

    for var_indx in range(nrows):
        subplt_indx = (nrows*100) + (1*10) + (var_indx+1)
        plt.subplot(subplt_indx)
        plot_figures('','',y[var_indx,:], xlabels[var_indx],ylabels[var_indx],x,legends[var_indx],legendlocs[var_indx],legendncols[var_indx],plt_only=True)
        # ax[var_indx].plot(x,y[var_indx,:])

    fig.tight_layout()
    filename = 'fig_' + desc
    if save_pdf:
        plt.savefig(output_path + filename + '.pdf')
    if save_eps:
        plt.savefig(output_path + filename + '.eps')
    plt.savefig(output_path + filename + '.png')
    if gen_pkl:
        save_object1(fig, output_path + 'pkl/' + filename + '.pkl')
    if show_plot:
        plt.show()
    plt.clf()
    plt.close()

def plot_figures_old(output_path, desc, y, xlabel, ylabel, x=None, legend=None, legendloc=None, legendncol=None, show_plot=False, gen_pkl=True, save_pdf=False, save_eps=False):

    # rcParams.update({'font.size': 20})
    # plt.ioff()
    # plt.rc('axes', prop_cycle=cycler('color',['black', 'red', 'blue', 'black', 'red', 'blue', 'black','red', 'blue', 'black', 'red', 'blue', 'black']) + cycler('marker', ['*', '+', 'x', 'o', '<', '>', 'v', '^', ',', "_", '.', '|', 'X']) + cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.',':', '-', '--', '-.',':','-']))
    # markersize=10
    # linewidth=3
    # plt.rc('lines', linewidth=linewidth, markersize=markersize)
    #
    # # plt.gca().set_prop_cycle(cycler('color',['red', 'green', 'blue', 'red', 'green', 'blue','red']))
    # # markers = ['*', '+', 'x', 'o', '<', '>', ',']  # http://matplotlib.org/api/markers_api.html
    # # linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # http://matplotlib.org/api/lines_api.html
    # fig = plt.figure(1, figsize=(15, 10))  # width, height
    rcParams.update({'font.size': 20})
    plt.ioff()
    plt.rc('axes', prop_cycle=cycler('color',['black', 'red', 'blue', 'black', 'red', 'blue', 'black','red', 'blue', 'black', 'red', 'blue', 'black']) + cycler('marker', ['*', '+', 'x', 'o', '<', '>', 'v', '^', ',', "_", '.', '|', 'X']) + cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.',':', '-', '--', '-.',':','-']))
    markersize=11
    linewidth=3
    capsize = 6 # not recognized in plt.rc
    elinewidth = 3 # same
    markeredgewidth = 1
    plt.rc('lines', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    # plt.gca().set_prop_cycle(cycler('color',['red', 'green', 'blue', 'red', 'green', 'blue','red']))
    # markers = ['*', '+', 'x', 'o', '<', '>', ',']  # http://matplotlib.org/api/markers_api.html
    # linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # http://matplotlib.org/api/lines_api.html
    # fig = plt.figure(1, figsize=(11.25,7.5))  # width, height
    fig = plt.figure(1, figsize=(7.5,7.5))  # width, height

    if isinstance(y, list) and len(y) != 0:
        if isinstance(y[0], list):
            lengths = [len(obj) for obj in y]
            minlength = min(lengths)
            y = [obj[:minlength] for obj in y]
    y = np.array(y)
    shape = y.shape
    if len(shape) == 1:
        ncols = shape[0]
        nrows = 1
    else:
        nrows, ncols = shape
    if x is None:
        x = range(1,ncols+1)

    if nrows == 1:
        plt.plot(x, y)
    else:
        for var_indx in range(nrows):
            plt.plot(x, y[var_indx, :])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend, loc=legendloc, ncol=legendncol)
    plt.grid()
    fig.tight_layout()
    filename = 'fig_' + desc
    if save_pdf:
        plt.savefig(output_path / (filename + '.pdf'))
    if save_eps:
        plt.savefig(output_path / (filename + '.eps'))
    plt.savefig(output_path / (filename + '.png'))
    if gen_pkl:
        save_object1(fig, output_path / 'pkl/' / (filename + '.pkl'))
    if show_plot:
        plt.show()
    plt.clf()
    plt.close()

def sort_pair_of_lists(list1, list2, reverse=False):
    # sorting will be based on the values of list1 (not list2)
    zipped_pair = zip(list1, list2)
    sorted_zip = sorted(zipped_pair, reverse=reverse)
    list1_sorted = [x for x, _ in sorted_zip]
    list2_sorted = [x for _, x in sorted_zip]
    return [list1_sorted, list2_sorted]

def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    s = str(s)
    if f:
        f.write(s)
        if new_line:
            f.write("\n")
    # stdout
    print(s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()

def save_object1(obj1, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:
        pickle.dump(obj1, output, pickle.HIGHEST_PROTOCOL)

def save_object2(obj1, obj2, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:
        pickle.dump(obj1, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj2, output, pickle.HIGHEST_PROTOCOL)

def read_object1(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def read_object2(filename):
    with open(filename, 'rb') as input:
        first = pickle.load(input)
        second = pickle.load(input)
        return first, second

# Function to get index of ceiling of x in arr[low..high]*/
def ceilSearch(arr, low, high, x):
    # If x is smaller than or equal to the first element,
    # then return the first element */
    if x <= arr[low]:
        return low

        # If x is greater than the last element, then return -1 */
    if x > arr[high]:
        return -1

        # get the index of middle element of arr[low..high]*/
    mid = int ((low + high) / 2)  # low + (high - low)/2 */

    # If x is same as middle element, then return mid */
    if arr[mid] == x:
        return mid

    # If x is greater than arr[mid], then either arr[mid + 1]
    # is ceiling of x or ceiling lies in arr[mid+1...high] */
    # elif arr[mid] < x:
    #     if mid + 1 <= high and x <= arr[mid + 1]:
    #         return mid + 1
    #     else:
    elif arr[mid] < x:
        return ceilSearch(arr, mid + 1, high, x)

    # If x is smaller than arr[mid], then either arr[mid]
    # is ceiling of x or ceiling lies in arr[mid-1...high] */
    else:
        # if mid - 1 >= low and x > arr[mid - 1]:
        #     return mid
        # else:
        return ceilSearch(arr, low, mid, x)

# Binary search function to get index of floor of x in arr[low..high]*/
def floorSearch(arr, low, high, x):
    # If x is smaller than or equal to the first element,
    # then return the first element */
    if x >= arr[high]:
        return high

    # If x is greater than the last element, then return -1 */
    if x < arr[low]:
        return -1

    # get the index of middle element of arr[low..high]*/
    mid = int ((low + high) / 2)  # low + (high - low)/2 */

    # If x is same as middle element, then return mid */
    if arr[mid] == x:
        return mid

    # If x is greater than arr[mid], then floor of x lies in arr[mid...high] */
    # elif arr[mid] < x:
    #     if mid + 1 <= high and x <= arr[mid + 1]:
    #         return mid + 1
    #     else:
    elif arr[mid] < x:
        if x < arr[mid+1]: # this is done to avoid infinite recursion; consider [2,8] and floor(3)
            return mid
        return floorSearch(arr, mid, high, x)

    # If x is smaller than arr[mid], then floor of x lies in arr[low...mid-1] */
    else:
        # if mid - 1 >= low and x > arr[mid - 1]:
        #     return mid
        # else:
        return floorSearch(arr, low, mid-1, x)


