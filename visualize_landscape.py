from definitions import PROJECT_ROOT
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import argparse
import os
from scipy.ndimage import gaussian_filter


"""
Script for plotting contour plots of optimization landscapes. Needs pre-computed point grids generated with
generate_landscape_data.py.
"""


def load_data(data_path, file, filter_sigma):
    """
    Load landscape data from the specified file.
    :param data_path: directory of saved .npz files
    :param file: name of the file to load
    :param filter_sigma: standard deviation of Gaussian filtering
    :return: grid of evaluated points, 'good' optimization path, 'bad' optimization path, extent of slice, slice number
    """
    data = np.load(f'{data_path}/{file}', allow_pickle=True)
    values = gaussian_filter(data['values'], sigma=filter_sigma)
    good_path = data['good_path']
    bad_path = data['bad_path']
    extent = data['extent']
    stage = int(file.split('.')[0].split('_')[-1])
    return values, good_path, bad_path, extent, stage


def plot_data(args, axis, data_path, file, plot_good_path=True, plot_bad_path=True, subplot_titles=None):
    """
    Plot the contours of a slice.
    :param args: script arguments
    :param axis: figure axis
    :param data_path: directory of saved .npz files
    :param file: name of the file to plot
    :param plot_good_path: whether to plot the 'good' optimization path
    :param plot_bad_path: whether to plot the 'bad' optimization path
    :param subplot_titles: title of this subplot
    :return: contour set
    """
    values, good_path, bad_path, extent, stage = load_data(data_path, file, args.filter_sigma)
    if subplot_titles is not None:
        title = subplot_titles[stage - 1]
        axis.set_title(title, fontsize=14)

    min_v = args.min
    max_v = args.max

    levels = np.linspace(min_v, max_v, args.contour_levels)
    z_vals = np.swapaxes(values[0, :, :], 0, 1)
    cset = axis.contour(z_vals, vmin=min_v, vmax=max_v, levels=levels, cmap='viridis',
                        extent=(-extent, extent, -extent, extent))
    if args.show_labels:
        axis.clabel(cset, cset.levels, inline=True, fmt='{:.0f}'.format, fontsize='smaller', inline_spacing=0)

    if good_path is not None and plot_good_path:
        path_x = good_path[0, :]
        path_y = good_path[1, :]
        axis.plot(path_x[:stage], path_y[:stage], marker='.', color='k', linestyle='-')
        #axis.plot(path_x[0], path_y[0], color='g', marker='o', linestyle='None', zorder=12)
        axis.plot(path_x[stage-1], path_y[stage-1], color='k', marker='o', linestyle='None', zorder=12)
        # subfig.plot(path_x[ind], path_y[ind], color='k', marker='o', linestyle='None', zorder=12)

    if None not in bad_path and plot_bad_path:
        # x_diffs = path_x[1:] - path_x[:-1]
        # y_diffs = path_y[1:] - path_y[:-1]
        path_x = bad_path[0, :]
        path_y = bad_path[1, :]
        axis.plot(path_x[:stage], path_y[:stage], marker='.', color='k', linestyle='-')
        # ax.quiver(path_x[:-1], path_y[:-1], x_diffs, y_diffs, angles='xy', zorder=10)
        #axis.plot(path_x[0], path_y[0], color='g', marker='o', linestyle='None', zorder=12)
        axis.plot(path_x[stage-1], path_y[stage-1], color='k', marker='o', linestyle='None', zorder=12)
        # subfig.plot(path_x[ind], path_y[ind], color='k', marker='o', linestyle='None', zorder=12)

    lims = [-extent / args.zoom, extent / args.zoom]
    axis.set_xlim(lims)
    axis.set_ylim(lims)

    return cset


def plot_slices(args):
    """
    Plot the slices from all specified experiments and arrange the figure layout.
    :param args: script arguments
    :return: finished figure
    """
    rows = [int(r) for r in args.plot_rows.split(',')]
    exp_dirs = [args.exp_dirs.split(',')[r] for r in rows]
    base_data_path = f'{PROJECT_ROOT}/plots/data'
    path_flags = [(True, False), (False, True), (False, False)]
    files = [file for file in next(os.walk(f'{base_data_path}/{exp_dirs[0]}'))[2]]
    stages = list(set([int(f.split('.')[0].split('_')[-1]) for f in files]))
    fig_size = (len(stages) * 3.5, len(exp_dirs) * 4)

    fig = plt.figure(figsize=fig_size)
    subplots = fig.subplots(len(exp_dirs), len(stages), sharex='col', sharey='row',
                            subplot_kw={'adjustable': 'box', 'aspect': 1})

    row_ylabels = ['Full curriculum', 'Action noise only', 'Assistance only']

    if len(rows) > 1:
        for i in range(len(rows)):
            subplots[i][0].set_ylabel(row_ylabels[rows[i]], fontsize=16, labelpad=35)
    else:
        subplots[0].set_ylabel(row_ylabels[rows[0]], fontsize=16, labelpad=35)

    sigma_range = np.linspace(float(args.sigma_range.split(',')[0]), float(args.sigma_range.split(',')[1]),
                              len(stages))
    curriculum_range = np.linspace(float(args.quantity_range.split(',')[0]), float(args.quantity_range.split(',')[1]),
                                   len(stages) - 1)
    curriculum_range = np.concatenate((curriculum_range, [curriculum_range[-1]]))

    subplot_titles = [[f'{args.curriculum_quantity}={round(curriculum_range[i], 1)}{args.unit}\n'
                       f'$\\sigma$={round(sigma_range[i], 3)}' for i in range(len(stages))],
                      [f'$\\sigma$={round(sigma_range[i], 3)}' for i in range(len(stages))],
                      [f'{args.curriculum_quantity}={round(curriculum_range[i], 1)}{args.unit}'
                       for i in range(len(stages))]]

    cset = None
    for row, exp_dir in enumerate(exp_dirs):
        data_path = f'{base_data_path}/{exp_dir}'
        files = next(os.walk(data_path))[2]

        for col, file in enumerate(files):
            if len(exp_dirs) > 1:
                cset = plot_data(args, subplots[row][col], data_path, file, path_flags[rows[row]][0],
                                 path_flags[rows[row]][1], subplot_titles[rows[row]])
            else:
                cset = plot_data(args, subplots[col], data_path, file, path_flags[rows[row]][0],
                                 path_flags[rows[row]][1], subplot_titles[rows[row]])

    cax = fig.add_axes([0.2, 0.14, 0.6, 0.02])
    norm = matplotlib.colors.Normalize(vmin=cset.cvalues.min(), vmax=cset.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cset.cmap)
    sm.set_array(np.array([]))

    if len(exp_dirs) > 1:
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', shrink=0.75, aspect=55)
    else:
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', shrink=0.75, aspect=55, ticks=cset.levels)

    cbar.set_label('Average episodic return', fontsize=15)
    alignment_label = ', PCA alignment' if args.alignment == 'pca' else ''
    fig.suptitle(f'{args.figure_title}{alignment_label}', fontsize=20)

    if args.alignment == 'pca':
        axis_name = 'principal axis'
    else:
        axis_name = 'basis direction'

    fig.text(0.5, 0.2, f'1st {axis_name}', ha='center', fontsize=14)
    fig.text(0.035, 0.56, f'2nd {axis_name}', va='center', rotation='vertical', fontsize=14)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, hspace=0.25)

    return fig


def plot_single_slice(args):
    """
    Plot a single slice and arrange the figure layout.
    :param args: script arguments
    :return: finished figure
    """
    exp_dir = args.exp_dirs.split(',')[0]
    base_data_path = f'{PROJECT_ROOT}/plots/data'
    data_path = f'{base_data_path}/{exp_dir}'
    files = [file for file in next(os.walk(f'{base_data_path}/{exp_dir}'))[2]]
    plot_file = files[args.plot_slice-1]

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    subplot = gs.subplots(sharex='col', sharey='row', subplot_kw={'adjustable': 'box', 'aspect': 1})
    cset = plot_data(args, subplot, data_path, plot_file, args.plot_good_path, args.plot_bad_path)
    cbar = fig.colorbar(cset, location='right', shrink=0.75, aspect=55)
    cbar.ax.set_ylabel('Average episodic return', rotation=270, labelpad=20, fontsize=15)
    fig.text(0.5, 0.4, '1st basis direction', ha='center')
    fig.text(0.5, 0.4, '2nd basis direction', va='center', rotation='vertical')
    return fig


def main():
    """
    Parse script arguments and call the plotting function.
    """
    parser = argparse.ArgumentParser('Plot the landscape visualization data')
    parser.add_argument('--exp-dirs', type=str,
                        default='humanoid_ac_pca_full,'
                                'humanoid_ac_pca_noise,'
                                'humanoid_ac_pca_det',
                        help='Comma-separated list of directories to npz files containing landscape data and'
                             'optimization paths.')
    parser.add_argument('--plot-rows', type=str, default='0,1,2', help='Indices of the rows to plot, comma-separated.')
    parser.add_argument('--plot-slice', type=int, default=None,
                        help='Index of slice to plot. Leave to None (default) to plot all slices.')
    parser.add_argument('--plot-good-path', action='store_true', default=False,
                        help='Option to plot good optimization path when plotting single figure.')
    parser.add_argument('--plot-bad-path', action='store_true', default=False,
                        help='Option to plot bad optimization path when plotting single figure.')
    parser.add_argument('--contour-levels', type=int, default=100, help='Number of contour levels')
    parser.add_argument('--zoom', type=float, default=5.0, help='Zoom factor')
    parser.add_argument('--figure-title', type=str,
                        default='Humanoid optimization landscapes')
    parser.add_argument('--alignment', type=str, default='pca',
                        help='Alignment type used in the visualizations, "model", "pca", or "random".')
    parser.add_argument('--filter-sigma', type=float, default=1.0,
                        help='Standard deviation of the Gaussian filtering used for post-processing the data points.')
    parser.add_argument('--min', type=float, default=-1000, help='Minimum return value, lowest contour level')
    parser.add_argument('--max', type=float, default=3000, help='Maximum return value, highest contour level')
    parser.add_argument('--show-labels', action='store_true', default=False, help='Plot the contour level labels')
    parser.add_argument('--curriculum-quantity', type=str, default='Assistance',
                        help='Name of the quantity affected by the curriculum, e.g. "Action weight" or "Assistance"')
    parser.add_argument('--quantity-range', type=str, default='100,0',
                        help='Value range for curriculum quantity, comma-separated list.')
    parser.add_argument('--unit', type=str, default='%', help='Unit of the curriculum quantity.')
    parser.add_argument('--sigma-range', type=str, default='1.0,0.1',
                        help='Standard deviation range, comma-separated list.')
    parser.add_argument('--subplot-titles', type=str,
                        default=r'Assistance=100%, $\sigma$=0.5;Assistance=75%, $\sigma$=0.402;'
                                r'Assistance=50%, $\sigma$=0.304;Assistance=25%, $\sigma$=0.206;'
                                r'Assistance=0%, $\sigma$=0.108;Assistance=0%, $\sigma$=0.01',
                        help='Semicolon-separated list of titles for subplots. If None or empty string, titles will be'
                             '"Stage N" format.')
    parser.add_argument('--plot-path', action='store_true', default=True, help='Toggle to plot the optimization path')
    parser.add_argument('--dpi', type=int, default=90, help='Dpi value for saved images.')
    parser.add_argument('--save-dir', type=str, default='plots/images/fixed_layout',
                        help='Directory for saving images. Leave to None to not save.')
    parser.add_argument('--show-image', action='store_true', default=False, help='Show the image')

    args = parser.parse_args()
    #args.subplot_titles = None

    if args.zoom < 1.0:
        print('Specified zoom is too small, defaulting to minimum zoom (1.0).')
        args.zoom = 1.0

    exp_dirs = args.exp_dirs.split(',')

    if args.plot_slice is not None:
        fig = plot_single_slice(args)
    else:
        fig = plot_slices(args)

    #plt.tight_layout()
    if args.show_image:
        plt.show()
    if args.save_dir is not None:
        save_dir = f'{PROJECT_ROOT}/{args.save_dir}/'
        save_name = f'{save_dir}{exp_dirs[0].split("_")[0].lower()}{"_pca" if args.alignment == "pca" else ""}' \
                    f'_landscapes_{int(args.zoom)}x.jpeg'
        fig.savefig(save_name, dpi=args.dpi)
    plt.close(fig)


if __name__ == '__main__':
    main()
