import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from layer import Layer
from layer_group import LayerGroup
from mpl_toolkits.mplot3d import Axes3D
from thermal_history import ThermalHistory
from scipy.interpolate import make_interp_spline, BSpline


THERMAL_HISTORY_GROUP_NAME = '/ThermalHistories'
SYNTH_DC = '/DataContainers/CroppedDataContainer'
REAL_DC = '/DataContainers/VertexDataContainer'
SYNTH_POINTS = SYNTH_DC + '/_SIMPL_GEOMETRY/SharedVertexList'  #computed thermal histories
REAL_POINTS = REAL_DC + '/_SIMPL_GEOMETRY/SharedVertexList'  #points scan vectors
SYNTH_TSBM = SYNTH_DC + '/VertexData/TvT'
SYNTH_DIST = SYNTH_DC + '/VertexData/DistFromStart'
SYNTH_VECTOR = SYNTH_DC + '/VertexData/VectorId'
REAL_TSBM = REAL_DC + '/VertexData/TvT'
REAL_DIST = REAL_DC + '/VertexData/DistFromStart'
REAL_VECTOR = REAL_DC + '/VertexData/VectorId'


def read_thermal_histories(path, figs=None):
    f = h5py.File(path, 'r')
    ths = []
    total = len(f[THERMAL_HISTORY_GROUP_NAME].keys())
    counter = 0
    incr = total / 20
    prog = 1
    for n, d in f[THERMAL_HISTORY_GROUP_NAME].items():
        ths.append(ThermalHistory(np.array(d.value), int(n), figs))
        if counter > prog:
            progress = int((counter / total) * 100.0)
            print('Reading Thermal Histories | File: {} | {}% Complete'.format(os.path.basename(path), progress))
            prog += incr
        counter += 1
    f.close()
    return ths


def read_thermal_histories_csv(path, bname, extension, nfiles):
    ths = []
    for i in range(nfiles):
        fname = path + bname + str(i) + extension
        f = open(fname, "r")
        l = f.readline()
        l = l.strip("\n\r")
        tokens = l.split(sep=",")
        pos = [float(tokens[2]), float(tokens[3]), 0.0]
        data = np.genfromtxt(fname, dtype=np.double, comments='#')
        ths.append(ThermalHistory(data, i, pos))
        print("Imported curve {} of {}".format(i + 1, nfiles))
    return ths


def read_layer(path, curves, index=None, figs=None, real=False):
    f = h5py.File(path, 'r')
    if real:
        points = f[REAL_POINTS].value
        dist = f[REAL_DIST].value
        vector = f[REAL_VECTOR].value
        tsbm = f[REAL_TSBM].value
    else:
        points = f[SYNTH_POINTS].value
        dist = f[SYNTH_DIST].value
        vector = f[SYNTH_VECTOR].value
        tsbm = f[SYNTH_TSBM].value
    l = Layer(curves, points, index, figs)
    l.distance = dist
    l.vector = vector
    l.tsbm = tsbm
    return l

# dpath = 'E:/publications/am_analytics/data/'
# fpath = 'E:/publications/am_analytics/MaterialsCharacterization_Submission/revision/Figures/'
# dpath = "./data/"
# fpath = "./data/test_figs/base_k_means_2/"
files = [0, 2, 3, 36, 39, 88, 91, 103, 106, 127, 130]
prefix = 'ThermalHistories'
layers = []


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [5, 2, 11, 16, 10, 12, 5, 2, 17, 3]
plt.scatter(x, y)
plt.show()

x = np.linspace(0, 0.5, 50)
y = np.sin(2*np.pi*x*4)/np.exp(x*.5)*.1 + np.sin(2*np.pi*x*4.5)/np.exp(x*.5)*3 + np.sin(2*np.pi*x*4.8)/np.exp(x*1.5)
y = (np.sin(2 * np.pi * x * 3.0) / np.exp(x * 2.5))**2
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2, labelsize='large')
plt.plot(y, marker='o')
breaks = [0.1, 0.2, 0.4]
colors = ['g', 'y', 'm']
plt.hlines(breaks, xmin=0, xmax=len(x), linestyles='dashed', colors=colors)
plt.text(50, 0.05, 'a', fontsize='large', fontweight='bold', color='g')
plt.text(50, 0.15, 'b', fontsize='large', fontweight='bold', color='y')
plt.text(50, 0.30, 'c', fontsize='large', fontweight='bold', color='m')
plt.text(50, 0.50, 'd', fontsize='large', fontweight='bold', color='r')
plt.show()

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 2, 11, 16, 13, 12, 4, 2, 17, 3]

xtmp = [0, 1, 1.66, 3, 4, 5, 5.59, 7, 8.28, 9]
ytmp = [1, 2, 8, 16, 13, 12, 7.66, 2, 18.6, 3]

xnew = np.linspace(0, 9, 200)
spline = make_interp_spline(x, y, k=3)
ynew = spline(xnew)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2, labelsize='large')
plt.scatter(xtmp, ytmp, marker='o')
plt.plot(xnew, ynew)
breaks = [5, 10, 15]
colors = ['g', 'y', 'm']
plt.hlines(breaks, xmin=0, xmax=len(x), linestyles='dashed', colors=colors)
plt.text(10, 2.5, 'a', fontsize='large', fontweight='bold', color='g')
plt.text(10, 7.5, 'b', fontsize='large', fontweight='bold', color='y')
plt.text(10, 12.5, 'c', fontsize='large', fontweight='bold', color='m')
plt.text(10, 17.5, 'd', fontsize='large', fontweight='bold', color='r')
name = 'featurization.png'
plt.savefig(fpath + name, bbox_inches='tight', dpi=1200)

# raw string latex works in plot

# for file in files:
#     th = dpath + file + '.hdf5' if isinstance(file, str) else dpath + prefix + '_' + str(file) + '.hdf5'
#     d3d = dpath + file + '.dream3d' if isinstance(file, str) else dpath + prefix + str(file) + '.dream3d'
#     isreal = isinstance(file, str)
#     layer = read_layer(d3d, None, file, fpath, isreal)
#     layers.append(layer)

# lg = LayerGroup(layers, fpath)

# lg.kmeans(clusters=4, standardize=True)
# lg.plot_kmeans(max_clusters=4)
# lg_kmeans = lg.kmeans_fit
#
# csv_path = 'E:/data/additive_manufacturing/simulations/thermal_modeling/cylinder_curves/curves/'
# csv_ths = read_thermal_histories_csv(csv_path, 'curve_', '.csv', 5614)
# cs = []
# for th in csv_ths:
#     cs.append(th.position)
# points = np.array(cs)
# cylinder = Layer(csv_ths, points, index=0, figs=fpath)
# cylinder.compute_sax()
# cylinder.compute_tsbm_ivt()
# cylinders = [cylinder]
# cylinder_lg = LayerGroup(cylinders, fpath)
# cylinder_lg.kmeans_seeded(lg_kmeans.cluster_centers_, clusters=4, standardize=True)
# cylinder_lg.plot_kmeans(max_clusters=4)

# lg.layers[0].kmeans(4, standardize=True)
# lg.layers[0].plot_tsbm(index=91)
# lg.layers[0].plot_kmeans(max_clusters=4, suffix='nojumps')

# lg.layers[0].kmeans(clusters=4)
# lg.layers[0].plot_kmeans(max_clusters=4, suffix='testcolors')
#
# lg.layers[0].kmeans(clusters=20)
# lg.layers[0].plot_kmeans(max_clusters=20, suffix='clusters20')

# lg.layers[0].plot_tsbm(13966)

# lg.pca(standardize=True)
# lg.plot_pca()

# for i in range(2, 10):
#     lg.kmeans(clusters=i, standardize=True)
#     lg.plot_group_tsbm(suffix=str(i))

# print(lg.group_tsbm.shape)
# k, gapdf = lg.gap_statistic(sample=5000, refs=5, max_clusters=35)
# print('Optimal k: {}'.format(k))
# lg.plot_gap_statistic(gapdf, k)

# labels = ['aa', 'ab', 'ac', 'ad', 'ae',
#           'ba', 'bb', 'bc', 'bd', 'be',
#           'ca', 'cb', 'cc', 'cd', 'ce',
#           'da', 'db', 'dc', 'dd', 'de',
#           'ea', 'eb', 'ec', 'ed', 'ee']
#
# # a = 600
# # b = 995
# # c = 1650
# # d = 3287
#
# vals = [ 0,  600,  995,  1650,  3287,
#         -600,  0,  395,  1050,  2687,
#         -995, -395,  0,  655,  2292,
#         -1650, -1050, -665,  0,  1637,
#         -3287, -2687, -2292, -1637,  0]
#
# # vals = [ 0,  1,  2,  3,  4,
# #         -1,  0,  1,  2,  3,
# #         -2, -1,  0,  1,  2,
# #         -3, -2, -1,  0,  1,
# #         -4, -3, -2, -1,  0]
#
# cmap = plt.cm.rainbow
# cmap.set_bad((1, 1, 1, 1))
# data = np.array(vals).reshape(5, 5).astype(float)

# data[0, 2] = np.nan
# data[0, 3] = np.nan
# data[0, 4] = np.nan
# data[1, 3] = np.nan
# data[1, 4] = np.nan
# data[2, 0] = np.nan
# data[2, 4] = np.nan
# data[3, 0] = np.nan
# data[3, 1] = np.nan
# data[4, 0] = np.nan
# data[4, 1] = np.nan
# data[4, 2] = np.nan

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)
# plt.imshow(data, cmap=cmap)
# plt.colorbar()
# zcbar = plt.colorbar(ticks=[-1, 1])
# cbar.ax.set_yticklabels(['Cooling', 'Heating'], fontsize='large', fontweight='bold')
# cbar = plt.colorbar(ticks=[-1500, -1000, -500, 0, 500, 1000, 1500])
# cbar = plt.colorbar(ticks=[-3000, -2000, -1000, 0, 1000, 2000, 3000])
# cbar.ax.set_yticklabels(['-3000 °C', '-2000 °C', '-1000 °C', '0 °C', '1000 °C', '2000 °C', '3000 °C'], fontsize='large', fontweight='bold')
# # cbar.ax.set_yticklabels(['-1500 °C', '-1000 °C', '-500 °C', '0 °C', '500 °C', '1000 °C', '1500 °C'], fontsize='large', fontweight='bold')
# labels = np.array(labels).reshape(5, 5)
# for (j, i), label in np.ndenumerate(labels):
#     ax.text(i, j, label, ha='center', va='center', fontsize='large', fontweight='bold')
# name = 'Fig2a_full_subword_space' + '.png'
# plt.savefig(fpath + name, bbox_inches='tight', dpi=1200)

# x = np.linspace(0, 0.5, 50)
# y = np.sin(2*np.pi*x*4)/np.exp(x*.5)*.1 + np.sin(2*np.pi*x*4.5)/np.exp(x*.5)*3 + np.sin(2*np.pi*x*4.8)/np.exp(x*1.5)
# y = (np.sin(2 * np.pi * x * 3.0) / np.exp(x * 2.5))**2
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)
# ax.tick_params(width=2, labelsize='large')
# plt.plot(y, marker='o')
# breaks = [0.1, 0.2, 0.4]
# colors = ['g', 'y', 'm']
# plt.hlines(breaks, xmin=0, xmax=len(x), linestyles='dashed', colors=colors)
# plt.text(50, 0.05, 'a', fontsize='large', fontweight='bold', color='g')
# plt.text(50, 0.15, 'b', fontsize='large', fontweight='bold', color='y')
# plt.text(50, 0.30, 'c', fontsize='large', fontweight='bold', color='m')
# plt.text(50, 0.50, 'd', fontsize='large', fontweight='bold', color='r')

# sax = ''
# for t in y:
#     if t < 0.1:
#         sax += 'a'
#     elif t < 0.2:
#         sax += 'b'
#     elif t < 0.4:
#         sax += 'c'
#     else:
#         sax += 'd'
#
# print(sax)
#
# tsbm = []
# aa = len(re.findall('(?=aa)', sax))
# tsbm.append(aa)
# ab = len(re.findall('(?=ab)', sax))
# tsbm.append(ab)
# ac = len(re.findall('(?=ac)', sax))
# tsbm.append(ac)
# ad = len(re.findall('(?=ad)', sax))
# tsbm.append(ad)
# ba = len(re.findall('(?=ba)', sax))
# tsbm.append(ba)
# bb = len(re.findall('(?=bb)', sax))
# tsbm.append(bb)
# bc = len(re.findall('(?=bc)', sax))
# tsbm.append(bc)
# bd = len(re.findall('(?=bd)', sax))
# tsbm.append(bd)
# ca = len(re.findall('(?=ca)', sax))
# tsbm.append(ca)
# cb = len(re.findall('(?=cb)', sax))
# tsbm.append(cb)
# cc = len(re.findall('(?=cc)', sax))
# tsbm.append(cc)
# cd = len(re.findall('(?=cd)', sax))
# tsbm.append(cd)
# da = len(re.findall('(?=da)', sax))
# tsbm.append(da)
# db = len(re.findall('(?=db)', sax))
# tsbm.append(db)
# dc = len(re.findall('(?=dc)', sax))
# tsbm.append(dc)
# dd = len(re.findall('(?=dd)', sax))
# tsbm.append(dd)
#
# print(tsbm)
#
# labels = ['aa', 'ab', 'ac', 'ad',
#           'ba', 'bb', 'bc', 'bd',
#           'ca', 'cb', 'cc', 'cd',
#           'da', 'db', 'dc', 'dd']
# labels = np.array(labels)
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)
# ax.tick_params(width=2, labelsize='large')
# d = np.array(tsbm)
# plt.xticks(range(len(d)), labels)
# plt.yticks(range(0, 21, 2))
# plt.ylabel('Frequency', fontsize='large', fontweight='bold')
# name = 'example_tsbm' + '.png'
# plt.plot(d, linewidth=3, marker='o', linestyle='--')
# plt.savefig(fpath + name, bbox_inches='tight', dpi=1200)

# name = 'sax_schematic.png'
# plt.savefig(fpath + name, bbox_inches='tight', dpi=1200)
