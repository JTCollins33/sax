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
import math


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


#this reads thermal histories from hdf5 file, creating a list of TH objects (layer)
def get_ths(hdf5_file, figs = None):
    file = h5py.File(hdf5_file, 'r')
    ths = []

    print("\nReading thermal histories of "+str(hdf5_file)+"...")
    #read data from file and put in thermal histories list
    for n,d in file['/ThermalHistories'].items():
        ths.append(ThermalHistory(np.array(d.value), int(n), figs))

    print("Done reading "+str(hdf5_file)+"\n")

    file.close()
    return ths

def get_layer_features(ths, figs, hdf5_file, index):
    #get dream3d file to read positions
    if (hdf5_file.find('_')!=-1):
        strs = hdf5_file.split('_')
        period = strs[1].split('.')
        d3d_str = strs[0]+period[0]+".dream3d"
    else:
        d3d_str = hdf5_file[0:len(hdf5_file)-5]+".dream3d"

    d3d_file = h5py.File(d3d_str, 'r')


    if (hdf5_file.find('_')!=-1):
        points = d3d_file[SYNTH_POINTS].value
        dist = d3d_file[SYNTH_DIST].value
        vector = d3d_file[SYNTH_VECTOR].value
        tsbm = d3d_file[SYNTH_TSBM].value
    else:
        points = d3d_file[REAL_POINTS].value
        dist = d3d_file[REAL_DIST].value
        vector = d3d_file[REAL_VECTOR].value
        tsbm = d3d_file[REAL_TSBM].value

    layer = Layer(ths, points, index=index, figs=fpath)
    layer.distance = dist
    layer.vector = vector

    #set decision tree features
    layer.get_dt_features()

    d3d_file.close()

    return layer

#this method finds the index of the current file
def find_index(file):
    return int(re.findall(r'\d+', file.split('.')[0])[0])

if __name__ == "__main__":
    dir_path = "./data/"
    fpath = "./data/featurization_figs/decision_trees/num_peaks_peak_dist_and_max/"
    layers = []
    cnt = 0
    for file in os.listdir(dir_path):
        layers=[]
        if file.endswith(".hdf5") and not file.endswith("_2.hdf5"):
            ths = get_ths(dir_path+file)
            index = find_index(file)

            layer = get_layer_features(ths, fpath, dir_path+file, index)

            #perform agglomerative clustering
            layer.agglomerative_clustering()

            layer.plot_dt(max_clusters=4)

            cnt+=1
