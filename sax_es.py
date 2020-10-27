# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:02:01 2017

@author: schwalej
"""
import itertools
import numpy as np
import scipy



def bin_signal(signal, bin_edges):
    """
    return of same length as incoming (floating point) signal
    """
    n_bins = len(bin_edges) + 1
    binned_signal = np.ones(signal.shape, dtype=np.int)*(n_bins-1)
    
    for i in range(n_bins-1):
        if i == 0:
            binned_signal[ signal < bin_edges[i]] = i
        if i > 0:
            binned_signal[ (signal<bin_edges[i])*(signal>=bin_edges[i-1])] = i
    return binned_signal
    



def equal_bins(signal, n_bins):
    """
    Compute bin edges from the data set.  bin edges chosen to have equal 
    fractions of the data in each bin.
    """
    sordid = np.sort(signal)
    di = int(np.floor(float(len(signal))/n_bins))
    bin_edges = []
    for i in range(n_bins-1):
        bin_edges.append(sordid[(i+1)*di])
    bin_edges = np.array(bin_edges)
    return bin_edges
    


def normal_bins(signal, n_bins):
    """
    Compute bin edges for roughly equal counts in each bin, assuming the signal 
    is normally distributed.
    """
    frac_per_bin = 1./float(n_bins)
    cumulative_fracs = frac_per_bin * (np.arange(n_bins-1)+1.)    #number of bin edges is 1 less than n_bins
    
    sig_mean = np.mean(signal)
    sig_std = np.std(signal)    
    
    #quantile function for a normal distribution:
    bin_edges = sig_mean + sig_std*np.sqrt(2)*scipy.special.erf(2.*cumulative_fracs - 1)
    return bin_edges
    

def pairwise_counts(signal, alphabet_size):
    """
    count all occurances of the pairs of each element of the alphabet
    """    
    pair_counts = np.zeros((alphabet_size, alphabet_size), dtype=int)
        
    #this does explicitly loop through the array, but seems fast enough for now.
    for i in range(len(signal)-1):
        pair_counts[ signal[i], signal[i+1]  ] += 1    
    
    #generate an array containing labels for the pair counts
    pair_count_labels = np.zeros(alphabet_size**2, dtype=np.int)
    lbl_str = ""
    for i in range(alphabet_size):
        lbl_str = lbl_str + str(i)
    for i, pair in enumerate(itertools.product(lbl_str, repeat=2)):
        pair_count_labels[i] = "".join(pair)
    
    pair_count_labels = pair_count_labels.reshape((alphabet_size,alphabet_size))
    
    return pair_counts, pair_count_labels