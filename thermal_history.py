import matplotlib.pyplot as plt
import numpy as np
import re
import pywt

#this class is about converting thermal curves into alphabetical strings and converting these strings into
#arrays with counts of each transition

class ThermalHistory:
    def __init__(self, curve, index=None, position=None, figs=None):
        self.curve = curve
        self.index = index
        self.position = position
        self.sax = None
        #tsbm is array containing count of letter transitions/thermal temperature transitions
        self.tsbm = None
        self.wavelets = None
        self.figs = figs

    #shrink curve size to minimum size
    def reduce_size_to_min(self, min):
        self.curve = self.curve[0:min]    


    #this computes the wavelet transform for the given thermal history curve
    def compute_wt(self):
        self.wavelets, freqs = pywt.dwt(self.curve, 'db1')

    #this shrinks an array to length min and centers it around its max value
    def shrink_around_max(self, min):
        max_index = 0
        max_val = float(self.wavelets[0])
        for i in range(len(self.wavelets)):
            # if((isinstance(self.wavelets[i], int) or isinstance(self.wavelets[i], float)) and float(self.wavelets[i])>max_val):
            if(float(self.wavelets[i])>max_val):
                max_index = i
                max_val = float(self.wavelets[i])
        
        side_length = int(min/2)
        rem = min%2

        #get start and end indices for new array
        if(rem==1):
            start = max_index-side_length
        elif(rem==0):
            start = max_index-side_length+1
        end = max_index+side_length

        #make sure start and end indices are valid indices in array
        while(start<0 or end>=len(self.wavelets)):
            if(start<0):
                start+=1
                end+=1
            if(end>=len(self.wavelets)):
                start-=1
                end-=1

        return self.wavelets[start:end+1]

    #this centers the array around the max value
    #and makes it size length, regardless of original size
    def center_around_max(self, length):
        max_index = 0
        max_val = float(self.wavelets[0])
        for i in range(len(self.wavelets)):
            if((isinstance(self.wavelets[i], int) or isinstance(self.wavelets[i], float)) and float(self.wavelets[i])>max_val):
            # if(float(self.wavelets[i])>max_val):
                max_index = i
                max_val = float(self.wavelets[i])
        
        side_length = int(length/2)
        rem = length%2

        #get start and end indices for new array
        if(rem==1):
            start = max_index-side_length
        elif(rem==0):
            start = max_index-side_length+1
        end = max_index+side_length

        new_arr = [0]*length

        #fill new array
        for i in range(side_length):
            if(start>=0):
                new_arr[i]=self.wavelets[start]
            start+=1

        new_arr[side_length+rem] = max_val

        for j in range(side_length):
            if(end<len(self.wavelets)):
                new_arr[len(new_arr)-j-1] = self.wavelets[end]
            end-=1
        
        return new_arr        

    def fix_wt_length(self, max_size):
        add_cols = max_size-len(self.wavelets)
        self.wavelets = self.wavelets.tolist()
        self.wavelets.append([0]*add_cols)
        self.wavelets = np.array(self.wavelets)

    #this computes the selected features of the ths curves and returns them
    #FEATURES: max value and # points above specified value
    def get_dt_features(self, threshold=500.0):
        #find max value
        max_val = max(self.curve)

        # #find number of points above certain threshold
        # count = sum(i>threshold for i in self.curve)

        n_peaks = 0
        peaks = []
        #find number of peaks in dataset
        for j in range(len(self.curve)):
            if j>0 and j< (len(self.curve)-1):
                if (self.curve[j] > (self.curve[j-1]+20)) and (self.curve[j] > (self.curve[j+1]+20)):
                    n_peaks += 1
                    peaks.append(j)

        peak_distance = 0
        #find max distance between peaks
        if len(peaks)>1:
            for i in range(len(peaks)-1):
                if((peaks[i+1]-peaks[i]) > peak_distance):
                    peak_distance = (peaks[i+1]-peaks[i])

        # return max_val, count, n_peaks, peak_distance
        return max_val, n_peaks, peak_distance



    #this translates thermal temperatures into leters a-e
    def compute_sax(self):
        self.sax = ''
        for t in self.curve:
            if t < 600:
                self.sax += 'a'
            elif t < 995:
                self.sax += 'b'
            elif t < 1650:
                self.sax += 'c'
            elif t < 3287:
                self.sax += 'd'
            else:
                self.sax += 'e'

    #this converts string of a-e letters into an array containing the count of each letter transition
    def compute_tsbm(self):
        if self.sax is None:
            self.compute_sax()
        self.tsbm = [0] * 25

        aa = len(re.findall('(?=aa)', self.sax))
        self.tsbm[0] = aa

        ab = len(re.findall('(?=ab)', self.sax))
        self.tsbm[1] = ab

        ac = len(re.findall('(?=ac)', self.sax))
        self.tsbm[2] = ac

        ad = len(re.findall('(?=ad)', self.sax))
        self.tsbm[3] = ad

        ae = len(re.findall('(?=ae)', self.sax))
        self.tsbm[4] = ae

        ba = len(re.findall('(?=ba)', self.sax))
        self.tsbm[5] = ba

        bb = len(re.findall('(?=bb)', self.sax))
        self.tsbm[6] = bb

        bc = len(re.findall('(?=bc)', self.sax))
        self.tsbm[7] = bc

        bd = len(re.findall('(?=bd)', self.sax))
        self.tsbm[8] = bd

        be = len(re.findall('(?=be)', self.sax))
        self.tsbm[9] = be

        ca = len(re.findall('(?=ca)', self.sax))
        self.tsbm[10] = ca

        cb = len(re.findall('(?=cb)', self.sax))
        self.tsbm[11] = cb

        cc = len(re.findall('(?=cc)', self.sax))
        self.tsbm[12] = cc

        cd = len(re.findall('(?=cd)', self.sax))
        self.tsbm[13] = cd

        ce = len(re.findall('(?=ce)', self.sax))
        self.tsbm[14] = ce

        da = len(re.findall('(?=da)', self.sax))
        self.tsbm[15] = da

        db = len(re.findall('(?=db)', self.sax))
        self.tsbm[16] = db

        dc = len(re.findall('(?=dc)', self.sax))
        self.tsbm[17] = dc

        dd = len(re.findall('(?=dd)', self.sax))
        self.tsbm[18] = dd

        de = len(re.findall('(?=de)', self.sax))
        self.tsbm[19] = de

        ea = len(re.findall('(?=ea)', self.sax))
        self.tsbm[20] = ea

        eb = len(re.findall('(?=eb)', self.sax))
        self.tsbm[21] = eb

        ec = len(re.findall('(?=ec)', self.sax))
        self.tsbm[22] = ec

        ed = len(re.findall('(?=ed)', self.sax))
        self.tsbm[23] = ed

        ee = len(re.findall('(?=ee)', self.sax))
        self.tsbm[24] = ee

    #this updates the count array with all the new set of transitions (with all intermediate transitions)
    def compute_tsbm_ivt(self):
        if self.sax is None:
            self.compute_sax()
        self.tsbm = [0] * 25

        aa = len(re.findall('(?=aa)', self.sax))
        self.tsbm[0] += aa

        ab = len(re.findall('(?=ab)', self.sax))
        self.tsbm[1] += ab

        ac = len(re.findall('(?=ac)', self.sax))
        self.tsbm[1] += ac
        self.tsbm[7] += ac

        ad = len(re.findall('(?=ad)', self.sax))
        self.tsbm[1] += ad
        self.tsbm[7] += ad
        self.tsbm[13] += ad

        ae = len(re.findall('(?=ae)', self.sax))
        self.tsbm[1] += ae
        self.tsbm[7] += ae
        self.tsbm[13] += ae
        self.tsbm[19] += ae

        ba = len(re.findall('(?=ba)', self.sax))
        self.tsbm[5] += ba

        bb = len(re.findall('(?=bb)', self.sax))
        self.tsbm[6] += bb

        bc = len(re.findall('(?=bc)', self.sax))
        self.tsbm[7] += bc

        bd = len(re.findall('(?=bd)', self.sax))
        self.tsbm[7] += bd
        self.tsbm[13] += bd

        be = len(re.findall('(?=be)', self.sax))
        self.tsbm[7] += be
        self.tsbm[13] += be
        self.tsbm[19] += be

        ca = len(re.findall('(?=ca)', self.sax))
        self.tsbm[10] += ca

        cb = len(re.findall('(?=cb)', self.sax))
        self.tsbm[11] += cb

        cc = len(re.findall('(?=cc)', self.sax))
        self.tsbm[12] += cc

        cd = len(re.findall('(?=cd)', self.sax))
        self.tsbm[13] += cd

        ce = len(re.findall('(?=ce)', self.sax))
        self.tsbm[13] += ce
        self.tsbm[19] += ce

        da = len(re.findall('(?=da)', self.sax))
        self.tsbm[15] += da

        db = len(re.findall('(?=db)', self.sax))
        self.tsbm[16] += db

        dc = len(re.findall('(?=dc)', self.sax))
        self.tsbm[17] += dc

        dd = len(re.findall('(?=dd)', self.sax))
        self.tsbm[18] += dd

        de = len(re.findall('(?=de)', self.sax))
        self.tsbm[19] += de

        ea = len(re.findall('(?=ea)', self.sax))
        self.tsbm[20] += ea

        eb = len(re.findall('(?=eb)', self.sax))
        self.tsbm[21] += eb

        ec = len(re.findall('(?=ec)', self.sax))
        self.tsbm[22] += ec

        ed = len(re.findall('(?=ed)', self.sax))
        self.tsbm[23] += ed

        ee = len(re.findall('(?=ee)', self.sax))
        self.tsbm[24] += ee

    def plot_thermal_curve(self):
        time_res = 0.0001
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        breaks = [600, 995, 1650, 3287]
        colors = ['g', 'y', 'm', 'c']
        plt.hlines(breaks, xmin=0, xmax=len(self.curve) * time_res, linestyles='dashed', colors=colors)
        plt.ylabel('Temperature ($^\circ$C)', fontsize='large', fontweight='bold')
        plt.xlabel('Time (s)', fontsize='large', fontweight='bold')
        x = []
        for i in range(len(self.curve)):
            x.append(i * time_res)
        plt.plot(x, self.curve, c='r', linewidth=3)
        name = 'thermal_curve_' + str(self.index) + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)
