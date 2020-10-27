import matplotlib.pyplot as plt
import numpy as np
import re

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
        self.figs = figs

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
