# -*- coding: utf-8 -*-
"""
Created on Wednesday February 7th 2018

@author: Enrique Alejandro

This library contains functions to process raw AFM spectrosocpy data.
"""

#from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import splrep, splev, sproot


def xiny(x, y, ret=2, op=all):
    """
    Description: this is an auxiliary function to the loadibw function
    ret=0: return True, False array
    ret=1: return True index array
    ret=2: return True data array
    op=all: and operation
    op=any: or operation
    """
    if type(x)==type(()) or type(x)==type([]) or type(x)==type(np.array(0)):
        if len(x)==0:
            a = np.array( [True]*len(y) )
        else:
            a = np.array( [op([j in i for j in x]) for i in y] )
    else:
        a = np.array([str(x) in i for i in map(str,y)])
    if ret==0: return a
    if ret==1: return np.arange(a.size)[a]
    if ret==2: return np.array(y)[a]

def findstr(src, soi):
    """
    Description: this is an auxiliary function to the loadibw function
    src: string source
    soi: string of interest
    """
    if len(soi)==0:
        return src
    else:
        if type(soi[0])==str:
            return map(str, xiny(soi, src))
        else:
            a = []
            for i in soi:
                a += map(str, xiny(i, src))
            return a
        

def loadibw(f,cal=0):
    """
    x,y,z,ch,note = loadibw(f)    
    """
    datatype = {
        0:np.dtype('a1'),
        2:np.float32,
        4:np.float64,
        8:np.int8,
        16:np.int16,
        32:np.int32,
        72:np.uint8,
        80:np.uint16,
        96:np.uint32
    }
    # igor header
    fp = open(f,'rb');
    fp.seek(  0); ih  = []
    fp.seek(  0); ih += [('version', np.fromfile(fp,np.int16,1)[0])]
    fp.seek(  8); ih += [('fmsize' , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 12); ih += [('ntsize' , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 68); ih += [('cdate'  , np.fromfile(fp,np.uint32,1)[0])]
    fp.seek( 72); ih += [('mdate'  , np.fromfile(fp,np.uint32,1)[0])]
    fp.seek( 76); ih += [('dsize'  , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 80); ih += [('dtype'  , np.fromfile(fp,np.uint16,1)[0])]
    fp.seek(132); ih += [('shape'  , np.fromfile(fp,np.int32,4))]
    fp.seek(132); ih += [('ndim'   , (ih[-1][1]>0).sum())]
    fp.seek(148); ih += [('sfa'    , np.fromfile(fp,np.float64,4))]
    fp.seek(180); ih += [('sfb'    , np.fromfile(fp,np.float64,4))]
    fp.seek(212); ih += [('dunit'  , np.fromfile(fp,np.dtype('a4'),1)[0])]
    fp.seek(216); ih += [('dimunit', np.fromfile(fp,np.dtype('a4'),4))]
    ih  = dict(ih)
    ih['shape'] = ih['shape'][:ih['ndim']][::-1]
    # images data
    fp.seek(384); z = np.fromfile(fp,datatype[ih['dtype']],ih['dsize']).reshape(ih['shape'])
    fp.seek(ih['fmsize'],1)
    ah = np.fromfile(fp,np.dtype('a%d'%ih['ntsize']),1)[0].split('\r')
    # asylum note
    note = []
    for i in ah:
        if i.find(':')>0:
            j = i.split(':',1)
            try:
                note += [(j[0],float(j[1]))]
            except:
                note += [(j[0],j[1].replace(' ','',1))]
    note = dict(note)
    if type(note['ScanRate']) == type(''):
        note['ScanRate'] = float(note['ScanRate'].split('@')[0])
    # channel & type
    fp.seek(-10,2)
    ch = fp.read()
    fp.seek(-int(ch[:4]),2)
    if ch[-5:] == 'MFP3D':
        ch = findstr(fp.readline().split(';'),'List')[0].split(':')[1].split(',')
        x = np.linspace(-note['FastScanSize']/2.,note['FastScanSize']/2.,note['ScanPoints']) + note['XOffset']
        y = np.linspace(-note['SlowScanSize']/2.,note['SlowScanSize']/2.,note['ScanLines']) + note['YOffset']
    elif ch[-5:] == 'Force':
        ch = findstr(fp.readline().split(';'),'Types')[0].split(':')[1].split(',')[:-1]
        ch.insert(0,ch.pop())
        x = y = []
    else:
        ch = []
        x = y = []
    if ch[-1] == '':
        ch = ch[:-1]
    fp.close()
    
    if cal and (x<>[]):
        x -= x[0]
        y -= y[0]
        x *= 1e6
        y *= 1e6
        for i,j in enumerate(ch):
            if ('Height' in j)or('ZSensor' in j)or('Amplitude' in j)or('Current' in j):
                z[i] *= 1e9
    
    return x,y,z,ch,note


   
def offsety(defl):
    """This function receives a deflection and applies offset in y direction"""
    """Receives tip and deflection in meters"""
    defl_at = defl[:defl.argmin()-100]
    offset = np.mean(defl_at)
    defl -= offset
    return defl
    
def offsetx(defl, zs):
    """This function receives a deflection and applies offset in x direction"""
    """Receives tip and deflection in meters"""
    b,a = signal.butter(10, 0.1, "low")
    df_smooth = signal.filtfilt(b,a,defl)
    zero = df_smooth.argmin()
    zs = zs - zs[zero] + defl[zero]
    return defl, zs, zero

def average_FDcurves(files, k = 1, invols = 1, max_indent = 1):
    #GETTING NAME OF FOLDER WHERE FILES ARE LOCATED
    #fn = os.path.basename(os.path.normpath(os.getcwd()))
    
    #Defining new axes for the averaging of the FD curves
    Z = np.linspace(-75.0e-9,40.0e-9,1000) #New z axis
    D = np.zeros(len(Z)) #Deflection to be corrected
    T = np.zeros(len(Z)) #Time
    N = 0 #counter for the number of FD curves
        
    k_a = []   #array containing stiffness of cantilever on each force spectroscopy
    inv_a = []  #array containing values of invols for each FD curve
    
    for f in files:
        x,y,z,ch,note = loadibw(f) 
        if k == 1:
            k = note['SpringConstant']  
        inv = note['Invols']
        if invols ==1:   #no correction needed
            correction = 1.0
        else: #correction needed due to late calibration
            correction = invols/inv
        defl = z[ch.index('Defl')]
        defl = defl*correction
        zs = z[ch.index('ZSnsr')]
        fs = note['NumPtsPerSec']    #sampling frequency
        t = np.arange(len(defl))/fs   #retrieving time array with the aid of sampling frequency
            
        #attaching values to lists containing stiffnesses and invols values
        k_a.append(k)
        inv_a.append(inv)
        
        #GETTING ONLY APPROACH PORTION
        maxi = zs.argmax()
        zs = zs[:maxi]
        defl = defl[:maxi]
        t = t[:maxi]
        
        #APPLYING OFFSETS TO MAKE CURVES FALL ONE OVER EACH OTHER     
        defl_oy = offsety(defl)  
        defl, zs, zero = offsetx(defl_oy, zs)
                
        plt.figure(1)
        plt.plot(zs, defl)
        plt.xlabel('Z_sensor, nm', fontsize=10)
        plt.ylabel('Deflection, nm', fontsize=10)
        plt.xlim(-zs[len(zs)-1], zs[len(zs)-1])
        
        #interpolation of deflection with respect to new axis zs
        tck = splrep(zs,defl,k=1) #this returns a tuple with the vector of knots, the B-spline coefficients, and the degree of the spline
        #at this point occurs the reduction of number of points, from original size to the size of Z (default 1000 points)
        df = splev(Z,tck) #this returns an array of values representing the spline function evaluated at the points in x 
        D += df
        tck = splrep(zs,t,k=1)  #interpolation of time with respect to new axis zs
        t = splev(Z,tck) #at this point occurs the reduction of number of points, from original size to the size of Z (default 1000 points)
        T += t
        N += 1   #this counter is useful for the final averaging, this counts how many FD curves are being averaged.
        
    # Average
    D /= N
    T /= N
    
    plt.figure(2)
    plt.plot(Z-D,D)
    plt.xlabel('Averaged Tip Position, nm', fontsize=10)
    plt.ylabel('Averaged Deflection, nm', fontsize=10)
    plt.xlim( - ( Z[len(Z) -1] - D[len(Z) - 1]  ),  (Z[len(Z) -1] - D[len(Z) - 1]) )
    
    plt.figure(3)
    plt.plot(Z,D)
    plt.xlabel('Averaged z-sensor, nm', fontsize=10)
    plt.ylabel('Averaged Deflection, nm', fontsize=10)
    
    return T, Z, D, Z-D, k_a, inv_a
    

def repulsiveFD(t,zs,defl):
    """This function takes the average FD curve obtained through averageFD 
    function and gives back the repulsive portion of the curve"""
    offset_pos = np.argmin(defl)
    N = len( defl[offset_pos:] )
    delta_t = t[2] - t[1]
    delta_zs = zs[2] -zs[1]
    t_rep = np.linspace(0,(N-1)*delta_t,N)
    z_rep = np.linspace(0,(N-1)*delta_zs,N)
    defl_rep = defl[offset_pos:] - defl[offset_pos]
    tip_rep = z_rep - defl_rep
    tip_r = tip_rep[tip_rep > 0]
    defl_r = defl_rep[tip_rep > 0]
    t_r = t_rep[tip_rep > 0]
    z_r = z_rep[tip_rep > 0]
    return t_r, z_r, defl_r, tip_r     
    
    
