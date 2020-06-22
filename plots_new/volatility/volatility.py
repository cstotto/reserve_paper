#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as pl 
from scipy.stats import norm
from math import *
import os.path as path
import sys, warnings
import pandas as pd
import argparse

from matplotlib.ticker import MultipleLocator
module_path = [path.abspath('/home/christian/projects/multi_twist')]
for mp in module_path:
     if mp not in sys.path:
          sys.path.append(mp)
          
from misc.plot_scripts.plot_parameters import *
from includes.convenience_functions import *

parser = argparse.ArgumentParser(
    description='compare volatility of different scenarios')
parser.add_argument(
'--basefolder', type=str, default='/home/cotto/projects/multi_twist/output/five_regions/biannual/2018/',
    help='comma separated list of basefolders')
parser.add_argument(
     '--folders', type=str, default='trends_only/main,baseline/main,full/main/baseline,full/main/reserve',
     help='comma separted list of path to output file')
parser.add_argument(
     '--column_labels', type=str, default='only_trends,baseline,policies,with_int_reserve',
    help='comma separted list of column labels')
parser.add_argument(
     '--start_year', type=int, default='1980',
    help='start year of comparison')
parser.add_argument(
     '--end_year', type=int, default='2017',
     help='end (agricultural) year for comparison')

parser.add_argument(
    '--png', action='store_true',
    help='set output format to png')

args = parser.parse_args()

# data to determine explained volatility
basefolder = args.basefolder
folders = args.folders.split(',')
column_labels = args.column_labels.split(',')
ds=[]
for _, filename in enumerate(folders):
    print  path.join(basefolder,filename,'output.nc')
    ds.append(Dataset(path.join(basefolder,filename,'output.nc'),'r'))
if len(ds)==1:
   ds=ds[0]

p_dic=parse_vec_in_dic(ds[0].variables['prod_names'][:],reversed=True)
#parameters
pars = read_nc_attributes_into_dic(ds[0].groups['parameters'])

start_ind_ = np.where(ds[0].variables['years'][:] > args.start_year)[0][0]-1
end_ind_ = np.where(ds[0].variables['years'][:] <= args.end_year)[0][-1] + 1
times = netCDF4.num2date(ds[0].variables['time'][start_ind_:end_ind_],ds[0].variables['time'].units)
P_def  =  ds[0].variables['P_def'][start_ind_:end_ind_]
Ps = pd.DataFrame(P_def,index=times,columns=['P_rep'])

for i, d in enumerate(ds):
     start_ind = np.where(d.variables['years'][:] > args.start_year)[0][0]-1
     end_ind = np.where(d.variables['years'][:] <= args.end_year)[0][-1]+1
     print start_ind, end_ind, column_labels[i], Ps.index.values.shape,d.variables['P_sim'][start_ind:end_ind].shape[0]
     #Ps[column_labels[i]] = d.variables['P_sim'][start_ind:end_ind]
     Ps[column_labels[i]] = d.variables['P_sim'][start_ind:]
#exp_vol = per_cent(Ps.diff().corr().ix['P_rep'].as_matrix()[1:])
    
#exp_vol = per_cent(Ps.corr().ix['P_rep'].as_matrix()[1:])
#exp_vol = per_cent(Ps.diff().corr().ix['P_rep'].as_matrix()[1:]**2)
exp_vol = per_cent(Ps.corr().ix['P_rep'].as_matrix()[1:]**2)


# bar plot with explained volatility
# fig, ax = pl.subplots();
# fig_params_presentation(widthfrac=1.,
#                         heightfrac=.8,
#                         left=80,
#                         right=10,
#                         bottom=45,
#                         top=10,
#                         hspace=60,
#                         wspace=90,
#                         width= 600,#418.25368,
#                         labelsize=14,
#                         fontsize=14,tickdir='out',fig=1)

# y_pos_bars = np.arange(len(column_labels))
# ax.barh(y_pos_bars,exp_vol,xerr=None,align='center',color=pik_cols['blue'],ecolor=None); # 
# ax.set_yticks(y_pos_bars);
# ax.set_yticklabels(column_labels);
# ax.invert_yaxis();  # labels read top-to-bottom
# ax.set_xlabel(r'explained price volatility (detrended) / [$\%$]');
# pl.savefig('exp_price_volatility_bars.pdf');

# stackplot of explained volatility

diffs_exp_vol = np.append(exp_vol[0],np.diff(exp_vol))
ps_diff_exp_vol = np.cumsum(diffs_exp_vol)

# fig, ax = pl.subplots();
# fig_params_presentation(widthfrac=1.,
#                         heightfrac=.8,
#                         left=50,
#                         right=120,
#                         bottom=10,
#                         top=30,
#                         hspace=60,
#                         wspace=90,
#                         width= 600,#418.25368,
#                         labelsize=14,
#                         fontsize=14,tickdir='out',fig=1)
# c_list = pik_color_list(len(ds)-1);
# x_pos_bars = [0];
# width = .5
# rects=[]
# rects.append(ax.bar(x_pos_bars,diffs_exp_vol[0],xerr=None,align='center',color=c_list[0],ecolor=None,width=.5));
# for i in np.arange(1,len(ds)-1):
#     rects.append(ax.bar(x_pos_bars,diffs_exp_vol[i],bottom = ps_diff_exp_vol[i-1],xerr=None,align='center',color=c_list[i],ecolor=None,width=.5));

# ax.set_xticks([]);
# ax.set_xlim([x_pos_bars[0]-.6*width,x_pos_bars[-1]+.6*width]);
# #ax.set_xticklabels(['full']);
# ax.invert_xaxis();  # labels read top-to-bottom
# ax.set_ylabel(r'explained volatility / [$\%$]');

# leg = ax.legend(rects[::-1],column_labels[::-1],loc=(.95,.7),ncol=1)
# pl.savefig('explained_price_volatility_stack.pdf')

#explained volatilites and volatilites
# Ps_ext = Ps.copy()
# Ps_ext[['P_rep','policies','with_int_reserve','baseline','only_trends']]

##########
# two plots total and explained volatility
##########
# fig_params_presentation(widthfrac=1.,
#                         heightfrac=.9,
#                         left=40,
#                         right=0,
#                         bottom=95,
#                         top=20,
#                         hspace=60,
#                         wspace=70,
#                         width= 418.25368,
#                         labelsize=11,
#                         fontsize=11,tickdir='out',fig=None)


# fig, ax = pl.subplots(1,2);

# length = len(column_labels)

# format_ax(ax[0],twoAchsisVisible=True,xtickdir='out',both_y_achses_visible=False,xlabel=True)
# add_caption(ax[0],"(a)",fs=12)
# x_pos_bars = np.arange(length-1)
# ax[0].bar(x_pos_bars,exp_vol[:-1],xerr=None,align='center',color=pik_cols['blue'],ecolor=None,width=.5);
# ax[0].set_xticks(x_pos_bars);
# ax[0].set_xticklabels([r"supply & demand""\n""trends only",r"supply & demand""\n""variability",r"w/ policies"],rotation=90);
# #ax.invert_yaxis();  # labels read top-to-bottom
# ax[0].set_ylabel(r'explained price volatility / [$\%$]');

# print exp_vol[:-1]

# format_ax(ax[1],twoAchsisVisible=True,xtickdir='out',both_y_achses_visible=False,xlabel=True)
# add_caption(ax[1],"(b)",fs=12)
# x1_pos_bars = np.arange(Ps_ext.columns.values.shape[0]-1);
# ax[1].bar(x1_pos_bars,per_cent(Ps_ext[['P_rep','baseline','with_int_reserve','only_trends']].std()/Ps_ext['policies'].std()),xerr=None,align='center',color=pik_cols['orange'],width=.6);
# ax[1].set_xticks(x1_pos_bars);
# ax[1].set_xticklabels([r'reported',r"supply & demand""\n""variability",r"w/ policies""\n""& int. res.",r"supply & demand""\n""trends only"],rotation=90);
# ax[1].set_ylabel(r'volatility / [$\%$ volatility w/ policies]')
# ax[1].set_ylim(0,115)
# ax[1].set_yticks(np.arange(0,150,10),minor=True)

##########
# single plot explained volatility
##########

fig_params_presentation(widthfrac=1.,
                        heightfrac=1.6,
                        left=40,
                        right=0,
                        bottom=90,
                        top=20,
                        hspace=60,
                        wspace=70,
                        width= 246.09686,
                        labelsize=9,
                        ticksize=9,
                        fontsize=11,tickdir='out',fig=None)

fig, ax = pl.subplots();
length = len(column_labels)

format_ax(ax,twoAchsisVisible=True,xtickdir='out',both_y_achses_visible=False,xlabel=True)
#add_caption(ax[0],"(a)",fs=12)
x_pos_bars = np.arange(length-1)
ax.bar(x_pos_bars,exp_vol[:-1],xerr=None,align='center',color=pik_cols['blue'],alpha=.8,lw=0,width=.7);
ax.set_xticks([]);
ax.set_xticklabels([])
#ax.set_xticklabels([r"supply & demand""\n""trends",r"supply & demand""\n""variability",r"w/ policies"],rotation=90);
labels = [r"supply & demand""\n""trends",r"supply & demand""\n""variability",r"w/ policies"]
ylev = 10
for i,x in enumerate(x_pos_bars):
     ax.text(x,ylev,labels[i],rotation=90,verticalalignment='bottom',horizontalalignment='center')

#ax.invert_yaxis();  # labels read top-to-bottom
ax.set_ylabel(r'explained price volatility ($\%$)');
ax.set_ylim(0,100)
print exp_vol[:-1]
if args.png:
     pl.savefig('exp_vol_detr.png',bbox_inches='tight',dpi=600)
else:
     pl.savefig('exp_vol_detr.pdf',bbox_inches='tight')                                                                                                                

