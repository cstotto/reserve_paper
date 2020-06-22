#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as pl 
from scipy.stats import norm
from math import *
import os.path as path
import sys, warnings, inspect
from matplotlib.ticker import MultipleLocator
import pandas as pd
from scipy.stats.kde import gaussian_kde
from sklearn.linear_model import LinearRegression
import argparse
#from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser(
    description='statistical properties of grain reserve')

parser.add_argument(
    '--png', action='store_true',
    help='set output format to png')
args = parser.parse_args()

module_path = [path.abspath('/home/christian/projects/multi_twist')]
for mp in module_path:
     if mp not in sys.path:
          sys.path.append(mp)
from misc.plot_scripts.plot_parameters import *
from misc.postprocessing.post_processing import *

#b_folder = '/home/cotto/projects/multi_twist/output/five_regions/biannual/2018/arti_ts_2010_15'
#runs = np.arange(0,77,7) #width = 19
#runs = np.arange(1,77,7) #width = 20
# runs = np.arange(2,77,7) #width = 21
# #runs = np.arange(6,77,7) #width = 25

#b_folder = '/home/cotto/projects/multi_twist/output/five_regions/biannual/2018/arti_ts_2010_15_low_bw'
# runs = np.arange(0,44,4) #width = 14
#runs = np.arange(1,44,4) #width = 15
# runs = np.arange(2,44,4) #width = 16
#runs = np.arange(3,44,4) #width = 17

b_folder = '/home/cotto/projects/multi_twist/output/five_regions/biannual/2018/arti_ts_2010_15_bw_18'
runs = np.arange(11) #width = 18

labels, thres_price, floor_price, price_ceil, capac, = [], [], [], [], []
ds=[]
for i, num in enumerate(runs):
    print "run: ", num
    ds.append(Dataset(path.join(b_folder,'run%i'%(num),'output.nc'),'r'))
    pars = read_nc_attributes_into_dic(ds[i].groups['parameters'])
    labels.append(int(pars['cap_int_reserve']))
    capac.append(int(pars['cap_int_reserve']))
    thres_price.append(float(pars['threshold_price_mean_years']))
    floor_price.append(float(pars['price_floor']))
    price_ceil.append(float(pars['price_ceiling']))
if len(ds)==1:
   ds=ds[0]

t_per_year, prepend_N_years = read_nc_attributes_into_dic(ds[0].groups['parameters'])['t_per_year'], read_nc_attributes_into_dic(ds[0].groups['parameters'])['prepend_N_years'] 
start_time = t_per_year * prepend_N_years
end_time = -1
P_sim = pd.DataFrame(columns=labels)
I_int = pd.DataFrame(columns=labels)
pars = dict()
for i,fn in enumerate(labels):
    P_sim[fn] = ds[i].variables['P_sim'][start_time:end_time]
    I_int[fn] = ds[i].variables['I_int'][start_time:end_time]
    pars.update({fn: read_nc_attributes_into_dic(ds[i].groups['parameters'])})

# trend correction
P_sim_detr = P_sim.copy()
for name, P in P_sim.iteritems():
    X = [j for j in range(0, len(P))]
    X = np.reshape(X, (len(X), 1))
    model = LinearRegression()
    model.fit(X, P.values)
    # calculate trend
    trend = model.predict(X)
    P_sim_detr[name] = P - (trend-trend[0])
    
#q-percentiles
q = 10.
q_tiles = np.arange(1./q,1.01,1./q)
quantiles_baseline = P_sim_detr[0].quantile(q_tiles)
#len(P_sim_detr[40][P_sim_detr[40] <= quantiles_baseline.values[0]])/len(P_sim_detr[40])*100
bins = np.append(0,quantiles_baseline) # extend first bin down to zero
bins[-1]*=100 # extend last bin to infinity
quantiles = pd.DataFrame(columns=P_sim_detr.columns)
for name, P in P_sim_detr.iteritems():
    binned = pd.cut(P.values,bins=bins).value_counts()/P_sim_detr.shape[0]*100
    quantiles[name] = binned.values
quantile_changes = quantiles.copy()
for i in quantiles.index.values:
    quantile_changes.ix[i,:] = (quantiles.ix[i,:] - quantiles.ix[i,0]) / quantiles.ix[i,0] * 100
print "quantiles baseline: ", quantiles_baseline
# calculate costs
op_costs,fixed_op_costs = [15,36],[5,5]
iterables = [op_costs,['av_op_pa','av_total_pa']]
mIndex = pd.MultiIndex.from_product(iterables,names=['cost','cost_type'])
costs = pd.DataFrame(np.zeros((4,len(runs))),index=mIndex,columns=labels)
for j, op_c in enumerate(op_costs):
    for res_size, I in I_int.iteritems():
        out = calculate_costs(op_c,fixed_op_costs[j],P_sim_detr[res_size],mmt(I),**pars[res_size])
        #slice with partial index
        costs.ix[op_c,res_size] = bn(np.array([out[-2],out[-1]]))

volatility = P_sim_detr.diff().std().values/P_sim_detr.diff().std().ix[0] * 100
res_size = costs.columns.values
min_costs, max_costs = costs.ix[(15,'av_total_pa'),:], costs.ix[(36,'av_total_pa'),:];
marg_cost_per_vol_red_min = min_costs.diff()[1:]/np.abs(np.diff(volatility))
marg_cost_per_vol_red_max = max_costs.diff()[1:]/np.abs(np.diff(volatility))
p_max_red = np.abs((P_sim_detr.max() - P_sim_detr[0].max()) / P_sim_detr[0].max())*100 

cols = pik_color_list(P_sim.shape[1]/2.)
fs, lw = 16,2
fig_params_presentation(widthfrac=1.2,
                        heightfrac=1.7,
                        left=80,
                        right=10,
                        bottom=45,
                        top=10,
                        hspace=10,
                        wspace=300,
                        width= 246.09686*2.,
                        labelsize=fs,
                        fontsize=fs,
                        ticksize=fs,
                        tickdir='out',fig=None)

ax0 = pl.subplot2grid((8,10),(0,0),colspan=10,rowspan=4);
format_ax(ax0,twoAchsisVisible=True,xtickdir='out',ytickdir='in',both_y_achses_visible=False,xlabel=True)
add_caption(ax0,"A",weight='bold',fs=fs+2,loc=1)
ax1 = pl.subplot2grid((8,10),(5,0),rowspan=3,colspan=4);
format_ax(ax1,twoAchsisVisible=True,xtickdir='out',ytickdir='in',both_y_achses_visible=True,xlabel=True)
add_caption(ax1,"B",weight='bold',fs=fs+2,loc=1)
ax1b = ax1.twinx();
format_ax(ax1b,ytickdir='in',y2axis=True)
#ax1b.yaxis.set_tick_params(width=1, direction='in', length=5,pad=5)
ax2 = pl.subplot2grid((8,10),(5,6),rowspan=3,colspan=4)
format_ax(ax2,twoAchsisVisible=True,xtickdir='out',ytickdir='in',both_y_achses_visible=True,xlabel=True)
add_caption(ax2,"C",fs=fs+2,weight='bold',loc=1)
ax2b = ax2.twinx()
format_ax(ax2b,ytickdir='in',y2axis=True)
# remove long-term trends
    
i,j = 0,0
cols= [pik_color('gray',-1)] + cols
ymin,ymax = 1,0
for name, _ in P_sim_detr.iteritems():
     if i % 2 == 0:
          kde_det = gaussian_kde(P_sim_detr[name].as_matrix())
          dist_supp_detr = np.linspace( P_sim_detr[name].min(), P_sim_detr[name].max(), 500 )
          kde = kde_det(dist_supp_detr)
          ymin = np.min(kde) if np.min(kde) < ymin else ymin
          ymax = np.max(kde) if np.max(kde) > ymax else ymax
          ax0.plot(dist_supp_detr, kde,color=cols[j],lw=2,label=name);
          #vertical lines at means
          ys = np.linspace(0,kde_det(P_sim_detr[name].mean()),100)
          xs = np.ones((100))*P_sim_detr[name].mean()
          ax0.plot(xs,ys,dashes=[15,2],color=cols[j],lw=.5);
          ax0.plot([P_sim_detr[name].max()],[0.00],ms=8,mew=0,marker='o',color=cols[j],zorder=100)
          ax0.plot([P_sim_detr[name].min()],[0.00],ms=8,mew=0,marker='D',color=cols[j],zorder=100)
          j+=1
     i+=1
     
     leg = ax0.legend(loc='upper left',frameon=False, markerscale=1., ncol=2, numpoints=1., borderaxespad=.0, handlelength=1.,columnspacing=1.5,title=r'size int. reserve (MMT)');
pos_m_min,pos_m_max = [[105],[0.04]],[[105],[0.035]]      
ax0.plot(pos_m_min[0],pos_m_min[1],marker='D',ms=8,color='k',mew=0)
ax0.text(pos_m_min[0][0]+5,pos_m_min[1][0]-0.001,'min. price')     
ax0.plot(pos_m_max[0],pos_m_max[1],marker='o',ms=8,color='k',mew=0)
ax0.text(pos_m_max[0][0]+5,pos_m_max[1][0]-0.001,'max. price')     
leg.get_frame().set_linewidth(0);
ax0.set_xlabel(r'Price (USD per MT)');
set_ticks(lims([P_sim_detr.min(axis=1),P_sim_detr.max(axis=1)],frac_max = 1./20.,frac_min=1./20.) ,axis='x', major_loc = 20, minor_loc = 10 ,flag_labels=True,ax_=ax0,col='k')

ax0.set_ylabel(r'Probability density function');
set_ticks(lims([ymin,ymax],frac_max = 1./40.,frac_min=1./40.) ,axis='y', major_loc = 0.01, minor_loc = 0.005 ,flag_labels=True,ax_=ax0,Formatter = FormatStrFormatter('%.2f'),col='k')
#ax0.set_ylim([-0.003,0.06])

#shading
kde_no_res = gaussian_kde(P_sim_detr[0].as_matrix())
supp_tail = np.linspace(quantiles_baseline.iloc[-2],quantiles_baseline.iloc[-1],num=500)
ax0.fill_between(supp_tail,kde_no_res(supp_tail),lw=0,color=pik_color('gray',2),alpha=.3,zorder=0)
ax0.axhline(y=0,lw=1, color=pik_color('gray',-1),dashes=[10,5],zorder=0)

# volatility reduction
vol_red = 100 - volatility
ax1.plot(P_sim_detr.columns.values,vol_red,'o-',lw = 2,color = pik_color('orange'),mew=0,ms=7);
ax1.set_xlabel(r'Capacity int. rsv. (MMT)');
set_ticks(lims([np.min(P_sim_detr.columns.values),np.max(P_sim_detr.columns.values)],frac_max = 1./20.,frac_min=1./20.) ,axis='x', major_loc = 20, minor_loc = 10 ,flag_labels=True,ax_=ax1,col='k')
ax1.set_ylabel(r'Volatility reduction (%)',color=pik_color('orange'));
set_ticks(lims([np.min(vol_red),np.max(vol_red)],frac_max = 1./5.,frac_min=1./20.) ,axis='y', major_loc = 5, minor_loc = 2.5 ,flag_labels=True,ax_=ax1,col=pik_color('orange'))



# ax2.plot(P_sim_detr.columns.values,p_max_red,'o-',lw = 2,color = pik_color('orange'),mew=0,ms=7);
# ax2.set_ylabel(r'red. max price / [%]')

ax1b.plot(P_sim_detr.columns.values,quantile_changes.ix[9,:].values,'o-',lw = 2,color = pik_color('gray'),mew=2,ms=7,mfc='white',mec=pik_color('gray'),zorder=0);
ax1b.set_ylabel(r'Red. in prob. for P$\geq93$USD (%)',color=pik_color('gray'))
set_ticks(lims([-100,np.max(quantile_changes.ix[9,:].values)],frac_max = 1./20.,frac_min=1./30.) ,axis='y', major_loc = 20, minor_loc = 10 ,flag_labels=True,ax_=ax1b,col=pik_color('gray'))
ax1b.spines['left'].set_color(pik_color('orange'))
ax1b.spines['right'].set_color(pik_color('gray'))
ax1b.spines['top'].set_visible(False)
#cost plot
ax2.plot(res_size,min_costs,'o',lw = 2,color = pik_color('green'),mew=0,ms=5);
ax2.plot(res_size,max_costs,'o',lw = 2,color = pik_color('green'),mew=0,ms=5);
ax2.fill_between(res_size,min_costs,max_costs,color = pik_color('green'),alpha=.4);
ax2.set_xlabel(r'Capacity int. rsv. (MMT)')
set_ticks(lims([np.min(P_sim_detr.columns.values),np.max(P_sim_detr.columns.values)],frac_max = 1./20.,frac_min=1./20.) ,axis='x', major_loc = 20, minor_loc = 10 ,flag_labels=True,ax_=ax2,col='k')
ax2.set_ylabel(r'Annual costs (bn. USD)',color=pik_color('green'))
set_ticks(lims([np.min(min_costs),np.max(max_costs)],frac_max = 1./7.,frac_min=1./30.) ,axis='y', major_loc = 1, minor_loc = .5 ,flag_labels=True,ax_=ax2,col=pik_color('green'))

ax2b.plot(res_size[1:],marg_cost_per_vol_red_min,'o',lw = 2,color = pik_color('blue'),mew=0,ms=5)
ax2b.plot(res_size[1:],marg_cost_per_vol_red_max,'o',lw = 2,color = pik_color('blue'),mew=0,ms=5)
ax2b.fill_between(res_size[1:],marg_cost_per_vol_red_min,marg_cost_per_vol_red_max,color = pik_color('blue'),alpha=.4)
set_ticks(lims([np.min(marg_cost_per_vol_red_min),np.max(marg_cost_per_vol_red_max)],frac_max = 1./7.,frac_min=1./30.) ,axis='y', major_loc = 2, minor_loc = 1 ,flag_labels=True,ax_=ax2b,col= pik_color('blue'))
ax2b.set_ylabel(r'Marg. an. costs per % vol. red. (bn. USD)',color=pik_color('blue'))

ax2b.spines['left'].set_color(pik_color('green'))
ax2b.spines['right'].set_color(pik_color('blue'))
ax2b.spines['top'].set_visible(False)

if args.png:
     pl.savefig(r'pdf_and_costs2grid.png',bbox_inches='tight',dpi=600)
else:
     pl.savefig(r'pdf_and_costs2grid.pdf',bbox_inches='tight')

