import matplotlib.pyplot as plt
import numpy as np

nbins = 36
bins = np.linspace(-np.pi,np.pi,nbins)
# bins = np.linspace(0,2*np.pi,nbins)
print 'bins = {0}'.format(bins)

ubins = np.cos(bins)
print 'ubins = {0}'.format(ubins)

vbins = np.sin(bins)
print 'vbins = {0}'.format(vbins)

print 'norm = {0}'.format(np.linalg.norm([ubins,vbins],axis=0))

print 'len(ubins) = {0}'.format(len(ubins))
print 'len(vbins) = {0}'.format(len(vbins))
    
f,ax = plt.subplots()
C = np.linspace(0,1,len(ubins))
ax.quiver(np.zeros(len(ubins)),
          np.zeros(len(vbins)),
          ubins,
          vbins,
          C,
          cmap=plt.cm.gray,
          units='xy',
          angles='xy',
          scale_units='xy',
          scale=1)


ax.set_xlim(-2,2)
ax.set_ylim(-2,2)


#########################################################################
#EXPERIMENT 2

x = np.array([1,1,1,-1,-1,-1,-1,-1,-2])
y = np.zeros(len(x))

nbins = 36
extent = (-np.pi, np.pi)

weights = np.linalg.norm([x,y],axis=0)
print 'weights = {0}'.format(weights)

counts, bins = np.histogram(np.arctan2(y,x), 
                            bins=nbins, 
                            range = extent, 
                            weights = weights)

normalized_counts = counts/np.sum(counts)
print 'normalized_counts = {0}'.format(normalized_counts)

bar_step = 5
bar_locs = np.arange(0,bar_step*nbins,bar_step)
bins_deg = np.degrees(bins[:-1])
width = bar_step*0.5
print 'bins_deg = {0}'.format(bins_deg)
plt.figure()
plt.bar(bar_locs, normalized_counts, align='center')
plt.xticks(bar_locs+width/2.,bins_deg,rotation=90)

plt.tight_layout()
plt.show()
