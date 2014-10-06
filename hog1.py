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

#########################################################################
#EXPERIMENT 3

#Draw some random vectors

#put the dominant wind direction in on [0,15] and add some uniform noise
#distributed around the unit circle.
randtheta = np.hstack(
    (np.radians(np.random.normal(loc=15./2., scale=1., size=40)),
     np.radians(np.random.uniform(0,360,10))))
mags = np.hstack((np.random.uniform(4,6,40),
                  np.random.uniform(1,2,10)))

X = np.zeros(len(randtheta))
Y = X
U = mags*np.cos(randtheta)
V = mags*np.sin(randtheta)
C = mags/np.max(mags)
#plot the data
plt.figure()
plt.quiver(X,Y,U,V,C,
           units='xy',
           angles='xy',
           scale_units='xy',
           scale=1,
           cmap=plt.cm.cool)

plt.xlim(-7,7)
plt.ylim(-7,7)

unwrapped_theta = np.arctan2(V,U)#will put theta on -pi to pi
counts, bins = np.histogram(unwrapped_theta,bins=nbins,range=extent)
print 'counts = {0}'.format(counts)
print 'bins = {0}'.format(bins)
nonzero_idx = np.nonzero(counts)
counts_nonzero = (counts[nonzero_idx]).astype(float)/np.max(counts[nonzero_idx])
print 'counts_nonzero = {0}'.format(counts_nonzero)
C = counts_nonzero
bin_centers = (bins[:-1]+bins[1:])/2.
print 'bin_centers (rads) = {0}'.format(bin_centers)
print 'bin_centers (deg) = {0}'.format(np.degrees(bin_centers))
print 'bins_nonzero (rads) = {0}'.format(bins[nonzero_idx])
print 'bins_nonzero (degs) = {0}'.format(np.degrees(bins[nonzero_idx]))

Ubins = counts_nonzero*np.cos(bin_centers[nonzero_idx])
Vbins = counts_nonzero*np.sin(bin_centers[nonzero_idx])
print 'Ubins = {0}'.format(Ubins)
print 'len(Ubins) = {0}'.format(len(Ubins))
print 'Vbins = {0}'.format(Vbins)
X = np.zeros(len(Ubins))
Y = X

fig, ax = plt.subplots()
ax.quiver(X,Y,Ubins,Vbins,C,
           cmap=plt.cm.coolwarm,
           units='xy',
           angles='xy',
           scale_units='xy',
           scale=1)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)





