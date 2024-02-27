import matplotlib.pyplot as plt
import numpy as np

labels = ['Internal', 'Internal-p', 'Pyfof', 'Pyfof-p']
npart = np.array([140_000, 2_300_000, 7_100_000, 150_000_000])
times_npart1 = np.array([1.81, 45, 104, np.nan])
times_npart2 = np.array([1.03, 9, 38, np.nan])
times_npart3 = np.array([0.15, 4.1, 13, np.nan])
times_npart4 = np.array([0.75, 3, 9, np.nan])

def on2(npart):
    return npart**2

def on(npart):
    return npart

fig, ax = plt.subplots(dpi = 300)
ax.plot(npart, times_npart1, label = labels[0], linewidth=3, alpha=0.9)
ax.plot(npart, times_npart2, label = labels[1], linewidth=3, alpha=0.9)
ax.plot(npart, times_npart3, label = labels[2], linewidth=3, alpha=0.9)
ax.plot(npart, times_npart4, label = labels[3], linewidth=3, alpha=0.9)
ax.plot(npart, on2(np.array(npart))/90_000**2, label = 'O(n^2)', linewidth=2, linestyle='--', color='black')
ax.plot(npart, on(np.array(npart))/60_000, label = 'O(n)', linewidth=2, linestyle='dotted', color='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(xlabel='Number of particles', ylabel='Time (s)',
       title='PyHALMA (FoF): Scaling Test')
ax.grid()
ax.legend()
plt.show()


fig, ax = plt.subplots(dpi = 300)
ax.plot(npart, times_npart1/times_npart4, label = labels[0], linewidth=3, alpha=0.9)
ax.plot(npart, times_npart2/times_npart4, label = labels[1], linewidth=3, alpha=0.9)
ax.plot(npart, times_npart3/times_npart4, label = labels[2], linewidth=3, alpha=0.9)
ax.hlines(1, npart[0], npart[2], label = labels[3], linewidth=3, alpha = 0.9, color = 'red')
ax.set(xlabel='Number of particles', ylabel='Slower than Pyfof-p',
       title='PyHALMA (FoF): Scaling Test')
ax.grid()

ax.legend()
plt.show()


