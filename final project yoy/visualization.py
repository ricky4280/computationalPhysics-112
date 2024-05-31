import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def load_files(header,pattern='[0-9][0-9][0-9][0-9][0-9][0-9]'):
    """
    Load the data from the output file

    :param header: string, the header of the output file
    """
    fns = 'data_' + header+'/'+header +'_'+ pattern+'.dat'
    fns = glob.glob(fns)
    fns.sort()
    return fns

def save_movie(fns, N, lengthscale=1, filename='movie.gif',fps=30):

    scale = lengthscale
    fig, ax = plt.subplots()
    fig.set_linewidth(5)
    fig.set_size_inches(10, 10, forward=True)
    fig.set_dpi(72)
    line_cold, = ax.plot([], [], 'o', color='b', markersize=5)
    line_hot, = ax.plot([], [], 'o', color='r', markersize=5)

    def init():
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect('equal')
        ax.set_xlabel('X [code unit]', fontsize=18)
        ax.set_ylabel('Y [code unit]', fontsize=18)
        return line_cold, 

    def update(frame):
        fn = fns[frame]
        m,t,x,y,vx,vy,ax,ay=np.loadtxt(fn)
        line_cold.set_data(x[:int(N/2)], y[:int(N/2)])
        line_hot.set_data(x[int(N/2):], y[int(N/2):])
        plt.title("Frame ="+str(frame),size=18)
        return line_cold,

    ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)

    return 