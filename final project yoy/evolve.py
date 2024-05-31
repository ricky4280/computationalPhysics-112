
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
import scipy as sp
import Particle as pt
from pathlib import Path
from grid_25 import grid_25
from numba import njit,prange,set_num_threads,jit

set_num_threads(4)


class simulation:
    """
    has the b factor, the boundary, the method, the grid, the particle

    explain some functions:
    setup: set up the initial condition
        contains
        the boundary: bound
        the velocity of the boundary: v_bu,v_bd,v_bl,v_br
            v_bu is the velocity of the upper boundary
            v_bd is the velocity of the lower boundary
            v_bl is the velocity of the left boundary
            v_br is the velocity of the right boundary
        the b factor: b_0,b_1
            force is b_1*exp(-r*b_0)
        the number of grid: n_grid
        the method: method
            method can be RK2 or Euler
        the radius of the softening: r_soft
        the radius of the particle: radius

    evolve: evolve the system
        contains
        the time step:dt
        the maximum time: t_max
        the frequency of output: io_freq
        the header of the

    output: output the data
        same with teacher's code

    viscosity: calculate the viscosity
        contains
        the position of the particle: pos
        the acceleration of the particle: a
        the mass of the particle: masses
        the b factor: b_0,b_1
        the number of particle in the first group: lu1
        the number of particle in the second group: lu2
        the radius of the softening: r_soft

    partition: partition the particles into 25 groups
        contains
        the position of the particle: pos

    compare_boundary: compare the boundary
        contains
        the position of the particle: pos
        the velocity of the particle: vel
        the time step: dt

    evolve_RK2: evolve the system using RK2
        contains
        the time step: dt
        the maximum time: t_max
        the frequency of output: io_freq
        the header of the output: io_header

    evolve_euler: evolve the system using Euler
        contains
        the time step: dt
        the maximum time: t_max
        the frequency of output: io_freq
        the header of the output: io_header

    caculate_force: caculate the force 
        contains
        the position of the first group: pos1
        the velocity of the first group: vel1
        the position of the second group: pos2
        the velocity of the second group: vel2
        the radius of the softening: r_soft

    collision:evolve to calculate the collision
        contains
        the position of the particle: pos
        the velocity of the particle: vel

    collision1: calculate the collision
        contains
        the position of the first group: pos1
        the velocity of the first group: vel1
        the position of the second group: pos2
        the velocity of the second group: vel2
        the radius of the particle: radius
        the mass of the particle: masses

        equation:
        vcm=(m1*v1+m2*v2)/(m1+m2)
        v1=2*vcm-v1
        v2=2*vcm-v2

    how to use:

    sim=simulation(particle)
    sim.setup(bound,v_bu=0,v_bd=0,v_bl=0,v_br=0,b_0=0,b_1=0,n_grid=25,method="RK2",r_soft=0.01,radius=1)
    sim.evolve(dt,t_max,io_freq=10,io_header="output")

    """
    def __init__(self,particle):
        self.particle=particle
        self.dict_grid=grid_25()
        return
    
    def setup(self,bound,v_bu=0,v_bd=0,v_bl=0,v_br=0,b_0=0,b_1=0,n_grid=25,method="RK2",r_soft=0.01,radius=1):
        self.time=0
        self.boundyu=bound
        self.boundyd=-bound
        self.boundxl=-bound
        self.boundxr=bound
        self.v_bu=v_bu
        self.v_bd=v_bd
        self.v_bl=v_bl
        self.v_br=v_br
        self.b_0=b_0
        self.n_grid=n_grid
        self.method=method
        self.b_1=b_1
        self.uu=self.partition(self.particle.positions)
        self.r_soft=r_soft
        self.radius=radius
        return
    
    def compare_boundary(self,pos,vel,dt):
        #y<500
        self.boundyu = self.boundyu+self.v_bu*dt
        u=np.where(pos[:,1]>=self.boundyu)
        vel[u,1]=-vel[u,1]+2*self.v_bu
        pos[u,1]=self.boundyu+(pos[u,1]-self.boundyu)
        #y>-500
        self.boundyd = self.boundyd+self.v_bd*dt
        u=np.where(pos[:,1]<=self.boundyd)
        vel[u,1]=-vel[u,1]+2*self.v_bd
        pos[u,1]=self.boundyd+(pos[u,1]-self.boundyd)

        #x<500
        x_bound1 = self.boundxr+self.v_br*dt
        u=np.where(pos[:,0]>=x_bound1)
        vel[u,0]=-vel[u,0]+2*self.v_br
        pos[u,0]=x_bound1+(pos[u,0]-x_bound1)

        #x>-500
        x_bound2 = self.boundxl+self.v_bl*dt
        u=np.where(pos[:,0]<=x_bound2)
        vel[u,0]=-vel[u,0]+2*self.v_bl
        pos[u,0]= x_bound2+(pos[u,0]-x_bound2)

        return pos,vel
    
    def collision(self,pos,vel):
        for i in range(self.n_grid):
            arr=self.dict_grid[i+1]
            for j in arr:
                u1=self.uu[i]
                u2=self.uu[j-1]
                vel=collision1(u1,u2,pos,vel,len(u1),len(u2),self.radius,self.particle.masses[:,0])
        return vel

    
    # def collision1(self,pos1,vel1,pos2,vel2,self.radius):
    #     r=pos1-pos2
    #     r_norm=np.sqrt(np.sum(r**2))
    #     if r_norm<2*self.radius:
    #         v1=vel1-2*np.dot(vel1-vel2,r)*r/r_norm**2
    #         v2=vel2-2*np.dot(vel2-vel1,r)*r/r_norm**2
    #         return v1,v2
    #     else:
    #         return vel1,vel2
    
    
    def partition(self,pos):
        bx=[abs((self.boundxl-self.boundxr))*i/5-abs((self.boundxl-self.boundxr))/2 for i in range(6)]
        by=[abs((self.boundyd-self.boundyu))*i/5-abs((self.boundyd-self.boundyu))/2 for i in range(6)]
        uu=[]
        tags=0
        for i in range(5):
            for j in range(5):
                i=4-i
                tags+=1
                u0=np.where(pos[:,0]>=bx[j])
                u1=np.where(pos[:,0]<bx[j+1])
                u2=np.where(pos[:,1]>=by[i])
                u3=np.where(pos[:,1]<by[i+1])
                u=np.intersect1d(np.intersect1d(u0,u1),np.intersect1d(u2,u3))
                self.particle.tags[u]=tags
                uu.append(np.array(u))
        return uu
    
    def viscosity(self,pos):
        a=np.zeros_like(pos)
        if self.b_1==0:
            return a
        for i in range(self.n_grid):
            arr=self.dict_grid[i+1]
            for j in arr:
                u1=self.uu[i]
                u2=self.uu[j-1]
                a=caculate_force(u1,u2,pos,a,self.particle.masses,self.b_0,self.b_1,len(u1),len(u2),self.r_soft)
        return a
    

    def output(self,filename):
        masses=self.particle.masses
        pos=self.particle.positions
        vel=self.particle.velocities
        acc=self.particle.accelerations
        tags=self.particle.tags
        time=self.time
        header = """
                    ----------------------------------------------------
                    Data from a 2D direct N-body simulation. 

                    rows are i-particle; 
                    coumns are :mass, tag, x ,y, vx, vy, ax, ay

                    NTHU, Computational Physics 

                    ----------------------------------------------------
                    """
        header += "Time = {}".format(time)
        np.savetxt(filename,(tags[:],masses[:,0],pos[:,0],pos[:,1],
                                vel[:,0],vel[:,1],
                                acc[:,0],acc[:,1]),header=header)
        return
    
    def evolve_RK2(self,dt,t_max,io_freq=10,io_header="output"):
        k=int(t_max/dt)
        io_folder = "data_"+io_header
        Path(io_folder).mkdir(parents=True, exist_ok=True) 
        fn = io_header+"_"+str(0).zfill(6)+".dat"
        fn = io_folder+"/"+fn
        self.output(fn) 
        for i in range(1,k):
            self.time+=dt
            self.uu=self.partition(self.particle.positions)
            self.particle.accelerations=np.zeros((self.particle.nparticles,2))
            self.particle.accelerations=self.viscosity(self.particle.positions)
            acc1=self.particle.accelerations
            self.particle.velocities=self.collision(self.particle.positions,self.particle.velocities)
            vk1=self.particle.velocities
            self.uu=self.partition(self.particle.positions+vk1*dt)
            self.particle.accelerations=np.zeros((self.particle.nparticles,2))
            self.particle.accelerations=self.viscosity(self.particle.positions+vk1*dt)
            acc2=self.particle.accelerations
            vk2=vk1+acc1*dt
            a=(acc1+acc2)/2
            v=(vk1+vk2)/2
            self.particle.positions+=v*dt
            self.particle.velocities+=a*dt
            self.particle.positions,self.particle.velocities=self.compare_boundary(self.particle.positions,self.particle.velocities,dt)
            if i%io_freq==0:
                print(i)
                fn = io_header+"_"+str(i).zfill(6)+".dat"
                fn = io_folder+"/"+fn
                self.output(fn)
        return
    
    def evolve_euler(self,dt,t_max,io_freq=10,io_header="output"):
        k=int(t_max/dt)
        io_folder = "data_"+io_header
        Path(io_folder).mkdir(parents=True, exist_ok=True) 
        fn = io_header+"_"+str(0).zfill(6)+".dat"
        fn = io_folder+"/"+fn
        self.output(fn) 
        for i in range(int(t_max/dt)+1):
            self.time+=dt
            self.uu=self.partition(self.particle.positions)
            self.particle.accelerations=np.zeros((self.particle.nparticles,2))
            self.particle.accelerations=self.viscosity(self.particle.positions)
            # if max(abs(self.particle.accelerations[:,0])) >1e5:
            #     fn = "err"+"_"+str(i).zfill(6)+".dat"
            #     fn = io_folder+"/"+fn
            #     self.output(fn)
            #     print("Acceleration is too large!")
            self.particle.velocities=self.collision(self.particle.positions,self.particle.velocities)
            self.particle.positions+=self.particle.velocities*dt
            self.particle.velocities+=self.particle.accelerations*dt
            self.particle.positions,self.particle.velocities=self.compare_boundary(self.particle.positions,self.particle.velocities,dt)
            if (i % io_freq == 0):
                print(i)
                fn = io_header+"_"+str(i).zfill(6)+".dat"
                fn = io_folder+"/"+fn
                self.output(fn)
        return
    

    def evolve(self,dt,t_max,io_freq=10,io_header="output"):
        if self.method=="RK2":
            self.evolve_RK2(dt,t_max,io_freq,io_header)
        elif self.method=="Euler":
            self.evolve_euler(dt,t_max,io_freq,io_header)
        else:
            print("Method not supported!")
        return 

@njit(parallel=True)
def caculate_force(u1,u2,pos,a,masses,b_0,b_1,lu1,lu2,r_soft):
    for k0 in prange(lu1):
        for l0 in prange(lu2):
            r=pos[u1[k0],:]-pos[u2[l0],:]
            r_norm=np.sqrt(np.sum(r**2))+r_soft
            a[u1[k0],:]-=b_1*r*exp(-r_norm*b_0)/masses[u1[k0],0]
            a[u2[l0],:]+=b_1*r*exp(-r_norm*b_0)/masses[u2[l0],0]
    return a

@njit(parallel=True)
def collision1(u1,u2,pos,vel,lu1,lu2,radius,masses):
    for k0 in prange(lu1):
        for l0 in prange(lu2):
            r=pos[u1[k0],:]-pos[u2[l0],:]
            r_norm=np.sqrt(np.sum(r**2))
            if r_norm<2*radius and np.dot(vel[u1[k0],:]-vel[u2[l0],:],r)<0:
                vcm=(masses[u1[k0]]*vel[u1[k0],:]+masses[u2[l0]]*vel[u2[l0],:])/(masses[u1[k0]]+masses[u2[l0]])
                #2D elastic collision
                vel[u1[k0],:]=2*vcm-vel[u1[k0],:]
                vel[u2[l0],:]=2*vcm-vel[u2[l0],:]
    return vel

# N=30
# particle=pt.Particles(N)
# pos=np.random.rand(N,2)*1000-500
# vel=np.random.rand(N,2)*300-150
# particle.positions=pos
# particle.velocities=vel
# particle.masses=np.ones((N,1))
# particle.accelerations=np.zeros((N,2))
# pos,vel=evolve(particle,0.1,100)
#make a movie
# import matplotlib.animation as animation
# fig, ax = plt.subplots()
# ax.set_xlim(-500,500)
# ax.set_ylim(-500,500)
# scat = ax.scatter(pos[0,:,0],pos[0,:,1],s=10)
# def update(frame_number):
#     scat.set_offsets(np.c_[pos[frame_number,:,0],pos[frame_number,:,1]])
#     return scat,
# ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
# ani.save('movie.gif', writer='ffmpeg', fps=10)
# plt.show()







