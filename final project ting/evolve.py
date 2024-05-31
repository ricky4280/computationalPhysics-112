import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pathlib import Path
from grid_N import grid_N
from numba import njit, prange, set_num_threads, jit
import Particle as pt  # Ensure this module is available and correctly imported

set_num_threads(4)

class Simulation:
    """
    Has the b factor, the boundary, the method, the grid, the particle
    """
    def __init__(self, particle):
        self.particle = particle
    
    def setup(self, bound, bound_vel_up=0, bound_vel_down=0, bound_vel_left=0, bound_vel_right=0, b0=0, b1=0, N=5, method="RK2", r_soft=0.01, radius=1):
        self.time = 0
        self.pressure = np.zeros(4)
        self.bound = np.array([bound, -bound, -bound, bound])
        self.bound_vel = np.array([-bound_vel_up, bound_vel_down, bound_vel_left, -bound_vel_right])
        self.b = [b0, b1]
        self.N_grid = N
        self.dict_grid = grid_N(N)
        self.method = method
        self.r_soft = r_soft
        self.radius = radius
    
    def evolve(self, dt, t_max, io_freq=10, io_header="output"):
        if self.method == "RK2":
            self.evolve_RK2(dt, t_max, io_freq, io_header)
        elif self.method == "Euler":
            self.evolve_euler(dt, t_max, io_freq, io_header)
        else:
            print("Method not supported!")
    
    def output_particles(self, filename):
        data = np.column_stack((self.particle.tags, self.particle.masses[:, 0], self.particle.positions[:, 0], self.particle.positions[:, 1],
                                self.particle.velocities[:, 0], self.particle.velocities[:, 1],
                                self.particle.accelerations[:, 0], self.particle.accelerations[:, 1]))
        header = """
        ----------------------------------------------------
        Data from a 2D direct N-body simulation. 
        Rows are i-particle; 
        Columns are: mass, tag, x, y, vx, vy, ax, ay
        NTHU, Computational Physics 
        ----------------------------------------------------
        Time = {}
        """.format(self.time)
        np.savetxt(filename, data, header=header)
    
    def output_pressure(self, filename):
        data = self.pressure
        header = """
        ----------------------------------------------------
        Data from a 2D direct N-body simulation. 
        Rows are i-particle; 
        Columns are: mass, tag, x, y, vx, vy, ax, ay
        NTHU, Computational Physics 
        ----------------------------------------------------
        Time = {}
        """.format(self.time)
        np.savetxt(filename, data, header=header)
    
    def evolve_RK2(self, dt, t_max, io_freq=10, io_header="output"):
        k = int(t_max / dt)
        Path(f'data_{io_header}/particles').mkdir(parents=True, exist_ok=True)
        Path(f'data_{io_header}/pressure').mkdir(parents=True, exist_ok=True)
        self.output_particles(f'data_{io_header}/particles/{io_header}_pts_{str(0).zfill(6)}.dat')
        self.output_pressure(f'data_{io_header}/pressure/{io_header}_psr_{str(0).zfill(6)}.dat')
        for i in range(1, k+1):
            self.time += dt
            self.N_part, self.particle.tags = self.partition(self.bound, self.N_grid, self.particle.positions, self.particle.tags)
            self.viscosity(self.N_grid, self.dict_grid, self.N_part, self.particle.positions,\
                           self.particle.accelerations, self.particle.masses, *self.b, self.r_soft)
            self.collision(self.N_grid, self.dict_grid, self.N_part, self.particle.positions,\
                           self.particle.velocities, self.particle.masses, self.radius)
            acc1 = self.particle.accelerations
            vk1 = self.particle.velocities
            self.N_part, self.particle.tags = self.partition(self.bound, self.N_grid, self.particle.positions + vk1 * dt, self.particle.tags)
            self.viscosity(self.N_grid, self.dict_grid, self.N_part, self.particle.positions + vk1*dt,\
            	           self.particle.accelerations, self.particle.masses, *self.b, self.r_soft)
            
            acc2 = self.particle.accelerations
            vk2 = vk1 + acc1 * dt
            a = (acc1 + acc2) / 2
            v = (vk1 + vk2) / 2

            self.particle.positions += v * dt
            self.particle.velocities += a * dt
            self.bound_reflection(self.bound, self.bound_vel, self.particle.positions, self.particle.velocities, self.pressure, dt)
            if i % io_freq == 0:
                self.output_particles(f'data_{io_header}/particles/{io_header}_pts_{str(i).zfill(6)}.dat')
                self.output_pressure(f'data_{io_header}/pressure/{io_header}_psr_{str(i).zfill(6)}.dat')
    '''
    def evolve_euler(self, dt, t_max, io_freq=10, io_header="output"):
        k = int(t_max / dt)
        io_folder = "data_" + io_header
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        self.output(f"{io_folder}/{io_header}_000000.dat")
        for i in range(1, k + 1):
            self.time += dt
            self.N_part, self.particle.tags = self.partition(self.bound, self.N_grid, self.paticle.positions, self.particle.tags)
            self.viscosity(self.N_grid, self.dict_grid, self.N_part, self.particle.positions,\
                           self.particle.accelerations, self.particle.masses, self.radius)
            self.collision(self.N_grid, self.dict_grid, self.N_part, self.particle.positions,\
                           self.particle.velocities, self.particle.masses, self.radius)
            self.particle.positions += self.particle.velocities * dt
            self.particle.velocities += self.particle.accelerations * dt
            self.particle.positions, self.particle.velocities, self.bound, self.pressure = self.bound_reflection(self.bound, self.bound_vel,\
            											self.particle.positions, self.particle.velocities, dt)
            if i % io_freq == 0:
                self.output(f"{io_folder}/{io_header}_{str(i).zfill(6)}.dat")
    '''
    @staticmethod
    def bound_reflection(bound, bound_vel, positions, velocities, pressure, dt):
        col = [1, 1, 0, 0]
        relation = ['>=', '<=', '<=', '>=']
        boundlength = [bound[3]-bound[2], bound[3]-bound[2], bound[0]-bound[1], bound[0]-bound[1]]
        for i in range(4):
            bound[i]+=bound_vel[i]*dt
            u = np.where(eval(f'positions[:, col[i]] {relation[i]} bound[i]'))
            pressure[i] = np.abs(np.sum((-velocities[u, col[i]] + 2 * bound_vel[i] - velocities[u, col[i]])/dt)/boundlength[i])
            velocities[u, col[i]] = -velocities[u, col[i]] + 2 * bound_vel[i]
            positions[u, col[i]] = bound[i] - (positions[u, col[i]] - bound[i])
    
    @staticmethod
    def partition(bound, N_grid, positions, tags):
        bx = [bound[2] + (bound[3] - bound[2]) * i / N_grid for i in range(N_grid+1)]	#the x coordinate of the boundary for each grid 
        by = [bound[1] + (bound[0] - bound[1]) * i / N_grid for i in range(N_grid+1)]	#the y coordinate of the boundary for each grid
        N_part = []
        tag = 0
        for i in range(N_grid):
            for j in range(N_grid):
                u = np.where((positions[:, 0] >= bx[j]) & (positions[:, 0] < bx[j + 1]) & (positions[:, 1] >= by[i]) & (positions[:, 1] < by[i + 1]))[0]
                N_part.append(u)
                tag += 1
                tags[u] = tag
        return N_part, tags
        
    @staticmethod
    def viscosity(N_grid, dict_grid, N_part, positions, accelerations, masses, b0, b1, r_soft):
        if b1 == 0: 
            accelerations = np.zeros_like(accelerations)
            return
        for i in range(N_grid**2):
            for j in dict_grid[i + 1]:
                u1 = N_part[i]
                u2 = N_part[j - 1]
                calculate_acceleration(u1, u2, positions, accelerations, masses, b0, b1, r_soft)
    
    @staticmethod 
    def collision(N_grid, dict_grid, N_part, positions, velocities, masses, radius):
        if radius == 0: return 
        for i in range(N_grid**2):
            for j in dict_grid[i+1]:
                u1 = N_part[i]
                u2 = N_part[j-1]
                calculate_collision(u1, u2, positions, velocities, masses, radius)

@njit(parallel=True)
def calculate_acceleration(u1, u2, positions, accelerations, masses, b0, b1, r_soft):
    for k0 in prange(len(u1)):
        for l0 in prange(len(u2)):
            r_vector = positions[u1[k0], :] - positions[u2[l0], :]
            r_norm = np.sqrt(np.sum(r_vector**2)) + r_soft
            if r_norm == 0:
                continue
            force = b1 * np.exp(-b0 * r_norm) * r_vector / (r_norm)
            accelerations[u1[k0], :] -= force/masses[u1[k0], 0]
            accelerations[u2[l0], :] += force/masses[u2[l0], 0]

@njit(parallel=True)
def calculate_collision(u1, u2, positions, velocities, masses, radius):
    for k0 in prange(len(u1)):
        for l0 in prange(len(u2)):
            r_vector = positions[u1[k0],:] - positions[u2[l0],:]
            r_norm = np.sqrt(np.sum(r_vector**2))
            if r_norm < 2*radius and np.dot(velocities[u1[k0],:]-velocities[u2[l0],:], r_vector) < 0:
                vcm = (masses[u1[k0], 0]*velocities[u1[k0],:]+masses[u2[l0], 0]*velocities[u2[l0],:])/(masses[u1[k0], 0]+masses[u2[l0], 0])
                #2D elastic collision
                velocities[u1[k0],:] = 2*vcm-velocities[u1[k0],:]
                velocities[u2[l0],:] = 2*vcm-velocities[u2[l0],:]
