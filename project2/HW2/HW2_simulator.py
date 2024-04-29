import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from HW2_particles import Particles 
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies

"""

class NBodySimulator:

    def __init__(self, particles: Particles): # tpying: particles attribute is belong to Particles class

        # TODO
        self.particles = particles
        self.time      = particles.time
        self.setup()    # set up the default setting
        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to output data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        
        # TODO
        # set up the class attributes from the input

        self.G = G
        self.rsoft = rsoft
        self.method = method
        if io_freq <= 0: io_freq = np.inf   # no output
        self.io_freq = io_freq              # output frequency  
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization
        return


    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        # TODO
        self.dt = dt
        self.tmax = tmax

        nsteps = int(np.ceil(tmax/dt))  #四捨五入    
        time = self.time      
        particles = self.particles

        # setup numerical meothd
        method = self.method

        if method.lower() == "euler":
            _advance_particles = self._advance_particles_Euler
        elif method.lower() == "rk2":
            _advance_particles = self._advance_particles_RK2
        elif method.lower() == "rk4":
            _advance_particles = self._advance_particles_RK4
        elif method.lower() == "lfs":
            _advance_particles = self._advance_particles_LFS
        else:
            raise ValueError("Unknown method")

        # ===============================
        # Start the simulation
        # The main loop
        # ===============================

        for n in range(nsteps):

            # check if the time step exceeds the total time
            if (time+dt > tmax): dt = tmax - time

            # advance the particles
            particles = _advance_particles(dt, particles)

            # output data every io_freq steps
            if (n % self.io_freq == 0):

                # print message on screen
                if self.io_screen:
                    print("n=",n , "Time: ", time, " dt: ", dt)
                
                 # prepare for the output 
                io_folder = "data_"+self.io_header                  # create a folder name
                Path(io_folder).mkdir(parents=True, exist_ok=True)  # create a folder

                # output the data
                fn = self.io_header+"_"+str(n).zfill(6)+".dat"
                fn = io_folder+"/"+fn
                particles.output(fn)    # ouput the data by output function in particles.py

            # update the time
            time += dt
        
        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        G = self.G
        rsoft = self.rsoft
        
        accelerations = _calculate_acceleration_kernal(nparticles, masses, positions, accelerations, G, rsoft)
        return accelerations
        
    def _advance_particles_Euler(self, dt, particles):
        
        # TODO
        nparticles = particles.nparticles
        masses = particles.masses
        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, masses, pos)

        # do the Euler update
        pos += vel*dt
        vel += acc*dt

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles

    def _advance_particles_RK2(self, dt, particles):

        # TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, mass, pos)
    
        # do the RK2 update
        pos2 = pos + vel*dt 
        vel2 = vel + acc*dt
        acc2 = self._calculate_acceleration(nparticles, mass, pos2) 

        pos2 = pos2 + vel2*dt
        vel2 = vel2 + acc2*dt

        # average
        pos = 0.5*(pos + pos2)
        vel = 0.5*(vel + vel2)
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles
    

    def _advance_particles_RK4(self, dt, particles):
        
        #TODO
        nparticles = particles.nparticles
        mass = particles.masses

        # y0
        pos = particles.positions
        vel = particles.velocities # k1
        acc = self._calculate_acceleration(nparticles, mass, pos) # k1

        dt2 = dt/2
        # y1
        pos1 = pos + vel*dt2
        vel1 = vel + acc*dt2 # k2
        acc1 = self._calculate_acceleration(nparticles, mass, pos1) # k2
        
        # y2
        pos2 = pos + vel1*dt2
        vel2 = vel + acc1*dt2 # k3
        acc2 = self._calculate_acceleration(nparticles, mass, pos2) # k3

        # y3
        pos3 = pos + vel2*dt
        vel3 = vel + acc2*dt # k4
        acc3 = self._calculate_acceleration(nparticles, mass, pos3) # k4

        # rk4
        pos = pos + (vel + 2*vel1 + 2*vel2 + vel3)*dt/6
        vel = vel + (acc + 2*acc1 + 2*acc2 + acc3)*dt/6
        acc = self._calculate_acceleration(nparticles, mass, pos)

        # update the particles
        particles.set_particles(pos, vel, acc)

        return particles
    
    def _advance_particles_LFS(self, dt, particles):
        
        nparticles = particles.nparticles
        masses = particles.masses
        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, masses, pos)

        # do the leap frog scheme
        vel += acc*dt/2
        pos += vel*dt
        
        particles.set_particles(pos, vel, acc)

        return particles




# Most important part of the code
@njit(parallel=True) # numba to accelerate the code
def _calculate_acceleration_kernal(nparticles, masses, positions, accelerations, G,rsoft):
    """
    Calculate the acceleration of the particles.
    the kernel function for numba out of class.
    
    :param particles: Particles, the particles to calculate the acceleration
    """

    # kernel for acceleration calculation
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * masses[i,0] * masses[j,0] * rij / r**3
                accelerations[i,:] += force[:] / masses[i,0]
                accelerations[j,:] -= force[:] / masses[j,0]

    return accelerations


if __name__ == "__main__":
    
    pass