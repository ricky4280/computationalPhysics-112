import numpy as np
import matplotlib.pyplot as plt


class Particles:
    """
    Particle class to store particle properties
    """
    
    def __init__(self, N:int=0): # given N initial value
        self.nparticles = N
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        self._tags = np.arange(N)
        self._time = 0
    pass


    @property
    def masses(self):
        return self._masses
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def velocities(self):
        return self._velocities
    
    @property
    def accelerations(self):
        return self._accelerations
    
    @property
    def tags(self):
        return self._tags
    
    @property
    def time(self):
        return self._time
    

    @masses.setter # use (self.particles ,1) instead of np.zeros((self.nparticles,1)).shape?
    def masses(self, m):
        if m.shape != (self.nparticles,1):
            print("Number of particles does not match!")
            raise ValueError
        self._masses = m
        return
    
    @positions.setter
    def positions(self, pos):
        if pos.shape != (self.nparticles,3):
            print("Number of particles does not match!")
            raise ValueError
        self._positions = pos
        return
    
    @velocities.setter
    def velocities(self, vel):
        if vel.shape != (self.nparticles,3):
            print("Number of particles does not match!")
            raise ValueError
        self._velocities = vel
        return
    
    @accelerations.setter
    def accelerations(self, acc):
        if acc.shape != (self.nparticles,3):
            print("Number of particles does not match!")
            raise ValueError
        self._accelerations = acc
        return
    
    @tags.setter
    def tags(self, some_tags):
        if some_tags.shape != np.arange(self.nparticles).shape:
            print("Number of particles does not match!")
            raise ValueError
        self._tags = some_tags
        return
    
    @time.setter
    def time(self, new_time):  
        self._tags = new_time
        return
    
    def set_particles(self, pos, vel, acc):
        """
        Set particle properties for the N-body simulation

        :param pos: positions of particles
        :param vel: velocities of particles
        :param acc: accelerations of particles

        assign ndarray to attributes with same format
        """
        self.positions = pos
        self.velocities = vel
        self.accelerations = acc
        return
    
    def add_particles(self, mass, pos, vel, acc):
        """
        Add N particles to the N-body simulation at once

        :param pos: positions of particles
        :param vel: velocities of particles
        :param acc: accelerations of particles
        """
        self.nparticles += mass.shape[0]
        self.masses = np.vstack((self.masses, mass)) 
        self.positions = np.vstack((self.positions, pos))
        self.velocities = np.vstack((self.velocities, vel))
        self.accelerations = np.vstack((self.accelerations, acc))
        self.tags = np.arange(self.nparticles)
    
        """
        np.vstack: Stack arrays in sequence vertically (row wise).
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        np.vstack((a,b))
         array([[1, 2, 3],
                [4, 5, 6]])
        """
        return
        
    def output(self, filename, time=0):
        """
        Output particle properties to a file

        :param filename: output file name
        """
        masses = self.masses
        pos = self.positions
        vel = self.velocities
        acc = self.accelerations
        tags = self.tags
        KE, PE = self.energy(G=1, rsoft=0.1)
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics 

                ----------------------------------------------------
                """
        header += "Time = {}\n".format(time)
        header += "                Total Kinetic Energy = {}\n".format(KE)
        header += "                Total Potential Energy = {}\n".format(PE)
        header += "                Total Energy = {}\n".format(PE+KE)

        np.savetxt(filename,(tags[:],masses[:,0],
                            pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)
        return

    def draw(self, dim=2, size=10):
        """
        Draw particles in 3D space
        """
        fig = plt.figure()

        if dim == 2: #because of getter, don't need _positions
            ax = fig.add_subplot(111)
            ax.scatter(self.positions[:,0], self.positions[:,1], s=size) 
            ax.set_xlabel('X [code unit]')
            ax.set_ylabel('Y [code unit]')

        elif dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=size)
            ax.set_xlabel('X [code unit]')
            ax.set_ylabel('Y [code unit]')
            ax.set_zlabel('Z [code unit]')

        else:
            print("Invalid dimension!")

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

        return fig, ax

    def energy(self, G, rsoft):
        """
        Compute the total kinetic energy and the potential energy of the particles.
        """
        # kinetic energy/ sum three directions/ turn into (N,1) array to match masses
        KE = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1)[:, np.newaxis])

        # potential energy
        PE = 0
        for i in range(self.nparticles):
            for j in range(self.nparticles):
                if (j>i): 
                    rij = self.positions[i,:] - self.positions[j,:]
                    r = np.sqrt(np.sum(rij**2) + rsoft**2)
                    PE += - G * self.masses[i,0] * self.masses[j,0] / r
        return KE, PE 