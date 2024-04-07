import numpy as np
"""
This program solves Initial Value Problems (IVP).
We support three numerical meothds: Euler, Rk2, and Rk4

Example Usage:

    def func(t,y,a,b,c):
        yderive = np.zeros(len(y))
        yderive[0] = 0
        yderive[1] = a, ...
        return yderive

    y0  = [0,1]
    t_span = (0,1)
    t_eval =np.linspace(0,1,100)

    sol = solve_ivp(func, t_span, y0, 
                    method="RK4",t_eval=t_eval, args=(K,M))


    See `solve_ivp` for detailed description. 

Author: Kuo-Chuan Pan, NTHU 2022.10.06
                            2024.03.08
For the course, computational physics
"""
def solve_ivp(func, t_span, y0, method, t_eval, args):
    """
    Solve Initial Value Problems. 

    :param func: a function to describe the derivative of the desired function
    :param t_span: 2-tuple of floats. the time range to compute the IVP, (t0, tf)
    :param y0: an array. The initial state
    :param method: string. Numerical method to compute. 
                   We support "Euler", "RK2" and "RK4".
    :param t_eval: array_like. Times at which to store the computed solution, 
                   must be sorted and lie within t_span.
    :param *args: extra arguments for the derive func.

    :return: array_like. solutions. 

    Note: the structe of this function is to mimic the scipy.integrate
          In the numerical scheme we designed, we didn't check the consistentcy between
          t_span and t_eval. Be careful. 

    """
    time = t_span[0]
    y = y0
    sol = np.zeros((len(y0), len(t_eval)))
   
    '''
    define the shape of the solution
    sol = np.zeros((3, 5))
     array([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])

    sol[:,3] = 1
     array([[0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0.]])
    '''
    # set the numerical solver based on "method"
    if method=="Euler":
        _update = _update_euler
    elif method=="RK2":
        _update = _update_rk2
    elif method=="RK4":
        _update = _update_rk4
    else:
        print("Error: mysolve doesn't supput the method",method)
        quit()
    
    for n, t in enumerate(t_eval):
        dt = t_eval[1]-t_eval[0]
        if dt > 0:
            y = _update(func, y, dt, t, *args)

        # record the solution, nth column = y
        sol[:,n]=y
        time+=dt

    return sol

def _update_euler(func,y0,dt,t,*args):
    """
    Update the IVP with the Euler's method
    :return: the next step solution y
    """
    yderv = func(y0, dt, t, *args)
    ynxt = y0 + yderv*dt
    return ynxt


def _update_rk2(func,y0,dt,t,*args):
    """
    Update the IVP with the RK2 method
    :return: the next step solution y
    """
    yderv = func(y0, dt, t, *args)
    k1 = yderv(t, y0, *args)
    k2 = yderv(t+dt, y0 + dt*k1, *args)
    ynxt = y0 + dt/2*(k1+k2) 

    return ynxt 

def _update_rk4(derive_func,y0,dt,t,*args):
    """
    Update the IVP with the RK4 method
    """
    dt2 = 0.5*dt 
    k1  = derive_func(t,y0,*args)
    y1  = y0 + k1 * dt2
    k2  = derive_func(t+dt2,y1,*args)
    y2  = y0 + k2 * dt2
    k3  = derive_func(t+dt2,y2,*args)
    y3  = y0 + k3 * dt
    k4  = derive_func(t+dt,y3,*args)
    ynxt = y0 + dt*(k1+ 2*k2 + 2*k3 + k4)/6.0   
    return ynxt


if __name__=='__main__':


    """
    Testing solver.solve_ivp()

    Kuo-Chuan Pan 2022.10.07

    """

 