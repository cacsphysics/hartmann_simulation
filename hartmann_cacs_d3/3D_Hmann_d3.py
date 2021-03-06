"""2D Hartmann flow example
Uses vector potential form of MHD; incompressible Navier-Stokes, no further approximations.
06/13/22 edits: author Carlos A. Cartagena-Sanchez

reference paper or document needed
"""
import os
import sys
import numpy as np
import pathlib
import h5py
import time
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
from mpi4py import MPI


import logging
logger = logging.getLogger(__name__)


Lx = 10.
Ly = 2
Lz = 1
nx = 64
ny = 64
nz = 64



mesh = [8,8]
stop_time = 0.2
data_dir = "checkpoints"
dealias = 3/2
# domain, distributor, and base
coords = de.CartesianCoordinates('x','y','z')
dist = de.Distributor(coords, dtype=np.float64, mesh=mesh)


xbasis = de.RealFourier(coords['x'], size=nx, bounds=(0,Lx), dealias=dealias)
ybasis = de.RealFourier(coords['y'], size=ny, bounds=(0,Ly), dealias=dealias)
zbasis = de.ChebyshevT(coords['z'], size=nz, bounds=(-Lz,Lz), dealias=dealias)


#x = de.Fourier('x', nx, interval=[0,Lx], dealias=3/2)
#y = de.Fourier('y', ny, interval=[-Ly, Ly], dealias=3/2)
#z = de.Chebyshev('z', nz, interval=[-Lz,Lz], dealias=3/2) #the coupled dimension
#domain = de.Domain([x,y,z],grid_dtype='float', mesh=mesh)


# problem, variables, and parameters
#### Vector Fields
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))

#### Scale Fields
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
t = dist.Field()
##### Taus
tau_p1 = dist.Field(name='tau_p1', bases = (xbasis, ybasis))
tau_p2 = dist.Field(name='tau_p2', bases = (xbasis, ybasis))

tau_v1 = dist.VectorField(coords, name='tau_v1', bases = (xbasis, ybasis))
tau_v2 = dist.VectorField(coords, name='tau_v2', bases = (xbasis, ybasis))
tau_a1 = dist.VectorField(coords, name='tau_a1', bases = (xbasis, ybasis))
tau_a2 = dist.VectorField(coords, name='tau_a2', bases = (xbasis, ybasis))
#### Problem 
ex, _, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(2)
lift = lambda A: de.Lift(A, lift_basis, -1)
grad_v = de.grad(v) + ez*lift(tau_v1) # First-order reduction
grad_a = de.grad(A) + ez*lift(tau_a1) # First-order reduction
#dy = lambda A: d3.Differentiate(A, coords['y'])

#### Substitutions
Ha = 20. # change this parameter
Re = 1.
Rm = 1.
Pi = 1.
tau = 0.1
B = de.Curl(A)
J = -de.Laplacian(A)

##### Forcing Terms
#f = dist.Field(name='f')
f = Pi*np.exp(-t/tau)*ex

hartmann = de.IVP([v, A, p, tau_p1, tau_p2, tau_v1, tau_v2, tau_a1, tau_a2], time=t, namespace=locals())
# Navier Stokes
hartmann.add_equation("trace(grad_v) + tau_p1 = 0") #first order form
hartmann.add_equation("trace(grad_a) + tau_p2 = 0")
hartmann.add_equation("dt(v)+ grad(p) - lap(v)/Re = -v@grad(v) - Ha**2/(Re*Rm)*cross(J, B) - (f  - 1) + lift(tau_v2)")

# div(v) = 0, incompressible
hartmann.add_equation("integ(p) = 0")
#hartmann.add_equation("div(v) + tau_v1 = 0")
#hartmann.add_equation("div(A) + tau_a1 = 0")
#hartmann.add_equation("dx(vx) + dy(vy) + vz_z = 0")

# Az Induction Equation
hartmann.add_equation("dt(A) - lap(A)/Rm = cross(v, B) + lift(tau_a2)")
#hartmann.add_equation("dt(Ay) - Lap(Ay, Ay_z)/Rm = vx*Bz - vz*Bx")

# first order form
#hartmann.add_equation("dz(Ay) - Ay_z = 0") #taken careof above.
#hartmann.add_equation("dz(vx) - vx_z = 0")
#hartmann.add_equation("dz(vy) - vy_z = 0")
#hartmann.add_equation("dz(vz) - vz_z = 0")

# boundary conditions: nonslip at wall, pressure concentrated on the left


hartmann.add_equation("v(x = 'left') = 0")
hartmann.add_equation("v(x = 'right') = 0", condition ="(nx == 0)")
#hartmann.add_equation("v(x='left') = 0")
#hartmann.add_equation("v(x='right') = 0")
#hartmann.add_equation("v(y='left') = 0")
#hartmann.add_equation("v(y='right') = 0")
#hartmann.add_equation("v(z='left') = 0")
#hartmann.add_equation("v(z='right') = 0", condition = "(nx != 0)")
hartmann.add_equation("p(x='right') = 0", condition = "(nx==0)")
hartmann.add_equation("A(x = 'left') = 0")
#hartmann.add_equation("A(x = 'right') = 0")
#hartmann.add_bc("left(vx) = 0")
#hartmann.add_bc("right(vx) = 0")
#hartmann.add_bc("left(vy) = 0")
#hartmann.add_bc("right(vy) = 0")
#hartmann.add_bc("left(vz) = 0")
#hartmann.add_bc("right(vz) = 0", condition="(nx != 0)")
#hartmann.add_bc("right(p) = 0", condition="(nx == 0)")
#hartmann.add_bc("left(Ay) = 0")
#hartmann.add_bc("right(Ay) = 0")

# build solver
solver = hartmann.build_solver(de.MCNAB2)
logger.info("Solver built")

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 5*24*60.*60
solver.stop_iteration = np.inf
dt = 1e-3

# Initial conditions are zero by default in all fields

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1e-3, max_writes=200)

snap.add_task(B, name='bfield')
snap.add_task(v, name='velocity')
#snap.add_task("Bx", scales=1)
#snap.add_task("Bz", scales=1)
#snap.add_task("Ay", scales=1)
#snap.add_task("vx", scales=1)
#snap.add_task("vy", scales=1)
#snap.add_task("vz", scales=1)

analysis_tasks.append(snap)

#integ = solver.evaluator.add_file_handler(os.path.join(data_dir,'integrals'), sim_dt=1e-3, max_writes=200)
#integ.add_task("Avg_x(vx)", name='vx_x', scales=1)
#integ.add_task("Avg_x(vy)", name='vy_x', scales=1)
#integ.add_task("Avg_x(vz)", name='vz_x', scales=1)
#integ.add_task("Avg_x(Bx)", name='Bx_x', scales=1)
#integ.add_task("Avg_x(Bz)", name='Bz_x', scales=1)
#analysis_tasks.append(integ)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), sim_dt=1e-2)
timeseries.add_task("0.5*integ(v@v)",name='Ekin')
timeseries.add_task("0.5*integ(Bx**2 + Bz**2)",name='Emag')
analysis_tasks.append(timeseries)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property(v@v/2, name='Ekin')
#flow.add_property("0.5*(vx**2 + vy**2 + vz**2)", name='Ekin')

try:
    logger.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max E_kin = %17.12e' %flow.max('Ekin'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %f' %(end_run_time-start_run_time))


logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    if os.path.isfile(dir_path+"/scratch/integrals/integrals.h5"):
        os.remove(dir_path+"/scratch/integrals/integrals.h5")
set_paths = list(pathlib.Path('scratch/integrals').glob("*.h5"))
post.merge_sets('checkpoints/scratch/integrals/integrals.h5', set_paths)
