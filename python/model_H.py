"""
Active Matter Model H
A script to study active matter

Pass it a .cfg file to specify parameters for the particular solve.
To run using 4 processes, you would use:
    $ mpiexec -n 4 python3 mri.py mri_params.cfg
"""

import time
import configparser
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import sys
import h5py
import dedalus.public as de
from mpi4py import MPI

CW = MPI.COMM_WORLD
import logging

logger = logging.getLogger(__name__)

# Parses .cfg filename passed to script
filename = Path(sys.argv[-1])
outbase = Path("data")

# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))

logger.info('Running model_H.py with the following parameters:')
logger.info(config.items('parameters'))

Nx = config.getint('parameters', 'Nx')
Ny = config.getint('parameters', 'Ny')
kappa = config.getfloat('parameters', 'kappa')
kappa_hat = config.getfloat('parameters', 'kappa_hat')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')


# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Fourier('x', Nx, interval=(0., Nx))
y_basis = de.Fourier('y', Ny, interval=(0., Ny))

domain = de.Domain([x_basis, y_basis], grid_dtype=np.float)

# Active Matter Model H

problem_variables = ['p', 'vx', 'vy', 'phi']
problem = de.IVP(domain, variables=problem_variables)

# Local parameters

problem.parameters['a'] = a
problem.parameters['b'] = b
problem.parameters["ll"] = ll
problem.parameters['kappa'] = kappa
problem.parameters['kappa_hat'] = kappa_hat
problem.parameters['ν'] = ν

# Operator substitutions for Laplacian
problem.substitutions['Lap(A)'] = "dx(dx(A))+dy(dy(A))"

# Variable substitutions
problem.substitutions['phi_x'] = 'dx(phi)'
problem.substitutions['phi_y'] = 'dy(phi)'
problem.substitutions['grad_phi_2'] = 'phi_x**2 + phi_y**2'
# Hydro equations: p, vx, vz

problem.add_equation("dx(vx) + dy(vy) = 0")
problem.add_equation("dt(vx) + dx(p) - ν*Lap(vx) = -vx*dx(vx) - vy*dy(vx) - kappa_hat*(dx(phi_x**2) - 0.5*dx(grad_phi_2) + dy(phi_x*phi_y))")
problem.add_equation("dt(vy) + dy(p) - ν*Lap(vx) = -vx*dx(vy) - vy*dy(vy) - kappa_hat*(dy(phi_y**2) - 0.5*dy(grad_phi_2) + dx(phi_x*phi_y))")

# Phase field
problem.add_equation("dt(phi) + kappa*Lap(Lap(phi)) = -vx*dx(phi) - vy*dy(phi) + a*Lap(phi) + b*Lap(phi**3)")

solver = problem.build_solver(de.timesteppers.SBDF2)

# Analysis
analysis_tasks = []
snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), iter=100, max_writes=200)
snap.add_task("phi")
snap.add_task("vx")
snap.add_task("vy")
analysis_tasks.append(snap)

# Create initial conditions
gshape = ch.domain.dist.grid_layout.global_shape(scales=ch.domain.dealias)
slices = ch.domain.dist.grid_layout.slices(scales=ch.domain.dealias)
rand = np.random.RandomState(seed=100)
noise = rand.standard_normal(gshape)[slices]

phi0 = solver.state['phi']
phi0['g'] = ampl*noise
filter_field(phi0)

solver.stop_wall_time = 24*3600
solver.stop_sim_time = 8000.
solver.stop_iteration = np.inf#100

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(phi)", name="Cint")

start  = time.time()
while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}, Time {:f}".format(solver.iteration,solver.sim_time))
        logger.info("Integrated Concentration = {:10.7e}".format(flow.max("Cint")))
stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
