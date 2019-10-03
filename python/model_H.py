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
B = config.getfloat('parameters', 'B')


# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Fourier('x', Nx, interval=(-Lx / 2, Lx / 2))
y_basis = de.Fourier('y', Ny, interval=(-Ly / 2, Ly / 2))

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

# Operator substitutions for y,z, and t derivatives

problem.substitutions['Lap(A)'] = "dx(dx(A))+dy(dy(A))"


# Variable substitutions

problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['jx'] = "dy(bz) - dz(by)"
problem.substitutions['jy'] = "dz(bx) - dx(bz)"
problem.substitutions['jz'] = "dx(by) - dy(bx)"

# Hydro equations: p, vx, vy, vz, ωy, ωz

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = 0")
problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = 0")
problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = 0")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx

problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0")

problem.add_equation("Dt(bx) - B*dz(vx)            + η*( dy(jz) - dz(jy) )                   = 0")
problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - η*( dx(jxx) + dy(dy(jx)) + dz(dz(jx)) ) = 0")

problem.add_equation("jxx - dx(jx) = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_bc("left(vx)   = 0")
problem.add_bc("left(ωy)   = 0")
problem.add_bc("left(ωz)   = 0")
problem.add_bc("left(bx)   = 0")
problem.add_bc("left(jxx)  = 0")

problem.add_bc("right(vx)  = 0")
problem.add_bc("right(ωy)  = 0")
problem.add_bc("right(ωz)  = 0")
problem.add_bc("right(bx)  = 0")
problem.add_bc("right(jxx) = 0")

# GO

solver = problem.build_solver()


def ideal_2D(kz):
    kk, BB = kz * kz, B * B
    a = kx * kx + kk
    b = kk * (2 * BB * a + f * (f + S))
    c = (BB * kk ** 2) * (BB * a + f * S)
    return np.sqrt((- b + np.sqrt(b * b - 4 * a * c + 0j)) / (2 * a))


# Create function to compute max growth rate for given ky, kz
def growth_rate(ky, kz, target, N=15, dense=False):
    eigvec = np.zeros((10, Nx), dtype=np.complex128)
    # Change ky, kz parameters
    problem.namespace['ky'].value = ky
    problem.namespace['kz'].value = kz
    # Solve for eigenvalues with sparse search near target, rebuilding NCCs
    solver_failed = False
    try:
        if dense:
            solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
        else:
            solver.solve_sparse(solver.pencils[0], N=N, target=target, rebuild_coeffs=True)
    except:
        logger.warning("Solver failed for (ky, kz) = (%f, %f)" % (ky, kz))
        solver_failed = True

    if solver_failed:
        gamma_r = np.nan
        gamma_i = np.nan
        gamma = []
        gamma.append(gamma_r + 1j * gamma_i)
    else:
        gamma = solver.eigenvalues
        if dense:
            gamma = gamma[np.abs(gamma) < dense_threshold]
        index = np.argsort(-gamma.real)
        gamma = gamma[index]

        eigvec = np.zeros((10, Nx), dtype=np.complex128)

        for k in range(10):
            solver.set_state(index[0])
            eigvec[k, :] = solver.state[problem_variables[k]]['g']

        gamma_r = gamma.real[0]
        gamma_i = gamma.imag[0]

        if np.abs(gamma_r) <= 1e-6: gamma_r = 0.0
        if np.abs(gamma_i) <= 1e-6: gamma_i = 0.0

    logger.info('(ky,kz,gamma,omega) = (%20.15f,%20.15f,%20.15f,%20.15f)' % (ky, kz, gamma_r, gamma_i))

    # Return complex growth rate
    return gamma[0], eigvec


ky_global = np.linspace(kymin, kymax, Nky)
kz_global = np.linspace(kzmin, kzmax, Nkz)
gamma_global = np.zeros((Nky, Nkz), dtype=np.complex128)
eigvec_global = np.zeros((Nky, Nkz, 10, Nx), dtype=np.complex128)

# Compute growth rate over local wavenumbers
kz_local = kz_global[CW.rank::CW.size]
gamma_local = gamma_global[:, CW.rank::CW.size]
eigvec_local = eigvec_global[:, CW.rank::CW.size]

t1 = time.time()
for k, kz in enumerate(kz_local):
    if ky_start_guess:
        soln = growth_rate(ky_global[0], kz, ky_start_guess, N=Nmodes, dense=dense)
    elif ky_start_dense:
        logger.info("Using dense solve at ky=0 for initial guess.")
        soln = growth_rate(0., kz, ideal_2D(kz), dense=True)
    else:
        soln = growth_rate(0., kz, ideal_2D(kz), N=Nmodes, dense=dense)
    gamma_local[0, k] = soln[0]
    eigvec_local[0, k] = soln[1]

for i in range(1, Nky):
    for (k, kz) in enumerate(kz_local):
        soln = growth_rate(ky_global[i], kz, gamma_local[i - 1, k], N=Nmodes, dense=dense)
        gamma_local[i, k] = soln[0]
        eigvec_local[i, k, :, :] = soln[1]
# Reduce growth rates to root process
gamma_global[:, CW.rank::CW.size] = gamma_local
eigvec_global[:, CW.rank::CW.size] = eigvec_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE, gamma_global, op=MPI.SUM, root=0)
    CW.Reduce(MPI.IN_PLACE, eigvec_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(gamma_global, gamma_global, op=MPI.SUM, root=0)
    CW.Reduce(eigvec_global, eigvec_global, op=MPI.SUM, root=0)
t2 = time.time()
logger.info('Elapsed total solve time for all processes: %f' % (t2 - t1))

# Save either or both eigenvalues and eigenvectors to a single .h5 file
# Output file will be the .cfg file name with _output.h5
if CW.rank == 0:
    output_file_name = Path(filename.stem + '_output.h5')
    output_file = h5py.File(outbase / output_file_name, 'w')
    if config.getboolean('output', 'gamma'):
        dset_gamma = output_file.create_dataset('gamma', data=gamma_global)
        dset_kz = output_file.create_dataset('kz', data=kz_global)
        dset_ky = output_file.create_dataset('ky', data=ky_global)
    if config.getboolean('output', 'eigvec'):
        dset_evec = output_file.create_dataset('eigvec', data=eigvec_global)
        dset_x = output_file.create_dataset('x', data=x_basis.grid())

find_max = False
if find_max:
    if CW.rank == 0:
        max_growth_guess = gamma_global.max()
        max_where = np.unravel_index(gamma_global.argmax(), gamma_global.shape)
        max_guess = [ky_global[max_where[0]], kz_global[max_where[1]]]
        func = lambda x: -growth_rate(x[0], x[1], max_growth_guess, N=Nmodes)[0].real
        t1 = time.time()
        # max_growth_opt = opt.minimize(func, max_guess, method="Nelder-Mead")
        max_growth_opt = opt.minimize(func, max_guess, method="Powell")
        t2 = time.time()

        if max_growth_opt.success:
            max_location = max_growth_opt.x
            max_growth = -max_growth_opt.fun
            dset_gamma.attrs.create("max growth rate", max_growth)
            dset_gamma.attrs.create("max ky", max_location[0])
            dset_gamma.attrs.create("max kz", max_location[1])
            logger.info("Time to find maximum growth rate: {:f}".format(t2 - t1))
            logger.info(
                "Maximum growth rate {:20.15f} at (ky, kz) = ({:20.15f}, {:20.15f})".format(max_growth, max_location[0],
                                                                                            max_location[1]))
            gamma_max = growth_rate(max_location[0], max_location[1], max_growth_guess, N=Nmodes)[0].real
            logger.info("Solver at max: {:22.20e}".format(gamma_max))
            logger.info("(gamma_max - gamma_guess)/gamma_guess = {:22.20e}".format(
                (max_growth - max_growth_guess) / max_growth_guess))
            kdist = np.sqrt((max_guess[0] - max_location[0]) ** 2 + (max_guess[1] - max_location[1]) ** 2)
            logger.info("kdist from guess: {:22.20e}".format(kdist))
            logger.info("Message from optimzier: {}".format(max_growth_opt.message))
        else:
            logger.info("Optimizer failed! Message from optimzier: {}".format(max_growth_opt.message))

if CW.rank == 0:
    output_file.close()