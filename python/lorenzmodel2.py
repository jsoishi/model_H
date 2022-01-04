import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

def lorenz(rho, sigma):
    # rho = 10
    # sigma = 10
    # beta = 8.0 / 3.0


    def f(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (rho * y - x + rho*z*y), -x * z + x - y, -2 * z + x * y  # Derivatives

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 40.0, 0.01)

    states = odeint(f, state0, t)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()


def main():
    lorenz(10, 10)
    lorenz(6, 30)
    lorenz(6, 30)
    lorenz(15, 30)

main()
