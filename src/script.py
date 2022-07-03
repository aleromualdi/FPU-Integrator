from classes import FPU
import matplotlib.pyplot as plt


"""
System inputs
"""

NUM_ATOMS = 32  # number of particles equals to N in FPUT equations
T_MAX = 40000  # maximum time of simulation
INTEGRATION_TIME_STEP = 0.6
ALPHA = 1.0  # non-linearity coefficient (quadratic term), alpha must be <= 0.25
BETA = 2.0  # non-linearity coefficient (cubic term), beta can be 0.3, 1, 3 etc.

# Initial mode inputs
INITIAL_MODE_NUMBER = 1  # = 1, ...NUM_ATOMS
INITIAL_MODE_AMPLITUDE = 1.0
NUM_MODES = 3  # number of modes to be observed

"""
FPU
"""

fpu = FPU(
    num_atoms=NUM_ATOMS,
    num_modes=NUM_MODES,
    initial_mode_number=INITIAL_MODE_NUMBER,
    initial_mode_amplitude=INITIAL_MODE_AMPLITUDE,
    t_step=INTEGRATION_TIME_STEP,
    t_max=T_MAX,
    alpha=ALPHA,
    beta=BETA,
)

time_steps, mode_energies = fpu.run()


"""
Plots
"""

plt.plot(time_steps, mode_energies["mode_1"], "k-", linewidth=1.5, label="Mode 1")
plt.plot(time_steps, mode_energies["mode_2"], "g-", linewidth=1.5, label="Mode 2")
plt.plot(time_steps, mode_energies["mode_3"], "r-", linewidth=1.5, label="Mode 3")


plt.xlabel("$t$ ")
plt.ylabel("Energy ")

# plt.ylim(0., 50)
plt.xlim(0.0, T_MAX)
legend = plt.legend(loc="upper right", shadow=True, fontsize="x-small")
plt.savefig("../outputs/fput1.pdf")
plt.show()

# sys.exit()
