import os

import numpy as np

from fput.integrator import FPUT_Integrator

NUM_ATOMS = 32  # number of particles equals to N in FPUT equations
INTEGRATION_TIME_STEP = 0.05
T_MAX = 20  # maximum time of simulation
NUM_MODES = 3  # number of modes to be observed
INITIAL_MODE_NUMBER = 1
INITIAL_MODE_AMPLITUDE = 10
ALPHA = 0.0
BETA_RANGE = np.arange(0.1, 3.0, 0.1, dtype=float)
OUTPUT_NAMES = ["times", "q", "p", "mode_energies"]


def main():

    # create output dataset path
    output_path = os.path.abspath(__file__ + "/../../dataset")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for beta in BETA_RANGE:

        os.makedirs(os.path.join(output_path, "%.2f" % beta))

        print("processing beta:", "%.2f" % beta)
        fpu = FPUT_Integrator(
            num_atoms=NUM_ATOMS,
            num_modes=NUM_MODES,
            initial_mode_number=INITIAL_MODE_NUMBER,
            initial_mode_amplitude=INITIAL_MODE_AMPLITUDE,
            t_step=INTEGRATION_TIME_STEP,
            t_max=T_MAX,
            alpha=ALPHA,
            beta=beta,
        )

        data = fpu.run(method="verlet")

        for x, name in zip(data, OUTPUT_NAMES):
            np.save(os.path.join(output_path, "%.2f" % beta, name), x)

    print("saved dataset to:", output_path)


if __name__ == "__main__":
    main()
