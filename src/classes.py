from typing import Tuple
import numpy as np
from tqdm import tqdm


class FPU(object):
    """Fermi-Pasta-Ulam integrator based on Verlet algorithm [...]"""

    def __init__(
        self,
        num_atoms: int,
        num_modes: int,
        initial_mode_number: int,
        initial_mode_amplitude: float,
        t_step: float,
        t_max: int,
        alpha: float,
        beta: float,
    ):
        self.num_atoms = num_atoms
        self.num_modes = num_modes
        self.initial_mode_number = initial_mode_number
        self.initial_mode_amplitude = initial_mode_amplitude
        self.t_step = t_step
        self.t_max = t_max
        self.alpha = alpha
        self.beta = beta

        # initialisation of displacements and velocities (conjugate moments)
        self.q = self._initialise_displacements()
        self.p = np.zeros(shape=(num_atoms,))

    def _initialise_displacements(self) -> np.array:
        """Initialise displacements in a given mode number."""

        q = np.zeros(shape=(self.num_atoms,))

        coef = np.sqrt(2.0 / (self.num_atoms + 1))
        for i in range(0, self.num_atoms):
            # formula says i * k * pi but array q starts with index 0
            const = (i + 1) * self.initial_mode_number * np.pi
            sin_arg = const / (self.num_atoms + 1)
            q[i] = self.initial_mode_amplitude * coef * np.sin(sin_arg)
        return q

    def _compute_force(self, q: np.array) -> np.array:
        """Gives linear (Hook's law) plus nonlinear (quadratic, cubic) terms dictated by alpha and beta,
         respectively."""
        force = np.zeros(shape=(self.num_atoms,))

        n_f = self.num_atoms - 1
        n_f_minus = n_f - 1

        force[0] = (
            q[1]
            - 2.0 * q[0]
            + self.alpha * (q[1] - q[0]) ** 2
            - self.alpha * q[0] ** 2
            + self.beta * (q[1] - q[0]) ** 3
            - self.beta * q[0] ** 3
        )
        force[n_f] = (
            q[n_f_minus]
            - 2.0 * q[n_f]
            + self.alpha * q[n_f] ** 2
            - self.alpha * (q[n_f] - q[n_f_minus]) ** 2
            + self.beta * (-q[n_f]) ** 3
            - self.beta * (q[n_f] - q[n_f_minus]) ** 3
        )
        for i in range(1, n_f):
            force[i] = (
                q[i + 1]
                + q[i - 1]
                - 2.0 * q[i]
                + self.alpha * (q[i + 1] - q[i]) ** 2
                - self.alpha * (q[i] - q[i - 1]) ** 2
                + self.beta * (q[i + 1] - q[i]) ** 3
                - self.beta * (q[i] - q[i - 1]) ** 3
            )

        return force

    def _compute_mode_energy(self, q: np.array, p: np.array, mode_number: int) -> float:
        coef = np.sqrt(2.0 / (self.num_atoms + 1))
        q_new = np.zeros(shape=self.num_atoms)
        p_new = np.zeros(shape=self.num_atoms)

        for j in range(0, self.num_atoms):
            sin_arg = (j * mode_number * np.pi) / (self.num_atoms + 1)
            term = np.sin(sin_arg) * q[j]
            q_new[j] = coef * term
        q_sum = np.sum(q_new)
        qBigSq = 0.5 * q_sum ** 2

        for j in range(0, self.num_atoms):
            sin_arg = (j * mode_number * np.pi) / (self.num_atoms + 1)
            term = np.sin(sin_arg) * p[j]
            p_new[j] = coef * term
        p_sum = np.sum(p_new)
        pBigSq = 0.5 * p_sum ** 2

        return pBigSq + qBigSq

    def _perform_verlet_step(
        self, q: np.array, p: np.array
    ) -> Tuple[np.array, np.array]:
        """Perform one step of the velocity Verlet algorithm."""

        p_half = p + 0.5 * self.t_step * self._compute_force(q)
        q = q + self.t_step * p_half  # q at tstep
        p = p_half + 0.5 * self.t_step * self._compute_force(q)

        return q, p

    def run(self) -> Tuple[np.array, np.array]:
        """Run Fermi-Pasta-Ulam integrator.

        Returns
        -------
        array of time steps and nd array of mode energies
        """
        time_steps = []
        mode_energies = {}
        for mode_num in range(1, self.num_modes + 1):
            mode_energies["mode_" + str(mode_num)] = []

        for tstep in tqdm(range(1, self.t_max + 1)):
            # print 'time-step', tstep
            # velocity Verlet, do not change the order of the statements
            p_half = self.p + 0.5 * self.t_step * self._compute_force(self.q)
            self.q = self.q + self.t_step * p_half  # q at tstep
            self.p = p_half + 0.5 * self.t_step * self._compute_force(self.q)

            time_steps.append(tstep)

            for mode_num in range(1, self.num_modes + 1):
                mode_energy = self._compute_mode_energy(self.q, self.p, mode_num)
                mode_energies["mode_" + str(mode_num)].append(mode_energy)

        return np.array(time_steps), mode_energies
