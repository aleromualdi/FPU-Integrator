import numpy as np
from typing import Tuple
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
        self.n_time_steps = int(self.t_max / self.t_step)

        self.alpha = alpha
        self.beta = beta

        self._initialise_displacements()
        self._initialise_momenta()
        self._initial_mode_energies()

    def _initialise_displacements(self):
        """Initialise displacements in initial mode."""

        q = np.zeros(shape=(self.num_atoms, self.n_time_steps))

        coef = np.sqrt(2.0 / (self.num_atoms + 1))
        for i in range(0, self.num_atoms):
            # formula says i * k * pi but array q starts with index 0
            const = (i + 1) * self.initial_mode_number * np.pi
            sin_arg = const / (self.num_atoms + 1)
            q[i][0] = self.initial_mode_amplitude * coef * np.sin(sin_arg)
        self.q = q

    def _initialise_momenta(self):
        self.p = np.zeros(shape=(self.num_atoms, self.n_time_steps))

    def _initial_mode_energies(self):

        mode_energies = np.zeros(shape=(self.num_modes, self.n_time_steps))

        for mode_number in range(self.num_modes):
            mode_energies[mode_number][0] = self._compute_mode_energy(
                self.q[:, 0], self.p[:, 0], mode_number
            )

        self.mode_energies = mode_energies

    def run(
        self, method="verlet"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run Fermi-Pasta-Ulam integrator.

        Returns
        -------
        array of time steps and nd array of mode energies
        """

        times = [0]
        current_time = 0
        for t in tqdm(range(self.n_time_steps - 1)):
            if method == "verlet":
                self.q[:, t + 1], self.p[:, t + 1] = self._perform_verlet_step(
                    self.q[:, t], self.p[:, t]
                )
            elif method == "runge-kutta":
                self.q[:, t + 1], self.p[:, t + 1] = self._perform_runge_kutta_step(
                    self.q[:, t], self.p[:, t]
                )

            current_time += self.t_step
            times.append(current_time)

            for mode_idx in range(self.num_modes):
                mode_number = mode_idx + 1
                mode_energy = self._compute_mode_energy(
                    self.q[:, t + 1], self.p[:, t + 1], mode_number
                )
                self.mode_energies[mode_idx][t + 1] = mode_energy

        return np.array(times), self.q, self.p, self.mode_energies

    def _perform_verlet_step(
        self, q: np.array, p: np.array
    ) -> Tuple[np.array, np.array]:
        """Perform one step of the velocity Verlet algorithm."""

        p_half = p + 0.5 * self.t_step * self._compute_force(q)
        q = q + self.t_step * p_half  # q at tstep
        p = p_half + 0.5 * self.t_step * self._compute_force(q)

        return q, p

    def _perform_runge_kutta_step(
        self, q: np.array, p: np.array
    ) -> Tuple[np.array, np.array]:
        """Perform 4th order Runge-Kutta algorithm."""
        k1_q = p
        k1_p = self._compute_force(q)
        k2_q = p + 0.5 * k1_q * self.t_step
        q_s = q + 0.5 * k1_p * self.t_step
        k2_p = self._compute_force(q_s)
        k3_q = p + 0.5 * k2_q * self.t_step
        q_s1 = q + 0.5 * k2_p * self.t_step
        k3_p = self._compute_force(q_s1)
        k4_q = p + k3_q + self.t_step
        q_s2 = q + k3_p * self.t_step
        k4_p = self._compute_force(q_s2)
        sumKq = k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
        sumKp = k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
        prefact = (1.0 / 6.0) * self.t_step
        q = q + prefact * sumKq
        p = p + prefact * sumKp
        return q, p

    def _compute_force(self, q: np.array) -> np.array:
        """Gives linear (Hook's law) plus nonlinear (quadratic, cubic) terms
        dictated by alpha and beta, respectively."""
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
        argMode = np.pi / (self.num_atoms + 1)
        for j in range(0, self.num_atoms):
            sin_arg = (j * mode_number * np.pi) / (self.num_atoms + 1)
            term = np.sin(sin_arg) * q[j]
            omega_mode = 2.0 * np.sin(mode_number * argMode)
            q_new[j] = coef * omega_mode * term
        q_sum = np.sum(q_new)
        qBigSq = 0.5 * q_sum ** 2

        for j in range(0, self.num_atoms):
            sin_arg = (j * mode_number * np.pi) / (self.num_atoms + 1)
            term = np.sin(sin_arg) * p[j]
            p_new[j] = coef * term
        p_sum = np.sum(p_new)
        pBigSq = 0.5 * p_sum ** 2

        return pBigSq + qBigSq
