from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
from dataclasses import dataclass
from datetime import datetime, timezone

# ============================================================
#  Physical constants
#  (yes, we typed these a thousand times in our lives)
# ============================================================

hbar = 1.054571817e-34
eV_to_J = 1.602176634e-19


# ============================================================
#  Configuration dataclasses
#  Tiny knobs that we kept tweaking until things stopped lying
# ============================================================

@dataclass
class GrapheneConfig:
    """Tight-binding parameters for a honeycomb lattice (graphene-like)."""
    a: float = 2.46e-10      # lattice constant (m)
    t_eV: float = 2.8        # nearest-neighbor hopping (eV)
    tprime_eV: float = 0.0   # next-nearest-neighbor hopping (eV)
    mass_eV: float = 0.0     # sublattice mass term (eV)


@dataclass
class CoreAuditConfig:
    """Numerical parameters controlling the Dirac-point and scaling audits."""
    q_rel_default: float = 1e-3
    h_rel_default: float = 1e-9
    directions_ring: int = 64
    directions_curvature: int = 64
    berry_dirs: int = 256
    berry_radius_rel: float = 1e-9
    newton_max_iter: int = 100
    det_tol: float = 1e-24
    vf_tol_rel: float = 1e-6
    berry_tol_abs: float = 1e-6
    curvature_min_abs_bound: float = 1e-12
    scaling_R2_min: float = 0.999999
    scaling_points: int = 13


@dataclass(frozen=True)
class ReportAuditConfig:
    """Tolerance and reporting thresholds for the audit.
    These are the lines we drew in the sand and refused to cross.
    """
    periodicity_atol: float = 1e-12
    hermiticity_atol: float = 1e-12
    gauge_atol: float = 1e-12
    curvature_min_abs_bound: float = 1e-12
    scaling_R2_min: float = 0.999999
    vf_tol_rel: float = 1e-6
    berry_tol_abs: float = 1e-6
    tprime_relax_threshold: float = 0.3
    tprime_isotropy_relax: float = 5e-6
    scaling_points: int = 13


REPORT_CFG = ReportAuditConfig()


# ============================================================
#  Graphene tight-binding model (C3-consistent)
#  The core: simple Hamiltonian, endless debugging
# ============================================================

class GrapheneModel(eqx.Module):
    """Honeycomb tight-binding model with optional mass and t' terms."""

    cfg: GrapheneConfig = eqx.field(static=True)
    a1: jax.Array
    a2: jax.Array
    A: jax.Array
    b1: jax.Array
    b2: jax.Array
    a_cc: float
    deltas: jax.Array
    nnn: jax.Array
    tJ: float
    tprimeJ: float
    massJ: float

    def __init__(self, cfg: GrapheneConfig):
        # This object.__setattr__ dance is the price of using eqx + dataclasses
        object.__setattr__(self, "cfg", cfg)

        a = cfg.a
        a_cc = a / jnp.sqrt(3.0)

        # primitive real-space vectors
        a1 = jnp.array([a / 2.0, jnp.sqrt(3.0) * a / 2.0])
        a2 = jnp.array([-a / 2.0, jnp.sqrt(3.0) * a / 2.0])
        A = jnp.stack([a1, a2], axis=1)

        # reciprocal lattice (the usual 2π A^{-1})
        B = 2.0 * jnp.pi * jnp.linalg.inv(A)
        b1, b2 = B[:, 0], B[:, 1]

        # nearest neighbors (C3 symmetric, length = a_cc)
        deltas = a_cc * jnp.array([
            [0.0, -1.0],
            [jnp.sqrt(3.0) / 2.0, 0.5],
            [-jnp.sqrt(3.0) / 2.0, 0.5],
        ])

        # next-nearest neighbors (C3 symmetric)
        nnn = a * jnp.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.5, jnp.sqrt(3.0) / 2.0],
            [-0.5, -jnp.sqrt(3.0) / 2.0],
            [0.5, -jnp.sqrt(3.0) / 2.0],
            [-0.5, jnp.sqrt(3.0) / 2.0],
        ])

        # convert to Joules, because the universe insists on SI
        tJ = cfg.t_eV * eV_to_J
        tprimeJ = cfg.tprime_eV * eV_to_J
        massJ = cfg.mass_eV * eV_to_J

        object.__setattr__(self, "a1", a1)
        object.__setattr__(self, "a2", a2)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "a_cc", float(a_cc))
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "nnn", nnn)
        object.__setattr__(self, "tJ", float(tJ))
        object.__setattr__(self, "tprimeJ", float(tprimeJ))
        object.__setattr__(self, "massJ", float(massJ))

    @property
    def vF_analytic(self):
        # textbook v_F = (3/2) a_cc t / ħ
        return (1.5 * self.a_cc * self.tJ) / hbar

    def f_k(self, k):
        # off-diagonal structure factor, the heart of the Dirac cone
        return -self.tJ * jnp.sum(jnp.exp(1j * (self.deltas @ k)))

    def eps_diag(self, k):
        # diagonal term from t' (next-nearest neighbors)
        # breaks particle-hole symmetry but does NOT open a gap
        if self.tprimeJ == 0.0:
            return 0.0
        return -self.tprimeJ * jnp.sum(jnp.cos(self.nnn @ k))

    def h_k(self, k):
        # 2x2 Bloch Hamiltonian
        f = self.f_k(k)
        e = self.eps_diag(k)
        Δ = self.massJ
        return jnp.array([[e + Δ, f],
                          [jnp.conj(f), e - Δ]], dtype=jnp.complex128)

    def energies(self, k):
        # eigenvalues only; we’ve stared at these more than we’d like to admit
        return jnp.linalg.eigvalsh(self.h_k(k))

    def grad_f(self, k):
        # gradient of f(k), used for analytic v_F
        phase = jnp.exp(1j * (self.deltas @ k))
        return -self.tJ * jnp.sum((1j * self.deltas) * phase[:, None], axis=0)

    def analytic_K(self):
        # analytic K point for ideal graphene
        kx = 4.0 * jnp.pi / (3.0 * jnp.sqrt(3.0) * self.a_cc)
        return jnp.array([kx, 0.0])

    def analytic_Kprime(self):
        # analytic K' point
        kx = -4.0 * jnp.pi / (3.0 * jnp.sqrt(3.0) * self.a_cc)
        return jnp.array([kx, 0.0])


# ============================================================
#  Berry phase from band eigenvectors (gauge-fixed, unwrapped)
#  This part fought us. Hard.
# ============================================================

def berry_phase_band_loop(model: GrapheneModel, K, radius_rel, dirs, band_index=0):
    """
    Compute Berry phase along a small loop around K using eigenvectors.

    We:
    - walk on a circle in k-space,
    - fix the phase of the first component at each point,
    - accumulate overlaps,
    - unwrap phase jumps to stay in [-π, π].

    After several rounds of "why is this 4π instead of π", this is the version
    that behaves like physics instead of chaos.
    """
    a = model.cfg.a
    r = radius_rel / a
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, int(dirs), endpoint=False)
    ks = K + r * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    def eigvec(k):
        # eigenvectors are only defined up to a phase; we pin that phase
        _, v = jnp.linalg.eigh(model.h_k(k))
        v = v[:, band_index]
        phase = jnp.angle(v[0])
        v = v * jnp.exp(-1j * phase)
        return v

    vs = jax.vmap(eigvec)(ks)

    overlaps = jnp.sum(jnp.conj(vs[:-1]) * vs[1:], axis=1)
    dphi = jnp.angle(overlaps)
    # unwrap to [-π, π] so we don't accumulate fake 2π jumps
    dphi = jnp.mod(dphi + jnp.pi, 2 * jnp.pi) - jnp.pi

    overlap_close = jnp.sum(jnp.conj(vs[-1]) * vs[0])
    dphi_close = jnp.angle(overlap_close)
    dphi_close = jnp.mod(dphi_close + jnp.pi, 2 * jnp.pi) - jnp.pi

    berry = jnp.sum(dphi) + dphi_close
    return berry


# ============================================================
#  Symmetry helpers
#  Little sanity checks that saved us from silent disasters
# ============================================================

def c3_rotation_matrix():
    c = -0.5
    s = jnp.sqrt(3.0) / 2.0
    return jnp.array([[c, -s],
                      [s, c]])


def audit_c3_invariance(f_k, b1, b2, samples=6, atol=REPORT_CFG.periodicity_atol):
    R = c3_rotation_matrix()
    ks = [
        0.15 * b1 + 0.10 * b2,
        0.33 * b1 + 0.07 * b2,
        0.42 * b1 + 0.21 * b2,
        0.05 * b1 + 0.47 * b2,
        0.27 * b1 + 0.36 * b2,
        0.49 * b1 + 0.12 * b2,
    ][:int(samples)]
    for k in ks:
        lhs = jnp.abs(f_k(k))
        rhs = jnp.abs(f_k(R @ k))
        if not bool(jnp.allclose(lhs, rhs, atol=atol)):
            return False
    return True


def energies_with_tau(k, tau_shift, a1, a2, tJ, nnn, eps_diag_base, massJ):
    # same physics, different choice of origin (tau shift)
    deltas_shift = jnp.stack([tau_shift, tau_shift - a1, tau_shift - a2], axis=0)
    phase = jnp.exp(1j * (deltas_shift @ k))
    f = -tJ * jnp.sum(phase)
    e = eps_diag_base(k)
    Δ = massJ
    H = jnp.array([[e + Δ + 0.0j, f],
                   [jnp.conj(f), e - Δ + 0.0j]], dtype=jnp.complex128)
    return jnp.linalg.eigvalsh(H)


def audit_tau_gauge_invariance(a1, a2, b1, b2, tJ, nnn, eps_diag_base, massJ, atol=REPORT_CFG.gauge_atol):
    shifts = [jnp.array([0.0, 0.0]), a1, a2, a1 + a2, -a1, -a2]
    kpoints = [
        0.23 * b1 + 0.41 * b2,
        0.37 * b1 + 0.19 * b2,
        0.11 * b1 + 0.29 * b2,
    ]
    base = [energies_with_tau(k, jnp.array([0.0, 0.0]), a1, a2, tJ, nnn, eps_diag_base, massJ) for k in kpoints]
    for sh in shifts:
        cur = [energies_with_tau(k, sh, a1, a2, tJ, nnn, eps_diag_base, massJ) for k in kpoints]
        for E0, E1 in zip(base, cur):
            if not bool(jnp.allclose(E0, E1, atol=atol)):
                return False
    return True


# ============================================================
#  Dirac auditor: vF, Berry, curvature, scaling
#  This is where we turned paranoia into code
# ============================================================

class DiracAuditor(eqx.Module):
    """Auditor for Dirac physics near K/K' in the graphene model."""

    model: GrapheneModel
    cfg: CoreAuditConfig = eqx.field(static=True)

    def newton_refine(self, k0):
        """
        Newton-Raphson refinement of the Dirac point.

        We started from analytic K, but we don't trust anything blindly anymore,
        so we solve f(k) = 0 numerically and let this converge.
        """
        det_tol = self.cfg.det_tol

        def body(carry, _):
            k, alpha = carry
            f = self.model.f_k(k)
            g = self.model.grad_f(k)
            J = jnp.stack([jnp.real(g), jnp.imag(g)], axis=0)
            F = jnp.array([jnp.real(f), jnp.imag(f)])
            detJ = jnp.linalg.det(J)

            def solve(_):
                dk = jnp.linalg.solve(J, F)
                return dk, jnp.minimum(alpha * 2.0, 1.0)

            def damp(_):
                # when detJ is tiny, we back off instead of exploding
                return jnp.zeros_like(k), alpha * 0.5

            dk, new_alpha = lax.cond(jnp.abs(detJ) >= det_tol, solve, damp, None)
            return (k - new_alpha * dk, new_alpha), None

        (k_final, _), _ = lax.scan(body, (k0, 1.0), None, length=self.cfg.newton_max_iter)
        return k_final

    def vF_from_grad(self, K):
        g = self.model.grad_f(K)
        g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))
        return g_norm / (jnp.sqrt(2.0) * hbar)

    def vF_ring_at_q(self, K, q_rel):
        """
        Estimate v_F by finite differences on a ring around K.

        This is the "let's see if the gradient story survives numerics" check.
        """
        q_abs = q_rel / self.model.a_cc
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, self.cfg.directions_ring, endpoint=False)
        us = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        def d_dir(u):
            fp = self.model.f_k(K + q_abs * u)
            fm = self.model.f_k(K - q_abs * u)
            return (fp - fm) / (2.0 * q_abs)

        d_vals = jax.vmap(d_dir)(us)
        mags2 = jnp.abs(d_vals) ** 2
        grad_f_norm = jnp.sqrt(2.0 * jnp.mean(mags2))
        vF = grad_f_norm / (jnp.sqrt(2.0) * hbar)
        spread = jnp.std(jnp.abs(d_vals) / hbar)
        return vF, spread

    def vF_ring_fit(self, K):
        return self.vF_ring_at_q(K, self.cfg.q_rel_default)

    def berry_discrete(self, K, radius_rel, dirs):
        """
        Berry phase from arg f(k) winding.

        This one behaved nicely from the start, unlike the eigenvector version.
        """
        a = self.model.cfg.a
        r = radius_rel / a
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, dirs, endpoint=False)
        ks = K + r * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        phases = jax.vmap(lambda k: jnp.angle(self.model.f_k(k)))(ks)
        diffs = jnp.diff(jnp.concatenate([phases, phases[:1]]))
        diffs = jnp.arctan2(jnp.sin(diffs), jnp.cos(diffs))
        wind = jnp.sum(diffs)
        return 0.5 * wind, wind

    def berry_adaptive(self, K):
        base_r = self.cfg.berry_radius_rel
        base_n = self.cfg.berry_dirs
        candidates = [
            (base_r, base_n),
            (base_r, base_n * 2),
            (base_r * 2.0, base_n),
            (base_r / 2.0, base_n * 2),
        ]
        results = []
        for r, n in candidates:
            g, w = self.berry_discrete(K, r, n)
            results.append((g, w, r, n))
        winds = jnp.array([w for (_, w, _, _) in results])
        idx = int(jnp.argmin(jnp.abs(jnp.abs(winds) - 2.0 * jnp.pi)))
        return results[idx], results

    def curvature(self, K):
        """
        Second derivative of the upper band at K along many directions.

        In a perfect Dirac cone this should be ~0. We made sure it's not
        "zero by accident" but actually below a physically meaningful floor.
        """
        a = self.model.cfg.a
        h_rel = self.cfg.h_rel_default
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, self.cfg.directions_curvature, endpoint=False)
        us = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        def curv_one(u):
            def E(k):
                return jnp.max(self.model.energies(k))

            h1 = h_rel / a
            h2 = (h_rel / 2.0) / a
            E0 = E(K)
            k1 = (E(K + h1 * u) - 2.0 * E0 + E(K - h1 * u)) / (h1 * h1)
            k2 = (E(K + h2 * u) - 2.0 * E0 + E(K - h2 * u)) / (h2 * h2)
            k_true = (4.0 * k2 - k1) / 3.0
            err = jnp.abs(k2 - k_true)
            return jnp.abs(k_true), err

        ks, es = jax.vmap(curv_one)(us)
        return jnp.max(ks), jnp.max(es)

    def scaling_t(self):
        """
        Check v_F ∝ t by scanning t and fitting a line.

        If this fails, something is deeply wrong with the model or our sanity.
        """
        tJ0 = self.model.tJ
        a_cc = self.model.a_cc
        deltas = self.model.deltas
        ts = jnp.linspace(0.5 * tJ0, 1.5 * tJ0, self.cfg.scaling_points)

        def vF_for_t(tJ):
            def f_k_t(k):
                return -tJ * jnp.sum(jnp.exp(1j * (deltas @ k)))

            def grad_f_t(k):
                phase = jnp.exp(1j * (deltas @ k))
                return -tJ * jnp.sum((1j * deltas) * phase[:, None], axis=0)

            K0 = jnp.array([4.0 * jnp.pi / (3.0 * jnp.sqrt(3.0) * a_cc), 0.0])

            def body(carry, _):
                k, alpha = carry
                f = f_k_t(k)
                g = grad_f_t(k)
                J = jnp.stack([jnp.real(g), jnp.imag(g)], axis=0)
                F = jnp.array([jnp.real(f), jnp.imag(f)])
                detJ = jnp.linalg.det(J)

                def solve(_):
                    dk = jnp.linalg.solve(J, F)
                    return dk, jnp.minimum(alpha * 2.0, 1.0)

                def damp(_):
                    return jnp.zeros_like(k), alpha * 0.5

                dk, new_alpha = lax.cond(jnp.abs(detJ) >= 1e-24, solve, damp, None)
                return (k - new_alpha * dk, new_alpha), None

            (Kf, _), _ = lax.scan(body, (K0, 1.0), None, length=100)
            g = grad_f_t(Kf)
            g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))
            return g_norm / (jnp.sqrt(2.0) * hbar)

        vfs = jax.vmap(vF_for_t)(ts)
        c = (ts @ vfs) / (ts @ ts)
        residuals = vfs - c * ts
        rss = jnp.sum(residuals ** 2)
        tss = jnp.sum((vfs - jnp.mean(vfs)) ** 2)
        r2 = 1.0 - rss / tss
        slope_theory = (1.5 * a_cc) / hbar
        return c, r2, slope_theory

    def scaling_a(self):
        """
        Check v_F ∝ a by scanning lattice constant.

        Another sanity check: if this breaks, we know where to cry.
        """
        a0 = self.model.cfg.a
        tJ = self.model.tJ
        a_vals = jnp.linspace(0.5 * a0, 1.5 * a0, self.cfg.scaling_points)

        def vF_for_a(a):
            a_cc = a / jnp.sqrt(3.0)
            deltas = a_cc * jnp.array([
                [0.0, -1.0],
                [jnp.sqrt(3.0) / 2.0, 0.5],
                [-jnp.sqrt(3.0) / 2.0, 0.5],
            ])

            def f_k_a(k):
                return -tJ * jnp.sum(jnp.exp(1j * (deltas @ k)))

            def grad_f_a(k):
                phase = jnp.exp(1j * (deltas @ k))
                return -tJ * jnp.sum((1j * deltas) * phase[:, None], axis=0)

            K0 = jnp.array([4.0 * jnp.pi / (3.0 * jnp.sqrt(3.0) * a_cc), 0.0])

            def body(carry, _):
                k, alpha = carry
                f = f_k_a(k)
                g = grad_f_a(k)
                J = jnp.stack([jnp.real(g), jnp.imag(g)], axis=0)
                F = jnp.array([jnp.real(f), jnp.imag(f)])
                detJ = jnp.linalg.det(J)

                def solve(_):
                    dk = jnp.linalg.solve(J, F)
                    return dk, jnp.minimum(alpha * 2.0, 1.0)

                def damp(_):
                    return jnp.zeros_like(k), alpha * 0.5

                dk, new_alpha = lax.cond(jnp.abs(detJ) >= 1e-24, solve, damp, None)
                return (k - new_alpha * dk, new_alpha), None

            (Kf, _), _ = lax.scan(body, (K0, 1.0), None, length=100)
            g = grad_f_a(Kf)
            g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))
            return g_norm / (jnp.sqrt(2.0) * hbar)

        vfs = jax.vmap(vF_for_a)(a_vals)
        c = (a_vals @ vfs) / (a_vals @ a_vals)
        residuals = vfs - c * a_vals
        rss = jnp.sum(residuals ** 2)
        tss = jnp.sum((vfs - jnp.mean(vfs)) ** 2)
        r2 = 1.0 - rss / tss
        slope_theory = (1.5 / jnp.sqrt(3.0)) * tJ / hbar
        return c, r2, slope_theory

    def linear_regime_radius(self, K):
        """
        Find how far in q we can go before v_F deviates too much.

        This is basically asking: "how honest is the Dirac cone numerically?"
        """
        vF_ref = self.vF_from_grad(K)
        qs_rel = jnp.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 1e-6, 5e-6, 1e-5])

        def dev(q_rel):
            vF_q, _ = self.vF_ring_at_q(K, q_rel)
            return jnp.abs(vF_q - vF_ref)

        devs = jax.vmap(dev)(qs_rel)
        mask = devs < (self.cfg.vf_tol_rel * vF_ref)
        idx = jnp.argmax(mask)
        return jnp.where(mask.any(), qs_rel[idx], 0.0)

    def run_core(self):
        # Find Dirac points
        K = self.newton_refine(self.model.analytic_K())
        Kp = self.newton_refine(self.model.analytic_Kprime())

        # Fermi velocities
        vf_grad_K = self.vF_from_grad(K)
        vf_grad_Kp = self.vF_from_grad(Kp)
        vf_fit_K, spread_K = self.vF_ring_fit(K)
        vf_fit_Kp, spread_Kp = self.vF_ring_fit(Kp)

        # Berry from f(k)
        (gK, wK, rK, nK), _ = self.berry_adaptive(K)
        (gKp, wKp, rKp, nKp), _ = self.berry_adaptive(Kp)

        # Berry from eigenvectors
        gamma_band_K = berry_phase_band_loop(self.model, K, rK, int(nK), band_index=0)
        gamma_band_Kp = berry_phase_band_loop(self.model, Kp, rKp, int(nKp), band_index=0)

        # Curvature
        curvK, errK = self.curvature(K)
        curvKp, errKp = self.curvature(Kp)

        # Scaling
        slope_t, R2_t, slope_t_theory = self.scaling_t()
        slope_a, R2_a, slope_a_theory = self.scaling_a()

        # Linear regime
        linR_K = self.linear_regime_radius(K)
        linR_Kp = self.linear_regime_radius(Kp)

        return {
            "K": K,
            "Kp": Kp,
            "vF_analytic": self.model.vF_analytic,
            "vf_grad_K": vf_grad_K,
            "vf_grad_Kp": vf_grad_Kp,
            "vf_fit_K": vf_fit_K,
            "vf_fit_Kp": vf_fit_Kp,
            "spread_K": spread_K,
            "spread_Kp": spread_Kp,
            "berry_K": (gK, wK, rK, nK),
            "berry_Kp": (gKp, wKp, rKp, nKp),
            "berry_band_K": gamma_band_K,
            "berry_band_Kp": gamma_band_Kp,
            "curvature_K": (curvK, errK),
            "curvature_Kp": (curvKp, errKp),
            "scaling_t": (slope_t, R2_t, slope_t_theory),
            "scaling_a": (slope_a, R2_a, slope_a_theory),
            "linear_radius_K": linR_K,
            "linear_radius_Kp": linR_Kp,
        }


# ============================================================
#  Reporting helpers
# ============================================================

def format_float(x):
    return f"{float(x):.6e}"


# ============================================================
#  Full audit
#  The big “is this thing physically honest?” button
# ============================================================

def run_full_audit(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, mass_eV=0.1):
    model = GrapheneModel(GrapheneConfig(a=a, t_eV=t_eV, tprime_eV=tprime_eV, mass_eV=mass_eV))
    auditor = DiracAuditor(model, CoreAuditConfig())
    core = auditor.run_core()

    tJ = model.tJ
    tprimeJ = model.tprimeJ
    massJ = model.massJ
    tprime_ratio = float(tprime_eV / t_eV) if t_eV != 0 else 0.0

    a1 = model.a1
    a2 = model.a2
    b1 = model.b1
    b2 = model.b2
    a_cc = model.a_cc
    deltas = model.deltas
    nnn = model.nnn

    # nearest-neighbor sanity
    delta_lengths = jnp.linalg.norm(deltas, axis=1)
    delta_ok = bool(jnp.allclose(delta_lengths, a_cc, atol=1e-14))

    K = core["K"]
    Kp = core["Kp"]

    fK = model.f_k(K)
    fKp = model.f_k(Kp)
    dirac_K = bool(jnp.abs(fK) < 1e-12 * tJ)
    dirac_Kp = bool(jnp.abs(fKp) < 1e-12 * tJ)

    ktest = 0.1 * b1 + 0.1 * b2
    H = model.h_k(ktest)
    hermiticity = bool(jnp.allclose(H, H.conj().T, atol=REPORT_CFG.hermiticity_atol))

    periodic_b1 = bool(jnp.allclose(model.f_k(ktest + b1), model.f_k(ktest), atol=REPORT_CFG.periodicity_atol))
    periodic_b2 = bool(jnp.allclose(model.f_k(ktest + b2), model.f_k(ktest), atol=REPORT_CFG.periodicity_atol))

    c3_ok = audit_c3_invariance(model.f_k, b1, b2)
    tau_ok = audit_tau_gauge_invariance(a1, a2, b1, b2, tJ, nnn, model.eps_diag, massJ)

    vF_analytic = float(core["vF_analytic"])
    vf_grad_K = float(core["vf_grad_K"])
    vf_grad_Kp = float(core["vf_grad_Kp"])
    vf_fit_K = float(core["vf_fit_K"])
    vf_fit_Kp = float(core["vf_fit_Kp"])
    vf_spread_K = float(core["spread_K"])
    vf_spread_Kp = float(core["spread_Kp"])

    gK, wK, rK, nK = core["berry_K"]
    gKp, wKp, rKp, nKp = core["berry_Kp"]
    gamma_band_K = float(core["berry_band_K"])
    gamma_band_Kp = float(core["berry_band_Kp"])

    curvK, errK = core["curvature_K"]
    curvKp, errKp = core["curvature_Kp"]

    slope_t_fit, R2_t, slope_t_theory = core["scaling_t"]
    slope_a_fit, R2_a, slope_a_theory = core["scaling_a"]

    linR_K = float(core["linear_radius_K"])
    linR_Kp = float(core["linear_radius_Kp"])

    # Assertions: the part where we refuse to accept quiet nonsense
    assert delta_ok, "Nearest-neighbor lengths do not match a_cc."
    assert hermiticity, "Hamiltonian is not Hermitian."
    assert periodic_b1 and periodic_b2, "Bloch periodicity failed."
    assert dirac_K and dirac_Kp, "Dirac condition failed (f(K) ≠ 0)."
    assert c3_ok, "C3 rotational invariance failed."
    assert tau_ok, "Tau gauge invariance failed."

    if massJ != 0.0:
        # massive Dirac: gap should be 2Δ at K
        def E_plus(k): return jnp.max(model.energies(k))
        def E_minus(k): return jnp.min(model.energies(k))

        gap_K = float(E_plus(K) - E_minus(K))
        rel_err_gap = abs(gap_K - 2.0 * massJ) / (2.0 * massJ)
        assert rel_err_gap < 1e-6, "Gap mismatch at K."

        print()
        print("-- Massive gap check --")
        print(f"gap(K) actual      = {gap_K / eV_to_J:.6f} eV")
        print(f"gap(K) expected    = {2.0 * massJ / eV_to_J:.6f} eV")
        print(f"gap rel error      = {rel_err_gap:.3e}")

    a = model.cfg.a
    curv_floor = REPORT_CFG.curvature_min_abs_bound * tJ / (a * a)
    assert float(curvK) < max(10.0 * float(errK), curv_floor)
    assert float(curvKp) < max(10.0 * float(errKp), curv_floor)

    assert abs(float(slope_t_fit - slope_t_theory) / float(slope_t_theory)) < REPORT_CFG.vf_tol_rel
    assert float(R2_t) > REPORT_CFG.scaling_R2_min
    assert abs(float(slope_a_fit - slope_a_theory) / float(slope_a_theory)) < REPORT_CFG.vf_tol_rel
    assert float(R2_a) > REPORT_CFG.scaling_R2_min

    now = datetime.now(timezone.utc).isoformat()

    print(f"=== Graphene Dirac Audit (massive Dirac, full symmetry & scaling checks) ===")
    print(f"timestamp: {now}")
    print(f"a = {model.cfg.a:.2e} m, t = {model.cfg.t_eV} eV, t' = {model.cfg.tprime_eV} eV, Δ = {model.cfg.mass_eV} eV")
    print(f"a_cc = {model.a_cc:.16e} m, t'/t = {tprime_ratio:.3f}")
    print()
    print("-- Fermi velocity --")
    print(f"vF_analytic        = {format_float(vF_analytic)} m/s")
    print(f"vF_grad_K          = {format_float(vf_grad_K)} m/s")
    print(f"vF_grad_K'         = {format_float(vf_grad_Kp)} m/s")
    print(f"vF_ring_fit_K      = {format_float(vf_fit_K)} m/s")
    print(f"vF_ring_fit_K'     = {format_float(vf_fit_Kp)} m/s")
    print(f"spread_K           = {format_float(vf_spread_K)} m/s")
    print(f"spread_K'          = {format_float(vf_spread_Kp)} m/s")
    print()
    print("-- Berry phase --")
    print(f"gamma_K (from f)   = {format_float(gK)} rad")
    print(f"gamma_K' (from f)  = {format_float(gKp)} rad")
    print(f"gamma_K (band)     = {format_float(gamma_band_K)} rad")
    print(f"gamma_K' (band)    = {format_float(gamma_band_Kp)} rad")
    print(f"winding_K          = {format_float(wK)}")
    print(f"winding_K'         = {format_float(wKp)}")
    print(f"selected (K):   radius_rel={float(rK):.3e}, dirs={int(nK)}")
    print(f"selected (K'):  radius_rel={float(rKp):.3e}, dirs={int(nKp)}")
    print()
    print("-- Dirac points & symmetry --")
    print(f"Dirac at K         = {dirac_K}, |f(K)|/t = {float(jnp.abs(fK)/tJ):.3e}")
    print(f"Dirac at K'        = {dirac_Kp}, |f(K')|/t = {float(jnp.abs(fKp)/tJ):.3e}")
    print(f"K coordinates      = ({float(K[0])}, {float(K[1])})")
    print(f"K' coordinates     = ({float(Kp[0])}, {float(Kp[1])})")
    print(f"Hermitian          = {hermiticity}")
    print(f"Periodicity b1/b2  = {periodic_b1}, {periodic_b2}")
    print(f"C3 invariance      = {c3_ok}")
    print(f"Tau gauge invariance= {tau_ok}")
    print()
    print("-- Curvature & linear regime --")
    print(f"curvature_max_K    = {format_float(curvK)} J")
    print(f"curvature_err_K    = {format_float(errK)} J")
    print(f"curvature_max_K'   = {format_float(curvKp)} J")
    print(f"curvature_err_K'   = {format_float(errKp)} J")
    print(f"linear_radius_rel_K    = {format_float(linR_K)}")
    print(f"linear_radius_rel_K'   = {format_float(linR_Kp)}")
    print()
    print("-- Scaling --")
    print(f"slope_t_fit        = {format_float(slope_t_fit)}")
    print(f"slope_t_theory     = {format_float(slope_t_theory)}")
    print(f"R2_t               = {float(R2_t):.6f}")
    print(f"slope_a_fit        = {format_float(slope_a_fit)}")
    print(f"slope_a_theory     = {format_float(slope_a_theory)}")
    print(f"R2_a               = {float(R2_a):.6f}")
    print()
    print("-- Mass & gap --")
    print(f"Δ (J)              = {format_float(massJ)}")
    print(f"Expected gap       = {format_float(2.0 * massJ)} J")

    return core


# ============================================================
#  Extra diagnostics
#  These are the “show me the numbers” helpers
# ============================================================

def vf_ring_diagnostics():
    model = GrapheneModel(GrapheneConfig(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, mass_eV=0.1))
    auditor = DiracAuditor(model, CoreAuditConfig())
    K = auditor.newton_refine(model.analytic_K())
    Kp = auditor.newton_refine(model.analytic_Kprime())

    vf_grad_K = auditor.vF_from_grad(K)
    vf_grad_Kp = auditor.vF_from_grad(Kp)
    vf_ring_K, spread_K = auditor.vF_ring_fit(K)
    vf_ring_Kp, spread_Kp = auditor.vF_ring_fit(Kp)

    print("=== vF Diagnostics (ring-based vs gradient) ===")
    print(f"vF analytic  = {model.vF_analytic:.6f}")
    print(f"vF grad K    = {vf_grad_K:.6f}")
    print(f"vF ring K    = {vf_ring_K:.6f}")
    print(f"spread K     = {spread_K:.6e}")
    print()
    print(f"vF grad K'   = {vf_grad_Kp:.6f}")
    print(f"vF ring K'   = {vf_ring_Kp:.6f}")
    print(f"spread K'    = {spread_Kp:.6e}")


def linear_regime_debug():
    model = GrapheneModel(GrapheneConfig(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, mass_eV=0.1))
    auditor = DiracAuditor(model, CoreAuditConfig())
    K = auditor.newton_refine(model.analytic_K())
    vF_ref = auditor.vF_from_grad(K)

    qs_rel = jnp.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 1e-6, 5e-6, 1e-5])

    print("=== Linear Regime Debug (vF_ring_at_q) ===")
    for qr in qs_rel:
        vF_q, spread_q = auditor.vF_ring_at_q(K, qr)
        dev = jnp.abs(vF_q - vF_ref)
        print(
            f"q_rel={qr:.1e}, vF(q)={vF_q:.6e}, dev={dev:.6e}, "
            f"dev_rel={dev/vF_ref:.3e}, spread={spread_q:.3e}"
        )


# ============================================================
#  “Answering the criticism” tests
#  These are literally us arguing with our own doubts
# ============================================================

def berry_massless_radius_sweep():
    """
    Test 1: massless case, Berry phase vs radius.

    We know eigenvectors are ill-defined exactly at the Dirac point.
    This sweep shows how gamma_band behaves as we move the loop radius.
    """
    print("=== Berry eigenvector vs radius (massless) ===")
    cfg = GrapheneConfig(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, mass_eV=0.0)
    model = GrapheneModel(cfg)
    auditor = DiracAuditor(model, CoreAuditConfig())
    K = auditor.newton_refine(model.analytic_K())

    radii = jnp.array([1e-10, 5e-10, 1e-9, 5e-9, 1e-8])
    dirs = 256

    for r in radii:
        gamma_band = float(berry_phase_band_loop(model, K, r, int(dirs), band_index=0))
        gamma_f, wind = auditor.berry_discrete(K, r, int(dirs))
        print(
            f"radius_rel={r:.1e} -> "
            f"gamma_band={gamma_band:.6f}, "
            f"gamma_f={float(gamma_f):.6f}, "
            f"winding={float(wind):.6f}"
        )


def tprime_dirac_energy_scan():
    """
    Test 2: scan t' and check Dirac energy & gap.

    t' should shift the Dirac point energy but must NOT open a gap
    as long as the mass term is zero.
    """
    print("\n=== t' scan: Dirac energy & gap (massless) ===")
    a = 2.46e-10
    t_eV = 2.8
    tprimes = jnp.array([0.0, 0.1, 0.2, 0.3])

    for tp in tprimes:
        cfg = GrapheneConfig(a=a, t_eV=t_eV, tprime_eV=float(tp), mass_eV=0.0)
        model = GrapheneModel(cfg)
        auditor = DiracAuditor(model, CoreAuditConfig())
        K = auditor.newton_refine(model.analytic_K())

        E = model.energies(K)
        E_min = float(jnp.min(E))
        E_max = float(jnp.max(E))
        gap = E_max - E_min

        print(
            f"t'={float(tp):.2f} eV -> "
            f"E_min={E_min/eV_to_J:.6f} eV, "
            f"E_max={E_max/eV_to_J:.6f} eV, "
            f"gap={gap/eV_to_J:.6e} eV"
        )


def newton_refine_drift():
    """
    Test 3: how far does Newton refinement move from analytic K?

    In the ideal case, the drift is tiny. But this test is here
    for the day we start deforming the lattice and want to see
    how much the Dirac point actually shifts.
    """
    print("\n=== Newton refine drift from analytic K ===")
    cfg = GrapheneConfig(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, mass_eV=0.0)
    model = GrapheneModel(cfg)
    auditor = DiracAuditor(model, CoreAuditConfig())

    K_analytic = model.analytic_K()
    K_refined = auditor.newton_refine(K_analytic)

    dK = K_refined - K_analytic
    print(f"K_analytic = ({float(K_analytic[0]):.6e}, {float(K_analytic[1]):.6e})")
    print(f"K_refined  = ({float(K_refined[0]):.6e}, {float(K_refined[1]):.6e})")
    print(f"|ΔK|       = {float(jnp.linalg.norm(dK)):.6e} 1/m")


# ============================================================
#  Main
#  This is the “run everything and see if it screams” entry point
# ============================================================

if __name__ == "__main__":
    # Core audit: massive Dirac, full symmetry & scaling checks
    core = run_full_audit(mass_eV=0.1)

    # Extra diagnostics for v_F and linear regime
    vf_ring_diagnostics()
    linear_regime_debug()

    # “Answer the criticism” tests:
    # 1) massless Berry vs radius
    # 2) t' only shifts energy, no gap
    # 3) Newton refinement drift from analytic K
    berry_massless_radius_sweep()
    tprime_dirac_energy_scan()
    newton_refine_drift()
