<svg width="460" height="140" viewBox="0 0 460 140" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#020617"/>
    </linearGradient>
    <linearGradient id="grad-hex" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect x="0" y="0" width="460" height="140" fill="url(#grad-bg)"/>

  <!-- Honeycomb / k-space motif -->
  <g transform="translate(80,70) scale(1.1)">
    <polygon points="0,-28 24,-14 24,14 0,28 -24,14 -24,-14"
             fill="none" stroke="url(#grad-hex)" stroke-width="2.2"/>
    <circle cx="0" cy="-14" r="3.2" fill="#f97316"/>
    <circle cx="0" cy="14" r="3.2" fill="#22c55e"/>
    <circle cx="24" cy="0" r="2.6" fill="#38bdf8"/>
    <circle cx="-24" cy="0" r="2.6" fill="#38bdf8"/>
    <circle cx="12" cy="21" r="2.4" fill="#a855f7"/>
    <circle cx="-12" cy="-21" r="2.4" fill="#a855f7"/>
  </g>

  <!-- Title -->
  <text x="150" y="60" fill="#e5e7eb"
        font-family="SF Mono, Menlo, Consolas, monospace"
        font-size="18" letter-spacing="0.08em">
    Graphene Massive Dirac Auditor
  </text>

  <!-- Subtitle -->
  <text x="150" y="84" fill="#9ca3af"
        font-family="SF Mono, Menlo, Consolas, monospace"
        font-size="11" letter-spacing="0.20em">
    SYMMETRY • BERRY PHASE • SCALING • JAX
  </text>
</svg>




[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-orange.svg)](https://github.com/google/jax)
[![Field: Computational Physics](https://img.shields.io/badge/Field-Condensed--Matter-blue)](#)

## 📦 Overview

This repository contains a monolithic **JAX-accelerated implementation** of a fully-audited massive Dirac model on the honeycomb lattice. It serves as a symmetry-aware, audit-grade reference for researchers exploring tight-binding physics in graphene.

The script performs a comprehensive physical audit, including:
*   **Dirac point refinement** and precision localization.
*   **Fermi velocity extraction** via multiple independent numerical methods.
*   **Berry phase evaluation** using both structure factors and band eigenvectors.
*   **Symmetry verification** (C3 rotation, Bloch periodicity, and gauge invariance).
*   **Scaling analysis** of hopping and lattice parameters.

**Goal:** To provide a research-grade, reproducible reference for tight-binding Dirac physics using **JAX + Equinox**.

---

## 🚀 Key Features

### 🔹 Dirac Point Refinement
Newton-based refinement of the analytic K and K′ points ensures numerical precision down to machine tolerance.

### 🔹 Fermi Velocity Extraction
Three independent methods providing strong internal consistency:
- **Analytic:** $v_F = \frac{3}{2} a_{cc} t / \hbar$  
- **Gradient-based:** $\nabla f(k)$ at the Dirac point.
- **Ring-based:** Finite-difference estimator.

### 🔹 Berry Phase & Topology
Two complementary approaches yielding the expected $\pm\pi$ (or $\pm2\pi$ depending on gauge):
- Phase winding of the complex structure factor $f(k)$.
- Band-eigenvector Berry loop integration.

### 🔹 Symmetry & Gauge Audits
Rigorous verification of the physical model's integrity:
- **C3 Invariance:** Rotational symmetry of the honeycomb lattice.
- **Bloch Periodicity:** Invariance under reciprocal lattice shifts.
- **Tau-Gauge Invariance:** Stability under sublattice origin shifts.

### 🔹 Scaling & Linear Regime
- **Curvature Estimation:** Local dispersion analysis near K/K′.
- **Scaling Tests:** Verifies $v_F \propto t$ and $v_F \propto a$ with $R^2 \approx 1.000000$.

---

## 🗂 File Structure (Monolithic)

This repository utilizes a single, self-contained Python file organized as follows:

1.  **Physical constants** & configuration dataclasses.
2.  **GrapheneModel:** Lattice geometry and tight-binding Hamiltonian.
3.  **Berry phase utilities.**
4.  **Symmetry & gauge** invariance checks.
5.  **DiracAuditor:** Core engine for refinement, $v_F$, and scaling.
6.  **Full audit routine** & Diagnostics.
7.  **Main entry point.**

---

## 💻 Usage

### 🔧 Dependencies
- Python 3.10+
- JAX (CPU or GPU)
- Equinox

```bash
pip install jax jaxlib equinox
```

### ▶ Running the Audit
Execute the monolithic script to generate a detailed physical report:

```bash
honeycomb_dirac_verification.py
```

### 📊 Example Output Snippet
```text
=== Graphene Dirac Audit (massive Dirac, full symmetry & scaling checks) ===
vF_analytic        = 9.062708e+05 m/s
gamma_K (from f)   = 3.141593e+00 rad
C3 invariance      = True
Tau gauge invariance= True
R2_t               = 1.000000
R2_a               = 1.000000
```

---

## 🧠 Why This Project Exists

This implementation is designed as a reference for researchers working on:
*   **Dirac materials** and Tight-binding models.
*   **Symmetry-protected physics** and Topological phases.
*   **Numerical audits** of physical models.
*   **JAX-based scientific computing** and high-performance simulation.

The emphasis is on **clarity, reproducibility, and physical correctness**, providing more than just numerical output.

---

## 📄 License

MIT License 

---

## 👤 Author

**hari hardiyan**  
*Research-grade Dirac Physics with JAX*
