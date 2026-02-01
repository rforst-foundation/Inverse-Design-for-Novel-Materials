# Project: Inverse Design of Heterogeneous QD-MOF Superlattices for Quantum Batteries

**A computational framework proposing the use of Active Learning and Equivariant Graph Neural Networks (EGNNs) to discover novel Quantum Dot-Metal Organic Framework (QD-MOF) superlattices.**

## Abstract

Quantum Batteries are a theoretical class of energy storage devices that leverage quantum mechanical phenomena, specifically coherence and entanglement, to store energy with vastly higher power density than classical chemical batteries. However, realizing these devices requires materials with very specific electronic properties, namely **"Flat Bands"** (which allow massive charge storage) that are also robust against environmental disorder.

This project proposes a solution: using **Machine Learning** to design "superlattices" made of Quantum Dots (QDs) connected by organic linkers (MOFs). Because the number of possible material and geometric combinations is too large to simulate one by one, we are building an **Autonomous Agent** that learns the physics of these materials on the fly, allowing us to find the optimal design without wasting millions of hours on unnecessary calculations.

---

## Problem Statement

To create a Quantum Battery, we need a material that acts like a "quantum capacitor." This requires three competing properties:

1.  **High Capacitance:** We need a high **Density of States (DOS)**. Conceptually, many "electron seats" available at the same energy level. In band theory, this appears as a **"Flat Band."** This allows the device to store more energy than a traditional chemical battery.
2.  **High Stability:** The battery should not leak or self-discharge energy while stored. This ensures the battery holds its charge for useful timescales under normal usage conditions.
3.  **Real-World Robustness:** The design must be resilient to defects and environmental noise.

**The Issue:**
Traditional simulation methods, such as Density Functional Theory (DFT), are too slow and expensive. A single fully simulated QD-MOF structure can take over 5,000 CPU hours and is limited to less than 1,000 atoms. The design space (QDs + Linker combinations) has over $10^{12}$ possibilities. Traditional screening is impossible, so a smarter search method is required.

---

## Proposed Solution: Materials Intelligence

We propose replacing traditional small-scale computations and simulations with an **Active-Learning Loop**. Instead of simulating random materials, we utilize an AI agent to decide which simulations to run based on what it has already learned.

Standard neural networks struggle with physics because they don't inherently understand 3D space (e.g. rotating a molecule shouldn't change its energy). We will use **NequIP**, an E(3)-equivariant graph neural network, which mathematically guarantees symmetry preservation. This allows it to learn accurate physics from about 300x less data than traditional models.

---

## Goals & Objectives

1.  **Build Model:** Train a NequIP model that predicts energies and forces of QD-MOF structures with DFT-level accuracy (< 2 meV/atom) but runs three orders of magnitude faster.
2.  **Automate Discovery:** Create a package that autonomously explores the design space, triggering expensive DFT calculations only when the AI is confused (high epistemic uncertainty).
3.  **Inverse Design:** Use the trained model to build new materials that maximize desired characteristics while minimizing strain.
4.  **Validate Feasibility:** Computationally "stress test" top candidates by adding defects and thermal noise to ensure they survive real-world conditions.

---

## Methodology

### Phase A: First Principles (In Progress)
* **Goal:** Generate the base datasets to teach the AI/ML model.
* **Problem:** High-quality training data for hybrid organic-inorganic interfaces is scarce.
* **Solution:** Create custom datasets and automate the hand-off between geometric generation (LAMMPS) and electronic structure calculation (DFT/VASP).
* **Process:**
    1.  **Generate DFT dataset:** Compute forces, energies, and stresses of experimentally realized systems (bulk soldis, interfaces, organic linkers).
    2.  **Validate:** Benchmark against experimental lattice constants and bandgaps in literature.
    3.  **Primitives:** Create data classes for QDs and Linkers.
    4.  **Perturbations:** Implement small random variations and defects to robustify the dataset.
* **Output:** Small, curated dataset of atomic forces for training.

### Phase B: Graph Neural Network Training (In Development)
* **Goal:** Develop an ML framework that understands QD-MOF physics at DFT fidelity.
* **Problem:** Standard neural networks are "data-hungry," often violate the conservation of energy, and cannot combine disparate DFT dataset due to inconsistent energy baselines.
* **Solution:** Utilize **NequIP-FLARE**, which respects physical symmetries (rotations/translations/inversion), and create a Multi-Fidelity Energy Referencing class paired with a Deep Ensemble architecture for epistemic uncertainty.
* **Process:**
    1.  **Deep Ensembles:** Train an ensemble of 5 separate NequIP-FLARE models.
    2.  **Uncertainty Quantification:** Explore the Potential Energy Surface (PES) using Molecular Dynamics.
    3.  **Decision Logic:**
        * *If models agree:* Infer forces instantly.
        * *If models disagree:* Trigger FLARE(VASP) to calculate ground truth (DFT).
    4.  **Loop:** Add new DFT data to training set to retrain models.
* **Output:** Validated Interatomic Potential (< 2 meV/atom error).

### Phase C: Multi-Objective Inverse Design
* **Goal:** Discover QD-MOF geometries that maximize storage without sacrificing stability/structural integrity.
* **Problem:** We need a way to balance competing design targets, such as "Flat Bands" and structural stability, while ensuring realistic geometric assemblies
* **Solution:** Gradient-descent optimization paired with Pareto ranking.
* **Process:**
    1.  **Loss Function:** Construct a multi-objective loss function: $\mathcal{L}_(\text{total}) = - \alpha(C_(\text{quantum})) + \beta(E_(\text{NequIP})) + \gamma(\sigma_(\text{strain}))$ where $C_(\text{quantum})$ is the Band Flatness/Capacitance, $E_(\text{NequIP})$ is the interatomic potential energy (stability), and $\sigma_(\text{strain})$ penalizes geometric distortion.
    2.  **Simulation:** Use the differentiable NequIP potential to update atomic coordinates via gradient descent.
    3.  **Pareto Ranking:** Plot structures on a Pareto Frontier to identify the optimal trade-offs.
* **Output:** 10-25 best crystal structures that are theoretically optimal and distinct from known materials.

### Phase D: Electronic and Stability Verification
* **Goal:** Ensure designs survive real-world manufacturing and have desired electronic properties.
* **Problem:** Materials that appear stable and electronically optimized in a static zero-temperature simulation often decompose due to thermal vibrations or manufacturing defects in the real world.
* **Solution:** Physics-based stress testing utilizing Phonon Dispersion analysis and Effective Hamiltonian mapping.
* **Process:**
    1.  **Electronic Confirmation:** Run high-precision static DFT on best candidates to confirm Density of States (DOS).
    2.  **Wannier Tight Binding Hamiltonian:** Project the DFT wavefunctions onto Maximally Localized Wannier Functions (MLWFs) to extract accurate hopping parameters and site energies.
    3.  **Phonon and Electron Interaction Stability:** Calculate dispersion curves to check for imaginary frequencies (dynamic instability).
    4.  **Hubbard Model:** Calculate the onsite Coulomb interaction($U$) to accurately model the quantum many-body correlations required for battery operation.
* **Output:** 3-5 validated, synthesizable candidate materials with full characterization of their quantum transport parameters and thermodynamic stability limits.

---

## Desired Conclusion

1.  **Open-Source Framework:** A documented tool for simulating hybrid QD-MOF interfaces.
2.  **Candidate List:** 3-5 specific material/geometric compositions theoretically capable of functioning as a Quantum Battery active medium.
3.  **Proof of Feasibility:** Computational evidence that "Materials Intelligence" can solve physics problems unmanagable for humans.

---

## Contact
**Riley Forst**
* Principal Investigator
* rwforst486@gmail.com

**Livia Guttieres**
* Research Collaborator
* Gradient Descent Framework Lead
* liviaguttieres@g.harvard.edu
