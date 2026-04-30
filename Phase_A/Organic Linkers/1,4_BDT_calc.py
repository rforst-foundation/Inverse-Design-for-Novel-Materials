import json, random, shutil
import numpy as np
from pathlib import Path
from ase import Atoms, units
from ase.calculators.cp2k import CP2K
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from tqdm.auto import tqdm

# 1,4-Benzenedithiol (BDT) — isolated molecule DFT dataset for NequIP
# Engine : CP2K / PBE+DFT-D3 / DZVP-MOLOPT-GTH / non-periodic (MT Poisson)
# SCF    : Orbital Transformation (OT) + DIIS + FULL_ALL preconditioner
#          → 2–3× faster than diagonalisation on CPU clusters for small molecules
#
# Pipeline: relax → C-S stretches → S-H stretches → thiol torsions
#           → frozen phonons → AIMD 300 K → AIMD 400 K
#
# Atom index map (fixed for all perturbation functions below):
#   C(0-5) benzene ring,  S(6) top,  S(7) bottom
#   H(8-11) ring,  H(12) thiol on S(6),  H(13) thiol on S(7)
#   Connectivity: C(0)–S(6)–H(12)  |  C(3)–S(7)–H(13)


_SYMBOLS = ['C','C','C','C','C','C','S','S','H','H','H','H','H','H']

_POSITIONS = [
    [ 0.000,  1.400,  0.000],  # C(0)  para, bonded to S(6)
    [ 1.212,  0.700,  0.000],  # C(1)
    [ 1.212, -0.700,  0.000],  # C(2)
    [ 0.000, -1.400,  0.000],  # C(3)  para, bonded to S(7)
    [-1.212, -0.700,  0.000],  # C(4)
    [-1.212,  0.700,  0.000],  # C(5)
    [ 0.000,  3.150,  0.000],  # S(6)  top thiol
    [ 0.000, -3.150,  0.000],  # S(7)  bottom thiol
    [ 2.150,  1.240,  0.000],  # H(8)  ring
    [ 2.150, -1.240,  0.000],  # H(9)  ring
    [-2.150, -1.240,  0.000],  # H(10) ring
    [-2.150,  1.240,  0.000],  # H(11) ring
    [ 0.000,  3.480,  1.340],  # H(12) thiol on S(6)
    [ 0.000, -3.480, -1.340],  # H(13) thiol on S(7)
]

# Bond topology for structural perturbations
# C-S: move S + its thiol H together so the C–S bond stretches without
#      distorting the internal S–H geometry
_CS_BONDS  = [(0, [6, 12]), (3, [7, 13])]
_SH_BONDS  = [(6, [12]),    (7, [13])]
# (c_idx, s_idx, h_idx) for each thiol arm
_THIOL_ROT = [(0, 6, 12),   (3, 7, 13)]

# ── CP2K input block ──────────────────────────────────────────────────────────
# Sections ASE does not generate automatically:
#   POISSON  : MT solver — correct choice for non-periodic (no 3-D Ewald)
#   VDW_POTENTIAL : DFT-D3 — mandatory for BDT; dispersion dominates S–ring stacking
#   OT / OUTER_SCF : Orbital Transformation SCF — no diagonalisation needed,
#                    typically 2–3× faster on CPUs for well-gapped molecules
# ASE merges this block with its own auto-generated FORCE_EVAL / DFT / SCF / XC
# sections, so only non-overlapping keywords are placed here.
_CP2K_INP = """\
&FORCE_EVAL
  &DFT
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER MT
    &END POISSON
    &XC
      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          REFERENCE_FUNCTIONAL PBE
          PARAMETER_FILE_NAME dftd3.dat
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
    &SCF
      SCF_GUESS RESTART
      &OT ON
        MINIMIZER DIIS
        PRECONDITIONER FULL_ALL
      &END OT
      &OUTER_SCF
        EPS_SCF 1.0E-6
        MAX_SCF 10
      &END OUTER_SCF
    &END SCF
  &END DFT
&END FORCE_EVAL
"""
# SCF_GUESS RESTART: CP2K automatically falls back to ATOMIC if no .wfn file
# exists, so the first step of each branch is safe. For AIMD, each step reuses
# the previous wavefunction → significant wall-time saving on large runs.

# ── Structure builder ─────────────────────────────────────────────────────────

def build_bdt() -> Atoms:
    mol = Atoms(symbols=_SYMBOLS, positions=_POSITIONS,
                cell=[15.0, 15.0, 15.0], pbc=[False, False, False])
    mol.center()  # ~4 Å vacuum on all sides — sufficient for MT Poisson
    return mol

# ── Calculator factory ────────────────────────────────────────────────────────

def make_cp2k(label: str, scratch_dir: Path) -> CP2K:
    work = scratch_dir / label
    work.mkdir(parents=True, exist_ok=True)
    return CP2K(
        label=str(work / label),
        xc='PBE',
        basis_set='DZVP-MOLOPT-GTH',
        basis_set_file='BASIS_MOLOPT',
        pseudo_potential='GTH-PBE',
        charge=0,
        # 400 Ry expressed in eV (ASE expects eV for this parameter).
        # 400 Ry is the standard production cutoff for C/H/S with MOLOPT basis.
        cutoff=400 * units.Ry,
        max_scf=50,  # OT inner-loop steps; outer loop controlled in inp block
        inp=_CP2K_INP,
    )

# ── DFT wrappers ──────────────────────────────────────────────────────────────

def relaxation(atoms: Atoms, calc, fmax=0.02, steps=80, label='relax') -> Atoms:
    a = atoms.copy()
    a.calc = calc
    pbar = tqdm(total=steps, desc=f'Relax ({label})', unit='step')
    opt = BFGS(a, logfile=None)
    def _cb():
        pbar.n = min(opt.nsteps, steps)
        pbar.refresh()
    opt.attach(_cb, interval=1)
    opt.run(fmax=fmax, steps=steps)
    pbar.close()
    # get_potential_energy / get_forces return cached values from the last SCF
    a.info['energy'] = float(a.get_potential_energy())
    a.arrays['forces'] = a.get_forces()
    a.info['tag'] = f'relaxed_{label}'
    a.calc = None
    return a


def single_point(atoms: Atoms, calc, tag: str) -> Atoms:
    a = atoms.copy()
    a.calc = calc
    # One SCF call; get_forces() reads the cached result — no second calculation
    a.info['energy'] = float(a.get_potential_energy())
    a.arrays['forces'] = a.get_forces()
    a.info['tag'] = tag
    a.calc = None
    return a


def aimd_snapshots(atoms: Atoms, calc, T: int, steps: int = 15,
                   dt_fs: float = 0.5) -> list:
    # dt=0.5 fs chosen for BDT: C-H stretch period ~11 fs → need dt < 2 fs
    # MaxwellBoltzmann assigns physically correct initial velocities at T
    frames = []
    md = atoms.copy()
    MaxwellBoltzmannDistribution(md, temperature_K=T)
    md.calc = calc
    dyn = Langevin(md, dt_fs * units.fs, temperature_K=T, friction=0.02)
    pbar = tqdm(total=steps, desc=f'AIMD {T}K', unit='step')
    for _ in range(steps):
        dyn.run(1)
        fr = md.copy()
        fr.info['energy'] = float(md.get_potential_energy())
        fr.arrays['forces'] = md.get_forces()
        fr.info['tag'] = f'aimd_{T}K'
        fr.calc = None
        frames.append(fr)
        pbar.update(1)
    pbar.close()
    return frames

# ── Structural perturbation helpers ──────────────────────────────────────────

def stretch_bond(atoms: Atoms, anchor_idx: int,
                 move_group: list, delta_A: float) -> Atoms:
    """Translate move_group rigidly along the anchor→group[0] bond by delta_A."""
    new = atoms.copy()
    direction = new.positions[move_group[0]] - new.positions[anchor_idx]
    direction /= np.linalg.norm(direction)
    for idx in move_group:
        new.positions[idx] += direction * delta_A
    return new


def rotate_thiol(atoms: Atoms, c_idx: int, s_idx: int,
                 h_idx: int, angle_deg: float) -> Atoms:
    """Rotate thiol H around the C–S bond axis by angle_deg."""
    new = atoms.copy()
    axis = new.positions[s_idx] - new.positions[c_idx]
    new.rotate(angle_deg, v=axis, center=new.positions[s_idx], indices=[h_idx])
    return new


def frozen_phonon(atoms: Atoms, magnitude: float = 0.01) -> Atoms:
    new = atoms.copy()
    noise = np.random.normal(scale=magnitude, size=new.positions.shape)
    new.set_positions(new.positions + noise)
    new.info['tag'] = 'frozen_phonon'
    return new

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    out     = Path(__file__).resolve().parent
    scratch = out / 'cp2k_work_BDT'
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    bdt    = build_bdt()
    frames = []

    # ── 1. Geometry optimisation ──────────────────────────────────────────────
    # fmax=0.02 eV/Å: tighter than the QD script because organic force
    # constants are softer and small errors compound into bad torsion barriers
    print('\n=== BDT Geometry Optimisation ===')
    rel = relaxation(bdt, make_cp2k('relax', scratch), fmax=0.02, steps=80, label='BDT')
    frames.append(rel)

    # ── 2. C–S bond stretches ─────────────────────────────────────────────────
    # BDT attaches to the QD surface via the S atom; NequIP must learn the
    # restoring force along this bond for correct adsorption dynamics.
    # S + its thiol H are moved together to preserve S-H geometry.
    print('\n=== C-S Bond Stretches ===')
    for anchor, group in _CS_BONDS:
        for delta in (-0.20, -0.10, 0.10, 0.20):
            tag = f'cs_C{anchor}_{delta:+.2f}A'
            frames.append(single_point(
                stretch_bond(rel, anchor, group, delta),
                make_cp2k(f'cs_{anchor}_{delta:+.2f}', scratch), tag))

    # ── 3. S–H bond stretches ─────────────────────────────────────────────────
    # Teaches dissociation restoring force; small deltas keep geometry physical
    print('\n=== S-H Bond Stretches ===')
    for anchor, group in _SH_BONDS:
        for delta in (-0.10, -0.05, 0.05, 0.10):
            tag = f'sh_S{anchor}_{delta:+.2f}A'
            frames.append(single_point(
                stretch_bond(rel, anchor, group, delta),
                make_cp2k(f'sh_{anchor}_{delta:+.2f}', scratch), tag))

    # ── 4. Thiol torsional rotations ──────────────────────────────────────────
    # CRITICAL: BDT twists when adsorbing onto a QD surface. NequIP must learn
    # the full C–S–H dihedral potential barrier (cis/gauche/trans landscape).
    # 7 angles × 2 thiols = 14 frames covering the complete 360° dihedral.
    print('\n=== Thiol Torsional Rotations ===')
    for c_idx, s_idx, h_idx in _THIOL_ROT:
        for angle in (45, 90, 135, 180, 225, 270, 315):
            tag = f'torsion_S{s_idx}_{angle}deg'
            frames.append(single_point(
                rotate_thiol(rel, c_idx, s_idx, h_idx, angle),
                make_cp2k(f'tor_S{s_idx}_{angle}', scratch), tag))

    # ── 5. Frozen phonon displacements ────────────────────────────────────────
    # 12 frames (was 6) — more draws needed to cover the 3N-6=36 normal modes
    # of a 14-atom molecule; cost is 12 single SCF calls (negligible for CP2K).
    print('\n=== Frozen Phonon Displacements ===')
    np.random.seed(7)
    for i in range(12):
        frames.append(single_point(
            frozen_phonon(rel, 0.01),
            make_cp2k(f'phonon_{i}', scratch), f'frozen_phonon_{i}'))

    # ── 6-8. Molecular AIMD ───────────────────────────────────────────────────
    # 300 K: room-temperature ring vibrations and S-tail wagging
    # 400 K: higher-energy conformations transiently populated during ligand exchange
    # 500 K: anharmonic regime (BDT stable to ~670 K); broadens PES coverage
    # steps=50 × dt=0.5 fs = 25 fs per run — samples several C-H vibration periods
    # (~11 fs) and ring-breathing modes (~30 fs) for diverse configurations.
    # (15 steps = 7.5 fs was barely one vibration period — essentially redundant frames.)
    for T in (300, 400, 500):
        print(f'\n=== AIMD {T}K ===')
        frames += aimd_snapshots(rel, make_cp2k(f'aimd_{T}K', scratch),
                                 T=T, steps=50, dt_fs=0.5)

    # ── Output ────────────────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(frames)
    n_tr = max(1, int(0.85 * len(frames)))
    write(out / 'dataset_train_BDT.extxyz', frames[:n_tr])
    write(out / 'dataset_val_BDT.extxyz',   frames[n_tr:])
    with open(out / 'manifest_BDT.json', 'w') as f:
        json.dump({'engine': 'CP2K/PBE+DFT-D3/DZVP-MOLOPT-GTH',
                   'n_frames': len(frames), 'n_train': n_tr,
                   'n_val': len(frames) - n_tr}, f, indent=2)
    print('\nWrote:', out / 'dataset_train_BDT.extxyz',
          'and',    out / 'dataset_val_BDT.extxyz')
    print(f'Total frames: {len(frames)}  ({n_tr} train / {len(frames)-n_tr} val)')


if __name__ == '__main__':
    main()
