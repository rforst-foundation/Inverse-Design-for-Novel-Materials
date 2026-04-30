import json, random, copy, shutil
import os
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from ase import Atoms, units
from ase.build import bulk
from ase.io import write
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm.auto import tqdm

# Defect branch of the Ondry et al. ACS Nano 2021 QD dataset.
# Applies N random vacancies to the relaxed perfect lattice, re-relaxes the
# defective structure, then runs the full pipeline (frozen phonons, biaxial
# strain, AIMD) on that defective structure so NequIP learns both the new
# equilibrium and the phonon modes associated with the vacancy.

@dataclass
class Spec:
    core: str = "CdSe"
    shell: str = "CdS"
    bridge: str = "CdS"
    hex_repeats: tuple = (4,4,1)
    vacuum_A: float = 8.0
    k: int = 2
    soc: bool = False


def hex_monolayer(spec: Spec) -> Atoms:
    slab = bulk(spec.shell, 'wurtzite', a=4.136, c=6.714).repeat(spec.hex_repeats)
    center = slab.get_cell().sum(axis=0)/2
    positions = slab.get_positions()
    core_radius = 5.0
    dists = ((positions[:, 0] - center[0])**2 + (positions[:, 1] - center[1])**2)**0.5
    mask_core = dists < core_radius
    symbols = np.array(slab.get_chemical_symbols())
    mask_sulfur = symbols == 'S'
    symbols[mask_core & mask_sulfur] = 'Se'
    slab.set_chemical_symbols(symbols)
    slab.center(vacuum=spec.vacuum_A, axis=2)
    slab.pbc = (True, True, False)
    slab.info["spec"] = asdict(spec)
    return slab


def calculations(engine: str, soc: bool, qe_pseudo_dir: str, spec: Spec):
    qe_pseudo_dir = str(Path(qe_pseudo_dir).expanduser().resolve())
    engine = engine.lower()
    kmesh = (spec.k, spec.k, 1)
    if engine == "qe":
        from ase.calculators.espresso import Espresso
        psp = {"Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF", "S": "s_pbe_v1.4.uspp.F.UPF", "Se": "Se_pbe_v1.uspp.F.UPF"}
        ctrl = dict(calculation="scf", prefix="qd", pseudo_dir=qe_pseudo_dir, outdir=".", tprnfor=True, restart_mode="from_scratch")
        syst = dict(ibrav=0, ecutwfc=70.0, ecutrho=560.0, occupations="smearing", degauss=0.01, nosym=False, noinv=False)
        qe_assume_isolated = os.environ.get("QE_ASSUME_ISOLATED")
        if qe_assume_isolated:
            syst["assume_isolated"] = qe_assume_isolated
        if soc:
            syst.update(noncolin=True, lspinorb=True)
        elec = dict(
            conv_thr=1e-5,
            mixing_mode="local-TF",  # better than plain for slab+vacuum; adapts mixing wavevector
            mixing_beta=0.3,
            mixing_ndim=12,          # Broyden history depth; more history → fewer oscillations
            electron_maxstep=100,
        )
        for sym, fname in psp.items():
            if not (Path(qe_pseudo_dir)/fname).exists():
                raise FileNotFoundError(f"Missing UPF for {sym}: {qe_pseudo_dir}/{fname}")
        command = os.environ.get("ASE_ESPRESSO_COMMAND") or os.environ.get("ESPRESSO_COMMAND") or "pw.x -in PREFIX.pwi > PREFIX.pwo"
        return Espresso(
            input_data={"control": ctrl, "system": syst, "electrons": elec},
            pseudopotentials=psp,
            kpts=kmesh,
            command=command,
        )
    elif engine == "stub":
        # Zero-force stub: returns E=0, F=0 for any structure.
        # Exercises the full ASE/extxyz pipeline instantly — not physical data.
        from ase.calculators.calculator import Calculator, all_changes
        class ZeroCalc(Calculator):
            implemented_properties = ["energy", "forces"]
            def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}
        return ZeroCalc()
    else:
        raise ValueError("Engine must be 'qe' or 'stub'")


def make_branch_calc(calc, label, base_outdir):
    c = copy.deepcopy(calc)
    ase_label = f"qd_{label}"
    outdir = Path(base_outdir) / ase_label
    outdir.mkdir(parents=True, exist_ok=True)

    if hasattr(c, "parameters") and "input_data" in c.parameters:
        c.label = str(outdir / ase_label)
        inp = c.parameters["input_data"]
        ctrl = inp["control"]
        ctrl["prefix"] = ase_label
        ctrl["outdir"] = str(outdir)
        ctrl["restart_mode"] = "from_scratch"
        inp["control"] = ctrl
        c.parameters["input_data"] = inp

    return c


def generate_vacancies(atoms, num_vacancies=1):
    defect_atoms = atoms.copy()
    indices_to_remove = random.sample(range(len(defect_atoms)), num_vacancies)
    del defect_atoms[indices_to_remove]
    defect_atoms.info["tag"] = f"vacancy_{num_vacancies}"
    return defect_atoms


def relaxation(atoms: Atoms, calc, fmax=0.03, steps=15, label="relaxation") -> Atoms:
    a = atoms.copy()
    a.calc = calc
    pbar = tqdm(total=steps, desc=f"Relaxation ({label})", unit="step")
    opt = BFGS(a, logfile=None)
    def print_step(*args, **kwargs):
        pbar.n = min(opt.nsteps, steps)
        pbar.refresh()
    opt.attach(print_step, interval=1)
    opt.run(fmax=fmax, steps=steps)
    pbar.close()
    a.info["energy"] = float(a.get_potential_energy())
    a.arrays["forces"] = a.get_forces()
    a.calc = None
    return a


def strain(atoms: Atoms, pct: float) -> Atoms:
    s = 1.0 + pct/100.0
    new = atoms.copy()
    cell = new.cell.copy()
    cell[0] *= s
    cell[1] *= s
    new.set_cell(cell, scale_atoms=True)
    return new


def aimd_snapshots(atoms: Atoms, calc, T=300, steps=30, dt_fs=1.0):
    frames = []
    md_atoms = atoms.copy()
    # Pre-sample velocities so every frame is immediately valid thermal data.
    # Without this the Langevin thermostat thermalizes from rest over ~3-4 steps,
    # wasting a significant fraction of the short run.
    MaxwellBoltzmannDistribution(md_atoms, temperature_K=T)
    md_atoms.calc = calc
    mdl = Langevin(md_atoms, dt_fs*units.fs, temperature_K=T, friction=0.02)
    pbar = tqdm(total=steps, desc=f"AIMD {T}K", unit="step")
    for i in range(steps):
        mdl.run(1)
        # Wavefunction warm-restart after first SCF — cuts subsequent SCF iterations
        # from ~40 to ~3-5 since atoms move only ~0.001 Å per step at 300 K.
        if i == 0 and hasattr(md_atoms.calc, "parameters") and "input_data" in md_atoms.calc.parameters:
            md_atoms.calc.parameters["input_data"]["control"]["restart_mode"] = "restart"
        energy = float(md_atoms.get_potential_energy())
        forces = md_atoms.get_forces()
        fr = md_atoms.copy()
        fr.info["energy"] = energy
        fr.arrays["forces"] = forces
        fr.calc = None
        frames.append(fr)
        pbar.update(1)
    pbar.close()
    return frames


def generate_displacements(atoms, displacement_magnitude=0.01):
    disp_atoms = atoms.copy()
    positions = disp_atoms.get_positions()
    noise = np.random.normal(scale=displacement_magnitude, size=positions.shape)
    disp_atoms.set_positions(positions + noise)
    disp_atoms.info["tag"] = "frozen_phonon_displacement"
    return disp_atoms


def main(engine="qe", qe_pseudo_dir=None, num_vacancies=1):
    out = Path(__file__).resolve().parent
    scratch_root = out / "qe_work_defect"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if qe_pseudo_dir is None:
        qe_pseudo_dir = str((out/"pseudos").resolve())

    spec = Spec()
    if "Hg" in (spec.core+spec.shell+spec.bridge):
        spec.soc = True
    base = hex_monolayer(spec)

    base_calc_template = calculations(engine, spec.soc, qe_pseudo_dir, spec)

    # Step 1: relax the perfect lattice to get a clean starting geometry
    print("\n=== Relaxing perfect base (seed for defect) ===")
    base_calc = make_branch_calc(base_calc_template, "base_seed", scratch_root)
    perfect_rel = relaxation(base, base_calc, fmax=0.03, steps=15, label="base_seed")

    # Step 2: introduce vacancies — they persist for the entire run
    print(f"\n=== Introducing {num_vacancies} vacancy/vacancies ===")
    defect = generate_vacancies(perfect_rel, num_vacancies)

    frames = []

    # Step 3: relax the defective structure so surrounding atoms find their new equilibrium
    print("\n=== Relaxing defective structure ===")
    d_rel_calc = make_branch_calc(base_calc_template, "defect_relax", scratch_root)
    d_rel = relaxation(defect, d_rel_calc, fmax=0.03, steps=15, label="defect_relax")
    d_rel.info["tag"] = f"relaxed_defect_{num_vacancies}vac"
    frames.append(d_rel)

    # Step 4: frozen phonons on the relaxed defect — 8 frames to sample harmonic basin
    for i in range(8):
        print(f"\n=== Frozen-Phonon {i+1}/8 (defect) ===")
        disp_struct = generate_displacements(d_rel, 0.01)
        disp_calc = make_branch_calc(base_calc_template, f"defect_phonon_{i}", scratch_root)
        disp_struct.calc = disp_calc
        en = disp_struct.get_potential_energy()
        forces = disp_struct.get_forces()
        disp_struct.info["energy"] = float(en)
        disp_struct.arrays["forces"] = forces
        disp_struct.calc = None
        frames.append(disp_struct)

    # Step 5: biaxial strain on the relaxed defect.
    # steps=20: at ±2% biaxial strain initial forces are ~0.3-0.8 eV/Å;
    # 8 steps (old) does not reach fmax=0.05.
    for s in (-2.0, 0.0, 2.0):
        print(f"\n=== Strain {s:+.1f}% relaxation (defect) ===")
        strain_calc = make_branch_calc(base_calc_template, f"defect_strain{s:+.1f}", scratch_root)
        rs = relaxation(strain(d_rel, s), strain_calc, fmax=0.05, steps=20, label=f"defect_strain_{s:+.1f}")
        rs.info["tag"] = f"defect_strain_{s:+.1f}pct"
        frames.append(rs)

    # Step 6: AIMD on the relaxed defect — samples defect phonon modes.
    # 300 K, 450 K, 600 K × 30 steps each (600 K added for anharmonic sampling).
    for T in (300, 450, 600):
        print(f"\n=== AIMD {T}K (defect) ===")
        aimd_calc = make_branch_calc(base_calc_template, f"defect_aimd{T}K", scratch_root)
        frames += aimd_snapshots(d_rel, aimd_calc, T=T, steps=30, dt_fs=1.0)

    random.seed(0); random.shuffle(frames)
    n_tr = max(1, int(0.85*len(frames)))
    write(out/"dataset_train_defect.extxyz", frames[:n_tr])
    write(out/"dataset_val_defect.extxyz", frames[n_tr:])
    with open(out/"manifest_defect.json", "w") as f:
        json.dump({"spec": asdict(spec), "engine": engine, "num_vacancies": num_vacancies,
                   "n_train": n_tr, "n_val": len(frames)-n_tr}, f, indent=2)

    print("Wrote:", out/"dataset_train_defect.extxyz", "and", out/"dataset_val_defect.extxyz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="qe", choices=["qe", "stub"])
    p.add_argument("--qe-pseudo-dir", default=None)
    p.add_argument("--num-vacancies", type=int, default=1, choices=[1, 2],
                   help="Number of atoms to remove (1 or 2).")
    args = p.parse_args()
    main(engine=args.engine, qe_pseudo_dir=args.qe_pseudo_dir, num_vacancies=args.num_vacancies)
