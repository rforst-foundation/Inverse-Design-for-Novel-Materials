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
from tqdm.auto import tqdm

# Replicated Wurtzite Crystal Structure for a CdS/CdSe Quantum Dot in Ondry et al. ACS Nano 2021

# Ondry geometry and DFT controls
@dataclass
class Spec:
    core: str = "CdSe"
    shell: str = "CdS"
    bridge: str = "CdS"
    hex_repeats: tuple = (4,4,1)
    vacuum_A: float = 8.0
    k: int = 2 #OPTIMIZED: reduced k-points from 3 to 2
    soc: bool = False #True if Hg is implemented

#MINIMAL HEXAGONAL MONOLAYER (placeholder builder)

def hex_monolayer(spec: Spec) -> Atoms:
    # 1. Create large supercell of the shell material (CdS)
    
    slab = bulk(spec.shell, 'wurtzite', a=4.136, c=6.714).repeat(spec.hex_repeats) #Creates large 2D slab
    
    # 2. Define region for the core (CdSe)
    
    center = slab.get_cell().sum(axis=0)/2 #center of the slab is the core
    positions = slab.get_positions()
    core_radius = 5.0 #Simple radial replacement for a cylindrical/hexagonal core, 10 is example angstrom

    #Identify atoms within the core radius and swap them to core material
    # 3. Calculate distances for all atoms at once using array slicing
    dists = ((positions[:, 0] - center[0])**2 + (positions[:, 1] - center[1])**2)**0.5
    # 4. Identify atoms inside the core radius
    mask_core = dists < core_radius
    # 5. Identify atoms that are sulfur ('S')
    symbols = np.array(slab.get_chemical_symbols())
    mask_sulfur = symbols == 'S'
    # 6. Combine them: only change atoms that are BOTH in the core AND are Sulfur
    mask_to_swap = mask_core & mask_sulfur
    # 7. Apply the swap instantly
    symbols[mask_to_swap] = 'Se'
    slab.set_chemical_symbols(symbols)

    # 3. Add Vacuum
    slab.center(vacuum=spec.vacuum_A, axis=2)
    slab.pbc = (True, True, False)
    slab.info["spec"] = asdict(spec) #stores run spec in atoms.info
    return slab

# COMPUTE THE ENERGY AND FORCES FOR ATOMIC STRUCTURE
# Using Quantum ESPRESSO (QE) and Effective Medium Theory (EMT)
# EMT is for analytical pair potential - Only for testing, runs instantly and produces eneries/forces for debugging the ASE/NequIP pipeline better


def calculations(engine: str, soc: bool, qe_pseudo_dir: str, spec: Spec):
    qe_pseudo_dir = str(Path(qe_pseudo_dir).expanduser().resolve())
    engine = engine.lower()
    kmesh = (spec.k, spec.k, 1)
    if engine == "qe":
        from ase.calculators.espresso import Espresso
        psp = {"Cd" : "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF", "S" : "s_pbe_v1.4.uspp.F.UPF", "Se" : "Se_pbe_v1.uspp.F.UPF"}
        ctrl = dict(calculation="scf", prefix="qd", pseudo_dir=qe_pseudo_dir, outdir=".", tprnfor = True, restart_mode="from_scratch")
        syst = dict(ibrav=0, ecutwfc=70.0, ecutrho=560.0, occupations="smearing", degauss = 0.01, nosym=False, noinv=False)
        qe_assume_isolated = os.environ.get("QE_ASSUME_ISOLATED")
        if qe_assume_isolated:
            syst["assume_isolated"] = qe_assume_isolated
        if soc:
            syst.update(noncolin=True, lspinorb=True)
        elec = dict(conv_thr=1e-5, mixing_beta=0.2, mixing_mode = "plain", electron_maxstep = 150)  # OPTIMIZED: relaxed conv_thr, increased mixing_beta for faster convergence, reduced maxstep
        for sym, fname in psp.items():
            if not (Path(qe_pseudo_dir)/fname).exists():
                raise FileNotFoundError(f"Mssing UPF for {sym}: {qe_pseudo_dir}/{fname}")
        command = os.environ.get("ASE_ESPRESSO_COMMAND") or os.environ.get("ESPRESSO_COMMAND") or "pw.x -in PREFIX.pwi > PREFIX.pwo"
        return Espresso(
            input_data={"control": ctrl, "system": syst, "electrons": elec},
            pseudopotentials=psp,
            kpts=kmesh,
            command=command,
        )
    elif engine == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    else:
        raise ValueError("Engine must be 'qe' or 'emt'")


def make_branch_calc(calc, label, base_outdir):
    #Create ONE calculator per branch (base / each strain / each AIMD T)
    c = copy.deepcopy(calc)
    ase_label = f"qd_{label}"
    outdir = Path(base_outdir) / ase_label
    outdir.mkdir(parents=True, exist_ok=True)

    c.label = str(outdir / ase_label)

    inp = c.parameters["input_data"]
    ctrl = inp["control"]
    
    ctrl["prefix"] = ase_label
    ctrl["outdir"] = str(outdir)
    ctrl["restart_mode"] = "from_scratch"  # warm-start within THIS branch

    inp["control"]= ctrl
    c.parameters["input_data"] = inp

    return c

# Creates a new calculator instance based on a template
# If using VASP/QE/GPAW, this sets directory

# ADD DEFECT GENERATION LOOP
def generate_vacancies(atoms, num_vacancies=1):
    defect_atoms = atoms.copy()
    indices_to_remove = random.sample(range(len(defect_atoms)), num_vacancies)
    del defect_atoms[indices_to_remove]
    defect_atoms.info["tag"] = f"vacancy_{num_vacancies}"
    return defect_atoms

# Creates a copy of atoms within N random vacancies

# CREATE RELAXATION AND ATTACH LABELS
# OPTIMIZED: Reduced max steps from 40 to 15 (BFGS typically converges in 10-15 steps for well-behaved systems)

def relaxation(atoms: Atoms, calc, fmax=0.03, steps=15, label = "relaxation") -> Atoms:
    a = atoms.copy()
    a.calc = calc
    pbar = tqdm(total=steps, desc=f"Relaxation ({label})", unit = "step" )
    opt = BFGS(a, logfile=None)
    def print_step(*args, **kwargs):
        done = min(opt.nsteps, steps)
        pbar.n = done
        pbar.refresh()
    opt.attach(print_step, interval = 1)
    opt.run(fmax=fmax, steps=steps) #updates atomic positions to minimize total energy, stops when all forces are < fmax=0.02 eV/A
    pbar.close()
    a.info["energy"] = float(a.get_potential_energy()) #total energy in eV
    a.arrays["forces"] = a.get_forces() #forces in eV/A
    a.calc =  None
    return a

# Runs a geometry optimization

#BIAXIAL STRAIN

def strain(atoms: Atoms, pct: float) -> Atoms:
    s = 1.0 + pct/100.0
    new = atoms.copy()
    cell = new.cell.copy()
    cell[0] *= s
    cell[1] *= s
    new.set_cell(cell, scale_atoms=True)
    return new

# CREATE SHORT FINITE-T MD SAMPLES (pseudo-AIMD)
# Good for sampling local termal displacements (phonon-like) without long AIMD
# OPTIMIZED: Reduced default steps from 20 to 8

def aimd_snapshots(atoms: Atoms, calc, T=300, steps=8, dt_fs=1.0):
    frames = []
    md_atoms = atoms.copy()
    md_atoms.calc = calc
    mdl = Langevin(md_atoms, dt_fs*units.fs, temperature_K=T, friction = 0.02)

    pbar = tqdm(total=steps, desc=f"AIMD {T}K", unit = "step")

    for _ in range(steps):
        mdl.run(1)
        
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

# T is the trarget temperature
# dt_fs is the timestep, 1 fs
# 0.02 is the frictional coefficient; weak damping so atoms sample near-equilibrum vibrations

# Langevin MD is used as it provides thermal perturbations around the equilibrium geometry and samples atomistic configurations that appear at finite T
# Necessary for the NequIP training as it teaches the network not only the perfect minimum but also the local curvature of the potential energy surface
# 50 * T supplies hundreds of labeld geometries spanning vibrational and strain space


# ADD DROZEN-PHONON DISPLACEMENT LOOP
def generate_displacements(atoms, displacement_magnitude=0.01):
    disp_atoms = atoms.copy()
    positions = disp_atoms.get_positions()
    #Add random noise vector
    noise = np.random.normal(scale=displacement_magnitude, size=positions.shape)
    disp_atoms.set_positions(positions + noise)
    disp_atoms.info["tag"] = "frozen_phonon_displacement"
    return disp_atoms

# Creates variants with small random displacements for force training
# A random noise vector was added





#MAIN BODY
# AGGRESSIVELY OPTIMIZED VERSION: Significantly reduced step counts and computational parameters:
# - k-points: 3→2 (faster SCF)
# - SCF convergence: 1e-6→1e-5 (faster convergence)
# - Base relaxation: 40→15 steps, fmax: 0.02→0.03
# - Frozen phonon: 5→3 structures
# - Strain points: 5→3 (-2, 0, +2%), steps: 20→8, fmax: 0.03→0.05
# - AIMD temperatures: 3→2 (300, 450K), steps: 20→8

def main(engine="qe", qe_pseudo_dir=None):
    out = Path(__file__).resolve().parent #sets output to current directory
    scratch_root = out / "qe_work"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if qe_pseudo_dir is None:
        qe_pseudo_dir = str((out/"pseudos").resolve())
   


    spec = Spec() #Instantiates Spec
    if "Hg" in (spec.core+spec.shell+spec.bridge): 
        spec.soc=True #auto-enables SOC if Hg is present
    base = hex_monolayer(spec)

    #write(out / "visual_check.xyz", base)
    #print(f"Geometry saved to {out}/visual_check.* for inspection")

    #exit()

    base_calc_template = calculations(engine, spec.soc, qe_pseudo_dir, spec)


    frames = []

    print("\n=== Starting base relaxation ===")
    
    #Base Relaxation and Parameters

    base_calc = make_branch_calc(base_calc_template, "base", scratch_root)
    rel = relaxation(base, base_calc, fmax=0.03, steps=15, label = "base")
    rel.info["tag"]="relaxed_base"
    frames.append(rel) #Relaxed based ground-state
    
    #Defect Loop
    for vac_count in [1, 2]:
        print(f"\n=== Generating Defects ===")
        # 1.Create structure
        defect_struct = generate_vacancies(rel, vac_count)
        # 2. Attach calculator
        d_calc = make_branch_calc(base_calc_template, f"vac_{vac_count}", scratch_root)
        defect_struct.calc = d_calc
        # 3. Run calculation
        en = defect_struct.get_potential_energy()
        forces = defect_struct.get_forces()
        # 4. Store Data and detach calculator
        defect_struct.info["energy"] = float(en)
        defect_struct.arrays["forces"] = forces
        defect_struct.calc = None

        frames.append(defect_struct)


    # Frozen-Phonon Loop
    for i in range(3):
        print(f"\n=== Generating Frozen-Phonons ===")
        # 1. Create structure
        disp_struct = generate_displacements(rel, 0.01)
        # 2. Attach calculator
        disp_calc = make_branch_calc(base_calc_template, f"phonon_disp_{i}", scratch_root)
        disp_struct.calc = disp_calc
        # 3. Run calculation
        en = disp_struct.get_potential_energy()
        forces = disp_struct.get_forces()
        # 4. Store data and detach calculator
        disp_struct.info["energy"] = float(en)
        disp_struct.arrays["forces"] = forces
        disp_struct.calc = None

        frames.append(disp_struct)
    
    #Strain and relaxation
    for s in (-2.0, 0.0, 2.0):
        print(f"\n=== Starting Strain {s:+.1f}% relaxation ===")
        strain_calc = make_branch_calc(base_calc_template, f"strain{s:+.1f}", scratch_root)
        rs = relaxation(strain(rel, s), strain_calc, fmax=0.05, steps = 8, label = f"strain_{s:+.1f}")
        rs.info["tag"]=f"strain_{s:+.1f}pct"
        frames.append(rs)
    
    
    
    #AIMD
    for T in (300, 450):
        print(f"\n=== Starting AIMD {T}K ===")
        aimd_calc = make_branch_calc(base_calc_template, f"aimd{T}K", scratch_root)
        frames += aimd_snapshots(rel, aimd_calc, T=T, steps=8, dt_fs=1.0)

    #shuffle and split
    random.seed(0); random.shuffle(frames)
    n_tr = max(1, int(0.85*len(frames))) #85% to train, 15% to val
    write(out/"dataset_train2.extxyz", frames[:n_tr])
    write(out/"dataset_val2.extxyz", frames[n_tr:])
    with open(out / "manifest.json", "w") as f:
        json.dump({"spec":asdict(spec), "engine":engine, "n_train":n_tr, "n_val":len(frames)-n_tr}, f, indent=2) #Writes a tiny manifest of spec, engine, counts, etc.

    print ("Wrote:", out/"dataset_train2.extxyz", "and", out/"dataset_val2.extxyz")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="qe", choices=["qe", "emt"], help="Calculator backend.")
    p.add_argument("--qe-pseudo-dir", default=None, help="Directory containing QE UPF pseudopotentials.")
    args = p.parse_args()
    main(engine=args.engine, qe_pseudo_dir=args.qe_pseudo_dir)
