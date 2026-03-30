# CFRP Oxidative Decomposition Modeling

This project presents a computational framework for modeling the thermal decomposition of Carbon Fiber Reinforced Polymers (CFRP) under fire conditions using Julia.

---

## Problem Statement

CFRP materials are widely used in aerospace and automotive industries. Under high temperatures, they undergo a complex multi-stage decomposition process involving:

- **Pyrolysis**: Thermal degradation of the resin matrix (independent of oxygen)
- **Oxidation**: Combustion of char and carbon fibers (oxygen-dependent)

To accurately simulate this behavior, kinetic parameters must be identified:

- Pre-exponential factor (A)
- Activation energy (E)
- Reaction order (n)
- Oxygen reaction order (m)

---

## Reaction Mechanism

The system consists of three reactions:

### 1. Matrix Pyrolysis (Anaerobic)

\[
r_1 = A_1 \exp\left(-\frac{E_1}{RT}\right) M^{n_1}
\]

\[
M \rightarrow \nu_{char} C + (1 - \nu_{char}) G_1
\]

---

### 2. Char Oxidation (Aerobic)

\[
r_2 = A_2 \exp\left(-\frac{E_2}{RT}\right) C^{n_2} (P_{O_2})^{m_2}
\]

\[
C + O_2 \rightarrow G_2
\]

---

### 3. Fiber Oxidation (Aerobic)

\[
r_3 = A_3 \exp\left(-\frac{E_3}{RT}\right) F^{n_3} (P_{O_2})^{m_3}
\]

\[
F + O_2 \rightarrow G_2
\]

---

## Temperature Model

Temperature increases linearly:

\[
T(t) = T_0 + \beta t
\]

---

## 🧪 Experimental Design

Four synthetic experiments are simulated:

| Exp | Heating Rate (K/min) | Oxygen (%) | Purpose |
|-----|----------------------|------------|---------|
| Exp1 | 2.5 | 21% | Baseline |
| Exp2 | 5.0 | 21% | Baseline |
| Exp3 | 10.0 | 21% | Activation Energy |
| Exp4 | 5.0 | 5% | Oxygen Effect |

---

## ⚙️ Methodology

### 🔹 Task A: Forward Model
- Built reaction system using `Catalyst.jl`
- Implemented Arrhenius kinetics
- Modeled temperature dynamically

### 🔹 Task B: Synthetic Data Generation
- Simulated TGA curves using ODE solver (`Rodas5P`)
- Generated mass loss:  
  \[
  \text{Mass} = M + C + F
  \]
- Added Gaussian noise (0.5%)

### 🔹 Task C: Inverse Problem Setup
- Formulated parameter estimation using `PEtab.jl`
- Defined observables and experimental conditions

### 🔹 Task D: Parameter Estimation & Validation
- Performed multi-start optimization
- Compared true vs recovered parameters
- Evaluated model accuracy

---

##  Results

### 🔹 Validation Plot (Exp 2 vs Exp 4)

![Validation Plot](results/validation_plot.png)

### 🔹 All Experiments

![All Experiments](results/all_experiments.png)

---

## Key Findings

- Oxygen concentration strongly influences oxidation reactions
- Lower oxygen levels delay decomposition to higher temperatures
- The model successfully captures multi-stage degradation behavior
- Parameter estimation accurately recovers kinetic parameters

---

##  How to Run

```bash
julia main.jl
