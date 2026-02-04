# Physics-Informed Neural Network (PINN) Architecture – HeatFlux Project

## 1. Problem Definition
The study addresses the one-dimensional transient heat-conduction equation:

\[
\rho c\,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},
\quad x \in [0,L],\ t \in [0,t_{\max}]
\]

where:
- \(T(x,t)\) is the temperature field,  
- \(k\) is thermal conductivity,  
- \(\rho c\) is volumetric heat capacity.

Boundary and initial conditions:
- Left boundary: **Dirichlet** – \(T(0,t) = T_L(t)\)
- Right boundary: **Neumann (unknown)** – \(-k\,T_x(L,t) = q''_R(t)\)
- Initial condition: \(T(x,0) = T_0(x)\)

The objective is to **recover the boundary heat flux** \(q''_R(t)\) from temperature data using a Physics-Informed Neural Network (PINN).

---

## 2. Forward Model (Reference Data)
A Crank–Nicolson finite-difference solver generates deterministic, noise-free datasets for benchmarking the PINN.  
This method is unconditionally stable and second-order accurate in both space and time, producing smooth fields ideal for supervised validation.

Generated datasets are stored in:
