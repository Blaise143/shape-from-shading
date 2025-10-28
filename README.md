# sfs
***
 [Variational Shape From Shading](https://people.csail.mit.edu/bkph/papers/Variational_SFS.pdf)
 ***
**Notes**

Rendering Equation describes how light is emited from a surface
$$
L_{\text{out}}(p,\mathbf v,\lambda)
\;=\;
L_{\text{emit}}(p,\mathbf v,\lambda)
\;+\;
\int_{\Omega}
\mathrm{BRDF}(p,\mathbf s,\mathbf v,\lambda)\,
L_{\text{in}}(p,\mathbf s,\lambda)\,
(-\mathbf n\!\cdot\!\mathbf s)\,
\mathrm d\mathbf s
$$
Simplification under lambertian surface assumption and $L_{emit}=0$
$$
L_{\text{out}} = \rho \cdot L_{\text{in}} \cdot (\mathbf{n} \cdot \mathbf{s})
=R(n)$$
***
**Shape From Shading**

Assumptions
- Constant albedo on entire surface
- light source at infinity 
  - Constant light vector across all pixels 
- Known camera at infinity 
  - Viewing direction constant across all pixels

$$
(p, q) = \left( -\frac{\partial z}{\partial x},\ -\frac{\partial z}{\partial y} \right)
$$
