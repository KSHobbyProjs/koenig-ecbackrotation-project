# Complex Scaling

$V(r) = V_0 \exp\left(-(r-r_0)^2/a^2\right)$


Normalizing under the Hermitian inner product fixes a state $|n\rangle$ modulo an arbitrary phase:
$$
    \langle e^{i\alpha} n | e^{i\alpha} n \rangle = e^{i\alpha}e^{-i\alpha} \langle n | n\rangle = \langle n | n \rangle.
$$
Normalizing under the c-product fixes a state $|n)$ only modulo $\pm 1$:
$$
    (e^{i\alpha}n | e^{i\alpha}n) = e^{2i\alpha}(n|n),
$$
so $|e^{i\alpha}n)$ can only normalize like $|n)$ if $2\alpha = 2\pi n$ $\forall$ $n \in \mathbb{Z}$. Thus, $e^{i\alpha}=(-1)^n$ $\forall$ $n \in \mathbb{Z}$ $\rightarrow$ $\pm |n)$. 

I.e., any state $|e^{i\alpha}n\rangle$ will have the same normalization once we specify $\langle n | n\rangle$, but only the states $\pm |n)$ will have the same normalization once we specify $(n | n)$. This is why we phase fix (see DVR code). If we receive a normalized state, we won't know if it's the exact same state another eigensolver output since they can differ by arbitrary phases. So, we fix the phase. Since we use the c-product, we fix the sign: the transition matrix element $(\psi_\text{res} | r | \psi_\text{bound})$ is only fixed up to a factor $\pm 1$, so we select one branch and compare everything to that branch. 

In the physical world, these arbitrary phases are irrelevant because we compute things like $|\psi|^2$, $\langle \psi |T | \psi \rangle$, $|\langle \psi_1 | T | \psi_2\rangle|^2$, and $\frac{\langle m | T_1 | n\rangle}{\langle m | T_2 | n\rangle}$ (in which case, the relative phase between the amplitudes for $T_1$ and $T_2$ matters, even though the arbitrary phases assigned to $|m\rangle$ and $|n\rangle$ don't). But, when you're purposely trying to compare how a computational technique performs against another, you want to fix the phases across the techniques, so the techniques don't look like they're outputing wildly different results even though they're outputting the same physical state modulo an arbitrary phase.