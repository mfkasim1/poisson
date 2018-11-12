import numbers
import numpy as np
from poisson.gradopt import Momentum

__all__ = ["solve"]

def solve(u0, phi=None, fix="bounds", dx=None):
    """
    Solve the Poisson equation solver given by:

        $$
        \nabla^2 u = phi
        $$

    from an initial value, `u0`, with a given `fix` map.

    Parameters
    ----------
    * `u0` : np.ndarray
        The initial guess of the solution.
    * `phi` : np.ndarray, optional
        The right hand side of the equation. The default is all zeros.
    * `fix` : np.ndarray or str, optional
        If it is np.ndarray, it is a boolean map to indicate which elements
        in `u` are fixed. If it is a str, acceptable values are "bounds". The
        default value is "bounds".
    * `dx` : sequence of floats, optional
        The pixel size in each dimension. The default is 1./shape[i] for the
        i-th dimension.

    Returns
    -------
    * np.ndarray
        The solution of the equation above of `u`.
    """
    if phi is None:
        phi = np.zeros_like(u0)
    phi = np.asarray(phi)
    u0 = np.asarray(u0)
    _assert_same_shape(phi, u0, "phi", "u0")
    shape = u0.shape

    # get the fix as a boolean map
    fix = _preprocess_fix(fix, shape)
    _assert_same_shape(fix, u0, "fix", "u0")

    # normalize dx
    if dx is None:
        dx = 1./np.asarray(shape)
    dx = np.asarray(dx)

    # solve it with proper input
    return _solve(u0, phi, fix, dx)

def _solve(u0, phi, fix, dx):
    # solve poisson using over-relaxation method
    ndim = np.ndim(u0)
    denom = 2*np.sum(1./dx**2)
    phidenom = phi/denom

    # get the sum of the surrounding pixels and accummulate to the residuals
    lidxs = [_get_idx(slice(1,-1,None), ndim, i, slice(None,-2,None)) \
        for i in range(ndim)]
    ridxs = [_get_idx(slice(1,-1,None), ndim, i, slice( 2,None,None)) \
        for i in range(ndim)]
    inv_dxsq = 1./(dx**2 * denom)
    mult_fix = 1.0 - fix
    def getresid(u):
        u_pad = np.pad(u, [(1,1)]*ndim, "edge")
        resid = phidenom + u
        for i in range(ndim):
            resid -= (u_pad[ridxs[i]] + u_pad[lidxs[i]]) * inv_dxsq[i]
        resid *= mult_fix # zeroing out the residual for fixed elements
        f = np.sum(resid*resid)
        return f, resid

    solver = Momentum()
    return solver.solve(getresid, u0)

def _assert_same_shape(a, b, na, nb):
    if a.shape != b.shape:
        raise ValueError("The %s and %s must have the same shape" % (na, nb))

def _preprocess_fix(fix, shape):
    if type(fix) == str:
        if fix == "bounds":
            ndim = len(shape)
            fix = np.ones(shape)
            idx_interior = tuple([slice(1,-1,None)]*ndim)
            fix[idx_interior] = 0.0
            return fix > 0.0
        else:
            raise ValueError("Unknown fix value: %s" % fix)
    elif hasattr(fix, "__iter__"):
        fix = np.asarray(fix)
        if not isinstance(fix.item(0), numbers.Number):
            raise TypeError("fix must be an array of numbers or boolean")
        return fix > 0.0
    else:
        raise TypeError("fix must be a str or an array")

def _get_idx(defidx, ndim, i, idx):
    s = [defidx] * ndim
    s[i] = idx
    return tuple(s)

def test():
    import matplotlib.pyplot as plt
    n = 300
    ndim = 2

    xline = [np.linspace(-1,1,n)] * ndim
    x = np.asarray(np.meshgrid(*xline))
    phi = np.exp(-np.sum(x**2, axis=0)/(2*0.2**2))
    u0 = np.zeros([n]*ndim)

    u = solve(u0, phi)

    # # plot the solutions
    # plt.subplot(1,2,1)
    # plt.imshow(phi)
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(u)
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    test()
