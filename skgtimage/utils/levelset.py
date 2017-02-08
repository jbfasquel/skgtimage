# http://ascratchpad.blogspot.fr/2011/01/level-sets-and-image-segmentation-with.html

import numpy as np
from scipy import ndimage
import scipy.ndimage.filters


def NeumannBoundCond(f):
    # Make a function satisfy Neumann boundary condition
    nrow, ncol = f.shape
    g = f

    g[0, 0] = g[2, 2]
    g[0, ncol - 1] = g[2, ncol - 3]
    g[nrow - 1, 0] = g[nrow - 3, 2]
    g[nrow - 1, ncol - 1] = g[nrow - 3, ncol - 3]

    g[0, 1:-1] = g[2, 1:-1]
    g[nrow - 1, 1:-1] = g[nrow - 3, 1:-1]

    g[1:-1, 0] = g[1:-1, 2]
    g[1:-1, ncol - 1] = g[1:-1, ncol - 3]

    return g


def Dirac(x, sigma):
    f = (1. / 2. / sigma) * (1. + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    f = f * b;
    return f


def curvature_central(nx, ny):
    [junk, nxx] = np.gradient(nx)
    [nyy, junk] = np.gradient(ny)
    K = nxx + nyy
    return K


def evolution(u0, g, lam, mu, alf, epsilon, delt, numIter):
    """ Exemple of use:
        #Gradient of grayscale image (after a possible noise removal, e.g. smoothin)
        Iy,Ix=np.gradient(image_smooth)
        f=Ix**2+Iy**2
        #Level set parameters
        g=1. / (1.+f)  # edge indicator function.
        epsilon=1.5 # the paramater in the definition of smoothed Dirac function 1.5
        timestep=10  # time step 5
        mu=0.2/timestep  # coefficient of the internal (penalizing) energy term P(\phi) - Note: the product timestep*mu must be less than 0.25 for stability!
        lam=5 # coefficient of the weighted length term Lg(\phi) 5
        alf=3 # coefficient of the weighted area term Ag(\phi); 3 ; # Note: Choose a positive(negative) alf if the initial contour is outside(inside) the object.
        #Level set initialization: "contour" initial
        c0=4
        initialLSF=c0*np.ones(image.shape)
        w=100
        initialLSF[500+1:-700-1, w+1:-w-1]=-c0
        u=initialLSF
        #Iterative invocation
        u=skit.evolution(u, g ,lam, mu, alf, epsilon, timestep, 1)
    """
    u = u0
    vy, vx = np.gradient(g)
    for k in range(numIter):
        u = NeumannBoundCond(u)
        [uy, ux] = np.gradient(u)
        normDu = np.sqrt(ux ** 2 + uy ** 2 + 1e-10)
        Nx = ux / normDu
        Ny = uy / normDu
        diracU = Dirac(u, epsilon)
        K = curvature_central(Nx, Ny)
        weightedLengthTerm = lam * diracU * (vx * Nx + vy * Ny + g * K)
        penalizingTerm = mu * (scipy.ndimage.filters.laplace(u) - K)
        weightedAreaTerm = alf * diracU * g
        u = u + delt * (weightedLengthTerm + weightedAreaTerm + penalizingTerm)  # update the level set function

    return u
