"""
ODE solvers for timestepping.

"""

from collections import deque
import numpy as np
from scipy.sparse import linalg

from ..data.system import CoeffSystem, FieldSystem
from ..tools.config import config


# Load config options
permc_spec = config['linear algebra']['permc_spec']
use_umfpack = config['linear algebra'].getboolean('use_umfpack')


class ErrorControl:

    def __init__(self, TimeStepper, tolerance, iter=1):

        self.TimeStepper = TimeStepper
        self.tolerance = tolerance
        self.iter = iter
        self.last_iter_div = 0

    def __call__(self, nfields, domain):

        self.domain = domain
        self.timestepper = self.TimeStepper(nfields, domain, extra=2)
        self.coeff_system_1 = CoeffSystem(nfields, domain)
        self.coeff_system_2 = CoeffSystem(nfields, domain)

        return self

    def __getattr__(self, attrname):
        return getattr(self.timestepper, attrname)

    def step(self, solver, dt, wall_time, analysis=True, force=False):
        """
        Notes
        -----
        Truncation error:
            ε = C h**(p+1)
        ==> |X1 - X2| = C h**(p+1) (1 - 2**(-p))
            (   D   ) = C h**(p+1) (     f     )

        Desired error:
            τ X hn = C hn**(p+1)
        ==> (hn / h)**(p+1) = (τ X hn) / (D / f)
            (hn / h)**p     = f τ / (D / X / h)
            (hn / h)**p     = f τ / (    R    )
        ==> hn = safety * h * (f τ / R)**(1/p)

        Where:
            p  = scheme order
            h  = current timestep
            hn = new timestep
            τ  = tolerance (relative error per unit sim time)
            X1 = solution after one h step
            X2 = solution after two h/2 steps
            X  = scaling magnitudes

        """

        timestepper = self.timestepper
        CS1 = self.coeff_system_1
        CS2 = self.coeff_system_2

        order = timestepper.order
        tolerance = self.tolerance

        iter_div = (solver.iteration + 1) // self.iter
        scheduled = (iter_div > self.last_iter_div)
        #print(solver.iteration, scheduled)
        if (scheduled or force) and (timestepper._iteration >= timestepper._min_iteration):
            self.last_iter_div = iter_div
            # Store initial state (X0)
            np.copyto(CS1.data, solver.state.data)
            # Compute two half-steps (X2)
            timestepper.step(solver, dt/2, wall_time, analysis=False)
            timestepper.step(solver, dt/2, wall_time, analysis=False)
            np.copyto(CS2.data, solver.state.data)
            # Revert to initial state (X0)
            solver.sim_time -= dt
            np.copyto(solver.state.data, CS1.data)
            timestepper.rollback(2)
            # Compute single full-step (X1)
            timestepper.step(solver, dt, wall_time, analysis=analysis)
            # X
            CS1.data[:] = np.maximum(np.maximum(np.abs(CS1.data), np.abs(CS2.data)), np.abs(solver.state.data))
            # D = |X2 - X1|
            np.subtract(CS2.data, solver.state.data, out=CS2.data)
            np.abs(CS2.data, out=CS2.data)
            #print(np.max(CS2.data.real))
            #$print(np.abs(solver.state.data))
            # X = |X1| + |X0-X1|
            #np.subtract(CS1.data, solver.state.data, out=CS1.data)
            #np.abs(CS1.data, out=CS1.data)
            #np.add(CS1.data, np.abs(solver.state.data), out=CS1.data) # memory!
            #CS1.data[:] = np.maximum(np.maximum(np.abs(solver.state.data), np.abs(CS1.data)), np.abs(CS1.data-solver.state.data))
            #CS1.data[:] = np.maximum(np.abs(solver.state.data), np.abs(CS1.data))
            #CS1.data[:] = 0.5*np.abs(solver.state.data) + 1*np.abs(CS1.data-solver.state.data) + 0.5*np.abs(CS1.data)
            #np.copyto(CS1.data, solver.state.data)
            #np.abs(CS1.data, out=CS1.data)
            # Take maximum R = D / X / h
            #np.divide(CS2.data, CS1.data, out=CS2.data)
            #print(CS2.data)
            R = np.nanmax(CS2.data / CS1.data)
            #R = np.nanmax(CS2.data) / np.nanmax(CS1.data)
            # Compute new timestep
            f = (1 - 2**(-order))
            print(self.domain.distributor.rank, R, np.argmax(CS2.data))
            dt_new = dt * (f * tolerance / R)**(1/(order+1))
            # Take minimum new timestop across ranks
            self.control_dt = dt_new
        else:
            timestepper.step(solver, dt, wall_time, analysis=analysis)


class MultistepIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    into the general form
        aj M.X(n-j) + bj L.X(n-j) = cj F(n-j)
    where j runs from {0, 0, 1} to {amax, bmax, cmax}.

    The system is then solved as
        (a0 M + b0 L).X(n) = cj F(n-j) - aj M.X(n-j) - bj L.X(n-j)
    where j runs from {1, 1, 1} to {cmax, amax, bmax}.

    References
    ----------
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).*

    * Our coefficients are related to those used by Wang as:
        amax = bmax = cmax = s
        aj = α(s-j) / k(n+s-1)
        bj = γ(s-j)
        cj = β(s-j)

    """

    min_iteration = None

    def __init__(self, nfields, domain, extra=0):

        self.RHS = CoeffSystem(nfields, domain)

        # Create deque for storing recent timesteps
        N = max(self.amax, self.bmax, self.cmax) + extra
        self.dt = deque([0.]*N)

        # Create coefficient systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()
        for j in range(self.amax + extra):
            MX.append(CoeffSystem(nfields, domain))
        for j in range(self.bmax + extra):
            LX.append(CoeffSystem(nfields, domain))
        for j in range(self.cmax + extra):
            F.append(CoeffSystem(nfields, domain))

        # Attributes
        self._iteration = 0

    def step(self, solver, dt, wall_time, analysis=True):
        """Advance solver by one timestep."""

        # Solver references
        pencils = solver.pencils
        evaluator = solver.evaluator
        state = solver.state
        Fe = solver.Fe
        Fb = solver.Fb
        sim_time = solver.sim_time
        iteration = solver.iteration

        # References
        MX = self.MX
        LX = self.LX
        F = self.F
        RHS = self.RHS

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(self.dt, self._iteration)
        self._iteration += 1

        # Run evaluator
        state.scatter()
        if analysis:
            evaluator.evaluate_scheduled(wall_time, sim_time, iteration)
        else:
            evaluator.evaluate_group('F', wall_time, sim_time, iteration)

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        a0 = a[0]
        b0 = b[0]

        for p in pencils:
            x = state.get_pencil(p)
            pFe = Fe.get_pencil(p)
            pFb = Fb.get_pencil(p)

            MX0.set_pencil(p, p.M*x)
            LX0.set_pencil(p, p.L*x)
            F0.set_pencil(p, p.G_eq*pFe + p.G_bc*pFb)

            np.copyto(p.LHS.data, a0*p.M.data + b0*p.L.data)

        # Build RHS
        RHS.data.fill(0)
        for j in range(1, len(c)):
            RHS.data += c[j] * F[j-1].data
        for j in range(1, len(a)):
            RHS.data -= a[j] * MX[j-1].data
        for j in range(1, len(b)):
            RHS.data -= b[j] * LX[j-1].data

        # Solve
        for p in pencils:
            A = p.LHS
            b = RHS.get_pencil(p)
            x = linalg.spsolve(A, b, use_umfpack=use_umfpack, permc_spec=permc_spec)
            state.set_pencil(p, x)

        # Update solver
        solver.sim_time += dt

    def rollback(self, iterations):
        self._iteration -= iterations
        self.dt.rotate(-iterations)
        self.LX.rotate(-iterations)
        self.MX.rotate(-iterations)
        self.F.rotate(-iterations)


class CNAB1(MultistepIMEX):
    """
    1st-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.5.3]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 1st-order Adams-Bashforth (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1
    order = 1
    _min_iteration = 0

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1

        return a, b, c


class SBDF1(MultistepIMEX):
    """
    1st-order semi-implicit BDF scheme [Wang 2008 eqn 2.6]

    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1
    order = 1
    _min_iteration = 0

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1
        c[1] = 1

        return a, b, c


class CNAB2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.9]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2
    order = 2
    _min_iteration = 1

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return CNAB1.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


class MCNAB2(MultistepIMEX):
    """
    2nd-order modified Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.10]

    Implicit: 2nd-order modified Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2
    order = 2
    _min_iteration = 1

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return CNAB1.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = (8 + 1/w1) / 16
        b[1] = (7 - 1/w1) / 16
        b[2] = 1 / 16
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


class SBDF2(MultistepIMEX):
    """
    2nd-order semi-implicit BDF scheme [Wang 2008 eqn 2.8]

    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation

    """

    amax = 2
    bmax = 2
    cmax = 2
    order = 2
    _min_iteration = 1

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return SBDF1.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = (1 + 2*w1) / (1 + w1) / k1
        a[1] = -(1 + w1) / k1
        a[2] = w1**2 / (1 + w1) / k1
        b[0] = 1
        c[1] = 1 + w1
        c[2] = -w1

        return a, b, c


class CNLF2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson leap-frog scheme [Wang 2008 eqn 2.11]

    Implicit: ?-order wide Crank-Nicolson
    Explicit: 2nd-order leap-frog

    """

    amax = 2
    bmax = 2
    cmax = 2
    order = 2
    _min_iteration = 1

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return CNAB1.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / (1 + w1) / k1
        a[1] = (w1 - 1) / k1
        a[2] = -w1**2 / (1 + w1) / k1
        b[0] = 1 / w1 / 2
        b[1] = (1 - 1/w1) / 2
        b[2] = 1 / 2
        c[1] = 1

        return a, b, c


class SBDF3(MultistepIMEX):
    """
    3rd-order semi-implicit BDF scheme [Wang 2008 eqn 2.14]

    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation

    """

    amax = 3
    bmax = 3
    cmax = 3
    order = 3
    _min_iteration = 2

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return SBDF2.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k2, k1, k0, *rest = timesteps
        w2 = k2 / k1
        w1 = k1 / k0

        a[0] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[1] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[2] = w2**2 * (w1 + 1/(1 + w2)) / k2
        a[3] = -w1**3 * w2**2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[0] = 1
        c[1] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[2] = -w2*(1 + w1*(1 + w2))
        c[3] = w1*w1*w2*(1 + w2) / (1 + w1)

        return a, b, c


class SBDF4(MultistepIMEX):
    """
    4th-order semi-implicit BDF scheme [Wang 2008 eqn 2.15]

    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation

    """

    amax = 4
    bmax = 4
    cmax = 4
    order = 4
    _min_iteration = 3

    @classmethod
    def compute_coefficients(self, timesteps, ts_iteration):

        if ts_iteration < self._min_iteration:
            return SBDF3.compute_coefficients(timesteps, ts_iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k3, k2, k1, k0, *rest = timesteps
        w3 = k3 / k2
        w2 = k2 / k1
        w1 = k1 / k0

        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3)
        A3 = 1 + w1*A2

        a[0] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[1] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[2] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[3] = -w2**3 * w3**2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[4] = (1 + w3) / (1 + w1) * A2 / A1 * w1**4 * w2**3 * w3**2 / A3 / k3
        b[0] = 1
        c[1] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[2] = -A2 * A3 * w3 / (1 + w1)
        c[3] = w2**2 * w3 * (1 + w3) / (1 + w2) * A3
        c[4] = -w1**3 * w2**2 * w3 * (1 + w3) / (1 + w1) * A2 / A1

        return a, b, c


class RungeKuttaIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    by constructing s stages
        M.X(n,i) - M.X(n,0) + k Hij L.X(n,j) = k Aij F(n,j)
    where j runs from {0, 0} to {i, i-1}, and F(n,i) is evaluated at time
        t(n,i) = t(n,0) + k ci

    The s stages are solved as
        (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
    where j runs from {0, 0} to {i-1, i-1}.

    The final stage is used as the advanced solution*:
        X(n+1,0) = X(n,s)
        t(n+1,0) = t(n,s) = t(n,0) + k

    * Equivalently the Butcher tableaus must follow
        b_im = H[s, :]
        b_ex = A[s, :]
        c[s] = 1

    References
    ----------
    U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Applied Numerical Mathematics (1997).

    """

    _min_iteration = 0

    def __init__(self, nfields, domain, extra=0):

        self.RHS = CoeffSystem(nfields, domain)
        self._iteration = 0

        # Create coefficient systems for stages
        self.MX0 = CoeffSystem(nfields, domain)
        self.LX = LX = [CoeffSystem(nfields, domain) for i in range(self.stages)]
        self.F = F = [CoeffSystem(nfields, domain) for i in range(self.stages)]

    def step(self, solver, dt, wall_time, analysis=True):
        """Advance solver by one timestep."""

        # Solver references
        pencils = solver.pencils
        evaluator = solver.evaluator
        state = solver.state
        Fe = solver.Fe
        Fb = solver.Fb
        sim_time_0 = solver.sim_time
        iteration = solver.iteration

        # Other references
        RHS = self.RHS
        MX0 = self.MX0
        LX = self.LX
        F = self.F
        A = self.A
        H = self.H
        c = self.c
        k = dt

        # Compute M.X(n,0)
        for p in pencils:
            pX0 = state.get_pencil(p)
            MX0.set_pencil(p, p.M*pX0)

        # Compute stages
        # (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
        for i in range(1, self.stages+1):

            # Compute F(n,i-1), L.X(n,i-1)
            state.scatter()
            if (i == 1) and analysis:
                evaluator.evaluate_scheduled(wall_time, solver.sim_time, iteration)
            else:
                evaluator.evaluate_group('F', wall_time, solver.sim_time, iteration)
            for p in pencils:
                pX = state.get_pencil(p)
                pFe = Fe.get_pencil(p)
                pFb = Fb.get_pencil(p)
                LX[i-1].set_pencil(p, p.L*pX)
                F[i-1].set_pencil(p, p.G_eq*pFe + p.G_bc*pFb)

            # Construct RHS(n,i)
            np.copyto(RHS.data, MX0.data)
            for j in range(0, i):
                RHS.data += k * A[i,j] * F[j].data
                RHS.data -= k * H[i,j] * LX[j].data

            for p in pencils:
                # Construct LHS(n,i)
                np.copyto(p.LHS.data, p.M.data + (k*H[i,i])*p.L.data)
                # Solve for X(n,i)
                pRHS = RHS.get_pencil(p)
                pX = linalg.spsolve(p.LHS, pRHS, use_umfpack=use_umfpack, permc_spec=permc_spec)
                state.set_pencil(p, pX)
            solver.sim_time = sim_time_0 + k*c[i]

        self._iteration += 1

    def rollback(self, iterations):
        self._iteration -= iterations


class RK111(RungeKuttaIMEX):
    """1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]"""

    stages = 1
    order = 1

    c = np.array([0, 1])

    A = np.array([[0, 0],
                  [1, 0]])

    H = np.array([[0, 0],
                  [0, 1]])


class RK222(RungeKuttaIMEX):
    """2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]"""

    stages = 2
    order = 2

    γ = (2 - np.sqrt(2)) / 2
    δ = 1 - 1 / γ / 2

    c = np.array([0, γ, 1])

    A = np.array([[0,  0 , 0],
                  [γ,  0 , 0],
                  [δ, 1-δ, 0]])

    H = np.array([[0,  0 , 0],
                  [0,  γ , 0],
                  [0, 1-γ, γ]])


class RK443(RungeKuttaIMEX):
    """3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8]"""

    stages = 4
    order = 3

    c = np.array([0, 1/2, 2/3, 1/2, 1])

    A = np.array([[  0  ,   0  ,  0 ,   0 , 0],
                  [ 1/2 ,   0  ,  0 ,   0 , 0],
                  [11/18,  1/18,  0 ,   0 , 0],
                  [ 5/6 , -5/6 , 1/2,   0 , 0],
                  [ 1/4 ,  7/4 , 3/4, -7/4, 0]])

    H = np.array([[0,   0 ,   0 ,  0 ,  0 ],
                  [0,  1/2,   0 ,  0 ,  0 ],
                  [0,  1/6,  1/2,  0 ,  0 ],
                  [0, -1/2,  1/2, 1/2,  0 ],
                  [0,  3/2, -3/2, 1/2, 1/2]])

