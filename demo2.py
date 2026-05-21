import mpmath as mp

mp.mp.dps = 50   # precision for overtones


# Parameters


ar = mp.mpf('0.49')
q  = mp.mpf('-0.9')

br = mp.sqrt(1 - 4*ar**2 - 4*q)
rminus = (1 - br)/2
rplus  = (1 + br)/2

s = -2
m = 2
l = 2

k1 = abs(m - s)/2
k2 = abs(m + s)/2

Alm = l*(l + 1) - s*(s + 1)   # initial guess

NITMAX = 200


# ============================================================
# Radial recurrence coefficients
# ============================================================

def Gamma(n, w):
    return (n - 2j*w) * (
        -2j*q*w*br
        + 1j*w*br
        - 2j*ar*m*br
        + 4*ar**2*(n + s - 1j*w)
    ) + (n - 2j*w)*(4*q - 1)*(n + s - 1j*w)


def Beta(n, w, Alm):
    return (
        -4*ar**4*w**2
        - 8*ar**3*m*w
        + 12*q*w**2*br
        - 4*w**2*br
        - 2j*w*br
        + 6j*q*w*br
        - 4j*n*w*br
        + 2*ar*m*(1j*br + 2*br*(w + 1j*n) - 4*q*w + w)
        + 12j*q*n*w*br
        + (1 - 4*q)*(Alm + 2*n*(n + 1) + s + 1)
        - 4*(4*q**2 - 5*q + 1)*w**2
        + 2j*(4*q - 1)*(2*n + 1)*w
        + ar**2*( w*( w*(8*br - 20*q + 17) + 4j*(br + 2) )
                 + 8j*n*( w*(br + 2) + 1j ) )
        - (4*Alm + 8*n**2 + 4*s + 4)*ar**2
    )


def Alpha(n, w):
    return (n + 1)*(
        -2j*q*w*br
        + 1j*w*br
        - 2j*ar*m*br
        + (n + 1)*(4*ar**2 + 4*q - 1)
    ) - (n + 1)*(4*ar**2 + 4*q - 1)*(s + 1j*w)


# ============================================================
# Radial continued fraction with overtone shift
# ============================================================

def radial_CF(w, Alm, n0):

    # ---- DOWNWARD PART ----
    R_down = mp.mpc(0)
    for n in reversed(range(n0 + 1, NITMAX)):
        R_down = Gamma(n, w) / (Beta(n, w, Alm) - Alpha(n, w)*R_down)

    down_part = Alpha(n0, w) * R_down

    # ---- UPWARD PART ----
    R_up = mp.mpc(0)
    for n in range(n0):
        R_up = Alpha(n, w) / (Beta(n, w, Alm) - Gamma(n, w)*R_up)

    up_part = Gamma(n0, w) * R_up if n0 > 0 else 0

    return Beta(n0, w, Alm) - up_part - down_part



# ============================================================
# Angular recurrence
# ============================================================

def Gamma_ang(n, w):
    return 2*ar*w*(n + k1 + k2 + s)


def Beta_ang(n, w, Sep):
    return (
        n*(n - 1)
        + 2*n*(k1 + k2 + 1 - 2*ar*w)
        - (2*ar*w*(2*k1 + s + 1)
           - (k1 + k2)*(k1 + k2 + 1))
        - (ar**2*w**2 + s*(s + 1) + Sep)
    )


def Alpha_ang(n):
    return -2*(n + 1)*(n + 2*k1 + 1)


def angular_CF(w, Sep):
    R = mp.mpc(-1)

    for n in reversed(range(1, NITMAX)):
        R = Gamma_ang(n, w) / (Beta_ang(n, w, Sep) - Alpha_ang(n)*R)

    return Beta_ang(0, w, Sep) - Alpha_ang(0)*R


# ============================================================
# Coupled solve for overtone n
# ============================================================

def compute_mode(n_overtone, w_guess):

    Sep_guess = l*(l + 1) - s*(s + 1)

    for _ in range(7):   # same as your Mathematica loop

        # Radial root
        w = mp.findroot(
            lambda ww: radial_CF(ww, Sep_guess, n_overtone),
            w_guess
        )

        # Angular root
        Sep_guess = mp.findroot(
            lambda SS: angular_CF(w, SS),
            Sep_guess
        )

        w_guess = w

    return w


# ============================================================
# initial guesses


initial_guesses = [
    0.52327057768813286464047353969288643566945066203288 - 0.1362786029864463867028189351919020206065185862693j,   # n = 0
    0.47891506910181746782770759208598775195896637968588 - 0.42635082773708161414980278287350787655841866293876j,   # n = 1
    0.40416762927739803046587923700170771350140908625398 - 0.76423495580808010842479958710672681901182880768887j,   # n = 2
    0.33056669196233815096788281532554135916087425699599 - 1.1539506279505343083372684149095994438688290768043j,   # n = 3
    0.27547431617735875477219070512309398376472249137509 - 1.5727275794261823751016093161614110307184971794732j,  # n = 4
    0.23645233788601378915819189203013402416325255363156 - 2.0027244916307850273154341556207856427516130588001j,      # n = 5
    0.1855599546486822787479819821464066431743363860178 - 2.871725816344217553825667591866469423088803187803j,                # n = 6
    0.17087017022264248562154695529051319273200410686885 - 3.3061410673711128085021626389500902384357288210673j        # n = 7
]


modes = []

for n, guess in enumerate(initial_guesses):
    w_mode = compute_mode(n, guess)
    modes.append(w_mode)
    print(f"n = {n}  -->  w = {w_mode}")
