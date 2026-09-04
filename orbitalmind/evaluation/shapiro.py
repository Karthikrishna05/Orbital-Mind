"""Own implementation of the Shapiro-Wilk normality test.

The competition (Note.pdf, Priority 1) requires teams to write their own SW code.
This is a faithful port of Royston's algorithm AS R94 (Royston 1992/1995), the
same algorithm used by MATLAB's ``swtest`` and R/scipy. Acceptance target: it must
reproduce W=0.9810, p=0.5840, H=0 on the organizer's 45-value reference dataset.

Self-contained: uses only numpy + stdlib ``math`` (inverse-normal via AS 241, the
normal tail via ``math.erfc``). scipy is used ONLY in tests to cross-check, never
here in the scored path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Royston AS R94 polynomial coefficients (constant term first) -------------
_C1 = (0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056)
_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_C3 = (0.544, -0.39978, 0.025054, -6.714e-4)
_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_C6 = (-0.4803, -0.082676, 0.0030302)
_G = (-2.273, 0.459)

_SMALL = 1e-19
_PI6 = 1.90985931710274      # 6 / pi
_STQR = 1.04719755119660     # asin(sqrt(3/4))


@dataclass
class ShapiroResult:
    W: float
    p: float
    H: int          # 1 = reject H0 (non-normal); 0 = fail to reject
    n: int
    method: str = "shapiro_wilk"

    def as_tuple(self):
        return self.W, self.p, self.H


def _poly(coef, x: float) -> float:
    """Evaluate coef[0] + coef[1]*x + coef[2]*x^2 + ... (Horner-free, per AS R94)."""
    result = coef[0]
    p = 1.0
    for c in coef[1:]:
        p *= x
        result += c * p
    return result


def _ppnd(p: float) -> float:
    """Inverse standard-normal CDF via AS 241 (Wichura, 1988), full precision."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
                    67265.770927008700853) * r + 45921.953931549871457) * r +
                  13731.693765509461125) * r + 1971.5909503065514427) * r +
                133.14166789178437745) * r + 3.387132872796366608)
        den = (((((((5226.495278852854561 * r + 28729.085735721942674) * r +
                    39307.89580009271061) * r + 21213.794301586595867) * r +
                  5394.1960214247511077) * r + 687.1870074920579083) * r +
                42.313330701600911252) * r + 1.0)
        return q * num / den
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
                    0.24178072517745061177) * r + 1.27045825245236838258) * r +
                  3.64784832476320460504) * r + 5.7694972214606914055) * r +
                4.6303378461565452959) * r + 1.42343711074968357734)
        den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
                    0.0151986665636164571966) * r + 0.14810397642748007459) * r +
                  0.68976733498510000455) * r + 1.6763848301838038494) * r +
                2.05319162663775882187) * r + 1.0)
        val = num / den
    else:
        r -= 5.0
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                    0.0012426609473880784386) * r + 0.026532189526576123093) * r +
                  0.29656057182850489123) * r + 1.7848265399172913358) * r +
                5.4637849111641143699) * r + 6.6579046435011037772)
        den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
                    1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
                  0.0148753612908506148525) * r + 0.13692988092273580531) * r +
                0.59983220655588793769) * r + 1.0)
        val = num / den
    return -val if q < 0 else val


def _normal_sf(z: float) -> float:
    """Upper-tail standard-normal survival function P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def shapiro_wilk(x, alpha: float = 0.05) -> ShapiroResult:
    """Compute the Shapiro-Wilk W statistic, p-value and hypothesis result.

    Parameters
    ----------
    x : array-like
        Sample values, 3 <= n <= 5000.
    alpha : float
        Significance level for the hypothesis decision (H=1 iff p < alpha).
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n < 3:
        raise ValueError("Shapiro-Wilk requires n >= 3")
    if n > 5000:
        raise ValueError("Shapiro-Wilk (AS R94) valid for n <= 5000")

    an = float(n)
    nn2 = n // 2
    a = np.zeros(nn2 + 1)  # 1-based weights a[1..nn2]

    # --- 1. normalized weights a[] ------------------------------------------
    if n == 3:
        a[1] = math.sqrt(0.5)
    else:
        an25 = an + 0.25
        summ2 = 0.0
        for i in range(1, nn2 + 1):
            a[i] = _ppnd((i - 0.375) / an25)
            summ2 += a[i] * a[i]
        summ2 *= 2.0
        ssumm2 = math.sqrt(summ2)
        rsn = 1.0 / math.sqrt(an)
        a1 = _poly(_C1, rsn) - a[1] / ssumm2

        if n > 5:
            i1 = 3
            a2 = -a[2] / ssumm2 + _poly(_C2, rsn)
            fac = math.sqrt((summ2 - 2.0 * a[1] ** 2 - 2.0 * a[2] ** 2) /
                            (1.0 - 2.0 * a1 ** 2 - 2.0 * a2 ** 2))
            a[1] = a1
            a[2] = a2
        else:
            i1 = 2
            fac = math.sqrt((summ2 - 2.0 * a[1] ** 2) / (1.0 - 2.0 * a1 ** 2))
            a[1] = a1
        for i in range(i1, nn2 + 1):
            a[i] = -a[i] / fac

    # --- 2. W statistic: squared correlation of ordered data with weights ----
    rng = x[-1] - x[0]
    if rng < _SMALL:
        # all values (nearly) identical -> degenerate; treat as perfectly normal
        return ShapiroResult(W=1.0, p=1.0, H=0, n=n)

    # signed full-length weight for order i (0-based): antisymmetric about center
    sa = 0.0
    sx = 0.0
    for i in range(n):
        j = n - 1 - i
        if i < j:
            w_i = -a[i + 1]
        elif i > j:
            w_i = a[j + 1]
        else:
            w_i = 0.0
        sa += w_i
        sx += x[i] / rng
    sa /= n
    sx /= n

    ssa = ssx = sax = 0.0
    for i in range(n):
        j = n - 1 - i
        if i < j:
            w_i = -a[i + 1]
        elif i > j:
            w_i = a[j + 1]
        else:
            w_i = 0.0
        asa = w_i - sa
        xsx = x[i] / rng - sx
        ssa += asa * asa
        ssx += xsx * xsx
        sax += asa * xsx

    ssassx = math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    W = 1.0 - w1

    # --- 3. p-value ----------------------------------------------------------
    if n == 3:
        pw = _PI6 * (math.asin(math.sqrt(W)) - _STQR)
        pw = min(max(pw, 0.0), 1.0)
        H = 1 if pw < alpha else 0
        return ShapiroResult(W=W, p=pw, H=H, n=n)

    y = math.log(w1)
    xx = math.log(an)
    if n <= 11:
        gamma = _poly(_G, an)
        if y >= gamma:
            pw = _SMALL
            H = 1 if pw < alpha else 0
            return ShapiroResult(W=W, p=pw, H=H, n=n)
        y = -math.log(gamma - y)
        m = _poly(_C3, an)
        s = math.exp(_poly(_C4, an))
    else:
        m = _poly(_C5, xx)
        s = math.exp(_poly(_C6, xx))

    z = (y - m) / s
    pw = _normal_sf(z)
    pw = min(max(pw, 0.0), 1.0)
    H = 1 if pw < alpha else 0
    return ShapiroResult(W=W, p=pw, H=H, n=n, method="shapiro_wilk")


def shapiro_francia(x, alpha: float = 0.05) -> ShapiroResult:
    """Shapiro-Francia (Weisberg-Bingham) normality test.

    W' = squared correlation between the ordered sample and the (Blom-approximated)
    expected normal order statistics; weights = m / ||m||. p-value via Royston
    (1993), valid for roughly 5 <= n <= 5000. For n < 5 we fall back to
    :func:`shapiro_wilk` (the SF p-value transform is undefined there).

    This is the statistic the competition's reference benchmark actually uses
    (see :func:`swtest`).
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n < 3:
        raise ValueError("Shapiro-Francia requires n >= 3")
    if n < 5:
        r = shapiro_wilk(x, alpha=alpha)
        return ShapiroResult(W=r.W, p=r.p, H=r.H, n=n, method="shapiro_francia_fallback_sw")

    mean = x.mean()
    ss = float(np.sum((x - mean) ** 2))
    if ss < _SMALL:
        return ShapiroResult(W=1.0, p=1.0, H=0, n=n, method="shapiro_francia")

    m = np.array([_ppnd((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)])
    w = m / math.sqrt(float(m @ m))
    W = float((w @ x) ** 2 / ss)
    W = min(W, 1.0)

    nu = math.log(n)
    u1 = math.log(nu) - nu
    u2 = math.log(nu) + 2.0 / nu
    mu = -1.2725 + 1.0521 * u1
    sigma = 1.0308 - 0.26758 * u2
    z = (math.log(1.0 - W) - mu) / sigma
    pw = _normal_sf(z)
    pw = min(max(pw, 0.0), 1.0)
    H = 1 if pw < alpha else 0
    return ShapiroResult(W=W, p=pw, H=H, n=n, method="shapiro_francia")


def _biased_kurtosis(x: np.ndarray) -> float:
    """MATLAB-style (biased) kurtosis: E[(x-mu)^4] / (E[(x-mu)^2])^2."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    if m2 <= 0:
        return 3.0
    return float(np.mean(d ** 4) / (m2 ** 2))


def swtest(x, alpha: float = 0.05) -> ShapiroResult:
    """Evaluator-faithful normality test, matching MATLAB ``swtest`` (BenSaida).

    Branches on biased kurtosis: kurtosis > 3 -> Shapiro-Francia (leptokurtic),
    else -> Shapiro-Wilk (Royston). This reproduces the organizer's reference
    (p=0.584 on the 45-value benchmark) and is the statistic our scoreboard
    optimizes against, because it is what the competition evaluator uses.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        raise ValueError("swtest requires n >= 3")
    if _biased_kurtosis(x) > 3.0 and n >= 5:
        res = shapiro_francia(x, alpha=alpha)
    else:
        res = shapiro_wilk(x, alpha=alpha)
    return ShapiroResult(W=res.W, p=res.p, H=res.H, n=res.n, method="swtest:" + res.method)


# Registry of normality statistics selectable by name.
# "swtest" is evaluator-faithful (what the competition uses); "shapiro_wilk" is
# the Note's literal wording; "shapiro_francia" is the leptokurtic branch alone.
STAT_FUNCS = {
    "swtest": swtest,
    "shapiro_wilk": shapiro_wilk,
    "shapiro_francia": shapiro_francia,
}
