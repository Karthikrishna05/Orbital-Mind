"""Deterministic solar-geometry features from the UTC timestamp.

Honest scope note: a *true* Beta angle and eclipse flag require the satellite's
orbital plane (RAAN, inclination) and longitude, which this dataset does not
provide (we only have error residuals + timestamps). What IS deterministically
computable from the timestamp are the *solar* drivers behind the thermal/eclipse
cycles: solar declination, the equation of time, the Greenwich solar hour angle,
and proximity to the equinox eclipse season. These are physically-motivated
proxies, cheap and additive, and pure functions of time (so they extend to any
future query timestamp).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

# Equinoxes (approximate day-of-year): vernal ~80 (Mar 20), autumnal ~266 (Sep 23).
_EQUINOX_DOY = (80.0, 266.0)
# GEO eclipse season half-width in days (satellite crosses Earth's shadow within
# roughly +/-23 days of equinox).
_ECLIPSE_HALFWIDTH = 23.0

FEATURE_NAMES = ("solar_decl", "eot_min", "sun_ha_sin", "sun_ha_cos",
                 "eclipse_season", "eclipse_midnight")


def _day_of_year_and_hour(t_seconds):
    t = np.asarray(t_seconds, dtype=float)
    doy = np.empty(t.size)
    utc_hour = np.empty(t.size)
    for i, ts in enumerate(t):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        doy[i] = dt.timetuple().tm_yday + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        utc_hour[i] = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return doy, utc_hour


def astro_features(t_seconds):
    """Return (X, names): (n, 6) matrix of solar-geometry features."""
    doy, utc_hour = _day_of_year_and_hour(t_seconds)

    # Solar declination (deg) -- Cooper's approximation.
    decl = -23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (doy + 10.0)))

    # Equation of time (minutes).
    B = np.deg2rad(360.0 / 364.0 * (doy - 81.0))
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    # Greenwich solar hour angle (deg): 0 at solar noon over Greenwich.
    ha = 15.0 * (utc_hour - 12.0) + eot / 4.0
    ha_rad = np.deg2rad(ha)

    # Eclipse-season proximity: Gaussian bump around the nearest equinox.
    dist = np.minimum(np.abs(doy - _EQUINOX_DOY[0]), np.abs(doy - _EQUINOX_DOY[1]))
    eclipse_season = np.exp(-(dist / _ECLIPSE_HALFWIDTH) ** 2)

    # Midnight-weighted eclipse proxy: shadow crossing is near local midnight and
    # only matters in season. cos(ha) ~ -1 near anti-solar (midnight) side.
    eclipse_midnight = eclipse_season * (0.5 * (1.0 - np.cos(ha_rad)))

    X = np.column_stack([decl, eot, np.sin(ha_rad), np.cos(ha_rad),
                         eclipse_season, eclipse_midnight])
    # standardize non-bounded columns for conditioning
    for j in (0, 1):
        s = X[:, j].std()
        if s > 0:
            X[:, j] = (X[:, j] - X[:, j].mean()) / s
    return X, list(FEATURE_NAMES)
