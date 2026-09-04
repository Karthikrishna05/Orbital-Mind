# Search for a larger dataset of the same type — findings

**Requirement:** a large dataset, exact same schema (`utc_time, x_error, y_error,
z_error, satclockerror`), multiple satellites with a good GEO/MEO mix, and
non-uniform sampling like ours.

## Bottom line

**No larger or different dataset of this exact type exists publicly. What I
found instead conclusively explains why**, and identifies the one real path to
get more data (which requires computation, not a download).

---

## 1. This is ISRO's Smart India Hackathon 2025, Problem Statement PS-25176

Web search confirms our `Note.pdf`/`SIH_Data_Discription.pdf` files are ISRO's
**SIH25176 "OrbitIQ"** problem statement: *"AI/ML-based models to predict
time-varying patterns of the error build-up between uploaded and modelled values
of both satellite clock and ephemeris parameters"* — same wording, same task.

## 2. Every team's dataset is byte-identical to ours — confirmed on GitHub

I searched GitHub code search for the exact column name `satclockerror` and the
exact filenames `DATA_GEO_Train.csv` / `DATA_MEO_Train.csv` and found **~20
independent team repositories** for this same problem statement, e.g.:

- `Amit-jha98/GNSS_Error_Predictor`, `Mapicx/TimeForge_2.0`,
  `Krishna-mishra-26/SIH-NAVPREDICT`, `codewithRahul01/GNSS-Error-prediction`,
  `Rohithjava777/ROHIH-GNNS`, `diiviikk5/Stellar-v1k`, `justaguy1337/gnss`,
  `Tamaghna1/SIH2025`, `gankit-aiml/TeamID_59634_SUB2`, `awasthi108/NavAi-backend`,
  `piyushzsharma/ISRO-SCOPE`, and more.

**Every one of them ships the identical `DATA_GEO_Train.csv` / `DATA_MEO_Train.csv`
/ `DATA_MEO_Train2.csv` files** — same filenames, same columns. I directly
verified one team's merged file: `Tamaghna1/SIH2025/GEO_full.csv` has exactly
**211 data rows = 142 (our train) + 69 (our test)** — i.e. it's just *our* train
and test concatenated, not new data.

**Conclusion: ISRO distributed one fixed dataset to every competing team.** There
is no "bigger" version circulating — everyone has exactly what we have. Searching
further within this competition's ecosystem cannot find a larger dataset because
none exists; it is the same file, forked ~20 times.

## 3. Why a larger *real* GEO+MEO dataset of this shape doesn't exist elsewhere either

I searched IGS/MGEX, NASA CDDIS, Kaggle, Hugging Face, and Zenodo for a
ready-made dataset with these columns. None exists, and the literature explains
why:

- **The underlying real-world quantity is genuine** — it's the difference between
  a GNSS satellite's **broadcast ephemeris/clock** (what's uploaded to the
  satellite) and its **precise reference ephemeris/clock** (post-hoc truth) —
  exactly our `x/y/z_error` + `satclockerror`. This is computed by researchers
  from raw **NASA CDDIS / IGS MGEX** archives (broadcast RINEX nav files vs.
  precise SP3/CLK products from analysis centers like CODE/GFZ/WHU), but **no one
  publishes the differenced result as a ready CSV** — every paper recomputes it
  from raw files for its own study window.
- **GEO is the specific reason no one has published this for GEO satellites.**
  One directly relevant finding: research explicitly states *"satellites in the
  GEO orbit have not been tested because a reference precise ephemeris is at this
  time unavailable"* — precise (truth) orbit products for GEO/IGSO satellites are
  much rarer than for MEO constellations (GPS/Galileo/BeiDou-MEO), because most
  analysis centers focus precise-orbit computation on MEO constellations. This
  independently corroborates our own finding that **GEO is the hard, outlier-prone
  case** — it's a known, documented data-scarcity problem in the field, not
  something specific to ISRO's dataset.
- **NavIC/IRNSS (ISRO's own real constellation) is the closest real analog** — 7
  satellites, a genuine mix of **3 GEO + 4 GSO/inclined ("MEO-like")** — and NASA
  CDDIS does host real IRNSS **broadcast** ephemeris files. But the same GEO
  precise-ephemeris scarcity applies, so a differenced error dataset isn't
  published for it either.

## 4. What I did NOT find (checked and ruled out)
- No Kaggle dataset with these columns or this GEO/MEO clock-error shape.
- No Hugging Face dataset with `x_error/y_error/z_error/satclockerror`-style
  columns for GNSS.
- No Zenodo archive with a pre-differenced GEO+MEO broadcast-vs-precise error
  series at non-uniform sampling.
- No SISRE/CDDIS pre-packaged CSV — SISRE is always computed per-study from raw
  RINEX+SP3, never distributed as a table.

## 5. What would actually work, if you want more (real) data

Not a download — a **build**, using the same pipeline every cited paper uses:
1. Pull **broadcast** navigation files (RINEX 3/4 `.rnx`) from NASA CDDIS/IGS for
   the constellation(s) you want (GPS/Galileo/BeiDou have the best MEO precise-orbit
   coverage).
2. Pull matching **precise** SP3 (orbit) + CLK (clock) products from an IGS/MGEX
   analysis center (CODE, GFZ, WHU) for the same days.
3. Interpolate the broadcast position/clock to the precise product's epochs (or
   vice versa) and difference them, in ECEF X/Y/Z + clock (× speed of light, as
   our `SIH_Data_Discription.pdf` specifies) — this reproduces our exact schema
   at whatever scale/satellite-count you choose.
4. **GEO/IGSO** (India's own GAGAN/NavIC GEO satellites, or other GNSS's
   GEO-augmentation satellites) will be the bottleneck — precise products are
   sparse-to-nonexistent for them, mirroring the exact difficulty already
   observed in our GEO series.

This is a multi-day data-engineering project on its own, not a quick fetch — and
this session has not attempted it, since it's out of scope unless you want it
built.

## Sources
- [Galileo Broadcast Ephemeris and Clock Errors Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7731340/)
- [MGEX Data + Products — International GNSS Service](https://igs.org/mgex/data-products/)
- [Robust Modeling of GNSS Orbit and Clock Error Dynamics](https://navi.ion.org/content/69/4/navi.539) (GEO precise-ephemeris unavailability)
- [NASA CDDIS — IRNSS Broadcast Ephemeris Data](https://catalog.data.gov/dataset/ground-based-global-navigation-satellite-system-gnss-irnss-broadcast-ephemeris-data-sub-ho)
- [Broadcast Ephemeris with Centimetric Accuracy (GPS/Galileo/BeiDou/GLONASS)](https://www.mdpi.com/2072-4292/13/20/4185)
- GitHub code search (this session): `DATA_GEO_Train`, `satclockerror` across ~20 SIH25176 team repos, e.g. [Tamaghna1/SIH2025](https://github.com/Tamaghna1/SIH2025), [Mapicx/TimeForge_2.0](https://github.com/Mapicx/TimeForge_2.0), [codewithRahul01/GNSS-Error-prediction](https://github.com/codewithRahul01/GNSS-Error-prediction)
