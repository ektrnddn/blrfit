"""Synthetic checks of the curve tests of ``blrfit.multiepoch.verdicts``: the motion test is calibrated with
shared-spectrum noise, does not depend on the reference, finds a steady drift and flags a zig-zag, leave-one-out finds a
single bad spectrum, a shared instrument offset does not stack, pairs that do not add up get their own error, the
wandering term matches a simulated random walk and keeps the test calibrated, and the motion matrix gives the test
and its power."""

import numpy as np

from blrfit.multiepoch import verdicts as V


def make(vel, sig, floor, mjd, rng, pairs="all"):
    """Pairs of spectra 0..n-1 with true velocities vel; spectrum noise shared by every pair of a spectrum."""
    n = len(vel)
    noise = rng.normal(0, sig)
    idx = (
        [(i, j) for i in range(n) for j in range(i + 1, n)]
        if pairs == "all"
        else [(0, j) for j in range(1, n)]
    )
    out = []
    for i, j in idx:
        d = (vel[j] + noise[j]) - (vel[i] + noise[i]) + rng.normal(0, floor)
        out.append(dict(a=f"s{i}", b=f"s{j}", d=d, floor=floor))
    sigd = {f"s{i}": sig[i] for i in range(n)}
    times = {f"s{i}": mjd[i] for i in range(n)}
    return out, sigd, times


def test_motion_test_calibrated_with_shared_noise():
    rng = np.random.default_rng(1)
    mjd = np.array([52000.0, 55000.0, 59600.0, 60300.0])
    sig = np.array([80.0, 60.0, 40.0, 50.0])
    ps = []
    for _ in range(3000):
        pr, sigd, times = make(np.zeros(4), sig, 50.0, mjd, rng)
        cl = V.clusters_of(times)
        p, dof, fit, comp = V.motion_p(pr, sigd, cl, {k: t / 365.25 for k, t in times.items()})
        ps.append(p)
    ps = np.array(ps)
    assert abs(np.mean(ps < 0.05) - 0.05) < 0.015
    assert abs(np.mean(ps < 0.01) - 0.01) < 0.006


def test_reference_does_not_matter():
    rng = np.random.default_rng(2)
    mjd = np.array([52000.0, 55000.0, 59600.0])
    pr, sigd, times = make(np.array([0.0, 300.0, 100.0]), np.array([70.0, 50.0, 40.0]), 40.0, mjd, rng)
    cl = V.clusters_of(times)
    t = {k: v / 365.25 for k, v in times.items()}
    f0 = V.gls(pr, sigd, cl, t, ref_cluster=0)
    f2 = V.gls(pr, sigd, cl, t, ref_cluster=2)
    assert abs((f0["chi2_const"] - f0["chi2_free"]) - (f2["chi2_const"] - f2["chi2_free"])) < 1e-6
    assert abs((f0["u"][1] - f0["u"][2]) - (f2["u"][1] - f2["u"][2])) < 1e-6


def test_steady_drift_and_zigzag():
    rng = np.random.default_rng(3)
    mjd = np.array([51500.0, 53500.0, 55500.0, 57500.0, 59500.0, 60500.0])
    yr = (mjd - mjd[0]) / 365.25
    sig = np.full(6, 30.0)
    steady, zig = [], []
    for _ in range(300):
        pr, sigd, times = make(40.0 * yr, sig, 20.0, mjd, rng)
        cl = V.clusters_of(times)
        fit = V.gls(pr, sigd, cl, {k: t / 365.25 for k, t in times.items()}, 0)
        steady.append(V.stats.chi2.sf(fit["chi2_lin"] - fit["chi2_free"], len(fit["clusters"]) - 2))
        pr, sigd, times = make(np.array([0, 600, 0, 600, 0, 600.0]), sig, 20.0, mjd, rng)
        fit = V.gls(pr, sigd, cl, {k: t / 365.25 for k, t in times.items()}, 0)
        zig.append(V.stats.chi2.sf(fit["chi2_lin"] - fit["chi2_free"], len(fit["clusters"]) - 2))
    assert np.mean(np.array(steady) < 0.01) < 0.03  # a drift passes the steady test
    assert np.mean(np.array(zig) < 0.01) > 0.97  # a zig-zag fails it


def test_leave_one_out_finds_the_bad_spectrum():
    rng = np.random.default_rng(4)
    mjd = np.array([52000.0, 55000.0, 59600.0, 60300.0, 60700.0])
    sig = np.full(5, 40.0)
    pr, sigd, times = make(np.array([0, 0, 900.0, 0, 0]), sig, 30.0, mjd, rng)
    cl = V.clusters_of(times)
    t = {k: v / 365.25 for k, v in times.items()}
    p, _, _, comp = V.motion_p(pr, sigd, cl, t)
    assert p < V.P_MOVE
    rests = []
    for k in sorted(comp):
        pk, _, _, _ = V.motion_p([q for q in pr if k not in (q["a"], q["b"])], sigd, cl, t)
        if not (np.isfinite(pk) and pk < V.P_MARG):
            rests.append(k)
    assert rests == ["s2"]


def test_shared_instrument_offset_does_not_stack():
    """Five SDSS spectra of one season against three DESI nights: an offset shared by the object's SDSS spectra must
    not become a detection by averaging the five; with the omega term the test stays calibrated, without it not."""
    rng = np.random.default_rng(5)
    mjd = np.array([51800.0, 51830.0, 51860.0, 51890.0, 51920.0, 59600.0, 59630.0, 59660.0])
    inst = {f"s{i}": ("SDSS" if i < 5 else "DESI") for i in range(8)}
    sig = np.full(8, 30.0)
    om = 60.0
    p_with, p_without = [], []
    for _ in range(1500):
        shift = rng.normal(0, om)
        vel = np.array([shift] * 5 + [0.0] * 3)  # no motion, only the shared instrument offset
        pr, sigd, times = make(vel, sig, 10.0, mjd, rng)
        cl = V.clusters_of(times)
        t = {k: v / 365.25 for k, v in times.items()}
        p_with.append(V.motion_p(pr, sigd, cl, t, {"SDSS": om}, inst)[0])
        p_without.append(V.motion_p(pr, sigd, cl, t)[0])
    assert abs(np.mean(np.array(p_with) < 0.05) - 0.05) < 0.02
    assert np.mean(np.array(p_without) < 0.05) > 0.3


def test_pairs_that_do_not_add_up_get_their_own_error():
    """Pairs with an independent, template-dependent error (150 km/s here) on top of small shared spectrum noise:
    the triangle closures recover it, and with it the motion test stays calibrated (without it, it would not)."""
    rng = np.random.default_rng(6)
    mjd = np.array([52000.0, 54000.0, 56000.0, 58000.0, 59600.0, 60300.0])
    sig = np.full(6, 30.0)
    eps_true = 150.0
    ps_obj, ps_pop, eps_hat = [], [], []
    for _ in range(800):
        pr, sigd, times = make(np.zeros(6), sig, eps_true, mjd, rng)
        e, n_tri, e_obj = V.pair_eps(pr, 26.0)
        eps_hat.append(e_obj)
        cl = V.clusters_of(times)
        t = {k: v / 365.25 for k, v in times.items()}
        ps_obj.append(V.motion_p([dict(p, floor=e) for p in pr], sigd, cl, t)[0])
        ps_pop.append(V.motion_p([dict(p, floor=26.0) for p in pr], sigd, cl, t)[0])
    assert abs(np.median(eps_hat) - eps_true) < 0.25 * eps_true
    assert np.mean(np.array(ps_obj) < 0.05) < 0.12
    assert np.mean(np.array(ps_pop) < 0.05) > 0.5


def drw(t_yr, amp, tau, rng):
    """A damped random walk (Ornstein-Uhlenbeck) of rms amp and timescale tau at sorted times t_yr."""
    x = np.empty(len(t_yr))
    x[0] = rng.normal(0, amp)
    for i in range(1, len(t_yr)):
        a = np.exp(-(t_yr[i] - t_yr[i - 1]) / tau)
        x[i] = a * x[i - 1] + rng.normal(0, amp * np.sqrt(1 - a * a))
    return x


def test_blr_term_covariance_matches_simulation():
    """-1/2 M Gamma M^T is the covariance of the changes of a damped random walk."""
    rng = np.random.default_rng(7)
    t = np.array([0.0, 0.1, 3.0, 12.0, 12.05, 15.0])
    amp, tau = 100.0, 4.0
    idx = [(i, j) for i in range(len(t)) for j in range(i + 1, len(t))]
    M = np.zeros((len(idx), len(t)))
    for r, (i, j) in enumerate(idx):
        M[r, j], M[r, i] = 1.0, -1.0
    G = 2 * amp**2 * (1 - np.exp(-np.abs(t[:, None] - t[None, :]) / tau))
    C = -0.5 * M @ G @ M.T
    sims = np.array([M @ drw(t, amp, tau, rng) for _ in range(20000)])
    emp = np.cov(sims.T)
    assert np.max(np.abs(emp - C)) < 0.05 * np.max(C)


def test_blr_term_keeps_the_motion_test_calibrated():
    """An object whose broad line only wanders (a damped random walk, no orbit), seen by SDSS once and by DESI over
    four years: with the wandering term the motion test keeps its false-alarm rate; without it, the wandering looks
    like motion."""
    rng = np.random.default_rng(8)
    mjd = np.array([52000.0, 58800.0, 59200.0, 59600.0, 60000.0, 60400.0])
    t_yr = mjd / 365.25
    sig = np.full(6, 25.0)
    amp, tau = 80.0, 5.0
    p_with, p_without = [], []
    for _ in range(1500):
        vel = drw(t_yr, amp, tau, rng)
        pr, sigd, times = make(vel, sig, 8.0, mjd, rng)
        cl = V.clusters_of(times)
        t = {k: v / 365.25 for k, v in times.items()}
        p_with.append(V.motion_p(pr, sigd, cl, t, blr=(amp, tau))[0])
        p_without.append(V.motion_p(pr, sigd, cl, t)[0])
    assert abs(np.mean(np.array(p_with) < 0.05) - 0.05) < 0.02
    assert abs(np.mean(np.array(p_with) < 0.01) - 0.01) < 0.008
    assert np.mean(np.array(p_without) < 0.01) > 0.5


def test_blr_term_still_finds_an_orbit():
    """A steady drift of 60 km/s per year over 15 years (900 km/s) stands out above a wandering of 80 km/s rms."""
    rng = np.random.default_rng(9)
    mjd = np.array([52000.0, 54000.0, 58800.0, 59200.0, 59600.0, 60000.0, 60400.0])
    t_yr = mjd / 365.25
    sig = np.full(7, 25.0)
    found = []
    for _ in range(300):
        vel = 60.0 * (t_yr - t_yr[0]) + drw(t_yr, 80.0, 5.0, rng)
        pr, sigd, times = make(vel, sig, 8.0, mjd, rng)
        cl = V.clusters_of(times)
        t = {k: v / 365.25 for k, v in times.items()}
        found.append(V.motion_p(pr, sigd, cl, t, blr=(80.0, 5.0))[0] < V.P_MOVE)
    assert np.mean(found) > 0.9


def test_motion_matrix_gives_the_test_and_its_power():
    """d^T Q d is the motion test's chi2 difference; for d = d0 + noise drawn from the covariance, the rate above the
    threshold equals the noncentral chi2 prediction (the binary-fraction limit relies on both)."""
    from scipy.stats import chi2, ncx2

    rng = np.random.default_rng(10)
    mjd = np.array([52000.0, 55000.0, 59600.0, 59640.0, 60300.0])
    sig = np.array([60.0, 50.0, 30.0, 30.0, 40.0])
    pr, sigd, times = make(np.zeros(5), sig, 15.0, mjd, rng)
    cl = V.clusters_of(times)
    t = {k: v / 365.25 for k, v in times.items()}
    blr, omega, inst = (40.0, 4.0), {"SDSS": 30.0}, {f"s{i}": ("SDSS" if i < 2 else "DESI") for i in range(5)}
    Q, dof = V.motion_matrix(pr, sigd, cl, t, omega, inst, blr)
    fit = V.gls(pr, sigd, cl, t, min(cl.values()), omega, inst, blr)
    d = np.array([p["d"] for p in pr])
    assert abs(float(d @ Q @ d) - (fit["chi2_const"] - fit["chi2_free"])) < 1e-6 * max(1.0, fit["chi2_const"])
    C, spec = V.covariance(pr, sigd, t, omega, inst, blr)
    L = np.linalg.cholesky(C)
    vel = {f"s{i}": 150.0 * i for i in range(5)}  # an orbit-like drift
    d0 = np.array([vel[p["b"]] - vel[p["a"]] for p in pr])
    thr = chi2.isf(V.P_MOVE, dof)
    sims = d0[None, :] + (L @ rng.normal(size=(len(pr), 20000))).T
    rate = np.mean(np.einsum("ij,jk,ik->i", sims, Q, sims) > thr)
    pred = ncx2.sf(thr, dof, float(d0 @ Q @ d0))
    assert abs(rate - pred) < 0.015


def test_leave_one_time_out_removes_a_whole_time():
    """Two spectra at one time that both carry the only change: leaving one spectrum out keeps the motion, leaving
    the time out removes it."""
    rng = np.random.default_rng(11)
    mjd = np.array([52000.0, 55000.0, 59600.0, 59610.0])  # the last two are one time
    sig = np.full(4, 30.0)
    pr, sigd, times = make(np.array([0.0, 0.0, 700.0, 700.0]), sig, 10.0, mjd, rng)
    cl = V.clusters_of(times)
    assert cl["s2"] == cl["s3"]
    t = {k: v / 365.25 for k, v in times.items()}
    p, _, _, comp = V.motion_p(pr, sigd, cl, t)
    assert p < V.P_MOVE
    for k in ("s2", "s3"):  # one spectrum of the time left out: the other still carries the change
        pk, _, _, _ = V.motion_p([q for q in pr if k not in (q["a"], q["b"])], sigd, cl, t)
        assert pk < V.P_MOVE
    drop = {k for k in comp if cl[k] == cl["s2"]}
    pt, _, _, _ = V.motion_p([q for q in pr if q["a"] not in drop and q["b"] not in drop], sigd, cl, t)
    assert not (np.isfinite(pt) and pt < V.P_MOVE)
