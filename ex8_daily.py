"""
ex8_daily.py -- Exercise 8 on real data (daily_airport_stats.csv).

QUESTION (one sentence for the top of your write-up):
  How well do validation and test losses from 120 days track the true error
  on the other 488 days, and does using contiguous 10-day blocks instead of
  random days change what we conclude (chosen H, optimism of the estimates)?

DATA
  One point per day (608 days, Jan 2025 - Aug 2026):
    x = day of year, rescaled to [-2, 2]
    y = national mean departure delay in minutes (airports weighted by
        scheduled departures)
  Jan-Aug appear twice (2025 and 2026) with different y: that is real noise.

DESIGN (per repeat; 15 repeats per scheme)
  'random' : 120 random days, shuffled into 30 train / 30 val / 60 test.
  'blocked': 12 random non-overlapping blocks of 10 consecutive days,
             3 blocks train / 3 val / 6 test (neighbouring days stay together).
  H is chosen by validation loss; the test set is used ONCE per repeat, for
  that chosen model only. 'Fresh' = all 488 days not sampled (true-error proxy).
  Losses are in standardised units (y standardised with TRAIN mean/sd only).

Run (needs fit.py next to it; a few minutes on a laptop CPU):
    python ex8_daily.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fit

HS = [0, 1, 2, 5, 10, 20]
EPOCHS = 3000
REPEATS = 15
BLOCK = 10
plt.rcParams.update({"font.size": 11})

# ---------------- data ----------------
d = pd.read_csv("daily_airport_stats.csv", parse_dates=["flight_date"])
d = d.dropna(subset=["dep_delay_mean"])
d["wd"] = d.dep_delay_mean * d.departures_scheduled
g = d.groupby("flight_date").agg(wd=("wd", "sum"), w=("departures_scheduled", "sum"))
Y = (g.wd / g.w).to_numpy()
dates = g.index
X = (dates.dayofyear.to_numpy() - 1) / 365.0 * 4.0 - 2.0
N = len(X)
print(f"{N} days, mean delay {Y.mean():.1f} min, sd {Y.std():.1f}")
xgrid = np.linspace(-2, 2, 400)


# ---------------- splitting schemes: return index arrays ----------------
def random_scheme(rng):
    idx = rng.choice(N, 120, replace=False)          # already in random order
    return idx[:30], idx[30:60], idx[60:]


def blocked_scheme(rng):
    starts = np.arange(0, N - BLOCK + 1, BLOCK)       # non-overlapping blocks
    pick = rng.choice(starts, 12, replace=False)
    blocks = [np.arange(s, s + BLOCK) for s in pick]
    cat = lambda bl: np.concatenate(bl)
    return cat(blocks[:3]), cat(blocks[3:6]), cat(blocks[6:])


SCHEMES = {"random": random_scheme, "blocked": blocked_scheme}


def one_repeat(scheme, r):
    rng = np.random.default_rng(100 + r)
    tr, va, te = SCHEMES[scheme](rng)
    used = np.concatenate([tr, va, te])
    fresh = np.setdiff1d(np.arange(N), used)           # 488 unseen days
    m, s = Y[tr].mean(), Y[tr].std()
    f = lambda v: (v - m) / s
    nets, vals = [], []
    for H in HS:
        net, _ = fit.train(X[tr], f(Y[tr]), hidden=H, epochs=EPOCHS)
        nets.append(net)
        vals.append(fit.mse(net, X[va], f(Y[va])))
    k = int(np.argmin(vals))
    net = nets[k]
    out = dict(H=HS[k], val=vals[k],
               test=fit.mse(net, X[te], f(Y[te])),     # test set: once per repeat
               fresh=fit.mse(net, X[fresh], f(Y[fresh])))
    return out, (net, m, s, tr)


results = {k: [] for k in SCHEMES}
example = None
for scheme in SCHEMES:
    for r in range(REPEATS):
        out, ex = one_repeat(scheme, r)
        results[scheme].append(out)
        if scheme == "random" and r == 0:
            example = ex
        print(f"[{scheme}] repeat {r:2d}: H={out['H']:3d} val={out['val']:.3f} "
              f"test={out['test']:.3f} fresh={out['fresh']:.3f}")

# ---------------- summary ----------------
rows = {}
for scheme in SCHEMES:
    R = pd.DataFrame(results[scheme])
    rows[scheme] = R
    print(f"\n[{scheme}] chosen H counts:\n{R.H.value_counts().sort_index().to_string()}")
    print(f"[{scheme}] mean val={R.val.mean():.3f}  test={R.test.mean():.3f}  "
          f"fresh={R.fresh.mean():.3f}")
    print(f"[{scheme}] mean (val-fresh)={np.mean(R.val-R.fresh):+.3f}  "
          f"mean (test-fresh)={np.mean(R.test-R.fresh):+.3f}")
    print(f"[{scheme}] sd(val-fresh)={np.std(R.val-R.fresh):.3f}  "
          f"sd(test-fresh)={np.std(R.test-R.fresh):.3f}")

# Figure 1: the data and one example fit
net, m, s, tr = example
plt.figure(figsize=(6, 4))
for yr, c in [(2025, "tab:blue"), (2026, "tab:orange")]:
    sel = dates.year == yr
    plt.plot(X[sel], Y[sel], ".", ms=4, color=c, label=str(yr))
plt.plot(X[tr], Y[tr], "k+", ms=7, label="train days")
plt.plot(xgrid, fit.predict(net, xgrid) * s + m, "r", label="chosen net")
plt.xlabel("day of year (rescaled)")
plt.ylabel("mean departure delay (min)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("ex8_data.png", dpi=120)

# Figure 2: chosen H and estimate errors
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for off, scheme in zip([-0.2, 0.2], SCHEMES):
    R = rows[scheme]
    cnt = R.H.value_counts().reindex(HS, fill_value=0)
    axes[0].bar(np.arange(len(HS)) + off, cnt.values, width=0.4, label=scheme)
axes[0].set_xticks(range(len(HS)))
axes[0].set_xticklabels(HS)
axes[0].set_xlabel("chosen H")
axes[0].set_ylabel("count")
axes[0].legend()
for ax, key in zip(axes[1:], ["val", "test"]):
    ax.boxplot([rows[s][key] - rows[s]["fresh"] for s in SCHEMES])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(list(SCHEMES))
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_ylabel(f"{key} loss - fresh loss")
plt.tight_layout()
plt.savefig("ex8_estimates.png", dpi=120)
plt.show()
