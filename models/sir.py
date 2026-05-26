"""
SIR model fitted to France's first COVID-19 wave using live OWID data.

Normalises against N_eff (cumulative wave cases) rather than the full
67 M population so that the infected fraction occupies a meaningful
range for the SIR solver and gradient-descent optimiser.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
INFECTIOUS_DAYS = 14
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "france_covid.csv")


def _load_owid_france():
    if os.path.exists(_CACHE_FILE):
        return pd.read_csv(_CACHE_FILE, parse_dates=["date"])
    import io, urllib.request
    req = urllib.request.urlopen(OWID_URL, timeout=10)
    df = pd.read_csv(
        io.BytesIO(req.read()),
        usecols=["iso_code", "date", "new_cases", "total_deaths"],
        parse_dates=["date"],
    )
    df = df[df["iso_code"] == "FRA"].copy()
    df = df[(df["date"] >= "2020-03-01") & (df["date"] <= "2020-06-30")]
    df = df.sort_values("date").reset_index(drop=True)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_csv(_CACHE_FILE, index=False)
    return df


# ---------------------------------------------------------------------------
# Live data pipeline
# ---------------------------------------------------------------------------

def fetch_france_first_wave():
    """
    Pull OWID COVID data for France, 2020-03-01 to 2020-06-30.

    Returns (t, I_data, N_eff, df) where:
      - I_data  = active-case fraction normalised by N_eff
      - N_eff   = cumulative confirmed cases over the wave
    """
    df = _load_owid_france()

    df["new_cases"] = df["new_cases"].fillna(0)
    df["smoothed"] = df["new_cases"].rolling(window=7, min_periods=1).mean()

    df["active"] = df["smoothed"].rolling(window=INFECTIOUS_DAYS, min_periods=1).sum()

    N_eff = df["smoothed"].sum()
    if N_eff == 0:
        N_eff = 1.0

    df["I_frac"] = df["active"] / N_eff

    # Trim leading near-zero rows so the optimizer isn't anchored by the
    # pre-epidemic flatline.  Threshold: 1 % of the wave peak.
    peak = df["I_frac"].max()
    onset_mask = df["I_frac"] >= peak * 0.01
    if onset_mask.any():
        first_idx = onset_mask.idxmax()
        df = df.loc[first_idx:].reset_index(drop=True)

    t = np.arange(len(df), dtype=float)
    I_data = df["I_frac"].values
    return t, I_data, N_eff, df


# ---------------------------------------------------------------------------
# SIR ODE system
# ---------------------------------------------------------------------------

def sir_deriv(S, I, R, beta, gamma):
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return dS, dI, dR


# ---------------------------------------------------------------------------
# Numerical solvers
# ---------------------------------------------------------------------------

def euler_forward(S0, I0, R0, beta, gamma, t):
    n = len(t)
    S, I, R = np.zeros(n), np.zeros(n), np.zeros(n)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        dS, dI, dR = sir_deriv(S[k], I[k], R[k], beta, gamma)
        S[k + 1] = S[k] + dt * dS
        I[k + 1] = I[k] + dt * dI
        R[k + 1] = R[k] + dt * dR
    return S, I, R


def heun(S0, I0, R0, beta, gamma, t):
    n = len(t)
    S, I, R = np.zeros(n), np.zeros(n), np.zeros(n)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        dS1, dI1, dR1 = sir_deriv(S[k], I[k], R[k], beta, gamma)
        S_p = S[k] + dt * dS1
        I_p = I[k] + dt * dI1
        R_p = R[k] + dt * dR1
        dS2, dI2, dR2 = sir_deriv(S_p, I_p, R_p, beta, gamma)
        S[k + 1] = S[k] + 0.5 * dt * (dS1 + dS2)
        I[k + 1] = I[k] + 0.5 * dt * (dI1 + dI2)
        R[k + 1] = R[k] + 0.5 * dt * (dR1 + dR2)
    return S, I, R


# ---------------------------------------------------------------------------
# Parameter fitting via gradient descent
# ---------------------------------------------------------------------------

def squared_error(params, t, S0, I0, R0, I_obs):
    beta, gamma = params
    if beta <= 0 or gamma <= 0:
        return 1e12
    _, I_sim, _ = heun(S0, I0, R0, beta, gamma, t)
    return np.sum((I_sim - I_obs) ** 2)


def fit_sir(t, S0, I0, R0, I_obs, beta_init=0.3, gamma_init=0.16,
            eta=1e-3, n_iter=6000, epsilon=1e-5, momentum=0.9,
            warmup=500, grad_tol=1e-10):
    beta, gamma = beta_init, gamma_init
    best_beta, best_gamma = beta, gamma
    best_loss = float("inf")
    v_beta, v_gamma = 0.0, 0.0

    for step in range(n_iter):
        args = (t, S0, I0, R0, I_obs)
        loss = squared_error(np.array([beta, gamma]), *args)

        if loss < best_loss:
            best_beta, best_gamma, best_loss = beta, gamma, loss

        grad_beta = (
            squared_error(np.array([beta + epsilon, gamma]), *args)
            - squared_error(np.array([beta - epsilon, gamma]), *args)
        ) / (2 * epsilon)
        grad_gamma = (
            squared_error(np.array([beta, gamma + epsilon]), *args)
            - squared_error(np.array([beta, gamma - epsilon]), *args)
        ) / (2 * epsilon)

        v_beta = momentum * v_beta + eta * grad_beta
        v_gamma = momentum * v_gamma + eta * grad_gamma

        beta = max(1e-4, beta - v_beta)
        gamma = max(1e-4, gamma - v_gamma)

        if step % 100 == 0:
            grad_mag = np.sqrt(grad_beta**2 + grad_gamma**2)
            print(f"Epoch {step} | Loss: {loss:.5f} | beta: {beta:.4f} | gamma: {gamma:.4f} | |grad|: {grad_mag:.2e}")

        if step >= warmup:
            grad_mag = np.sqrt(grad_beta**2 + grad_gamma**2)
            if grad_mag < grad_tol:
                print(f"Early stop at epoch {step} (|grad| = {grad_mag:.2e} < {grad_tol})")
                break

    return best_beta, best_gamma


# ---------------------------------------------------------------------------
# SIRD-S model (S, I, R, D with mortality mu and immunity loss omega)
# ---------------------------------------------------------------------------

def sirds_deriv(S, I, R, D, beta, gamma, mu, omega=0.0, u=0.0):
    infection = beta * (1.0 - u) * S * I
    dS = -infection + omega * R
    dI = infection - (gamma + mu) * I
    dR = gamma * I - omega * R
    dD = mu * I
    return dS, dI, dR, dD


def sirds_euler_forward(S0, I0, R0, D0, beta, gamma, mu, t,
                        omega=0.0, u=0.0):
    n = len(t)
    S, I, R, D = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    S[0], I[0], R[0], D[0] = S0, I0, R0, D0
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        dS, dI, dR, dD = sirds_deriv(S[k], I[k], R[k], D[k],
                                      beta, gamma, mu, omega, u)
        S[k + 1] = S[k] + dt * dS
        I[k + 1] = I[k] + dt * dI
        R[k + 1] = R[k] + dt * dR
        D[k + 1] = D[k] + dt * dD
    return S, I, R, D


def _sirds_heun(S0, I0, R0, D0, beta, gamma, mu, t, omega=0.0):
    n = len(t)
    S, I, R, D = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    S[0], I[0], R[0], D[0] = S0, I0, R0, D0
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        d1 = sirds_deriv(S[k], I[k], R[k], D[k], beta, gamma, mu, omega)
        Sp = S[k] + dt * d1[0]
        Ip = I[k] + dt * d1[1]
        Rp = R[k] + dt * d1[2]
        Dp = D[k] + dt * d1[3]
        d2 = sirds_deriv(Sp, Ip, Rp, Dp, beta, gamma, mu, omega)
        S[k+1] = S[k] + 0.5 * dt * (d1[0] + d2[0])
        I[k+1] = I[k] + 0.5 * dt * (d1[1] + d2[1])
        R[k+1] = R[k] + 0.5 * dt * (d1[2] + d2[2])
        D[k+1] = D[k] + 0.5 * dt * (d1[3] + d2[3])
    return S, I, R, D


def _sirds_loss(params, t, S0, I0, R0, D0, I_obs, D_obs, w_I, w_D):
    beta, gamma, mu = params
    if beta <= 0 or gamma <= 0 or mu <= 0:
        return 1e12
    _, I_sim, _, D_sim = _sirds_heun(S0, I0, R0, D0, beta, gamma, mu, t)
    return w_I * np.sum((I_sim - I_obs)**2) + w_D * np.sum((D_sim - D_obs)**2)


def fit_sird(t, S0, I0, R0, D0, I_obs, D_obs,
             beta_init=0.3, gamma_init=0.14, mu_init=0.003,
             eta=1e-3, n_iter=4000, epsilon=1e-5, momentum=0.9,
             warmup=300, grad_tol=1e-10):
    """Fit beta, gamma, mu jointly to I and D data. Returns (beta, gamma, mu)."""
    scale_I = max(np.max(np.abs(I_obs)), 1e-6)
    scale_D = max(np.max(np.abs(D_obs)), 1e-8)
    w_I = 1.0 / (scale_I ** 2)
    w_D = 1.0 / (scale_D ** 2)

    params = np.array([beta_init, gamma_init, mu_init])
    best = params.copy()
    best_loss = float("inf")
    vel = np.zeros(3)

    for step in range(n_iter):
        args = (t, S0, I0, R0, D0, I_obs, D_obs, w_I, w_D)
        loss = _sirds_loss(params, *args)
        if loss < best_loss:
            best_loss = loss
            best = params.copy()

        grad = np.zeros(3)
        for j in range(3):
            p_fwd = params.copy(); p_fwd[j] += epsilon
            p_bwd = params.copy(); p_bwd[j] -= epsilon
            grad[j] = (_sirds_loss(p_fwd, *args) - _sirds_loss(p_bwd, *args)) / (2 * epsilon)

        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        vel = momentum * vel + eta * grad
        params = np.maximum(1e-6, params - vel)

        if step % 500 == 0:
            g_mag = np.linalg.norm(grad)
            print(f"  fit_sird {step:4d} | loss={loss:.5f} | "
                  f"β={params[0]:.4f} γ={params[1]:.4f} μ={params[2]:.6f} | |g|={g_mag:.2e}")

        if step >= warmup and np.linalg.norm(grad) < grad_tol:
            print(f"  fit_sird early stop at {step}")
            break

    return float(best[0]), float(best[1]), float(best[2])


# ---------------------------------------------------------------------------
# France COVID data pipeline with deaths
# ---------------------------------------------------------------------------

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
POPULATION_FRANCE = 67_000_000

def fetch_france_data():
    """
    Pull OWID data for France, 2020-03-01 to 2020-06-30.
    Returns (t, I_data, D_data, N_eff) where both are normalised fractions.
    """
    df = _load_owid_france()

    df["new_cases"] = df["new_cases"].fillna(0)
    df["total_deaths"] = df["total_deaths"].ffill().fillna(0)
    df["smoothed"] = df["new_cases"].rolling(window=7, min_periods=1).mean()
    df["active"] = df["smoothed"].rolling(window=14, min_periods=1).sum()

    N_eff = max(df["smoothed"].sum(), 1.0)
    df["I_frac"] = df["active"] / N_eff
    df["D_frac"] = df["total_deaths"] / POPULATION_FRANCE

    peak = df["I_frac"].max()
    onset = df["I_frac"] >= peak * 0.01
    if onset.any():
        df = df.loc[onset.idxmax():].reset_index(drop=True)

    t = np.arange(len(df), dtype=float)
    return t, df["I_frac"].values, df["D_frac"].values, N_eff


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fit(t, I_data, I_model, beta, gamma, N_eff, df):
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = df["date"].values.astype("datetime64[D]")
    ax.plot(dates, I_data, "k.", markersize=3, alpha=0.5, label="OWID data (active, smoothed)")
    ax.plot(dates, I_model, "r-", linewidth=2,
            label=f"SIR fit  β={beta:.4f}  γ={gamma:.4f}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Infected fraction  (N_eff = {N_eff:,.0f})")
    ax.set_title("SIR Model vs France First Wave (OWID live data)")
    y_max = max(np.max(I_data), np.max(I_model)) * 1.15
    ax.set_ylim(0, y_max)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fetching OWID COVID-19 data for France...")
    t, I_data, N_eff, df = fetch_france_first_wave()
    print(f"  {len(t)} days (trimmed), range: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"  N_eff = {N_eff:,.0f}  |  I_data peak = {np.max(I_data):.4f}")
    print(f"  I_data[0] = {I_data[0]:.6f}  (dynamic initial condition)")

    I0 = I_data[0]
    S0 = 1.0 - I0
    R0_init = 0.0

    print("\nFitting beta and gamma via gradient descent...")
    beta_fit, gamma_fit = fit_sir(t, S0, I0, R0_init, I_data)
    print(f"\nResult: beta={beta_fit:.6f}, gamma={gamma_fit:.6f}")
    print(f"  R0 = beta/gamma = {beta_fit / gamma_fit:.2f}")

    _, I_model, _ = heun(S0, I0, R0_init, beta_fit, gamma_fit, t)

    fig = plot_fit(t, I_data, I_model, beta_fit, gamma_fit, N_eff, df)
    plt.show()
