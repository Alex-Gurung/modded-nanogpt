# ------------------------------------------------------------
# 1. Original offline Polar Express construction (Appendix G)
# ------------------------------------------------------------
from math import inf, sqrt
import numpy as np
import torch


def optimal_quintic(l, u):
    """
    Appendix G: find the degree-5 odd polynomial
        p(x) = a x + b x^3 + c x^5
    that (nearly) minimax-approximates 1 on [l, u].
    """
    assert 0 <= l <= u
    # When l ~ u we can use the scaled Newton–Schulz quintic in closed form
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomial is numerically equal to:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)

    # Remez-style iteration as in Algorithm 3
    q = (3 * l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l ** 3, l ** 5, 1],
            [q, q ** 3, q ** 5, -1],
            [r, r ** 3, r ** 5, 1],
            [u, u ** 3, u ** 5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        # Update interior equioscillation points using derivative roots
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * np.sqrt(9 * b ** 2 - 20 * a * c))
            / (10 * c)
        )

    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, cushion=0.02407327424182761,
                        safety_factor=None):
    """
    Appendix G: build the sequence of quintic polynomials p_t for t=1..T.

    l          : initial lower bound on singular values (ℓ_1)
    num_iters  : T
    cushion    : 'cushion factor' controlling the effective lower bound
    safety_factor : if not None, incorporate the runtime safety factor
                    (i.e. X is divided by (1 + safety_factor) before iterating)
    """
    u = 1.0
    coefficients = []
    for _ in range(num_iters):
        # 1) Find near-minimax polynomial on [max(l, cushion * u), u]
        a, b, c = optimal_quintic(max(l, cushion * u), u)

        # 2) Recentre it so it's symmetric around 1 on [l, u]
        pl = a * l + b * l ** 3 + c * l ** 5
        pu = a * u + b * u ** 3 + c * u ** 5
        rescalar = 2.0 / (pl + pu)
        a *= rescalar
        b *= rescalar
        c *= rescalar

        # 3) Optionally incorporate safety factor in the polynomial itself
        if safety_factor is not None:
            s = 1.0 + safety_factor
            a /= s
            b /= s ** 3
            c /= s ** 5

        coefficients.append((a, b, c))

        # 4) Update the interval [ℓ_{t+1}, u_{t+1}] = [p(l), 2 - p(l)]
        l = a * l + b * l ** 3 + c * l ** 5
        u = 2.0 - l

    return coefficients


# ------------------------------------------------------------
# 2. New refinement: keep very small singular values small
# ------------------------------------------------------------

def refine_with_small_singular_preservation(
    coeffs,
    ell=1e-3,
    small_thresh=1e-3,
    weight_small=1.0,
    n_grid=2048,
    lr=1e-2,
    n_steps=2000,
    device="cpu",
):
    """
    Take an initial sequence of (a,b,c) coefficients (e.g. from optimal_composition),
    and refine them so that:

        - On [ell, 1], the composed scalar map g(x) ~ 1 (polar-like).
        - On [0, small_thresh], g(x) ~ x (small singular values remain small).

    Args
    ----
    coeffs       : list[(a,b,c)], length T
    ell          : lower bound ℓ used when constructing coeffs (normalized domain)
    small_thresh : threshold δ; singular values in [0, δ] should behave ~ identity
    weight_small : relative weight for the "keep small" objective
    n_grid       : number of grid points in [0,1] for the discretized minimax
    lr           : learning rate for Adam
    n_steps      : number of refinement steps
    device       : 'cpu' or 'cuda'

    Returns
    -------
    new_coeffs : list[(a,b,c)] with refined values.
    """
    T = len(coeffs)

    # Initialize learnable parameters from given coeffs
    params = torch.nn.Parameter(
        torch.tensor(coeffs, dtype=torch.float64, device=device)
    )  # shape (T, 3)

    opt = torch.optim.Adam([params], lr=lr)

    # Build a dense grid in [0,1], biased a bit towards 0
    xs_uniform = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64, device=device)
    xs_near0 = torch.linspace(0.0, min(5 * small_thresh, 1.0),
                              n_grid // 2, dtype=torch.float64, device=device)
    xs = torch.unique(torch.cat([xs_uniform, xs_near0])).sort().values

    # masks
    mask_small = xs <= small_thresh
    mask_large = xs >= ell

    assert mask_small.any(), "small_thresh too small for chosen grid"
    assert mask_large.any(), "ell too large for chosen grid"

    target_identity = xs.clone()
    target_one = torch.ones_like(xs)

    for step in range(n_steps):
        opt.zero_grad()

        # Compose the polynomials p_T ∘ ... ∘ p_1 on xs
        y = xs.clone()
        for t in range(T):
            a, b, c = params[t]
            y = a * y + b * (y ** 3) + c * (y ** 5)

        # Errors:
        #  - On [ell,1]: want y ≈ 1
        #  - On [0, small_thresh]: want y ≈ x
        err_large = torch.abs(y[mask_large] - target_one[mask_large])
        err_small = torch.abs(y[mask_small] - target_identity[mask_small])

        # Approximate L∞ with log-sum-exp ("soft max")
        # and weight the small-region error.
        E_large = torch.logsumexp(torch.log(err_large + 1e-12), dim=0)
        E_small = torch.logsumexp(torch.log(err_small + 1e-12), dim=0)

        loss = torch.logsumexp(
            torch.stack([E_large, E_small + torch.log(torch.tensor(weight_small))]),
            dim=0,
        )

        loss.backward()
        opt.step()

        # (Optional: print occasionally if you want to monitor)
        # if step % 200 == 0:
        #     print(f"step {step}  loss {loss.item():.4e}")

    # Convert back to Python floats
    refined = params.detach().cpu().numpy().tolist()
    return [(float(a), float(b), float(c)) for (a, b, c) in refined]


# ------------------------------------------------------------
# 3. Convenience wrapper: what you actually call
# ------------------------------------------------------------

def build_polar_express_coeffs_with_small_preservation(
    num_iters=5,
    safety_factor=2e-2,
    cushion=0.02407327424182761,
    ell=1e-3,
    small_thresh=1e-3,
    weight_small=1.0,
    device="cpu",
):
    """
    High-level function:

      1. Build coefficients with the original Polar Express offline procedure
         (including your chosen safety_factor / cushion).
      2. Refine them so very small singular values are preserved.

    You can drop the returned coeffs into your polar_express() implementation.
    """
    base_coeffs = optimal_composition(
        l=ell,
        num_iters=num_iters,
        cushion=cushion,
        safety_factor=safety_factor,
    )

    refined_coeffs = refine_with_small_singular_preservation(
        coeffs=base_coeffs,
        ell=ell,
        small_thresh=small_thresh,
        weight_small=weight_small,
        device=device,
    )

    return refined_coeffs


if __name__ == "__main__":
    # Example: roughly reproduce your setup and then refine.
    num_iters = 9
    safety_factor = 2e-2
    # NOTE: cushion=2 only makes sense if you have modified the appendix-G code.
    # Here we stick with the original-style cushion < 1. You can tinker with it.
    # cushion = 0.05
    cushion = 0.05


    # Your existing coefficients
    polar_express_coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]
    """
    # Directly refine these, instead of calling optimal_composition(...)
    coeffs = refine_with_small_singular_preservation(
        coeffs=polar_express_coeffs,
        ell=1e-3,          # lower bound of “large-ish” singular values in normalized domain
        # small_thresh=1e-2, # what you consider “super close to zero”
        small_thresh=1e-4, # what you consider “super close to zero”
        # weight_small=2.0,  # or >1 to emphasize preserving small singular values
        weight_small=0.0,  # or >1 to emphasize preserving small singular values
        n_steps=2000,
        lr=1e-3,
        device="cuda"  # or "cpu"
    )"""
    """
    coeffs = build_polar_express_coeffs_with_small_preservation(
        num_iters=num_iters,
        safety_factor=safety_factor,
        cushion=cushion,
        ell=1e-3,
        # small_thresh=5e-4,   # "super close to zero" region
        small_thresh=1e-3,   # "super close to zero" region
        # small_thresh=0,   # "super close to zero" region
        # weight_small=1.0,    # emphasize preserving small singular values
        weight_small=0.0,    # emphasize preserving small singular values
        device="cuda:1",
    )
    """
    # coeffs = polar_express_coeffs

    coeffs = optimal_composition(
        l=1e-3,
        num_iters=num_iters,
        cushion=cushion,
        safety_factor=safety_factor,
    )


    print("Refined Polar Express coefficients:")
    for c in coeffs:
        print(c)

    funcs = "fghjklmnb"
    for i, c in enumerate(coeffs):
        name = funcs[i]
        print(f"{name}(x) = {c[0]}x + {c[1]}x^3 + {c[2]}x^5")
