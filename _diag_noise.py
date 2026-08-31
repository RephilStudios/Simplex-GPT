"""Scratch diagnostic: compare snoise2/snoise3 against scalar float64 references."""

import math

import torch

from simplex_thought_field import snoise2, snoise3


def perm(x: int) -> int:
    return ((x * 34) + 1) * x % 289


def fract(x: float) -> float:
    return x - math.floor(x)


def taylor_inv_sqrt(r: float) -> float:
    return 1.79284291400159 - 0.85373472095314 * r


def snoise2_ref(v0: float, v1: float) -> float:
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    s = (v0 + v1) * F2
    i = math.floor(v0 + s)
    j = math.floor(v1 + s)
    t = (i + j) * G2
    x0 = v0 - i + t
    y0 = v1 - j + t

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii, jj = i % 289, j % 289
    h0 = perm(perm(jj) + ii)
    h1 = perm(perm(jj + j1) + ii + i1)
    h2 = perm(perm(jj + 1) + ii + 1)

    inv = 1.0 / math.sqrt(2.0)
    grads = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (inv, inv),
        (inv, -inv),
        (-inv, inv),
        (-inv, -inv),
    ]

    def corner(x, y, h):
        n = max(0.5 - (x * x + y * y), 0.0)
        n = n**4
        gx, gy = grads[h % 8]
        return n * (gx * x + gy * y)

    return 70.0 * (corner(x0, y0, h0) + corner(x1, y1, h1) + corner(x2, y2, h2))


def snoise3_ref(v0: float, v1: float, v2: float) -> float:
    # Canonical Gustavson 3D simplex constants: F3 = 1/3, G3 = 1/6
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
    s = (v0 + v1 + v2) * F3
    i = math.floor(v0 + s)
    j = math.floor(v1 + s)
    k = math.floor(v2 + s)
    t = (i + j + k) * G3
    x0 = v0 - i + t
    y0 = v1 - j + t
    z0 = v2 - k + t

    if x0 >= y0:
        if y0 >= z0:
            i1, i2 = (1, 0, 0), (1, 1, 0)  # X Y Z order
        elif x0 >= z0:
            i1, i2 = (1, 0, 0), (1, 0, 1)  # X Z Y order
        else:
            i1, i2 = (0, 0, 1), (1, 0, 1)  # Z X Y order
    else:
        if y0 < z0:
            i1, i2 = (0, 0, 1), (0, 1, 1)  # Z Y X order
        elif x0 < z0:
            i1, i2 = (0, 1, 0), (0, 1, 1)  # Y Z X order
        else:
            i1, i2 = (0, 1, 0), (1, 1, 0)  # Y X Z order

    corners = [
        (x0, y0, z0, i, j, k),
        (
            x0 - i1[0] + G3,
            y0 - i1[1] + G3,
            z0 - i1[2] + G3,
            i + i1[0],
            j + i1[1],
            k + i1[2],
        ),
        (
            x0 - i2[0] + 2.0 * G3,
            y0 - i2[1] + 2.0 * G3,
            z0 - i2[2] + 2.0 * G3,
            i + i2[0],
            j + i2[1],
            k + i2[2],
        ),
        (
            x0 - 1.0 + 3.0 * G3,
            y0 - 1.0 + 3.0 * G3,
            z0 - 1.0 + 3.0 * G3,
            i + 1,
            j + 1,
            k + 1,
        ),
    ]
    grads = [
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
    ]

    total = 0.0
    for cx, cy, cz, ix, iy, iz in corners:
        h = perm(perm(perm(iz) + iy) + ix) % 12
        gx, gy, gz = grads[h]
        n = max(0.6 - (cx * cx + cy * cy + cz * cz), 0.0)
        n = n**4
        total += n * (gx * cx + gy * cy + gz * cz)
    return 32.0 * total


def check2():
    torch.manual_seed(0)
    pts = torch.randn(3000, 2) * 2.0
    mine = snoise2(pts).to(torch.float64)
    ref = torch.tensor([snoise2_ref(*p.tolist()) for p in pts], dtype=torch.float64)
    diff = (mine - ref).abs()
    print(
        f"2D: max diff vs reference = {diff.max().item():.3e}, mean = {diff.mean().item():.3e}"
    )
    d = (snoise2(pts + 1e-3) - snoise2(pts)).abs()
    print(
        f"2D: max |delta| eps=1e-3 = {d.max().item():.5f}, p95 = {d.kthvalue(int(0.95 * len(d))).values.max().item():.5f}"
    )
    i = int(d.argmax())
    p = pts[i]
    print(
        f"   worst point: {[round(c, 4) for c in p.tolist()]}, jump={d[i].item():.5f}"
    )
    xs = torch.linspace(p[0] - 0.3, p[0] + 0.3, 31)
    line_mine = [
        snoise2(torch.tensor([[float(x), float(p[1])]])).item() for x in xs.tolist()
    ]
    line_ref = [snoise2_ref(float(x), float(p[1])) for x in xs.tolist()]
    md = [abs(a - b) for a, b in zip(line_mine, line_ref)]
    print(f"   line scan near worst point: max |mine-ref| = {max(md):.3e}")
    return diff.max().item()


def check3():
    torch.manual_seed(1)
    pts = torch.randn(1500, 3) * 1.5
    mine = snoise3(pts).to(torch.float64)
    ref = torch.tensor([snoise3_ref(*p.tolist()) for p in pts], dtype=torch.float64)
    diff = (mine - ref).abs()
    print(
        f"3D: max diff vs reference = {diff.max().item():.3e}, mean = {diff.mean().item():.3e}"
    )
    d = (snoise3(pts + 1e-3) - snoise3(pts)).abs()
    print(
        f"3D: max |delta| eps=1e-3 = {d.max().item():.5f}, p95 = {d.kthvalue(int(0.95 * len(d))).values.max().item():.5f}"
    )
    return diff.max().item()


if __name__ == "__main__":
    check2()
    check3()
