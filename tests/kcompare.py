# kcompare.py <a.pt> <b.pt>: max abs / max rel difference per kcheck case
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for key in a:
    x, y = a[key], b[key]
    d = (x - y).abs()
    print(f"{key:14s} max|d| {d.max().item():.3e}  mean|d| {d.mean().item():.3e}  max|x| {x.abs().max().item():.3e}  "
          f"bit-identical {bool(torch.equal(x, y))}  finite {bool(torch.isfinite(y).all())}")
