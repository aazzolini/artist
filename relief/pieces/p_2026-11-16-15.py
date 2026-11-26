import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from studio import *
PID = "2026-11-16-15"
cx, nose, tail, hw = 0.45, 0.255, 0.615, 0.036      # ship centre line, nose tip, engine end, half width
gy, wy = 1.03, 1.075                                 # ground line, waterline

# sky: dark dome, haze near the ground, stars that fade toward the glow
gl = np.exp(-(((X - cx) / 0.42) ** 2 + ((Y - 0.86) / 0.36) ** 2))
sky = 0.012 + 0.07 * sm(0.5, 1.03, Y) ** 2 + 0.27 * gl
sky += stars(520, seed=11, ymax=0.95, big=7) * clamp(1 - 1.9 * gl) * sm(1.0, 0.7, Y)
f = sky.astype(np.float32)

# tower: lattice lit from the lower left by the engines
tl, tr, ttop = 0.640, 0.704, 0.185
def tower(p):
    for x in (tl, tr): p.line([(x, ttop), (x, gy)], 0.0065)
    y = ttop
    while y < gy - 0.01:
        p.line([(tl, y), (tr, y)], 0.0035); p.line([(tl, y), (tr, y + 0.042)], 0.003); p.line([(tr, y), (tl, y + 0.042)], 0.003)
        y += 0.042
    p.rect(tl - 0.004, ttop - 0.012, tr + 0.004, ttop)
    p.line([((tl + tr) / 2, ttop - 0.012), ((tl + tr) / 2, ttop - 0.075)], 0.003)
    p.rect(tl - 0.012, 0.40, tr + 0.012, 0.53)                       # carriage the arms ride on
tw = shape(tower)
tlit = 0.16 + 0.5 * np.exp(-((Y - 0.66) / 0.3) ** 2) + 0.12 * sm(0.3, 0.2, Y)
f = over(f, tw, tlit * (0.75 + 0.25 * fbm(X * 60, Y * 60, 3, 5)))
f = over(f, rect(tl - 0.006, 0.415, tr + 0.006, 0.52) * tw, 0.22 + 0.2 * sm(0.52, 0.4, Y))

# far arm (behind the ship)
f = over(f, poly([(tl, 0.432), (0.335, 0.448), (0.335, 0.463), (tl, 0.455)]), 0.3 + 0.25 * sm(0.3, 0.6, X))

# ship: ogive nose, steel body with a dark tiled half, flaps
def body(p):
    pts = []
    for i in range(0, 41):
        t = i / 40; yy = nose + (0.36 - nose) * t; w = hw * np.sin(t * np.pi / 2) ** 0.75
        pts.append((cx - w, yy))
    pts += [(cx - hw, tail), (cx + hw, tail)]
    for i in range(40, -1, -1):
        t = i / 40; yy = nose + (0.36 - nose) * t; w = hw * np.sin(t * np.pi / 2) ** 0.75
        pts.append((cx + w, yy))
    p.poly(pts)
ship = shape(body)
u = (X - cx) / hw
steel = 0.42 + 0.5 * np.exp(-((u + 0.45) / 0.3) ** 2) + 0.12 * np.exp(-((u - 0.75) / 0.2) ** 2)
steel *= 0.8 + 0.2 * np.cos((Y - nose) * 2 * np.pi / 0.028) ** 8 * 0 + 0.0
rings = 1 - 0.22 * (np.abs(((Y - 0.36) / 0.032) % 1 - 0.5) < 0.06)
tiles = sm(0.05, 0.22, u)                                             # windward side wears black tiles
lit = steel * rings * (1 - 0.72 * tiles) + 0.5 * sm(0.47, tail, Y) * (1 - 0.5 * tiles)
f = over(f, ship, clamp(lit))
flaps = shape(lambda p: (p.poly([(cx - hw * 0.9, 0.318), (cx - hw - 0.03, 0.352), (cx - hw - 0.03, 0.392), (cx - hw, 0.392)]),
                         p.poly([(cx + hw * 0.9, 0.318), (cx + hw + 0.03, 0.352), (cx + hw + 0.03, 0.392), (cx + hw, 0.392)]),
                         p.poly([(cx - hw, 0.522), (cx - hw - 0.042, 0.548), (cx - hw - 0.042, tail), (cx - hw, tail)]),
                         p.poly([(cx + hw, 0.522), (cx + hw + 0.042, 0.548), (cx + hw + 0.042, tail), (cx + hw, tail)])))
f = over(f, flaps, np.where(X < cx, 0.5 + 0.35 * sm(0.45, tail, Y), 0.1 + 0.25 * sm(0.5, tail, Y)))

# near arm (in front of the ship), with catch rail highlights
f = over(f, poly([(tl, 0.458), (0.31, 0.482), (0.31, 0.503), (tl, 0.486)]), 0.38 + 0.4 * sm(0.75, 0.3, X))
f = over(f, line([(tl, 0.459), (0.31, 0.483)], 0.003), 0.85)
truss = shape(lambda q: [q.line([(tl - k * 0.03, 0.4865 - k * 0.0015), (tl - k * 0.03 - 0.03, 0.4615 + (k + 1) * 0.0022)], 0.0022) for k in range(11)])
f = over(f, truss * 0.8, 0.14)

# engines: three flames with shock diamonds, then the wide glow
flame = zeros()
for dx, ln in ((-0.014, 0.19), (0.0, 0.215), (0.014, 0.185)):
    t = clamp((Y - tail) / ln)
    w = 0.0125 * (1 - 0.72 * t) + 0.002
    core = np.exp(-((X - cx - dx * (1 - 0.3 * t)) / w) ** 2) * sm(1.0, 0.75, t) * (Y >= tail)
    flame = np.maximum(flame, core * (0.72 + 0.28 * np.cos(t * 2 * np.pi * 5.5) ** 2))
f = screen(f, clamp(flame * 1.15))
f = screen(f, clamp(0.55 * blur(flame, 0.022) + 0.35 * blur(flame, 0.07)))

# exhaust cloud rolling across the pad, lit from its centre
n = fbm(X * 4.2 + 3, Y * 7.0, 5, 21)
reach = np.exp(-((X - cx) / 0.5) ** 2)
cloud = sm(0.3, 0.62, n + 1.05 * reach * sm(0.7, 1.0, Y) - 0.5) * sm(0.7, 0.9, Y)
low = sm(0.42, 0.7, fbm(X * 3.0 + 9, Y * 16.0, 4, 33)) * sm(0.93, 1.02, Y) * 0.8
cloud = np.maximum(cloud, low)
clit = 0.2 + 0.8 * np.exp(-(((X - cx) / 0.24) ** 2 + ((Y - 0.9) / 0.16) ** 2))
f = over(f, 0.94 * cloud, clamp(clit * (0.6 + 0.6 * fbm(X * 12, Y * 18, 4, 8))))

# ground: flat coast with tank farm and lamps
land = np.clip((Y - gy) * W + 0.5, 0, 1)
tanks = shape(lambda p: [p.rect(x, gy - h, x + 0.034, gy + 0.004) for x, h in ((0.07, 0.05), (0.112, 0.05), (0.154, 0.062), (0.196, 0.062), (0.238, 0.04), (0.84, 0.03), (0.885, 0.045))])
f = over(f, tanks, 0.05 + 0.2 * sm(0.0, 0.3, X) * (X < 0.5) + 0.1 * (X > 0.5))
f = over(f, land, 0.012)
for lx, ly in ((0.09, gy - 0.056), (0.175, gy - 0.068), (0.26, gy - 0.046), (0.33, gy - 0.012), (0.58, gy - 0.012), (0.78, gy - 0.014), (0.9, gy - 0.05), (0.955, gy - 0.01)):
    f += 0.9 * np.exp(-(((X - lx) ** 2 + (Y - ly) ** 2) / 0.0000042))

# water in the foreground mirrors the fire
f = clamp(f)
refl = water(np.where(Y < gy, f, 0), wy, sheen=1.0, rough=1.3, seed=4, stretch=2.5)
f = np.where(Y >= wy, 0.012 + refl, f)

out = save(f, PID, water_mask=(Y >= wy))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "o_%s.json" % PID), "w"))
