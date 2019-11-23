import sympy as sp
import pprint

fx, fy, cx, cy = sp.symbols('fx fy cx cy')
K = sp.Matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

x, y = sp.symbols('x y')
u = sp.Matrix([[x], [y], [1]])

tx, ty, tz = sp.symbols('tx ty tz')
t = sp.Matrix([[tx], [ty], [tz]])

Z = sp.symbols('Z')

v = u + (K * t) / Z

x_ = v[0] / v[2]
y_ = v[1] / v[2]


J = sp.Matrix([[sp.diff(x_, tx), sp.diff(x_, ty), sp.diff(x_, tz)],
        [sp.diff(y_, tx), sp.diff(y_, ty), sp.diff(y_, tz)]])

pprint.pprint(sp.simplify(J))
