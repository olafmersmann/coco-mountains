import sys
import numpy as np
import trimesh as tm

from tqdm.auto import tqdm, trange
from pathlib import Path
from cocoex.function import BenchmarkFunction


class VertexBuffer:
    def __init__(self):
        self._next_id = 0
        self.ptmap = {}

    def get(self, x, y, z):
        point = (x, y, z)
        try:
            return self.ptmap[point]
        except KeyError:
            id = self._next_id
            self._next_id += 1
            self.ptmap[point] = id
            return id

    def vertices(self):
        return np.array(sorted(self.ptmap.keys(), key=lambda pt: self.ptmap[pt]))


def heightmap_to_mesh(X1, X2, Y):
    n = X1.shape[0]
    vbuf = VertexBuffer()
    edges = []
    for i1 in trange(0, n - 1):
        for i2 in range(0, n - 1):
            v00 = vbuf.get(X1[i1, i2], X2[i1, i2], Y[i1, i2])
            v01 = vbuf.get(X1[i1, i2 + 1], X2[i1, i2 + 1], Y[i1, i2 + 1])
            v10 = vbuf.get(X1[i1 + 1, i2], X2[i1 + 1, i2], Y[i1 + 1, i2])
            v11 = vbuf.get(X1[i1 + 1, i2 + 1], X2[i1 + 1, i2 + 1], Y[i1 + 1, i2 + 1])
            edges.append((v00, v01, v10))
            edges.append((v11, v10, v01))

    for i1 in trange(0, n - 1):
        x1, x2, y = X1[i1, 0], X2[i1, 0], Y[i1, 0]
        v00 = vbuf.get(x1, x2, 0.0)
        v01 = vbuf.get(x1, x2, y)
        x1, x2, y = X1[i1 + 1, 0], X2[i1 + 1, 0], Y[i1 + 1, 0]
        v10 = vbuf.get(x1, x2, 0.0)
        v11 = vbuf.get(x1, x2, y)
        edges.append((v00, v01, v10))
        edges.append((v11, v10, v01))

    for i1 in trange(0, n - 1):
        x1, x2, y = X1[i1, n - 1], X2[i1, n - 1], Y[i1, n - 1]
        v00 = vbuf.get(x1, x2, 0.0)
        v01 = vbuf.get(x1, x2, y)
        x1, x2, y = X1[i1 + 1, n - 1], X2[i1 + 1, n - 1], Y[i1 + 1, n - 1]
        v10 = vbuf.get(x1, x2, 0.0)
        v11 = vbuf.get(x1, x2, y)
        edges.append((v00, v10, v01))
        edges.append((v11, v01, v10))

    for i1 in trange(0, n - 1):
        x1, x2, y = X1[0, i1], X2[0, i1], Y[0, i1]
        v00 = vbuf.get(x1, x2, 0.0)
        v01 = vbuf.get(x1, x2, y)
        x1, x2, y = X1[0, i1 + 1], X2[0, i1 + 1], Y[0, i1 + 1]
        v10 = vbuf.get(x1, x2, 0.0)
        v11 = vbuf.get(x1, x2, y)
        edges.append((v00, v10, v01))
        edges.append((v11, v01, v10))

    for i1 in trange(0, n - 1):
        x1, x2, y = X1[n - 1, i1], X2[n - 1, i1], Y[n - 1, i1]
        v00 = vbuf.get(x1, x2, 0.0)
        v01 = vbuf.get(x1, x2, y)
        x1, x2, y = X1[n - 1, i1 + 1], X2[n - 1, i1 + 1], Y[n - 1, i1 + 1]
        v10 = vbuf.get(x1, x2, 0.0)
        v11 = vbuf.get(x1, x2, y)
        edges.append((v00, v01, v10))
        edges.append((v11, v10, v01))

    ## Bottom
    v00 = vbuf.get(X1[0, 0], X2[0, 0], 0.0)
    v01 = vbuf.get(X1[0, n-1], X2[0, n-1], 0.0)
    v10 = vbuf.get(X1[n-1, 0], X2[n-1, 0], 0.0)
    v11 = vbuf.get(X1[n-1, n-1], X2[n-1, n-1], 0.0)
    edges.append((v00, v01, v10))
    edges.append((v11, v10, v01))

    print("Creating mesh...")
    mesh = tm.Trimesh(vbuf.vertices(), edges)
    del vbuf
    print(mesh.is_watertight)
    print("Simplify mesh...")
    mesh = mesh.simplify_quadric_decimation(0.9, aggression=3)
    return mesh


def main(args):
    side_length = 50 # mm
    resolution = 2 # mm
    fn = BenchmarkFunction("bbob", 20, 2, 1)

    n = int(side_length / resolution) + 1
    x = np.linspace(-5.0, 5.0, n)
    X1, X2 = np.meshgrid(x, x)
    Y = fn(np.column_stack((X1.flatten(), X2.flatten()))).reshape(X1.shape)
    Y = Y - np.min(Y)
    Y = 1.1 - Y / np.max(Y)

    # Rescale to side_length * side_length * side_length cube.
    X1 = side_length * (X1 + 5.0) / 10.0
    X2 = side_length * (X2 + 5.0) / 10.0
    Y = side_length * Y

    mesh = heightmap_to_mesh(X1, X2, Y)
    print(mesh.is_watertight)
    mesh.export("bbob-f7-i1-d2.stl")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
