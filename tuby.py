import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from tqdm import tqdm
from rivuletpy.utils.io import writetiff3d
from scipy.ndimage.filters import gaussian_filter
from rivuletpy.swc import SWC
from rivuletpy.trace import R2Branch
import argparse


# Make a volume with the skeleton in it
def make_vol(curves):
    x_max_sz = np.max([c.max_x() for c in curves])
    y_max_sz = np.max([c.max_y() for c in curves])
    z_max_sz = np.max([c.max_z() for c in curves])

    R = np.zeros(
            (int(x_max_sz) + 40,
             int(y_max_sz) + 40,
             int(z_max_sz) + 40
            ))
    for c in curves:
        for n in c:
            R[int(n.x), int(n.y), int(n.z)] = n.r

    R[R==0] = 1e-10
    R = gaussian_filter(R, sigma=3)

    # Make TIFF
    R = R / R.max() * 255
    return R

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """

    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


class Node3D(object):
    def __init__(self, x, y, z, r=1, id=None):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.id = id

    def toarray(self):
        return np.asarray((self.x, self.y, self.z))

class Curve3D(object):
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    def make_nodes(self, x, y, z, radius_type):
        sin_len = len(x) * 2
        for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
            if radius_type == 'uniform':
                r = 1
            elif radius_type == 'sin': # The radius changes according to a sin wave
                r = max(5 * np.sin( ((i % sin_len)/sin_len) * 2 * math.pi), 1)
            elif radius_type == 'random': # ranges from 1-5
                r = np.random.randint(1, 4)
            else:
                raise NotImplementedError
            if xx >= 0 and yy >= 0 and zz >= 0:
                self.add(Node3D(xx, yy, zz, r))
            else:
                break

    def size_x(self):
        lx = np.asarray([n.x for n in self.nodes])
        return lx.max() - lx.min()

    def size_y(self):
        ly = np.asarray([n.y for n in self.nodes])
        return ly.max() - ly.min()

    def size_z(self):
        lz = np.asarray([n.z for n in self.nodes])
        return lz.max() - lz.min()

    def max_x(self):
        return np.asarray([n.x for n in self.nodes]).max()

    def max_y(self):
        return np.asarray([n.y for n in self.nodes]).max()

    def max_z(self):
        return np.asarray([n.z for n in self.nodes]).max()

class Spiral(Curve3D):
    def __init__(self, N=1e6, scale=40, radius_type='uniform'):
        super(Spiral, self).__init__()
        theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
        z = np.linspace(-1, 1, N)
        r = z**2 + 1
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        # Shift the points
        x = (x - x.min()) * scale + scale // 2
        y = (y - y.min()) * scale + scale // 2
        z = (z - z.min()) * scale + scale // 2

        self.make_nodes(x, y, z, radius_type)

    def toswc(self):
        swc = np.zeros((len(self, 7)))
        for i, n in enumerate(self.nodes):
            pid = 0 if i == 0 else self.nodes[i-1].id
            swc[i, :] = [n.id, 1, n.x, n.y, n.z, n.r, pid]

        return swc


def rand_rot_angle(max_angle):
    return np.random.rand() * max_angle * 2 - max_angle

class Branch(Curve3D):
    def __init__(self, N=20, start_point=(0., 0., 0.), radius_type='uniform', id=0, parent_id=None):
        super(Branch, self).__init__()
        self.id = id

        x = np.zeros((N,))
        y = np.zeros((N,))
        z = np.zeros((N,))

        vel = np.random.rand(3)
        vel = vel / np.linalg.norm(vel) * 0.1
        p = start_point.toarray()

        for i in range(N):
            x[i], y[i], z[i] = p[0], p[1], p[2]
            p += vel

            # Disturb the orientation of the velocity a little
            vel = np.dot(rotation_matrix([0., 0., 1.], rand_rot_angle(np.pi/128)), vel)
            vel = np.dot(rotation_matrix([0., 1., 0.], rand_rot_angle(np.pi/128)), vel)
            vel = np.dot(rotation_matrix([1., 0., 0.], rand_rot_angle(np.pi/128)), vel)
        self.make_nodes(x, y, z, radius_type)

    def get_parent_id(self):
        return '-'.join(self.id.split('-')[:-1]) 

def make_layer(depth, start_point, nlayer, nchild, branchlen=20, radius_type='uniform', pid=None):
    new_branches = [Branch(np.random.randint(1, branchlen), start_point, radius_type, id=pid + '-' + str(i)) for i in range(np.random.randint(1, nchild+1))]
    print('depth=', depth, 'nchild=', len(new_branches), 'nlayer=', nlayer)
    if depth + 1 == nlayer:  # Base case
        return new_branches
    else:
        child_branches = []
        for b in new_branches:
            print(b.id)
            st = b.nodes[-1]
            child_branches += make_layer(depth + 1, st, nlayer, nchild, branchlen, radius_type, pid=b.id)
        return new_branches + child_branches

class Tree(object):
    def __init__(self, nlayer, nchild, branchlen=2000, radius_type='uniform'):
        self.branches = make_layer(0, Node3D(0., 0., 0.), nlayer, nchild, branchlen, radius_type, pid='R')

    def get_branch_by_id(self, id):
        if id == 'R':
            return None

        for b in self.branches:
            if b.id == id:
                return b

        print(id, 'not found')
        raise ValueError

    def toswc(self):
        swc = np.asarray([0, 1, 0, 0, 0, 1, -1])

        # Assign id to nodes
        id = 1
        for b in self.branches:
            for n in b.nodes:
                n.id = id
                id += 1

        for b in self.branches:
            branch_swc = np.zeros((len(b), 7))
            pbranch = self.get_branch_by_id(b.get_parent_id())
            pid = pbranch[-1].id if pbranch is not None else 0

            for i, n in enumerate(b):
                branch_swc[i, :] = [n.id, 1, n.x, n.y, n.z, n.r, b[i-1].id if i > 0 else pid]
            swc = np.vstack((swc, branch_swc))

        return swc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Synthetic Tubes.')
    parser.add_argument(
        '--type',
        type=str,
        default='spiral',
        required=False,
        help='The type of tubes to generate [line, spiral, tree, circle].')
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        default='syn',
        required=False,
        help='The path to save the output volume and swc.')
    parser.add_argument(
        '--radius',
        type=str,
        default='uniform',
        required=False,
        help='The type of radius to generate [uniform, sin, random]')
    parser.add_argument(
        '-l',
        '--length',
        type=int,
        default=100000,
        required=False,
        help='The number of nodes to generate in the skeleton')
    parser.add_argument(
        '--nlayer',
        type=int,
        default=4,
        required=False,
        help='The number of layers to generate a tree')
    parser.add_argument(
        '--nchild',
        type=int,
        default=4,
        required=False,
        help='The number of children for each tree node')
    parser.add_argument(
        '--branchlen',
        type=int,
        default=2e3,
        required=False,
        help='The length of each tree branch')
    args = parser.parse_args()

    if args.type == 'spiral':
        curves = [Spiral(args.length, radius_type=args.radius), ]
    elif args.type == 'tree':
        tree = Tree(args.nlayer, args.nchild, args.branchlen, radius_type=args.radius)
        saveswc(args.out+'.swc', tree.toswc())
        curves = tree.branches
    else:
        raise NotImplementedError

    D = make_vol(curves)
    writetiff3d(args.out + '.tif', D.astype('uint8'))
