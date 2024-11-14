import torch
from tqdm import tqdm
from . import _C


class OctreeGaussianNode():
    def __init__(self, xyz: torch.Tensor, opacity: torch.Tensor, covariance: torch.Tensor, depth=0, max_depth=5, min_points=10, verbose=False):
        self.children = [None] * 8
        self.depth = depth
        self.bb = torch.stack((torch.min(xyz[:,0]), torch.min(xyz[:,1]), torch.min(xyz[:,2]), torch.max(xyz[:,0]), torch.max(xyz[:,1]), torch.max(xyz[:,2])))

        if verbose:
            print(f"{'  ' * (depth - 1)}octree node depth={depth} size={xyz.shape[0]}")

        if len(xyz) > min_points and depth < max_depth:
            self.is_leaf = False
            cuts = [torch.median(xyz[:,0]), torch.median(xyz[:,1]), torch.median(xyz[:,2])]
            # cuts = [(self.bb[0] + self.bb[3])/2, (self.bb[1] + self.bb[4])/2, (self.bb[2] + self.bb[5])/2]
            mask0 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask0):
                self.children[0] = OctreeGaussianNode(xyz[mask0], opacity[mask0], covariance[mask0], self.depth + 1, max_depth, min_points, verbose)
            mask1 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask1):
                self.children[1] = OctreeGaussianNode(xyz[mask1], opacity[mask1], covariance[mask1], self.depth + 1, max_depth, min_points, verbose)
            mask2 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask2):
                self.children[2] = OctreeGaussianNode(xyz[mask2], opacity[mask2], covariance[mask2], self.depth + 1, max_depth, min_points, verbose)
            mask3 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask3):
                self.children[3] = OctreeGaussianNode(xyz[mask3], opacity[mask3], covariance[mask3], self.depth + 1, max_depth, min_points, verbose)
            mask4 = (xyz[:,0] > cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask4):
                self.children[4] = OctreeGaussianNode(xyz[mask4], opacity[mask4], covariance[mask4], self.depth + 1, max_depth, min_points, verbose)
            mask5 = (xyz[:,0] > cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask5):
                self.children[5] = OctreeGaussianNode(xyz[mask5], opacity[mask5], covariance[mask5], self.depth + 1, max_depth, min_points, verbose)
            mask6 = (xyz[:,0] > cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask6):
                self.children[6] = OctreeGaussianNode(xyz[mask6], opacity[mask6], covariance[mask6], self.depth + 1, max_depth, min_points, verbose)
            mask7 = (xyz[:,0] > cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask7):
                self.children[7] = OctreeGaussianNode(xyz[mask7], opacity[mask7], covariance[mask7], self.depth + 1, max_depth, min_points, verbose)
        else:
            self.is_leaf = True
            self.xyz = xyz.clone()
            self.opacity = opacity.clone()
            self.covariance = covariance.clone()

class OctreeGaussian():
    def __init__(self, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor) -> None:
        self.rootnode = OctreeGaussianNode(xyz, opacity, covariance, depth=0, max_depth=5, min_points=128, verbose=True)

    def is_intersect(self, beam: torch.Tensor, lidar_position: torch.Tensor, lidar_range: float, bb: torch.Tensor) -> bool:
        t_atmin = (bb[:3] - lidar_position) / beam
        t_atmax = (bb[3:] - lidar_position) / beam
        t_enter = torch.max(torch.min(torch.vstack((t_atmin, t_atmax)),dim=0).values)
        t_exit = torch.min(torch.max(torch.vstack((t_atmin, t_atmax)),dim=0).values)
        return t_enter < t_exit and 0 < t_exit < (lidar_range / beam.norm())
    
    def get_intersect_gaussians(self, beam: torch.Tensor, lidar_position: torch.Tensor, lidar_range: float):
        covariance = torch.zeros((0, 6), dtype=torch.float, device="cuda")
        xyz = torch.zeros((0, 3), dtype=torch.float, device="cuda")
        opacity = torch.zeros((0, 1), dtype=torch.float, device="cuda")
        nodestack = [self.rootnode]
        while len(nodestack) > 0:
            node = nodestack.pop()
            if node is None or not self.is_intersect(beam, lidar_position, lidar_range, node.bb):
                continue
            if node.is_leaf:
                covariance = torch.vstack((covariance, node.covariance))
                xyz = torch.vstack((xyz, node.xyz))
                opacity = torch.vstack((opacity, node.opacity))
            else:
                nodestack += node.children
                
        return covariance, xyz, opacity


class GaussianLidarRenderer():
    def __init__(self) -> None:
        pass

    def render(self, lidar_position: torch.Tensor, beams: torch.Tensor, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor, lidar_range: float=74):
        '''covariance stripped to n * 6'''
        octree_gaussian = OctreeGaussian(covariance, xyz, opacity)
         
        reflect_points = []
        weights = []

        beams = beams[torch.randperm(beams.shape[0])]
        progress_bar = tqdm(total=beams.shape[0])
        for beam in beams:
            covariance, xyz, opacity = octree_gaussian.get_intersect_gaussians(beam, lidar_position, lidar_range)
            if xyz.shape[0] <= 0:
                continue
            t, weight = _C.render_beam(lidar_position, beam, covariance, xyz, opacity)
            if t is not None:
                reflect_point = beam * t + lidar_position
                reflect_points.append(reflect_point)
                weights.append(weight)
                        
            progress_bar.set_postfix({"cuda mem": "{:.5f} G".format(torch.cuda.memory_allocated('cuda') / (1024 ** 3))})
            progress_bar.update()

        reflect_points = torch.stack(reflect_points)
        weights = torch.tensor(weights)

        return reflect_points, weights


            