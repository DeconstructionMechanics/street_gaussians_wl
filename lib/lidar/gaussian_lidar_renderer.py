import torch
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel


def unstrip_symmetric(stripped: torch.Tensor):
    L = torch.zeros((stripped.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:, 0, 0] = stripped[:, 0]
    L[:, 0, 1] = stripped[:, 1]
    L[:, 0, 2] = stripped[:, 2]
    L[:, 1, 0] = stripped[:, 1]
    L[:, 1, 1] = stripped[:, 3]
    L[:, 1, 2] = stripped[:, 4]
    L[:, 2, 0] = stripped[:, 2]
    L[:, 2, 1] = stripped[:, 4]
    L[:, 2, 2] = stripped[:, 5]
    return L

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
        self.rootnode = _OctreeGaussianNode(xyz, opacity, covariance, depth=0, max_depth=5, min_points=128, verbose=False)

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




class _OctreeGaussianNode():
    def __init__(self, xyz: torch.Tensor, opacity: torch.Tensor, covariance: torch.Tensor, depth=0, max_depth=5, min_points=10, verbose=False):
        self.children = [None] * 8
        self.bb = torch.stack((torch.min(xyz[:,0]), torch.min(xyz[:,1]), torch.min(xyz[:,2]), torch.max(xyz[:,0]), torch.max(xyz[:,1]), torch.max(xyz[:,2])))
        self.depth = depth

        if verbose:
            print(f"{'  ' * (depth - 1)}octree node depth={depth} size={xyz.shape[0]}")

        if len(xyz) > min_points and depth < max_depth:
            self.is_leaf = False
            cuts = [torch.median(xyz[:,0]), torch.median(xyz[:,1]), torch.median(xyz[:,2])]
            # cuts = [(self.bb[0] + self.bb[3])/2, (self.bb[1] + self.bb[4])/2, (self.bb[2] + self.bb[5])/2]
            mask0 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask0):
                self.children[0] = _OctreeGaussianNode(xyz[mask0], opacity[mask0], covariance[mask0], self.depth + 1, max_depth, min_points, verbose)
            mask1 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask1):
                self.children[1] = _OctreeGaussianNode(xyz[mask1], opacity[mask1], covariance[mask1], self.depth + 1, max_depth, min_points, verbose)
            mask2 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask2):
                self.children[2] = _OctreeGaussianNode(xyz[mask2], opacity[mask2], covariance[mask2], self.depth + 1, max_depth, min_points, verbose)
            mask3 = (xyz[:,0] <= cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask3):
                self.children[3] = _OctreeGaussianNode(xyz[mask3], opacity[mask3], covariance[mask3], self.depth + 1, max_depth, min_points, verbose)
            mask4 = (xyz[:,0] > cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask4):
                self.children[4] = _OctreeGaussianNode(xyz[mask4], opacity[mask4], covariance[mask4], self.depth + 1, max_depth, min_points, verbose)
            mask5 = (xyz[:,0] > cuts[0]) & (xyz[:,1] <= cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask5):
                self.children[5] = _OctreeGaussianNode(xyz[mask5], opacity[mask5], covariance[mask5], self.depth + 1, max_depth, min_points, verbose)
            mask6 = (xyz[:,0] > cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] <= cuts[2])
            if torch.any(mask6):
                self.children[6] = _OctreeGaussianNode(xyz[mask6], opacity[mask6], covariance[mask6], self.depth + 1, max_depth, min_points, verbose)
            mask7 = (xyz[:,0] > cuts[0]) & (xyz[:,1] > cuts[1]) & (xyz[:,2] > cuts[2])
            if torch.any(mask7):
                self.children[7] = _OctreeGaussianNode(xyz[mask7], opacity[mask7], covariance[mask7], self.depth + 1, max_depth, min_points, verbose)
        else:
            self.is_leaf = True
            self.xyz = xyz.clone()
            self.opacity = opacity.clone()
            self.covariance = covariance.clone()

class _OctreeGaussian():
    def __init__(self, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor) -> None:
        rootnode = _OctreeGaussianNode(xyz, opacity, covariance, depth=0, max_depth=5, min_points=128, verbose=False)
        
        children_list = []
        bb_list = []
        info_list = []
        covariance_list = []
        xyz_list = []
        opacity_list = []

        queue = [rootnode]
        node_id = 1
        gauss_id = 0
        while len(queue) > 0:
            dequeue_node = queue.pop(0)
            children_node = []
            for child_node in dequeue_node.children:
                if child_node is not None:
                    queue.append(child_node)
                    children_node.append(node_id)
                    node_id += 1
                else:
                    children_node.append(-1)
            children_list.append(children_node)
            bb_list.append(dequeue_node.bb)
            info_node = [dequeue_node.depth]
            if dequeue_node.is_leaf:
                info_node.append(gauss_id)
                n_gauss_node = dequeue_node.opacity.shape[0]
                info_node.append(n_gauss_node)
                gauss_id += n_gauss_node
                covariance_list.append(dequeue_node.covariance)
                xyz_list.append(dequeue_node.xyz)
                opacity_list.append(dequeue_node.opacity)
            else:
                info_node.append(-1)
                info_node.append(0)
            info_list.append(info_node)
                

        self.children_tensor = torch.tensor(children_list, dtype=torch.int, device="cuda").contiguous()
        self.bb_tensor = torch.stack(bb_list).to(dtype=torch.float, device="cuda").contiguous()
        self.info_tensor = torch.tensor(info_list, dtype=torch.int, device="cuda").contiguous()
        self.covariance_tensor = torch.cat(covariance_list, 0).to(dtype=torch.float, device="cuda").contiguous()
        self.xyz_tensor = torch.cat(xyz_list, 0).to(dtype=torch.float, device="cuda").contiguous()
        self.opacity_tensor = torch.cat(opacity_list, 0).to(dtype=torch.float, device="cuda").contiguous()

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
        nodestack = [0]
        while len(nodestack) > 0:
            node_id = nodestack.pop()
            if node_id < 0 or not self.is_intersect(beam, lidar_position, lidar_range, self.bb_tensor[node_id]):
                continue
            if self.info_tensor[node_id, 1] >= 0:
                covariance = torch.vstack((covariance, self.covariance_tensor[self.info_tensor[node_id, 1]: self.info_tensor[node_id, 1] + self.info_tensor[node_id, 2]]))
                xyz = torch.vstack((xyz, self.xyz_tensor[self.info_tensor[node_id, 1]: self.info_tensor[node_id, 1] + self.info_tensor[node_id, 2]]))
                opacity = torch.vstack((opacity, self.opacity_tensor[self.info_tensor[node_id, 1]: self.info_tensor[node_id, 1] + self.info_tensor[node_id, 2]]))
            else:
                for child_id in self.children_tensor[node_id]:
                    nodestack.append(child_id.item())
        return covariance, xyz, opacity





class GaussianLidarRenderer():
    def __init__(self) -> None:
        pass

    def render(self, lidar_positions: list, beams: list, lidar_ranges, covariance, xyz, opacity, angle_range: float=0.999, weight_treshold: float=-100):
        '''beams normed'''
        octree_gaussian = OctreeGaussian(covariance, xyz, opacity)
        # _octree_gaussian = _OctreeGaussian(covariance, xyz, opacity)
        if not isinstance(lidar_ranges, list):
            lidar_ranges = [lidar_ranges for _ in range(len(lidar_positions))]
        
        full_reflect_points = []
        full_weights = []
        progress_bar = tqdm(total=len(beams) * beams[0].shape[0])
        for lidar_position, beam_set, lidar_range in zip(lidar_positions, beams, lidar_ranges):
            reflect_points = []
            weights = []
            for beam in beam_set:
                covariance, xyz, opacity = octree_gaussian.get_intersect_gaussians(beam, lidar_position, lidar_range)
                # _covariance, _xyz, _opacity = _octree_gaussian.get_intersect_gaussians(beam, lidar_position, lidar_range)
                # if (opacity.shape[0] != _opacity.shape[0]):
                #     print(opacity.shape[0], _opacity.shape[0])
                covariance = unstrip_symmetric(covariance)
                reflect_point, weight = self.render_beam(lidar_range, angle_range, weight_treshold, lidar_position, beam, covariance, xyz, opacity)
                if reflect_point is not None:
                    reflect_points.append(reflect_point)
                    weights.append(weight)
                
                progress_bar.set_postfix({"cuda mem": "{:.5f} G".format(torch.cuda.memory_allocated('cuda') / (1024 ** 3))})
                progress_bar.update()

            reflect_points = torch.stack(reflect_points)
            weights = torch.tensor(weights)
            full_reflect_points.append(reflect_points)
            full_weights.append(weights)

        return full_reflect_points, full_weights

    def render_beam(self, lidar_range: float, angle_range: float, weight_treshold: float, lidar_position: torch.Tensor, beam: torch.Tensor, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor, w_gaussian: float=0.01, w_distance: float=0):
        distance = xyz - lidar_position
        line_distance = (distance - torch.einsum('i,j->ij', ((distance @ beam) / (beam @ beam)), beam)).norm(dim=1)
        intersect_position_coefficient = torch.einsum('bi,bij,j->b', distance, covariance, beam) / torch.einsum('i,bij,j->b', beam, covariance, beam)
        # front_mask = (intersect_position_coefficient > 0) & (distance @ beam / (beam.norm() * distance.norm(dim=1)) > angle_range) & (beam.norm() * intersect_position_coefficient < lidar_range)
        
        # distance = distance[front_mask]
        # line_distance = line_distance[front_mask]
        # intersect_position_coefficient = intersect_position_coefficient[front_mask]
        # covariance = covariance[front_mask]
        # xyz = xyz[front_mask]
        # opacity = opacity[front_mask]
        
        intersect_positions = torch.einsum('b,i->bi', intersect_position_coefficient, beam) + lidar_position
        intersect_mu = intersect_positions - xyz
        
        do_ablend = True
        if not do_ablend:
            pass
        #     weights = torch.log(opacity.reshape(-1)) + w_gaussian * (-torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu)) - w_distance * torch.log(line_distance)
        #     # print(f"weights = {torch.log(opacity.reshape(-1))} + {torch.log(front_mask)} + {w_gaussian * (-torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu))}")

        #     reflect_mask = weights > weight_treshold
        #     if torch.any(reflect_mask):
        #         reflect_points = intersect_positions[reflect_mask]
        #         reflect_point_coefficient = intersect_position_coefficient[reflect_mask]
        #         reflect_weights = weights[reflect_mask]
        #         # print(reflect_weights.shape[0], torch.max(reflect_weights), torch.mean(reflect_weights), torch.min(reflect_weights))
        #         # print(reflect_point_coefficient.shape[0], torch.max(reflect_point_coefficient), torch.mean(reflect_point_coefficient), torch.min(reflect_point_coefficient))
        #         response1_index = torch.argmin(0 * reflect_point_coefficient - 1 * reflect_weights)

        #         if (reflect_points[response1_index] - lidar_position).norm() > lidar_range:
        #             print('[weight] ', reflect_weights[response1_index])
        #             print('[distance] ', reflect_points[response1_index] - lidar_position)
        #             print('[mask] ', front_mask[reflect_mask][response1_index])
        #             print(f'[mask angle] ({distance[reflect_mask][response1_index]} @ {beam} / ({beam.norm()} * {distance[reflect_mask][response1_index].norm()} = ', (distance[reflect_mask][response1_index] @ beam / (beam.norm() * distance[reflect_mask][response1_index].norm())))
        #             print('[mask distance] ', beam.norm() * reflect_point_coefficient[response1_index])
        #             print('[coeff] ', reflect_point_coefficient[response1_index])
        #             raise Exception
                
        #         return reflect_points[response1_index].clone(), float(reflect_weights[response1_index])
        #     else:
        #         return None, None
            
        else:
            remaining_opacity = 1
            intersect_position_coefficient_toreturn = 0
            weight_sum = 0
            sort_index = torch.argsort(intersect_position_coefficient)
            for w, power, coefficient in zip(opacity.reshape(-1)[sort_index], -torch.einsum('bi,bij,bj->b', intersect_mu, covariance, intersect_mu)[sort_index], intersect_position_coefficient[sort_index]):
                if power < -2.5 or power > 0:
                    continue
                alpha = min(0.99, w * torch.exp(power).item())
                if alpha < 0.004:
                    continue

                weight_sum
                intersect_position_coefficient_toreturn += remaining_opacity * alpha * coefficient
                remaining_opacity *= 1 - alpha
                
            if remaining_opacity < 0.5:
                return (intersect_position_coefficient_toreturn / (1 - remaining_opacity)) * beam + lidar_position, 1 - remaining_opacity
            return None, None


            