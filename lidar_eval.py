import torch

def chamfer_distance(pointcloud1, mask1, pointcloud2, mask2):
    """
    Calculate Chamfer Distance between two point clouds.
    
    Args:
        pointcloud1 (torch.Tensor): Tensor of shape (N, 3) representing the first point cloud.
        mask1 (bool tensor): size (N)
        pointcloud2 (torch.Tensor): Tensor of shape (N, 3) representing the second point cloud.
        mask2 (bool tensor): size (N)
    
    Returns:
        torch.Tensor: Scalar value of Chamfer Distance.
    """
    pointcloud1 = pointcloud1[mask1]
    pointcloud2 = pointcloud2[mask2]
    diff1 = pointcloud1.unsqueeze(1) - pointcloud2.unsqueeze(0)  # Shape: (N, M, 3)
    dist1 = torch.sum(diff1 ** 2, dim=-1)  # Squared distances: Shape (N, M)

    min_dist1, _ = torch.min(dist1, dim=1)  # Shape: (N,)
    min_dist2, _ = torch.min(dist1, dim=0)  # Shape: (M,)

    chamfer_dist = torch.sum(min_dist1) + torch.sum(min_dist2)

    return chamfer_dist.item()

def f1_score(pointcloud1, mask1, pointcloud2, mask2, threshold=0.05):
    distance_close = torch.norm(pointcloud1 - pointcloud2, dim=1) < threshold
    tp = torch.sum(mask1 & mask2 & distance_close)
    fp = torch.sum(mask1) - tp
    tn = mask1.shape[0] - torch.sum(mask1 | mask2)
    fn = torch.sum(mask2) - torch.sum(mask1 & mask2)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def RMSE(pointcloud1, mask1, pointcloud2, mask2):
    pc1 = pointcloud1[mask1 & mask2]
    pc2 = pointcloud2[mask1 & mask2]
    assert pc1.shape == pc2.shape, "RMSE Point clouds must have the same shape."
    rmse = torch.sqrt(torch.mean(torch.sum((pc1 - pc2) ** 2, dim=1)))
    return rmse

def MAE(pointcloud1, mask1, pointcloud2, mask2):
    pc1 = pointcloud1[mask1 & mask2]
    pc2 = pointcloud2[mask1 & mask2]
    assert pc1.shape == pc2.shape, "MAE Point clouds must have the same shape."
    mae = torch.mean(torch.mean(torch.abs(pc1 - pc2), dim=1))
    return mae

def eval(pointcloud1, mask1, pointcloud2, mask2, verbose=True):
    cd = chamfer_distance(pointcloud1, mask1, pointcloud2, mask2)
    f_score = f1_score(pointcloud1, mask1, pointcloud2, mask2)
    rmse = RMSE(pointcloud1, mask1, pointcloud2, mask2)
    mae = MAE(pointcloud1, mask1, pointcloud2, mask2)
    if verbose:
        print(f'CD: {cd}, F-score: {f_score}, RMSE: {rmse}, MAE: {mae}')
    return cd, f_score, rmse, mae

for _ in range(10):
    pc1 = torch.rand((10000, 3))  # Random point cloud with 100 points
    pc2 = torch.rand((10000, 3))  # Random point cloud with 200 points
    mask1 = torch.ones(10000).to(torch.bool)
    mask2 = torch.ones(10000).to(torch.bool)
    eval(pc1, mask1, pc2, mask2, True)
