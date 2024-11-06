import cv2
import torch
import numpy as np
import math
class EllipssianComputer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.max_len = math.sqrt((width**2) + (height**2))


    def CreateMaps(self, cov_map, area_mask_list, edge_mask_list):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize center_map as a PyTorch tensor on CUDA
        center_map = torch.zeros((cov_map.shape[2], cov_map.shape[3]), device=device)  # Shape: (H, W)

        # Convert edge and area mask points to tensors on the GPU
        edge_index, y_edge, x_edge = np.where(edge_mask_list != 0)
        edge_points = torch.tensor(np.column_stack((x_edge, y_edge)), dtype=torch.int32, device=device)  # Shape: (N, 2)

        area_index, y_area, x_area = np.where(area_mask_list != 0)
        area_points = torch.tensor(np.column_stack((x_area, y_area)), dtype=torch.int32, device=device)  # Shape: (L, 2)

        means = []
        unique_indices = torch.unique(torch.tensor(edge_index, device=device))

        H_factor = cov_map.shape[2]
        W_factor = cov_map.shape[3]
        normalization_factor = cov_map.shape[2] * cov_map.shape[3]

        for i in unique_indices:
            # Find the indices where edge_index equals the current unique index `i`
            indices = torch.where(torch.tensor(edge_index, device=device) == i)[0]

            # Use these indices to select the corresponding points from edge_points and convert to float
            polygon_edge_points = edge_points[indices].float()

            # Check if polygon_edge_points is empty
            if polygon_edge_points.size(0) == 0:
                continue  # Skip this iteration if there are no edge points

            # Calculate the mean of the polygon edge points
            mean = torch.mean(polygon_edge_points, dim=0)
            means.append(mean)

            #############################################################
            ########## Mean for visualize ellipssians
            #############################################################

            # Calculate Euclidean distances from each point to the mean
            edge_distances = torch.norm(polygon_edge_points - mean, dim=1)

            # Check if edge_distances is empty, and skip if so
            if edge_distances.numel() == 0:
                continue  # Skip this iteration if no distances were computed

            # Find the min distance
            min_distance = torch.min(edge_distances)
            min_distance = torch.clamp(min_distance, min=0.1)

            # Filter area points for this polygon and convert to float
            indices_area = torch.where(torch.tensor(area_index, device=device) == i)[0]
            polygon_area_points = area_points[indices_area].float()
            all_polygon_points = torch.cat((polygon_edge_points, polygon_area_points), dim=0)  # len: M

            #############################################################
            ########## Center map
            #############################################################

            # Calculate distances for all area points relative to the mean
            area_distances = torch.norm(polygon_area_points - mean, dim=1)

            # Normalize distances and apply fast decay function
            normalized_distances = area_distances / min_distance
            normalized_distances = torch.clamp(normalized_distances, min=0.0, max=1.0)
            heat_map = 1 / (1 + 10 * normalized_distances)

            # Assign heat map values to center_map for the polygon's area
            y_coords, x_coords = polygon_area_points[:, 1].long(), polygon_area_points[:, 0].long()
            center_map[x_coords, y_coords] = heat_map

            #############################################################
            ########## Cov map
            #############################################################

            # Compute covariance matrices for all points at once (vectorized)
            polygon_edge_points_v = polygon_edge_points.unsqueeze(0)  # Shape: (1, N, 2)
            duplicated_polygon_edge_points = polygon_edge_points_v.expand(all_polygon_points.shape[0], -1,
                                                                          -1)  # Shape: (M, N, 2)
            all_polygon_points_v = all_polygon_points.unsqueeze(1)  # Shape: (M, 1, 2)

            # Compute centered_points relative to all_polygon_points
            centered_points = (duplicated_polygon_edge_points - all_polygon_points_v) * 1.0  # Shape: (M, N, 2)

            # Compute covariance matrices using torch.einsum for efficiency
            cov_matrices = torch.einsum('mni,mnj->mij', centered_points, centered_points) / (polygon_edge_points.shape[0] - 1)
            cov_matrices = cov_matrices.float()

            # Assign covariance matrices to cov_map on GPU
            all_y_coords, all_x_coords = all_polygon_points[:, 1].long(), all_polygon_points[:, 0].long()
            cov_map[:, :, all_x_coords, all_y_coords] = cov_matrices.permute(2, 1, 0)

        # Check if means is empty, and stack if not
        cov_map[0, 0, :, :] /= W_factor ** 2  # Normalize the variance along x-axis
        cov_map[1, 1, :, :] /= H_factor ** 2  # Normalize the variance along y-axis
        cov_map[0, 1, :, :] /= W_factor * H_factor  # Normalize the covariance between x and y
        cov_map[1, 0, :, :] /= W_factor * H_factor  # Normalize the covariance between x and y (symmetric entry)
        if means:
            means_tensor = torch.stack(means).to(device)
        else:
            means_tensor = torch.empty((0, 2), dtype=torch.float32, device=device)  # Empty tensor if no means

        return means_tensor, cov_map, center_map



    def draw_ellipssians_on_image(self, means, cov_map, image):
        # Convert means and covariances to numpy for easier manipulation
        means_np = means.cpu().numpy()
        cov_map_np = cov_map.cpu().numpy()

        result_img = image.copy()
        result_img.fill(255)

        H_factor = image.shape[0]
        W_factor = image.shape[1]

        cov_map_np[0, 0, :, :] *= W_factor ** 2  # Scale up the variance along x-axis
        cov_map_np[1, 1, :, :] *= H_factor ** 2  # Scale up the variance along y-axis
        cov_map_np[0, 1, :, :] *= W_factor * H_factor  # Scale up the covariance between x and y
        cov_map_np[1, 0, :, :] *= W_factor * H_factor  # Scale up the covariance between x and y (symmetric entry)


        # Vectorized indexing: extract all covariance matrices for the given (y, x) points
        x_coords, y_coords = means_np[:, 0].astype(int), means_np[:, 1].astype(int)

        # Extract covariances using advanced indexing
        extracted_covariances = cov_map_np[:, :, x_coords, y_coords]
        extracted_colors = image[y_coords, x_coords, :]


        for i in range(means_np.shape[0]):
            mean = means_np[i]  # Center of the Gaussian (x, y)
            cov = extracted_covariances[:, :, i]
            color = (int(extracted_colors[i, 0]), int(extracted_colors[i, 1]), int(extracted_colors[i, 2]))

            # Perform eigenvalue decomposition to get the axes lengths and orientation
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort eigenvalues and eigenvectors (to ensure correct assignment to major/minor axis)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Compute the semi-major and semi-minor axes (as 2 standard deviations for visualization)
            semi_major_axis = np.sqrt(eigenvalues[0])
            semi_minor_axis = np.sqrt(eigenvalues[1])

            # Compute the angle of the ellipse (in degrees)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # Convert the mean to int for OpenCV drawing
            center = (int(mean[0]), int(mean[1]))

            # Draw the ellipse using OpenCV
            cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)), angle, 0, 360, (0, 0, 0), 2)
            cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)), angle, 0, 360, color, -1)

        return result_img

