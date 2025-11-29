import cv2
import torch
import numpy as np
import random

class EllinetDatasetCreator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.voronoi = Voronoi(width=width, height=height)

    def CreateSingleData(self):
        self.voronoi.SetParams()
        voronoi_seeds = self.voronoi.generate_random_points()
        polygon_list, color_list, voronoi_img, voronoi_edges, gradient_map_np, polygon_mask_list, edge_mask_list =\
            self.voronoi.ComputeVoronoi(voronoi_seeds)

        identity_matrix = torch.eye(2).unsqueeze(-1).unsqueeze(-1).float().cuda()  # Shape: (2, 2, 1, 1)
        init_cov_map = identity_matrix.repeat(1, 1, voronoi_img.shape[1], voronoi_img.shape[0])  # Shape: (2, 2, N, M)
        mean_t, cov_map_t, center_map = self.CreateMaps(init_cov_map, polygon_mask_list, edge_mask_list)
        center_map_np = center_map.cpu().numpy()  # Transpose the array to swap dimensions

        return voronoi_img, voronoi_edges, voronoi_seeds, mean_t, gradient_map_np, center_map_np, cov_map_t


    def CreateMaps(self, cov_map, area_mask_list, edge_mask_list):
        '''
        :param cov_map: 2 x 2 x W x H
        :param area_mask_list:
        :param edge_mask_list:
        :return:
        '''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize center_map
        center_map = torch.zeros((cov_map.shape[3], cov_map.shape[2]), device=device)  # Shape: (H, W)

        # Find nonzero (non-background) points
        edge_label, y_edge, x_edge = np.where(edge_mask_list != 0)
        edge_points = torch.tensor(np.stack((x_edge, y_edge), axis=1), dtype=torch.int32, device=device)
        edge_label_tensor = torch.tensor(edge_label, dtype=torch.int64, device=device)

        area_label, y_area, x_area = np.where(area_mask_list != 0)
        area_points = torch.tensor(np.stack((x_area, y_area), axis=1), dtype=torch.int32, device=device)
        area_label_tensor = torch.tensor(area_label, dtype=torch.int64, device=device)

        means = []
        unique_labels = torch.unique(edge_label_tensor)

        for label in unique_labels:
            # Edge points belonging to current polygon
            mask_edge = (edge_label_tensor == label)
            polygon_edge_points = edge_points[mask_edge].float()

            if polygon_edge_points.size(0) == 0:
                continue

            mean = torch.mean(polygon_edge_points, dim=0)
            means.append(mean)

            # Edge distance calculation
            edge_distances = torch.norm(polygon_edge_points - mean, dim=1)
            if edge_distances.numel() == 0:
                continue

            min_distance = torch.clamp(torch.min(edge_distances), min=0.1)

            # Area points belonging to current polygon
            mask_area = (area_label_tensor == label)
            polygon_area_points = area_points[mask_area].float()

            all_polygon_points = torch.cat((polygon_edge_points, polygon_area_points), dim=0)  # (M, 2)

            # Center map update
            area_distances = torch.norm(polygon_area_points - mean, dim=1)
            normalized_distances = torch.clamp(area_distances / min_distance, 0.0, 1.0)
            heat_map = 1 / (1 + 10 * normalized_distances)

            # Make sure x, y indexing matches
            x_coords = polygon_area_points[:, 0].long()
            y_coords = polygon_area_points[:, 1].long()
            center_map[y_coords, x_coords] = heat_map  # Careful: y, x

            # Covariance map update
            polygon_edge_points_v = polygon_edge_points.unsqueeze(0)  # (1, N, 2)
            all_polygon_points_v = all_polygon_points.unsqueeze(1)    # (M, 1, 2)
            duplicated_polygon_edge_points = polygon_edge_points_v.expand(all_polygon_points.shape[0], -1, -1)

            centered_points = (duplicated_polygon_edge_points - all_polygon_points_v) * 1.0
            cov_matrices = torch.einsum('mni,mnj->mij', centered_points, centered_points) / (polygon_edge_points.shape[0] - 1)
            cov_matrices = cov_matrices.float()

            x_coords_all = all_polygon_points[:, 0].long()
            y_coords_all = all_polygon_points[:, 1].long()
            cov_map[:, :, x_coords_all, y_coords_all] = cov_matrices.permute(2, 1, 0)

        if means:
            means_tensor = torch.stack(means).to(device)
        else:
            means_tensor = torch.empty((0, 2), dtype=torch.float32, device=device)

        return means_tensor, cov_map, center_map

    def draw_ellipssians_on_image(self, means, cov_map, image):
        means_np = means.cpu().numpy()
        cov_map_np = cov_map.cpu().numpy()

        result_img = image.copy()
        result_img.fill(255)

        x_coords = means_np[:, 0].astype(int)
        y_coords = means_np[:, 1].astype(int)

        # Extract corresponding covariances
        extracted_covariances = cov_map_np[:, :, x_coords, y_coords]
        extracted_colors = image[y_coords, x_coords, :]

        for i in range(means_np.shape[0]):
            mean = means_np[i]
            cov = extracted_covariances[:, :, i]
            color = tuple(int(c) for c in extracted_colors[i])

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            semi_major_axis = np.sqrt(eigenvalues[0])
            semi_minor_axis = np.sqrt(eigenvalues[1])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            center = (int(mean[0]), int(mean[1]))
            cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)),
                        angle, 0, 360, (0, 0, 0), 2)
            cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)),
                        angle, 0, 360, color, -1)

        return result_img

class Voronoi:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.img_shape = (self.height, self.width, 3)  # Height, Width, Channels

        self.distribution = None
        self.num_points = None
        self.num_clusters = None
        self.num_colors = None
        self.points = None

    def SetParams(self):
        self.distribution = np.random.randint(1, 21)  # [1, 20]
        self.num_points = np.random.randint(10, 501)  # [10, 500]
        self.num_colors = np.random.randint(10, 101)  # [10, 100]
        if np.random.randint(3) == 0:
            self.num_clusters = self.num_points
        else:
            self.num_clusters = np.random.randint(1, 11)  # [1, 10]
        self.points = self.generate_random_points()

    # Function to generate random 2D points
    def generate_random_points(self):
        # Create random points in clusters
        cluster_points = self.num_points // self.num_clusters
        points = []

        for _ in range(self.num_clusters):
            cluster_center_margin = 30  # You can set this higher or lower
            cluster_center = np.random.rand(2) * [
                self.img_shape[1] - 2 * cluster_center_margin,
                self.img_shape[0] - 2 * cluster_center_margin
            ] + [cluster_center_margin,
                 cluster_center_margin]
            cluster_std_dev = np.array([self.img_shape[1], self.img_shape[0]]) // 20
            cluster_points_array = np.random.randn(cluster_points, 2) * cluster_std_dev + cluster_center
            cluster_points_array = np.clip(
                cluster_points_array,
                [cluster_center_margin, cluster_center_margin],
                [self.img_shape[1] - cluster_center_margin - 1, self.img_shape[0] - cluster_center_margin - 1]
            )

            for cluster_point in cluster_points_array:
                padding_w = np.random.randint(30, 51)  # [10, 30]
                padding_h = np.random.randint(30, 51)  # [10, 30]
                clipped_point = np.clip(cluster_point, [padding_w, padding_h],
                                        [self.img_shape[1] - padding_w, self.img_shape[0] - padding_h])
                points.append(clipped_point)

        points = np.vstack(points)
        points = np.clip(points, [0, 0],
                         [self.img_shape[1] - 30, self.img_shape[0] - 30])  # Ensure points are within image bounds
        return points.astype(int)

    def ComputeVoronoi(self, seed_points):
        padding = 10
        img = np.zeros(self.img_shape, dtype=np.uint8)
        lined_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lined_img.fill(255)  # or img[:] = 255
        polygon_masks = np.zeros((0, self.img_shape[0], self.img_shape[1]), dtype=np.uint8)
        polygon_edge_masks = np.zeros((0, self.img_shape[0], self.img_shape[1]), dtype=np.uint8)

        gradient_map = np.zeros(self.img_shape[:2], dtype=np.float32)

        black_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        colors = []
        for i in range(self.num_colors):
            red = np.random.randint(256)
            green = np.random.randint(256)
            blue = np.random.randint(256)
            colors.append((red, green, blue))

        # Create a Subdiv2D object
        subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))

        # Insert points into Subdiv2D
        for p in seed_points:
            subdiv.insert((int(p[0]), int(p[1])))

        # Get Voronoi facets and centers
        facets, centers = subdiv.getVoronoiFacetList([])

        # Define the clipping rectangle
        img_rect = [(0, 0), (img.shape[1] - 1, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, img.shape[0] - 1)]

        # Store the polygon points
        polygon_points = []

        # Draw the Voronoi diagram with random colors
        result_color = []
        for facet in facets:
            facet = np.array(facet, dtype=np.int32)
            clipped_facet = self.sutherland_hodgman_clip(facet.tolist(), img_rect)
            if len(clipped_facet) > 0:
                clipped_facet = np.array(clipped_facet, dtype=np.int32)
                color = random.choice(colors)
                result_color.append(color)

                # Fill and draw the polygons on the original images
                edge_mask = black_img.copy()
                cv2.fillConvexPoly(img, clipped_facet, color)
                cv2.polylines(lined_img, [clipped_facet], True, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.polylines(edge_mask, [clipped_facet], True, (255), 1, cv2.LINE_AA)

                # Create a new mask
                polygon_mask = black_img.copy()
                cv2.fillConvexPoly(polygon_mask, clipped_facet, (255))

                # Add padding to prevent edge issues in distance transform
                padded_mask = np.pad(polygon_mask, ((padding, padding), (padding, padding)), mode='constant',
                                     constant_values=0)

                dist_transform = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)

                # Remove the padding after the distance transform
                dist_transform = dist_transform[padding:-padding, padding:-padding]

                dist_transform = dist_transform * (polygon_mask > 0)
                max_distance = dist_transform.max()

                if max_distance > 0:
                    normalized_gradient = (dist_transform / max_distance)
                else:
                    normalized_gradient = np.zeros_like(dist_transform)
                normalized_gradient = normalized_gradient
                cv2.polylines(normalized_gradient, [clipped_facet], True, (0, 0, 0), 1, cv2.LINE_AA)

                gradient_map += normalized_gradient * (polygon_mask > 0)

                polygon_masks = np.append(polygon_masks, polygon_mask[np.newaxis, ...], axis=0)
                polygon_edge_masks = np.append(polygon_edge_masks, edge_mask[np.newaxis, ...], axis=0)
                polygon_points.append(clipped_facet)

        return polygon_points, result_color, img, lined_img, gradient_map, polygon_masks, polygon_edge_masks  # (x, y)

    def sutherland_hodgman_clip(self, subject_polygon, clip_polygon):
        def inside(p):
            return (clip_edge[1][0] - clip_edge[0][0]) * (p[1] - clip_edge[0][1]) > (
                    clip_edge[1][1] - clip_edge[0][1]) * (p[0] - clip_edge[0][0])

        def compute_intersection():
            dc = (clip_edge[0][0] - clip_edge[1][0], clip_edge[0][1] - clip_edge[1][1])
            dp = (s[0] - e[0], s[1] - e[1])
            n1 = clip_edge[0][0] * clip_edge[1][1] - clip_edge[0][1] * clip_edge[1][0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return ((n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3)

        output_list = subject_polygon
        clip_edge_count = len(clip_polygon)
        for j in range(clip_edge_count):
            clip_edge = (clip_polygon[j], clip_polygon[(j + 1) % clip_edge_count])
            input_list = output_list
            output_list = []
            if len(input_list) == 0:
                return output_list
            s = input_list[-1]
            for e in input_list:
                if inside(e):
                    if not inside(s):
                        output_list.append(compute_intersection())
                    output_list.append(e)
                elif inside(s):
                    output_list.append(compute_intersection())
                s = e
        return output_list