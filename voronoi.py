import numpy as np
import cv2
import random

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
            cluster_center = np.random.rand(2) * [self.img_shape[1], self.img_shape[0]]
            cluster_std_dev = np.array([self.img_shape[1], self.img_shape[0]]) // 20
            cluster_points_array = np.random.randn(cluster_points, 2) * cluster_std_dev + cluster_center
            points.append(cluster_points_array)

            for cluster_point in cluster_points_array:
                padding_w = np.random.randint(1, 31)  # [10, 30]
                padding_h = np.random.randint(1, 31)  # [10, 30]
                clipped_point = np.clip(cluster_point, [padding_w, padding_h],
                                        [self.img_shape[1] - padding_w, self.img_shape[0] - padding_h])
                # clipped_point = np.clip(cluster_point, [0, 0],
                #                         [self.img_shape[1] - 1, self.img_shape[0] - 1])
                points.append(clipped_point)

        points = np.vstack(points)
        points = np.clip(points, [0, 0], [self.img_shape[1] - 1, self.img_shape[0] - 1])  # Ensure points are within image bounds
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
        img_rect = [(0, 0), (img.shape[1]-1, 0), (img.shape[1]-1, img.shape[0]-1), (0, img.shape[0]-1)]

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

                # cv2.imshow("normalized_gradient", normalized_gradient)
                # cv2.imshow("masks", polygon_mask)
                # cv2.waitKey(0)

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


#
# voronoi = Vor