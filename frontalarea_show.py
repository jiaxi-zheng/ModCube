import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_structure(num_cubes,resolution=400):
    # Initialize positions with an example starting position
    positions = {(0, 0, 0)}
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while len(positions) < num_cubes:
        new_pos = random.choice(list(positions))
        shift = random.choice(directions)
        next_pos = tuple(np.array(new_pos) + np.array(shift))
        while next_pos in positions:
            new_pos = random.choice(list(positions))
            shift = random.choice(directions)
            next_pos = tuple(np.array(new_pos) + np.array(shift))
        positions.add(next_pos)

    # Convert set of positions to a numpy array for easier manipulation
    positions = np.array(list(positions))

    # Determine the number of points per dimension per cube (assuming cube size is 1)
    points_per_dimension = round(resolution**(1/3))  # For 20 points, get the cube root and round it
    step = 1 / points_per_dimension
    offsets = np.linspace(0, 1-step, points_per_dimension)  # Create evenly spaced points within [0, 1-step]
    
    # Generate the grid of points within a unit cube starting at (0, 0, 0)
    local_points = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
    
    # Initialize an array to hold all sampled points
    all_sampled_points = np.empty((0, 3), dtype=float)
    
    # Create sampled points for each cube by translating the local_points grid to the cube's position
    for pos in positions:
        sampled_points = local_points + pos
        all_sampled_points = np.vstack((all_sampled_points, sampled_points))

    return positions, all_sampled_points

def normalize_vector(v):
    return v / np.linalg.norm(v)

def get_rotation_matrix(direction):
    # direction = normalize_vector(direction)
    z_axis = np.array([0, 0, 1])
    v = np.cross(direction, z_axis)
    c = np.dot(direction, z_axis)
    s = np.linalg.norm(v)
    if s == 0:  # direction is parallel to z_axis
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def project_cubes(positions, direction):
    rotation_matrix = get_rotation_matrix(direction)

    cube_size=25
    rotated_positions = (cube_size * positions).dot(rotation_matrix.T)
    projected_positions = rotated_positions[:, :2]
    depths = rotated_positions[:, 2]

    # Create a projection map that stores the maximum depth for each 2D projected position
    projection_map = {}
    for pos, depth in zip(projected_positions, depths):
        pos_tuple = tuple(pos)
        if pos_tuple not in projection_map or projection_map[pos_tuple] < depth:
            projection_map[pos_tuple] = depth
    
    return projected_positions, projection_map

def calculate_visible_area_convex_hull(projected_positions):
    if len(projected_positions) < 3:
        return 0  # A minimum of three points is required to form a polygon
    hull = ConvexHull(projected_positions)
    return hull.volume  # For 2D, hull.volume gives the area

def generate_spherical_fibonacci_points(num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.vstack((x, y, z)).T

def plot_cubes(ax,positions,sampled_points):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    cube_size = 25  # mm
    cube_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    cube_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]]

    all_vertices = []

    for pos in positions:
        # Calculate the vertices for each cube
        vertices = cube_vertices * cube_size + np.array(pos) * cube_size
        all_vertices.extend(vertices)  # Collect all vertices for centroid calculation
        cube = Poly3DCollection([vertices[face] for face in cube_faces], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
        ax.add_collection3d(cube)

    # Calculate the geometric center using all vertices
    all_vertices = np.array(all_vertices)
    centroid = np.mean(all_vertices, axis=0)
    # ax.scatter(cube_size*sampled_points[:, 0], cube_size*sampled_points[:, 1], cube_size*sampled_points[:, 2], color='magenta', s=1)

    # Plot the centroid
    ax.scatter(*centroid, color='red', s=100, label='Geometric Center')
    ax.legend()

    # Set the limits for better visibility
    padding = cube_size * 5
    ax.set_xlim(centroid[0] - padding, centroid[0] + padding)
    ax.set_ylim(centroid[1] - padding, centroid[1] + padding)
    ax.set_zlim(centroid[2] - padding, centroid[2] + padding)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Structure of Cubes with Center')

def plot_global_search(ax,directions, areas):
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Visualizing Frontal Surface Areas")

    norm = Normalize(vmin=np.min(areas), vmax=np.max(areas))

    scaled_directions = directions * np.array(areas)[:, None]

    hull = ConvexHull(scaled_directions)

    for simplex in hull.simplices:
        triangle = scaled_directions[simplex]
        color = plt.cm.viridis(norm(np.mean([areas[i] for i in simplex])))
        poly = Poly3DCollection([triangle], color=[color], alpha=0.5, linewidths=0.2, edgecolors='k')
        ax.add_collection3d(poly)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(areas)
    max_extent = np.max(np.linalg.norm(scaled_directions, axis=1))
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_thrust_points(ax, points):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', marker='o')
    ax.set_title('Thrust Points Scatter Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def sync_views(fig, ax1, ax2):
    """Synchronize the view angles of two 3D subplots."""
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        fig.canvas.draw_idle()

    return on_move

def main():
    num_cubes = random.randint(5, 10)
    positions,sampled_points = generate_structure(num_cubes)

    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    plot_cubes(ax1, positions, sampled_points)

    num_points = 500  # Number of points/directions on the sphere
    directions = generate_spherical_fibonacci_points(num_points)
    directions /= np.linalg.norm(directions, axis=1)[:, None]  # Normalize

    visible_areas = []
    for dir in directions:
        proj_pos, proj_map = project_cubes(sampled_points, dir)
        visible_area = calculate_visible_area_convex_hull(proj_pos)
        visible_areas.append(visible_area)
    plot_global_search(ax2, directions, visible_areas)
    callback = sync_views(fig, ax1, ax2)
    fig.canvas.mpl_connect('button_release_event', callback)
    fig.canvas.mpl_connect('motion_notify_event', callback)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    scaled_directions = directions * np.array(visible_areas)[:, None]
    plot_thrust_points(ax3, np.array(scaled_directions))
    
    plt.show()

if __name__ == "__main__":
    main()
