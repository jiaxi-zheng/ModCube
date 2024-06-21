import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.transform import Rotation as R
np.set_printoptions(threshold=10000)  # 1000 is the default threshold
from scipy.special import sph_harm
from scipy.linalg import lstsq
from scipy.spatial import Delaunay

# For cube_v1 
thruster_positions = np.array([
    [0.07, 0.07, -0.07],
    [0.07, -0.07, -0.07],
    [-0.07, 0.07, 0.07],
    [-0.07, -0.07, 0.07],
    [0.07, 0.07, 0.07],
    [0.07, -0.07, 0.07],
    [-0.07, 0.07, -0.07],
    [-0.07, -0.07, -0.07]
])
thruster_directions = np.array([
    [ np.pi/2,  0      , -np.pi/4  ],
    [ np.pi/2,  0      ,  np.pi/4  ],
    [-np.pi/2,  0      , -3*np.pi/4],
    [-np.pi/2,  0      ,  3*np.pi/4],
    [-np.pi  , -np.pi/4, -np.pi/2  ],
    [ 0      , -np.pi/4,  np.pi/2  ],
    [ 0      ,  np.pi/4, -np.pi/2  ],
    [ np.pi  ,  np.pi/4,  np.pi/2  ]
])

# For kingfisher
# thruster_positions = np.array([
#     [0.43852, 0.22093, 0.0],
#     [0.43852, -0.22093, 0.0],
#     [-0.43852, 0.22093, 0.0],
#     [-0.43852, -0.22093, 0.0],
#     [0.23876, 0.213083, 0.01115],
#     [0.23876, -0.213083, 0.01115],
#     [-0.23876, 0.213083, 0.01115],
#     [-0.23876, -0.213083, 0.01115]
# ])
# thruster_directions = np.array([
#     [ np.pi/2,  0      , -np.pi/4  ],
#     [ np.pi/2,  0      ,  np.pi/4  ],
#     [-np.pi/2,  0      , -3*np.pi/4],
#     [-np.pi/2,  0      ,  3*np.pi/4],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ]
# ])

# For BlueRov2 heavy
# thruster_positions = np.array([
#     [0.156, 0.111, -0.085],
#     [0.156, -0.111, -0.085],
#     [-0.156, 0.111, -0.085],
#     [-0.156, -0.111, -0.085],
#     [0.120, 0.218, 0.0],
#     [0.120, -0.218, 0.0],
#     [-0.120, 0.218, 0.0],
#     [-0.120, -0.218, 0.0]
# ])
# thruster_directions = np.array([
#     [ np.pi/2,  0      , -np.pi/4  ],
#     [ np.pi/2,  0      ,  np.pi/4  ],
#     [-np.pi/2,  0      , -3*np.pi/4],
#     [-np.pi/2,  0      ,  3*np.pi/4],
#     [2*np.pi, np.pi/2,  2*np.pi  ],
#     [2*np.pi, np.pi/2,  2*np.pi  ],
#     [2*np.pi, np.pi/2,  2*np.pi  ],
#     [2*np.pi, np.pi/2,  2*np.pi  ]
# ])

# For Chasing
# thruster_positions = np.array([
#     [0.07, 0.07, -0.07],
#     [0.07, -0.07, -0.07],
#     [-0.07, 0.07, 0.07],
#     [-0.07, -0.07, 0.07],
#     [0.07, 0.07, 0.07],
#     [0.07, -0.07, 0.07],
#     [-0.07, 0.07, -0.07],
#     [-0.07, -0.07, -0.07]
# ])
# thruster_directions = np.array([
#     [ -0.556475,    2.7340882    , 2.8218116  ],
#     [ -0.556475,    2.7340882    , -2.8218116  ],
#     [-0.556475,    -0.4069118    , -2.8218116  ],
#     [-0.556475,    -0.4069118    , 2.8218116  ],
#     [-0.556475,    3.5479118    , 2.8218116],
#     [-0.556475,    3.5479118    , -2.8218116],
#     [-0.556475,    0.4069118    , -2.8218116],
#     [-0.556475,    0.4069118    , 2.8218116]
# ])


# For BlueRov2
# thruster_positions = np.array([
#     [0.156, 0.111, 0.085],
#     [0.156, -0.111, 0.085],
#     [-0.156, 0.111, 0.085],
#     [-0.156, -0.111, 0.085],
#     [0.0, 0.111, 0.0],
#     [0.0, -0.111, 0.0]
# ])
# thruster_directions = np.array([
#     [ np.pi/2,  0      , -np.pi/4  ],
#     [ np.pi/2,  0      ,  np.pi/4  ],
#     [-np.pi/2,  0      , -3*np.pi/4],
#     [-np.pi/2,  0      ,  3*np.pi/4],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ],
#     [-np.pi/2, -np.pi/2,  2*np.pi  ]
# ])

def draw_arrow(ax, origin, orientation, length=0.05, label=''):
    arrows = np.array([[length, 0, 0],
                       [0, length, 0],
                       [0, 0, length]])
    arrows = orientation.apply(arrows) + origin
    colors = ['r', 'g', 'b']

    for i in range(3):
        ax.quiver(*origin, *(arrows[i] - origin), color=colors[i])
    ax.text(*arrows[0], label, fontsize=10, color='blue')

def draw_cylinder(ax, origin, orientation, length=0.06, radius=0.01):
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    y_circle = radius * np.cos(theta)
    z_circle = radius * np.sin(theta)
    x_circle = np.linspace(0, length, 2)
    Yc, Xc = np.meshgrid(y_circle, x_circle)
    Zc, _ = np.meshgrid(z_circle, x_circle)
    cylinder = np.array([Xc.flatten(), Yc.flatten(), Zc.flatten()])
    cylinder = orientation.apply(cylinder.T).T + origin[:, np.newaxis]
    Xc = cylinder[0].reshape(Yc.shape)
    Yc = cylinder[1].reshape(Yc.shape)
    Zc = cylinder[2].reshape(Yc.shape)
    ax.plot_surface(Xc, Yc, Zc, color='c', alpha=0.5)

def generate_spherical_fibonacci_points(num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.vstack((x, y, z)).T

def normalize_vector(v):
    return v / np.linalg.norm(v)

def calculate_thrust(thrusters, directions, inv_tam):
    num_directions = len(directions)

    total_thrust_values = []

    for j in range(num_directions):
        direction = directions[j]
        thrust_vector = inv_tam @ np.hstack((np.zeros(3), direction))
        total_thrust_values.append(np.sum(abs(thrust_vector)))

    return total_thrust_values

def build_tam(thruster_positions, thruster_directions):
    num_thrusters = thruster_positions.shape[0]
    tam = np.zeros((6, num_thrusters))

    for i in range(num_thrusters):
        position = thruster_positions[i]
        direction = thruster_directions[i]
        rotation_matrix = R.from_euler('xyz', direction).as_matrix()
        force = rotation_matrix @ np.array([1, 0, 0])
        torque = np.cross(position, force)
        tau = np.hstack((force, torque))
        tam[:, i] = tau
    inv_tam = np.linalg.pinv(tam)
    return inv_tam

def plot_global_search(ax, directions, areas):
    norm = Normalize(vmin=np.min(areas), vmax=np.max(areas))
    scaled_directions = directions * np.array(areas)[:, None]
    hull = ConvexHull(scaled_directions)
    # print(" hull.volume : ", hull.volume)
    # print(" hull.area : ", hull.area)
    for simplex in hull.simplices:
        triangle = scaled_directions[simplex]
        color = plt.cm.viridis(norm(np.mean([areas[i] for i in simplex])))
        poly = Poly3DCollection([triangle], color=[color], alpha=0.5, linewidths=0.2, edgecolors='k')
        ax.add_collection3d(poly)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(areas)
    plt.colorbar(mappable, ax=ax)

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


def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)

def gaussian_curvature(points, hull):
    curvatures = []
    for simplex in hull.simplices:
        vertices = points[simplex]
        curvature = np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) / 2
        curvatures.append(curvature)

    average_curvature = np.mean(curvatures)
    print("Average Gaussian Curvature:", average_curvature)
    variance_curvature = np.var(curvatures)
    print("Variance of Curvature:", variance_curvature)
    return curvatures

def face_normal(points,face):
    u = points[face[1]] - points[face[0]]
    v = points[face[2]] - points[face[0]]
    return np.cross(u, v)

############################
def compute_centroid(points):
    return np.mean(points, axis=0)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def compute_spherical_harmonics(theta, phi, l_max):
    Y = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y.append(sph_harm(m, l, phi, theta))
    return np.array(Y).T

def compute_dirichlet_energy(points, l_max=10):
    centroid = compute_centroid(points)
    spherical_points = [cartesian_to_spherical(*(p - centroid)) for p in points]
    r, theta, phi = zip(*spherical_points)

    Y = compute_spherical_harmonics(theta, phi, l_max)
    a_lm, _, _, _ = lstsq(Y, r)

    dirichlet_energy = sum(l * (l + 1) * np.abs(a_lm[l]**2) for l in range(l_max + 1))
    return dirichlet_energy

def compute_mean_curvature(points):
    tri = Delaunay(points)
    mean_curvature = np.zeros(len(points))
    for i, point in enumerate(points):
        neighbors = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
        neighbor_points = points[neighbors]
        centroid = np.mean(neighbor_points, axis=0)
        mean_curvature[i] = np.linalg.norm(point - centroid)
    return mean_curvature

def compute_continuity_index(mean_curvature):
    return np.var(mean_curvature)

def compute_barycentric_area(points):
    tri = Delaunay(points)
    barycentric_area = np.zeros(len(points))
    for simplex in tri.simplices:
        for vertex in simplex:
            neighbors = points[simplex]
            triangle_area = np.linalg.norm(np.cross(neighbors[1] - neighbors[0], neighbors[2] - neighbors[0])) / 2
            barycentric_area[vertex] += triangle_area / 3
    return barycentric_area

def compute_willmore_energy(points):
    mean_curvature = compute_mean_curvature(points)
    barycentric_area = compute_barycentric_area(points)
    willmore_energy = np.sum((mean_curvature**2) * barycentric_area) - 4 * np.pi
    return willmore_energy
###########################

def main():
    num_points = 1000
    inv_tam = build_tam(thruster_positions, thruster_directions)

    directions = generate_spherical_fibonacci_points(num_points)
    directions = np.array([normalize_vector(dir) for dir in directions])

    total_thrust_values = calculate_thrust(thruster_positions, directions, inv_tam)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('Thruster Orientation Visualization')
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')

    for i, position in enumerate(thruster_positions):
        orientation = R.from_euler('xyz', thruster_directions[i])
        draw_arrow(ax1, position, orientation, label=str(i))
        draw_cylinder(ax1, position, orientation)

    fig1.tight_layout()
    save_figure(fig1, 'thruster_orientation_visualization.png')
    fig1.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title("Visualizing Thrust Areas Convex Hull")

    plot_global_search(ax2, directions, total_thrust_values)

    fig2.tight_layout()
    save_figure(fig2, 'thrust_areas_convex_hull.png')
    fig2.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    scaled_directions = directions * np.array(total_thrust_values)[:, None]
    plot_thrust_points(ax3, np.array(scaled_directions))
    
###########################
    # hull = ConvexHull(scaled_directions)
    # gaussian_curvature(scaled_directions,hull)

    # normals = np.array([face_normal(scaled_directions,face) for face in hull.simplices])
    # normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize

    # # Calculate the angles between adjacent faces
    # angles = []
    # for i in range(len(normals)):
    #     for j in range(i + 1, len(normals)):
    #         angle = np.arccos(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0))
    #         angles.append(np.degrees(angle))

    # # Calculate the variance of the angles
    # angle_variance = np.var(angles)
    # print("Variance of Angles Between Faces:", angle_variance)
###############################
    dirichlet_energy = compute_dirichlet_energy(scaled_directions)
    mean_curvature = compute_mean_curvature(scaled_directions)
    continuity_index = compute_continuity_index(scaled_directions)


    print(f"Dirichlet energy: {dirichlet_energy}")
    print(f"Continuity index: {continuity_index}")

    distances = np.linalg.norm(scaled_directions, axis=1)
    variance = np.var(distances)
    print(f"variance : {variance}")

    willmore_energy = compute_willmore_energy(scaled_directions)
    print(f"Willmore energy: {willmore_energy}")
    
    fig3.tight_layout()
    save_figure(fig3, 'thrust_points_scatter_plot.png')
    fig3.show()

    print(" Total_torque_values : ", np.sum(total_thrust_values)*np.sum(np.abs(thruster_positions)/1000))

    plt.show()

if __name__ == "__main__":
    main()
