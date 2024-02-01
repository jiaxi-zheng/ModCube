import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R

# Define the function to draw arrows
def draw_arrow(ax, origin, orientation, length=0.05, label=''):
    # Define the arrows for each axis
    arrows = np.array([[length, 0, 0],
                       [0, length, 0],
                       [0, 0, length]])
    arrows = orientation.apply(arrows) + origin

    # Colors for each axis
    colors = ['r', 'g', 'b']

    # Draw the arrows
    for i in range(3):
        ax.quiver(*origin, *(arrows[i] - origin), color=colors[i])

    ax.text(*arrows[0], label, fontsize=10, color='blue')

def draw_cylinder(ax, origin, orientation, length=0.06, radius=0.01):
    # Define the cylinder parameters
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    y_circle = radius * np.cos(theta)  # y is the circle part
    z_circle = radius * np.sin(theta)  # z is the circle part
    x_circle = np.linspace(0, length, 2)  # x is along the cylinder's length

    # Create a grid for the cylinder
    Yc, Xc = np.meshgrid(y_circle, x_circle)
    Zc, _ = np.meshgrid(z_circle, x_circle)

    # Stack the grid to create a 3D mesh
    cylinder = np.array([Xc.flatten(), Yc.flatten(), Zc.flatten()])

    # Apply rotation and translation
    cylinder = orientation.apply(cylinder.T).T + origin[:, np.newaxis]

    # Reshape and plot cylinder
    Xc = cylinder[0].reshape(Yc.shape)
    Yc = cylinder[1].reshape(Yc.shape)
    Zc = cylinder[2].reshape(Yc.shape)
    ax.plot_surface(Xc, Yc, Zc, color='c', alpha=0.5)

thrusters = {
    0: {'xyz': [0.07, -0.07, -0.07], 'rpy': [np.pi/2, 0, np.pi/4]},
    1: {'xyz': [0.07, 0.07, -0.07], 'rpy': [np.pi/2, 0, -np.pi/4]},
    2: {'xyz': [-0.07, -0.07, 0.07], 'rpy': [-np.pi/2, 0, 3*np.pi/4]},
    3: {'xyz': [-0.07, 0.07, 0.07], 'rpy': [-np.pi/2, 0, -3*np.pi/4]},
    4: {'xyz': [0.07, -0.07, 0.07], 'rpy': [-np.pi/2, -np.pi/4, np.pi/2]},
    5: {'xyz': [0.07, 0.07, 0.07], 'rpy': [-np.pi/2, -np.pi/4, -np.pi/2]},
    6: {'xyz': [-0.07, -0.07, -0.07], 'rpy': [np.pi/2, np.pi/4, -np.pi/2]},
    7: {'xyz': [-0.07, 0.07, -0.07], 'rpy': [np.pi/2, np.pi/4, np.pi/2]}
}

# thrusters = {
#     0: {'xyz': [0.07, -0.07, -0.07], 'rpy': [np.pi/2, -np.pi/4, 0]},
#     1: {'xyz': [0.07, 0.07, -0.07], 'rpy': [np.pi/2, np.pi/4, 0]},
#     2: {'xyz': [-0.07, -0.07, 0.07], 'rpy': [np.pi/2, np.pi/4, np.pi]},
#     3: {'xyz': [-0.07, 0.07, 0.07], 'rpy': [np.pi/2, -np.pi/4, -np.pi]},
#     4: {'xyz': [0.07, -0.07, 0.07], 'rpy': [-3*np.pi/4, np.pi/2, 0]},
#     5: {'xyz': [0.07, 0.07, 0.07], 'rpy': [-np.pi/4, -np.pi/2, 0]},
#     6: {'xyz': [-0.07, -0.07, -0.07], 'rpy': [np.pi/4, -np.pi/2, 0]},
#     7: {'xyz': [-0.07, 0.07, -0.07], 'rpy': [3*np.pi/4, np.pi/2, 0]}
# }

# <!-- [ x: np.pi/2, y: -np.pi/4, z: 0 ] -->
# <!-- [ x: np.pi/2, y: np.pi/4, z: -0 ] -->
# <!-- [ x: np.pi/2, y: np.pi/4, z: np.pi ] -->
# <!-- [ x: np.pi/2, y: -np.pi/4, z: -np.pi ] -->
# <!-- [ x: -3*np.pi/4, y: np.pi/2, z: 0 ] -->
# <!-- [ x: -np.pi/4, y: -np.pi/2, z: 0 ] -->
# <!-- [ x: np.pi/4, y: -np.pi/2, z: 0 ] -->
# <!-- [ x: 3*np.pi/4, y: np.pi/2, z: 0 ] -->

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the limits of the plot
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([-0.1, 0.1])

# Set the labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Plot each thruster and its cylinder
for id, thruster in thrusters.items():
    position = np.array(thruster['xyz'])
    orientation = R.from_euler('xyz', thruster['rpy'])
    draw_arrow(ax, position, orientation, label=str(id))
    draw_cylinder(ax, position, orientation)

# Set the title
ax.set_title('Thruster Orientation Visualization')

# Show the plot
plt.show()
