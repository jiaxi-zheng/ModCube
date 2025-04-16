import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Sample rotation matrix (3x3)
rotation_matrix = np.array([[0.70738827, -0.70682518, 0],
                            [0.0, 0.1, 0.0],
                            [0.0, 0.0, 0.1]])

# Extract axis vectors
x_axis = rotation_matrix[:, 0]
y_axis = rotation_matrix[:, 1]
z_axis = rotation_matrix[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each axis vector
ax.quiver(0, 0, 0, *x_axis, color='r', length=1.0)
ax.quiver(0, 0, 0, *y_axis, color='g', length=1.0)
ax.quiver(0, 0, 0, *z_axis, color='b', length=1.0)

# Set plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title("Visualization of Rotation Matrix")

# Show the plot
plt.show()
