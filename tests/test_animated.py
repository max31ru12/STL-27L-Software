import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define a function to generate random data (replace this with your actual function)
def make_coordinates_from_pc(data):
    x, y = data  # Assuming data is a tuple of (x, y) coordinates
    return x, y


# Initialize empty lists to store data
x_data = []
y_data = []

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

# Set the axis limits
ax.set_xlim(0, 10)  # Change limits according to your data range
ax.set_ylim(0, 10)  # Change limits according to your data range


# Define an update function for animation
def update(frame):
    # Generate new data (replace this with your actual data source)
    data = np.random.rand(2) * 10  # Generating random (x, y) coordinates
    x, y = make_coordinates_from_pc(data)

    print(x, y)

    # Append new data to the lists
    x_data.append(x)
    y_data.append(y)

    # Update the plot with new data
    line.set_data(x_data, y_data)
    return line,


# Create an animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=1000, blit=True)

# Show the plot
plt.show()