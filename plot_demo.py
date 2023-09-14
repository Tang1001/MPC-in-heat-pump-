import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Generate some sample data
x = np.linspace(0, 24, 1000)  # Simulating hours in a day
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# Create a rectangle from x=17 to x=18 and y=-1 to y=1
rectangle = patches.Rectangle((17, -1), 1, 2, color="green", alpha=0.5)
ax.add_patch(rectangle)

plt.xlabel("Time (hours)")
plt.ylabel("Value")
plt.title("Rectangle Between 17:00 and 18:00")
plt.grid(True)
plt.show()
