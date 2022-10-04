import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# creating a blank window
# for the animation
fig = plt.figure()
axis = plt.axes(xlim=(-50, 50),
                ylim=(-50, 50))

path, = axis.plot([], [], lw=2)
point, = axis.plot([], [], lw = 5)
label = axis.text(-45, 45, '', ha='left', va='top', fontsize=20, color="Red")


# what will our line dataset contain?
def init():
  path.set_data([], [])
  point.set_data([], [])
  return path, point


# initializing empty values for x and y co-ordinates
xdata, ydata = [], []


# animation function
def animate(i):
  # t is a parameter which varies with the frame number
  t = 0.1 * i

  # x, y values to be plotted
  x = t * np.sin(t)
  y = t * np.cos(t)

  # appending values to the previously empty x and y data holders
  xdata.append(x)
  ydata.append(y)
  path.set_data(xdata, ydata)

  point.set_data(xdata[-2:], ydata[-2:])

  label.set_text("%.1f" % t)

  return path, point, label


# calling the animation function
anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=100,
                               interval=10,
                               blit=True)
print('OK1')
plt.show(block=False)
print('OK2')
plt.pause(10)

# saves the animation in our desktop
# anim.save('growingCoil.mp4', writer='ffmpeg', fps=30)