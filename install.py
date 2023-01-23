import launch
import os

path = os.path.dirname(os.path.realpath(__file__))

launch.git_clone("https://github.com/WuZongWei6/Pixelization.git", os.path.join(path, "pixelization"), "pixelization", "b7142536da3a9348794bce260c10e465b8bebcb8")
