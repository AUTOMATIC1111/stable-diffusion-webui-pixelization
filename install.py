import launch
import os

path = os.path.dirname(os.path.realpath(__file__))

launch.git_clone("https://github.com/WuZongWei6/Pixelization.git", os.path.join(path, "pixelization"), "pixelization", "b7142536da3a9348794bce260c10e465b8bebcb8")

# we remove __init__ because it breaks BLIP - takes over the directory named models which BLIP also uses.
try:
    os.remove(os.path.join(path, "pixelization", "models", "__init__.py"))
except OSError as e:
    pass
