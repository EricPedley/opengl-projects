from depthmesh import GLFWDrawer,createVertexArray,createMeshIndices
from depth_prediction.predict import DepthMapGenerator
from PIL import Image
import matplotlib.pyplot as plt
gen = DepthMapGenerator()

img = Image.open("testimg.jpg")

pred = gen.getPrediction(img)
fig = plt.figure()
ii = plt.imshow(pred, interpolation='nearest')
fig.colorbar(ii)
plt.show()

del gen


