from depthmesh import GLFWDrawer,createVertexArray,createMeshIndices
from depth_prediction.predict import DepthMapGenerator
from PIL import Image
gen = DepthMapGenerator()
drawer = GLFWDrawer()

img = Image.open("testimg.jpg")
pred = gen.getPrediction(img)
mode="texture"

vertices = createVertexArray(depthmap=pred,mode=mode)
indices=createMeshIndices(*pred.shape)

drawer.set_vertices(vertices)
drawer.set_indices(indices)
drawer.set_img(Image.open("viridis colormap.png") if mode=="colormap" else img)

while not drawer.should_close():
    drawer.draw()

del gen
del drawer

