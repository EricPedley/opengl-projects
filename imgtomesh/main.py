from depthmesh import GLFWDrawer,createVertexArray,createMeshIndices
from depth_prediction.predict import DepthMapGenerator
from PIL import Image
import cv2
gen = DepthMapGenerator()

camMode = "video"
if camMode=="file":
    img = Image.open("testimg.jpg")
elif camMode=="video":
    cap = cv2.VideoCapture(0)
    s,img = cap.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

pred = gen.getPrediction(img)
mode="colormap"

drawer = GLFWDrawer(mode="dynamic" if camMode=="video" else "static")

vertices = createVertexArray(depthmap=pred,mode=mode)
drawer.set_vertices(vertices)
indices=createMeshIndices(*pred.shape)
drawer.set_indices(indices)

drawer.set_img(Image.open("viridis colormap.png") if mode=="colormap" else img)

while not drawer.should_close():
    if camMode=="video":
        s,img = cap.read()
        cv2.imshow("Webcam feed",img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        pred = gen.getPrediction(img)
        vertices = createVertexArray(depthmap=pred,mode=mode)
        drawer.set_vertices(vertices)
    drawer.draw()

del gen
del drawer

