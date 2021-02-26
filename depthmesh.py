import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from math import sqrt,pi
from PIL import Image
vertex_src = """
# version 330
in vec3 a_position;
layout(location = 1) in vec2 a_texture;
out vec2 v_texture;
uniform mat4 rotation;

void main()
{
    gl_Position = rotation*vec4(a_position, 1.0);
    v_texture=a_texture;
}
"""
#TODO: add color map instead of using grayscale values
fragment_src = """
# version 330
in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture,v_texture);   
}
"""

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# make the context current
glfw.make_context_current(window)

with open("pred.npy","rb") as file:
    depthmap = np.load(file)[0,:,:,0]

# from matplotlib import pyplot as plt
# fig = plt.figure()
# ii = plt.imshow(depthmap, interpolation='nearest')
# fig.colorbar(ii)
# plt.show()

mode = "texture"
vertices = np.zeros(shape=(depthmap.size*5),dtype=np.float32)
counter=0
print(depthmap.shape)
height = depthmap.shape[0]
width = depthmap.shape[1]
max_depth = np.max(depthmap)
for i in range(0,height):#this doesn't account for perspective, it treats the depthmap as coming from an orthogonal perspective
    for j in range(0,width):
        #set x coord
        vertices[counter*5] = j/width-0.5
        #set y coord
        vertices[counter*5+1] = i/height-0.5
        #set z coord (depth)
        depthNormalized = depthmap[i][j]/max_depth-0.5#ranges from -0.5 to 0.5
        vertices[counter*5+2] = depthNormalized
        if mode=="texture":
            vertices[counter*5+3] = j/width
            vertices[counter*5+4] = i/height
        elif mode == "colormap":
            vertices[counter*5+3] = 0.5-depthNormalized#ranges from 0 to 1. Lighter colors on the colormap are toward 1, so this makes closer things lighter
            vertices[counter*5+4] = 0.5#depthNormalized
        counter+=1


indices = np.empty(shape=((width-1)*(height-1)*6),dtype=np.uint32)
counter=0
hugeCount=0
for row in range(0,height-1):
    for col in range(0,width-1):
        i=row*width+col#stupid bug that was tripping me up: I was using row*height instead of row*width, which connected points at opposite ends of the mesh. 🤡
        indices[counter] = i
        indices[counter+1] = i+1
        indices[counter+2] = i+width
        indices[counter+3] = i+1
        indices[counter+4] = i+width
        indices[counter+5] = i+width+1
        #for debugging we need to know which of these points are the ones where the x coordinates of the vertices
        #they're connecting are really large. So, we need all the edges of the 3 triangles, which is 6 lines to check x coordinates
        #each of those lines is all combos of the first 3 and the combos of the second 3
        # for index in range(counter,counter+6):
        #     for index2 in range(counter,counter+6):
        #         if(abs(vertices[indices[index]*5]-vertices[indices[index2]*5])>0.8):
        #             print(f"dist huuuge for i={i}, counter={counter}, index1={index}({indices[index]}), index2={index2}({indices[index2]})")
        #             hugeCount+=1
        counter+=6
print(hugeCount)
# w = 960
# h=1011
# vertices = [
#             -0.5,  -0.5,  0.5,  471/w, 207/h,#front face
#              0.5, -0.5,  -0.5,  195/w, 498/h,
#              0.0,  0.2, -0.0,  38/w, 71/h,]

# # vertices = [-0.5,-0.5,0,0,1,
# #             0.5,-0.5,0,0,-1,
# #             0.0,0.2,-0.0,0.5,0]

# indices = [0,1,2]
# vertices = np.array(vertices, dtype=np.float32)
# indices = np.array(indices, dtype=np.uint32)
# tetrahedron for testing with simple shape
# vertices = [sqrt(8/9), -1/3, 0.0, 
#              -sqrt(2/9), -1/3, -sqrt(2/3), 
#              -sqrt(2/9),  -1/3, sqrt(2/3),
#              0.0, 1.0, 0.0]

# vertices = np.array(vertices, dtype=np.float32)

# indices = [0,1,2,
#         3,0,2,
#         0,3,1,
#         1,2,3]
# indices = np.array(indices,dtype=np.uint32)


# vertexShader = glCreateShader(GL_VERTEX_SHADER)
# vertexShader = glShaderSource(vertexShader,1,vertex_src)
# compileShader(vertex_src)
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

#don't need this in python for some reason?
# VAO = glGenVertexArrays(1)
# glBindVertexArray(VAO)

VBO = glGenBuffers(1)#VBO stands for "vertex buffer object", and this is just the integer ID of that object
glBindBuffer(GL_ARRAY_BUFFER, VBO)#binds the GL_ARRAY_BUFFER to the buffer we just made, so now referencing GL_ARRAY_BUFFER means referencing the vbo object we made
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)#puts the vertex data into the buffer we created

EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)

position = glGetAttribLocation(shader, "a_position")#gets the integer position of the a_position variable from the shader, which we could've set with "layout (location=0) in vec3 a_position"
glEnableVertexAttribArray(position)
glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, vertices.itemsize*5, ctypes.c_void_p(0))#starts at 0 and stride is 24 bytes, so it skips over the color info that follows each vertex

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
if mode=="colormap":
    image = Image.open("viridis colormap.png")#503x19
elif mode=="texture":
    image = Image.open("testimg.jpg")
    image.resize((width,height))
#image = image.transpose(Image.FLIP_TOP_BOTTOM)
img_data = image.convert("RGBA").tobytes()
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

# color = glGetAttribLocation(shader, "a_color")#integer location of a_color variable in the shader.
# glEnableVertexAttribArray(color)
# glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, vertices.itemsize*6, ctypes.c_void_p(12))#starts at 12 abd strude us 24, so it starts at the color and skips the vertex info

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)#sets clear color
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
def framebuffer_size_callback(window,width,height):
    glViewport(0,0,width,height)

def process_input(window):
    global rot_y_scalar, rot_x_scalar
    if glfw.get_key(window,glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window,True)
    if glfw.get_key(window,glfw.KEY_D) == glfw.PRESS:
        rot_y_scalar-=pi/24
    if glfw.get_key(window,glfw.KEY_A) == glfw.PRESS:
        rot_y_scalar+=pi/24
    if glfw.get_key(window,glfw.KEY_W) == glfw.PRESS:
        rot_x_scalar-=pi/24
    if glfw.get_key(window,glfw.KEY_S) == glfw.PRESS:
        rot_x_scalar+=pi/24


glfw.set_framebuffer_size_callback(window,framebuffer_size_callback)
rot_location = glGetUniformLocation(shader,"rotation")
# the main application loop
rot_y_scalar = pi
rot_x_scalar = pi
while not glfw.window_should_close(window):
    process_input(window)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    rot_y = pyrr.Matrix44.from_y_rotation(rot_y_scalar)#np.array([0,1,0,0.5*glfw.get_time()],dtype=np.float32)
    rot_x = pyrr.Matrix44.from_x_rotation(rot_x_scalar)

    #print(rot_matrix)
    glUniformMatrix4fv(rot_location,1,GL_FALSE,pyrr.matrix44.multiply(rot_y,rot_x))
    #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    #glDrawArrays(GL_TRIANGLES, 0,6)
    glDrawElements(GL_TRIANGLES,len(indices),GL_UNSIGNED_INT,ctypes.c_void_p(0))#this was the big error causing white screen of death in the program. The last argument should be None, not 0. The problem was that I just translated the parameters straight from the book, where the code is in C++

    glfw.poll_events()
    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()