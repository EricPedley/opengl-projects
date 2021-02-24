import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
uniform mat4 rotation;
out vec2 v_texture;
void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_texture = a_texture;
    
    //v_texture = 1 - a_texture;                      // Flips the texture vertically and horizontally
    //v_texture = vec2(a_texture.s, 1 - a_texture.t); // Flips the texture vertically
}
"""

fragment_src = """
# version 330
in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture, v_texture); // * vec4(v_color, 1.0f);
}
"""

# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)

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

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

w = 960
h=1011

vertices = [-0.5, -0.5,  -0.5,  486/w,770/h,#bottom rectangle
             0.5, -0.5,  -0.5,  756/w,486/h,
            -0.5,  -0.5,  0.5,  471/w, 207/h,
             0.5, -0.5,   0.5,  195/w, 498/h,

            -0.5,  -0.5,  0.5,  458/w, 204/h,#front face
             0.5, -0.5,  0.5,  189/w, 489/h,
             0.0,  0.5, 0.0,  42/w, 74/h,

             0.5,  -0.5,  0.5,  480/w, 203/h,#left ear
             0.5, -0.5,  -0.5,  765/w, 477/h,
             0.0,  0.5, 0.0,  896/w, 54/h,

             -0.5, -0.5,  0.5,  190/w, 507/h,
             -0.5, -0.5,  -0.5,  473/w,779/h,
             0.0,  0.5, 0.0,  59/w, 928/h,

            -0.5,  -0.5,  -0.5,  493/w, 780/h,
             0.5, -0.5,  -0.5,  765/w,493/h,
             0.0,  0.5, 0.0,  912/w, 909/h]

indices = [0, 1, 2, 3,2,1,
           4,5,6,
           7,8,9,
           10,11,12,
           13,14,15]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))

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
image = Image.open("obamium net.jpg")
#image = image.transpose(Image.FLIP_TOP_BOTTOM)
img_data = image.convert("RGBA").tobytes()
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

rotation_loc = glGetUniformLocation(shader, "rotation")

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    rot_x = pyrr.Matrix44.from_x_rotation(0)#0.5 * glfw.get_time())
    rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

    glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, pyrr.matrix44.multiply(rot_x, rot_y))
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()