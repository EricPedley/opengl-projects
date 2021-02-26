import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from math import sqrt,pi
from PIL import Image

class glfwDrawer:
    def __init__(self,x=400,y=200,width=1280,height=720):
        # initializing glfw library
        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        # creating the window
        window = glfw.create_window(width,height, "My OpenGL window", None, None)

        # check if window was created
        if not window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")
        self.window=window
        # set window's position
        glfw.set_window_pos(window, x, y)

        # make the context current
        glfw.make_context_current(window)
        vertex_src = """
        # version 330
        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        out vec2 v_texture;
        uniform mat4 rotation;

        void main()
        {
            gl_Position = rotation*vec4(a_position, 1.0);
            v_texture=a_texture;
        }
        """
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
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
                
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)

        VBO = glGenBuffers(1)#VBO stands for "vertex buffer object", and this is just the integer ID of that object
        glBindBuffer(GL_ARRAY_BUFFER, VBO)#binds the GL_ARRAY_BUFFER to the buffer we just made, so now referencing GL_ARRAY_BUFFER means referencing the vbo object we made
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # Set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        def framebuffer_size_callback(window,width,height):
            glViewport(0,0,width,height)
        self.framebuffer_size_callback=framebuffer_size_callback
        glUseProgram(shader)
        glClearColor(0, 0.1, 0.1, 1)#sets clear color
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.rot_location = glGetUniformLocation(shader,"rotation")
        # the main application loop
        self.rot_y_scalar = pi
        self.rot_x_scalar = pi


    #options for writing to buffers
    # GL_STREAM_DRAW: the data is set only once and used by the GPU at most a few times.
    # GL_STATIC_DRAW: the data is set only once and used many times.
    # GL_DYNAMIC_DRAW: the data is changed a lot and used many times. 

    #vertices array expects each vertex to be 5 values: x,y,z,texture x, and texture y, and the array should be 1-dimensional
    def set_vertices(self,vertices):
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)#puts the vertex data into the buffer we created
        glEnableVertexAttribArray(0)#the position is zero because I did layout(location = 0) in the shader source code
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize*5, ctypes.c_void_p(0))#starts at 0 and stride is 24 bytes, so it skips over the color info that follows each vertex
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))
    
    
    def set_indices(self,indices):
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)
        self.indices=indices

    def set_img(self,image):
        #image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.convert("RGBA").tobytes()
        # img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    

    
    def should_close(self):
        return glfw.window_should_close(self.window)

    def draw(self):
        def process_input(window):
            if glfw.get_key(window,glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window,True)
            if glfw.get_key(window,glfw.KEY_D) == glfw.PRESS:
                self.rot_y_scalar-=pi/24
            if glfw.get_key(window,glfw.KEY_A) == glfw.PRESS:
                self.rot_y_scalar+=pi/24
            if glfw.get_key(window,glfw.KEY_W) == glfw.PRESS:
                self.rot_x_scalar-=pi/24
            if glfw.get_key(window,glfw.KEY_S) == glfw.PRESS:
                self.rot_x_scalar+=pi/24
            glfw.set_framebuffer_size_callback(window,self.framebuffer_size_callback)
        process_input(self.window)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        rot_y = pyrr.Matrix44.from_y_rotation(self.rot_y_scalar)#np.array([0,1,0,0.5*glfw.get_time()],dtype=np.float32)
        rot_x = pyrr.Matrix44.from_x_rotation(self.rot_x_scalar)

        #print(rot_matrix)
        glUniformMatrix4fv(self.rot_location,1,GL_FALSE,pyrr.matrix44.multiply(rot_y,rot_x))
        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
        #glDrawArrays(GL_TRIANGLES, 0,6)
        glDrawElements(GL_TRIANGLES,len(self.indices),GL_UNSIGNED_INT,ctypes.c_void_p(0))#this was the big error causing white screen of death in the program. The last argument should be None, not 0. The problem was that I just translated the parameters straight from the book, where the code is in C++

        glfw.poll_events()
        glfw.swap_buffers(self.window)


with open("pred.npy","rb") as file:
    #the file still has the extra dimensions from the prediction model
    depthmap = np.load(file)[0,:,:,0]

mode = "colormap"
height = depthmap.shape[0]
width = depthmap.shape[1]

#creates array of vertices where each vertex's x and y coordinates are its x and y coordinates on the image and its z coordinate is its depth
def createVertexArray(depthmap,mode):
    vertices = np.zeros(shape=(depthmap.size*5),dtype=np.float32)
    counter=0
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
    return vertices

vertices = createVertexArray(depthmap,mode)

#generates numpy indices array that connects each vertex to all its surrounding vertices to create a mesh
def createMeshIndices(width,height):
    indices = np.empty(shape=((width-1)*(height-1)*6),dtype=np.uint32)
    counter=0
    for row in range(0,height-1):
        for col in range(0,width-1):
            i=row*width+col#stupid bug that was tripping me up: I was using row*height instead of row*width, which connected points at opposite ends of the mesh. 🤡
            indices[counter] = i
            indices[counter+1] = i+1
            indices[counter+2] = i+width
            indices[counter+3] = i+1
            indices[counter+4] = i+width
            indices[counter+5] = i+width+1
            counter+=6
    return indices

indices = createMeshIndices(width,height)





# load image
if mode=="colormap":
    image = Image.open("viridis colormap.png")#503x19
elif mode=="texture":
    image = Image.open("testimg.jpg")
    image.resize((width,height))
glObj = glfwDrawer()
glObj.set_vertices(vertices)
glObj.set_indices(indices)
glObj.set_img(image)

while not glObj.should_close():
    glObj.draw()
# terminate glfw, free up allocated resources
glfw.terminate()