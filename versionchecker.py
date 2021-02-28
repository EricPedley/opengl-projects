from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# make the context current
glfw.make_context_current(window)
for i in range(10,50):
    unsupported=False
    vertex_src = f"""
    # version {i}0
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec2 a_texture;
    uniform mat4 rotation;
    out vec2 v_texture;
    void main()
    {{
        gl_Position = rotation * vec4(a_position, 1.0);
        v_texture = a_texture;
    }}
    """
    try:
        compileShader(vertex_src, GL_VERTEX_SHADER)
    except Exception as e:
        if str(e).find("not supported")!=-1:
            unsupported=True
    finally:
        if not unsupported:
            print(f"{i}0 supported")