//
//  draw.cpp
//  artoolkitX Square Tracking Example
//
//  Copyright 2018 Realmax, Inc. All Rights Reserved.
//
//  Author(s): Philip Lamb
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//  this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//  notice, this list of conditions and the following disclaimer in the
//  documentation and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from this
//  software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//

#include "draw.h"
#include <ARX/ARController.h>
#if HAVE_GL
#  if ARX_TARGET_PLATFORM_MACOS
#    include <OpenGL/gl.h>
#  else
#    include <GL/gl.h>
#  endif
#endif // HAVE_GL
#if HAVE_GL3
#  include <ARX/ARG/mtx.h>
#  include <ARX/ARG/shader_gl.h>
#include <GL/glut.h>

#endif // HAVE_GL3

#define DRAW_MODELS_MAX 32

#if HAVE_GLES2 || HAVE_GL3
// Indices of GL program uniforms.
enum {
    UNIFORM_MODELVIEW_PROJECTION_MATRIX,
    UNIFORM_COUNT
};
enum {
    UNIFORM_MVP_MATRIX_GENERIC,
    UNIFORM_COUNT_GENERIC
};
enum {
    UNIFORM_MVP_MATRIX_TEX,
    UNIFORM_COUNT_TEX
};
// Indices of of GL program attributes.
enum {
    ATTRIBUTE_VERTEX,
    ATTRIBUTE_COLOUR,
    ATTRIBUTE_COUNT,

};
enum{
    ATTRIBUTE_VERTEX_GENERIC,
    ATTRIBUTE_COLOUR_GENERIC
};
enum{
    ATTRIBUTE_VERTEX_TEX,
    ATTRIBUTE_TEXTURE_TEX
};
static GLint uniforms[UNIFORM_COUNT] = {0};
static GLint uniformsGeneric[UNIFORM_COUNT_GENERIC] = {0};
static GLint uniformsTex[UNIFORM_COUNT_TEX] = {0};
static GLuint program = 0;
static GLuint programGeneric = 0;
static GLuint programTex = 0;

#if HAVE_GL3
static GLuint gCubeVAOs[2] = {0};
static GLuint gCubeV3BO = 0;
static GLuint gCubeC4BO = 0;
static GLuint gCubeCb4BO = 0;
static GLuint gCubeEABO = 0;
static GLuint axesVBO = 0;
#endif // HAVE_GL3

#endif // HAVE_GLES2 || HAVE_GL3

static ARG_API drawAPI = ARG_API_GL3;
static bool rotate90 = false;
static bool flipH = false;
static bool flipV = false;

static int32_t gViewport[4] = {0};
static float gProjection[16];
static float gProjectionGeneric[16];
static float gProjectionTex[16];
static float gView[16];
static float gViewGeneric[16];
static float gViewTex[16];
static bool gModelLoaded[DRAW_MODELS_MAX] = {false};
static float gModelPoses[DRAW_MODELS_MAX][16];
static float gCameraPoses[DRAW_MODELS_MAX][16];
static bool gModelVisbilities[DRAW_MODELS_MAX];

static void drawCube(float viewProjection[16], float pose[16]);
static void drawAxis(float viewProjection[16], float pose[16]);

void drawSetup(ARG_API drawAPI_in, bool rotate90_in, bool flipH_in, bool flipV_in)
{
    drawAPI = drawAPI_in;
    rotate90 = rotate90_in;
    flipH = flipH_in;
    flipV = flipV_in;
    //ARLOGi("Passou no drawSetup. \n");
    return;
}

void drawCleanup()
{
#if HAVE_GLES2 || HAVE_GL3
    if (drawAPI == ARG_API_GLES2 || drawAPI == ARG_API_GL3) {
        if (program) {
            glDeleteProgram(program);
            program = 0;
        }
#if HAVE_GL3
        if (drawAPI == ARG_API_GL3) {
            if (gCubeVAOs[0]) {
                glDeleteBuffers(1, &gCubeCb4BO);
                glDeleteBuffers(1, &gCubeEABO);
                glDeleteBuffers(1, &gCubeC4BO);
                glDeleteBuffers(1, &gCubeV3BO);
                glDeleteVertexArrays(2, gCubeVAOs);
                gCubeVAOs[0] = gCubeVAOs[1] = 0;
            }
        }
#endif // HAVE_GL3
    }
#endif // HAVE_GLES2 || HAVE_GL3
    for (int i = 0; i < DRAW_MODELS_MAX; i++) gModelLoaded[i] = false;

    return;
}

int drawLoadModel(const char *path)
{
    // Ignore path, we'll always draw a cube.
    for (int i = 0; i < DRAW_MODELS_MAX; i++) {
        if (!gModelLoaded[i]) {
            gModelLoaded[i] = true;
            //ARLOGi("Passou no drawLoadModel. \n");
            return i;
        }
    }
    return -1;
}

void drawSetViewport(int32_t viewport[4])
{
    gViewport[0] = viewport[0];
    gViewport[1] = viewport[1];
    gViewport[2] = viewport[2];
    gViewport[3] = viewport[3];
    //ARLOGi("Passou no drawSetViewport. \n");
}

void drawSetCamera(float projection[16], float view[16])
{
    if (projection) {
        if (flipH || flipV) {
            mtxLoadIdentityf(gProjection);
            mtxScalef(gProjection, flipV ? -1.0f : 1.0f,  flipH ? -1.0f : 1.0f, 1.0f);
            mtxMultMatrixf(gProjection, projection);
        } else {
            mtxLoadMatrixf(gProjection, projection);
        }
    } else {
        mtxLoadIdentityf(gProjection);
    }
    if (view) {
        mtxLoadMatrixf(gView, view);
        //mtxRotatef(gView, 180.0f, 0.0f, 0.0f, 1.0f);
        //mtxRotatef(gView, 90.0f, 1.0f, 0.0f, 0.0f);
    } else {
        mtxLoadIdentityf(gView);
        //mtxRotatef(gView, 180.0f, 0.0f, 0.0f, 1.0f);
        //mtxRotatef(gView, 90.0f, 1.0f, 0.0f, 0.0f);
    }
    //ARLOGi("Passou no drawSetCamera. \n");
}

void drawSetModel(int modelIndex, bool visible, float pose[16], float camPose[16])
{
    if (modelIndex < 0 || modelIndex >= DRAW_MODELS_MAX) return;
    if (!gModelLoaded[modelIndex]) return;
    
    gModelVisbilities[modelIndex] = visible;
    if (visible) {
        mtxLoadMatrixf(&(gModelPoses[modelIndex][0]), pose);
        mtxLoadMatrixf(&(gCameraPoses[modelIndex][0]), camPose);
    }
}

void draw()
{
    float viewProjection[16];

    glViewport(gViewport[0], gViewport[1], gViewport[2], gViewport[3]);

#if HAVE_GL
    if (drawAPI == ARG_API_GL) {
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(gProjection);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(gView);
    }
#endif

#if HAVE_GLES2 || HAVE_GL3
    if (drawAPI == ARG_API_GLES2 || drawAPI == ARG_API_GL3) {
        if (!program) {
            GLuint vertShader = 0, fragShader = 0;
            // A simple shader pair which accepts just a vertex position and colour, no lighting.
            const char vertShaderStringGLES2[] =
                "attribute vec4 position;\n"
                "attribute vec4 colour;\n"
                "uniform mat4 modelViewProjectionMatrix;\n"
                "varying vec4 colourVarying;\n"
                "void main()\n"
                "{\n"
                    "gl_Position = modelViewProjectionMatrix * position;\n"
                    "colourVarying = colour;\n"
                "}\n";
            const char fragShaderStringGLES2[] =
                "#ifdef GL_ES\n"
                "precision mediump float;\n"
                "#endif\n"
                "varying vec4 colourVarying;\n"
                "void main()\n"
                "{\n"
                    "gl_FragColor = colourVarying;\n"
                "}\n";
            const char vertShaderStringGL3[] =
                "#version 150\n"
                "in vec4 position;\n"
                "in vec4 colour;\n"
                "uniform mat4 modelViewProjectionMatrix;\n"
                "out vec4 colourVarying;\n"
                "void main()\n"
                "{\n"
                "gl_Position = modelViewProjectionMatrix * position;\n"
                "colourVarying = colour;\n"
                "}\n";
            const char fragShaderStringGL3[] =
                "#version 150\n"
                "in vec4 colourVarying;\n"
                "out vec4 FragColor;\n"
                "void main()\n"
                "{\n"
                "FragColor = colourVarying;\n"
                "}\n";

            if (program) arglGLDestroyShaders(0, 0, program);
            program = glCreateProgram();
            if (!program) {
                ARLOGe("draw: Error creating shader program.\n");
                return;
            }
            
            if (!arglGLCompileShaderFromString(&vertShader, GL_VERTEX_SHADER, drawAPI == ARG_API_GLES2 ? vertShaderStringGLES2 : vertShaderStringGL3)) {
                ARLOGe("draw: Error compiling vertex shader.\n");
                arglGLDestroyShaders(vertShader, fragShader, program);
                program = 0;
                return;
            }
            if (!arglGLCompileShaderFromString(&fragShader, GL_FRAGMENT_SHADER, drawAPI == ARG_API_GLES2 ? fragShaderStringGLES2 : fragShaderStringGL3)) {
                ARLOGe("draw: Error compiling fragment shader.\n");
                arglGLDestroyShaders(vertShader, fragShader, program);
                program = 0;
                return;
            }
            glAttachShader(program, vertShader);
            glAttachShader(program, fragShader);
            
            glBindAttribLocation(program, ATTRIBUTE_VERTEX, "position");
            glBindAttribLocation(program, ATTRIBUTE_COLOUR, "colour");
            if (!arglGLLinkProgram(program)) {
                ARLOGe("draw: Error linking shader program.\n");
                arglGLDestroyShaders(vertShader, fragShader, program);
                program = 0;
                return;
            }
            arglGLDestroyShaders(vertShader, fragShader, 0); // After linking, shader objects can be deleted.
            
            // Retrieve linked uniform locations.
            uniforms[UNIFORM_MODELVIEW_PROJECTION_MATRIX] = glGetUniformLocation(program, "modelViewProjectionMatrix");
        }
        glUseProgram(program);
        mtxLoadMatrixf(viewProjection, gProjection);
        mtxMultMatrixf(viewProjection, gView);
    }
#endif // HAVE_GLES2 || HAVE_GL3

    glEnable(GL_DEPTH_TEST);
    
    for (int i = 0; i < DRAW_MODELS_MAX; i++) {
        if (gModelLoaded[i] && gModelVisbilities[i]) {
            drawAxis(viewProjection, &(gModelPoses[i][0]));
            drawCube(viewProjection, &(gModelPoses[i][0]));
            //ARLOGd("Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", gCameraPoses[i][12], gCameraPoses[i][13], gCameraPoses[i][14], gCameraPoses[i][15]);
            //ARLOGi("Passou no drawCube. \n");
        }
    }
    //ARLOGi("Passou no draw. \n");
}

void drawAux()
{
    float viewProjection[16];

        if (!programGeneric) {
            GLuint vertShader = 0, fragShader = 0;
            // A simple shader pair which accepts just a vertex position and colour, no lighting.
            const char vertShaderStringGL3[] =
                    "#version 150\n"
                    "in vec4 position;\n"
                    "in vec4 colour;\n"
                    "uniform mat4 modelViewProjectionMatrix;\n"
                    "out vec4 colourVarying;\n"
                    "void main()\n"
                    "{\n"
                    "gl_Position = modelViewProjectionMatrix * position;\n"
                    "colourVarying = colour;\n"
                    "}\n";
            const char fragShaderStringGL3[] =
                    "#version 150\n"
                    "in vec4 colourVarying;\n"
                    "out vec4 FragColor;\n"
                    "void main()\n"
                    "{\n"
                    "FragColor = colourVarying;\n"
                    "}\n";

            if (programGeneric) arglGLDestroyShaders(0, 0, programGeneric);
            programGeneric = glCreateProgram();
            if (!programGeneric) {
                ARLOGe("draw: Error creating shader program.\n");
                return;
            }

            if (!arglGLCompileShaderFromString(&vertShader, GL_VERTEX_SHADER, vertShaderStringGL3)) {
                ARLOGe("draw: Error compiling vertex shader.\n");
                arglGLDestroyShaders(vertShader, fragShader, programGeneric);
                programGeneric = 0;
                return;
            }
            if (!arglGLCompileShaderFromString(&fragShader, GL_FRAGMENT_SHADER, fragShaderStringGL3)) {
                ARLOGe("draw: Error compiling fragment shader.\n");
                arglGLDestroyShaders(vertShader, fragShader, programGeneric);
                programGeneric = 0;
                return;
            }
            glAttachShader(programGeneric, vertShader);
            glAttachShader(programGeneric, fragShader);

            glBindAttribLocation(programGeneric, ATTRIBUTE_VERTEX_GENERIC, "position");
            glBindAttribLocation(programGeneric, ATTRIBUTE_COLOUR_GENERIC, "colour");
            if (!arglGLLinkProgram(programGeneric)) {
                ARLOGe("draw: Error linking shader program.\n");
                arglGLDestroyShaders(vertShader, fragShader, programGeneric);
                programGeneric = 0;
                return;
            }
            arglGLDestroyShaders(vertShader, fragShader, 0); // After linking, shader objects can be deleted.

            // Retrieve linked uniform locations.
            uniformsGeneric[UNIFORM_MVP_MATRIX_GENERIC] = glGetUniformLocation(programGeneric, "modelViewProjectionMatrix");
        }
        glUseProgram(programGeneric);

        mtxLoadIdentityf(gProjectionGeneric);
        mtxLoadIdentityf(gViewGeneric);

        mtxLoadMatrixf(viewProjection, gProjectionGeneric);
        mtxMultMatrixf(viewProjection, gViewGeneric);

        drawTriangle(viewProjection);
}

void drawTriangle(float viewProjection[16]) {

    float modelViewProjection[16];

    mtxLoadMatrixf(modelViewProjection, viewProjection);
    glUniformMatrix4fv(uniformsGeneric[UNIFORM_MVP_MATRIX_GENERIC], 1, GL_FALSE, modelViewProjection);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
            // positions         // colors
            0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f, 0.3, // bottom right
            -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f, 0.3,  // bottom left
            0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f, 0.3   // top

    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(ATTRIBUTE_VERTEX_GENERIC, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(ATTRIBUTE_VERTEX_GENERIC);
    // color attribute
    glVertexAttribPointer(ATTRIBUTE_COLOUR_GENERIC, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(ATTRIBUTE_COLOUR_GENERIC);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    // glBindVertexArray(0);

    // render the triangle
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Something to look at, draw a rotating colour cube.
static void drawCube(float viewProjection[16], float pose[16])
{
//    const GLfloat cube_vertices [8][3] = {
//            /* +z */ {5.5f, 5.5f, 5.5f}, {5.5f, -5.5f, 5.5f}, {-5.5f, -5.5f, 5.5f}, {-5.5f, 5.5f, 5.5f},
//            /* -z */ {5.5f, 5.5f, -5.5f}, {5.5f, -5.5f, -5.5f}, {-5.5f, -5.5f, -5.5f}, {-5.5f, 5.5f, -5.5f} };
    // Colour cube data.
    const GLfloat cube_vertices [8][3] = {
            /* +z */ {80.0f, 10.0f, 120.0f}, {80.0f, -10.0f, 120.0f}, {-80.0f, -10.0f, 120.0f}, {-80.0f, 10.0f, 120.0f},
            /* -z */ {80.0f, 10.0f, 0.0f}, {80.0f, -10.0f, 0.0f}, {-80.0f, -10.0f, 0.0f}, {-80.0f, 10.0f, 0.0f} };
//Transformadas para desenho no Optix
//    const GLfloat cube_vertices [8][3] = {
//            /* +z */ {-2.0f, 3.0f, 0.5f}, {-2.0f, 3.0f, -0.5f}, {2.0f, 3.0f, -0.5f}, {2.0f, 3.0f, 0.5f},
//            /* -z */ {-2.0f, 0.0f, 0.5f}, {-2.0f, 0.0f, -0.5f}, {2.0f, 0.0f, -0.5f}, {2.0f, 0.0f, 0.5f} };

//    const GLfloat cube_vertices [8][3] = {
//        /* +z */ {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
//        /* -z */ {0.5f, 0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f} };
    const GLubyte cube_vertex_colors [8][4] = {
        {255, 255, 255, 100}, {255, 255, 0, 100}, {0, 255, 0, 100}, {0, 255, 255, 100},
        {255, 0, 255, 100}, {255, 0, 0, 100}, {0, 0, 0, 100}, {0, 0, 255, 100} };
    const GLubyte cube_vertex_colors_black [8][4] = {
        {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
        {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };
    const GLushort cube_faces [6][4] = { /* ccw-winding */
        /* +z */ {3, 2, 1, 0}, /* -y */ {2, 3, 7, 6}, /* +y */ {0, 1, 5, 4},
        /* -x */ {3, 0, 4, 7}, /* +x */ {1, 2, 6, 5}, /* -z */ {4, 5, 6, 7} };
    int i;
    //ARLOGi("Cube colour data\n");
#if HAVE_GLES2 || HAVE_GL3
    float modelViewProjection[16];
#endif

#if HAVE_GL
    if (drawAPI == ARG_API_GL) {
        glPushMatrix(); // Save world coordinate system.
        glMultMatrixf(pose);
        //glScalef(40.0f, 40.0f, 40.0f);
        //glTranslatef(0.0f, 0.0f, 0.5f); // Place base of cube on marker surface.
        glDisable(GL_LIGHTING);
        //glDisable(GL_TEXTURE_2D);
        //glDisable(GL_BLEND);
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, cube_vertex_colors);
        glVertexPointer(3, GL_FLOAT, 0, cube_vertices);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        for (i = 0; i < 6; i++) {
            glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_SHORT, &(cube_faces[i][0]));
        }
        glDisableClientState(GL_COLOR_ARRAY);
        glColor4ub(0, 0, 0, 255);
        for (i = 0; i < 6; i++) {
            glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, &(cube_faces[i][0]));
        }
        glPopMatrix();    // Restore world coordinate system.
    }
#endif // HAVE_GL

#if HAVE_GLES2 || HAVE_GL3
    if (drawAPI == ARG_API_GLES2 || drawAPI == ARG_API_GL3) {
        //ARLOGi("Draw API 2 ou 3\n");
        mtxLoadMatrixf(modelViewProjection, viewProjection);
        mtxMultMatrixf(modelViewProjection, pose);
        //mtxScalef(modelViewProjection, 40.0f, 40.0f, 40.0f);
        //mtxTranslatef(modelViewProjection, 0.0f, 0.0f, 0.5f); // Place base of cube on marker surface.
        glUniformMatrix4fv(uniforms[UNIFORM_MODELVIEW_PROJECTION_MATRIX], 1, GL_FALSE, modelViewProjection);

        if (drawAPI == ARG_API_GL3) {
            //ARLOGi("Draw API 3\n");
            if (!gCubeVAOs[0]) {
                glGenVertexArrays(2, gCubeVAOs);
                glBindVertexArray(gCubeVAOs[0]);
                glGenBuffers(1, &gCubeV3BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeV3BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_VERTEX);
                glGenBuffers(1, &gCubeC4BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeC4BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_colors), cube_vertex_colors, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_COLOUR, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_COLOUR);
                glGenBuffers(1, &gCubeEABO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gCubeEABO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_faces), cube_faces, GL_STATIC_DRAW);
                glBindVertexArray(gCubeVAOs[1]);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeV3BO);
                glVertexAttribPointer(ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_VERTEX);
                glGenBuffers(1, &gCubeCb4BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeCb4BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_colors_black), cube_vertex_colors_black, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_COLOUR, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_COLOUR);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gCubeEABO);
            }
            
            glBindVertexArray(gCubeVAOs[0]);
    #ifdef DEBUG
            if (!arglGLValidateProgram(program)) {
                ARLOGe("drawCube() Error: shader program %d validation failed.\n", program);
                return;
            }
    #endif
            for (i = 0; i < 6; i++) {
                glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_SHORT, (void *)(i*4*sizeof(GLushort)));
                //ARLOGi("drawElements Faces\n");
            }
            glBindVertexArray(gCubeVAOs[1]);
            for (i = 0; i < 6; i++) {
                glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (void *)(i*4*sizeof(GLushort)));
                //ARLOGi("drawElements arestas\n");
            }
            glBindVertexArray(0);
        }
    }
#endif // HAVE_GLES2 || HAVE_GL3
}

static void drawCubeTranslucent(float viewProjection[16], float pose[16])
{
//    const GLfloat cube_vertices [8][3] = {
//            /* +z */ {5.5f, 5.5f, 5.5f}, {5.5f, -5.5f, 5.5f}, {-5.5f, -5.5f, 5.5f}, {-5.5f, 5.5f, 5.5f},
//            /* -z */ {5.5f, 5.5f, -5.5f}, {5.5f, -5.5f, -5.5f}, {-5.5f, -5.5f, -5.5f}, {-5.5f, 5.5f, -5.5f} };
    // Colour cube data.
    const GLfloat cube_vertices [8][3] = {
            /* +z */ {1.9f, 0.4f, 2.9f}, {1.9f, -0.4f, 2.9f}, {-1.9f, -0.4f, 2.9f}, {-1.9f, 0.4f, 2.9f},
            /* -z */ {1.9f, 0.4f, -0.1f}, {1.9f, -0.4f, -0.1f}, {-1.9f, -0.4f, -0.1f}, {-1.9f, 0.4f, -0.1f} };
//Transformadas para desenho no Optix
//    const GLfloat cube_vertices [8][3] = {
//            /* +z */ {-2.0f, 3.0f, 0.5f}, {-2.0f, 3.0f, -0.5f}, {2.0f, 3.0f, -0.5f}, {2.0f, 3.0f, 0.5f},
//            /* -z */ {-2.0f, 0.0f, 0.5f}, {-2.0f, 0.0f, -0.5f}, {2.0f, 0.0f, -0.5f}, {2.0f, 0.0f, 0.5f} };

//    const GLfloat cube_vertices [8][3] = {
//        /* +z */ {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
//        /* -z */ {0.5f, 0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f} };
    const GLubyte cube_vertex_colors [8][4] = {
            {255, 255, 255, 0}, {255, 255, 0, 0}, {0, 255, 0, 0}, {0, 255, 255, 0},
            {255, 0, 255, 0}, {255, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 255, 0} };
    const GLubyte cube_vertex_colors_black [8][4] = {
            {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255},
            {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255} };
    const GLushort cube_faces [6][4] = { /* ccw-winding */
            /* +z */ {3, 2, 1, 0}, /* -y */ {2, 3, 7, 6}, /* +y */ {0, 1, 5, 4},
            /* -x */ {3, 0, 4, 7}, /* +x */ {1, 2, 6, 5}, /* -z */ {4, 5, 6, 7} };
    int i;
    //ARLOGi("Cube colour data\n");
#if HAVE_GLES2 || HAVE_GL3
    float modelViewProjection[16];
#endif

#if HAVE_GLES2 || HAVE_GL3
    if (drawAPI == ARG_API_GLES2 || drawAPI == ARG_API_GL3) {
        //ARLOGi("Draw API 2 ou 3\n");
        mtxLoadMatrixf(modelViewProjection, viewProjection);
        mtxMultMatrixf(modelViewProjection, pose);
        mtxScalef(modelViewProjection, 40.0f, 40.0f, 40.0f);
        mtxTranslatef(modelViewProjection, 0.0f, 0.0f, 0.5f); // Place base of cube on marker surface.
        glUniformMatrix4fv(uniforms[UNIFORM_MODELVIEW_PROJECTION_MATRIX], 1, GL_FALSE, modelViewProjection);

        if (drawAPI == ARG_API_GL3) {
            //ARLOGi("Draw API 3\n");
            if (!gCubeVAOs[0]) {
                glGenVertexArrays(2, gCubeVAOs);
                glBindVertexArray(gCubeVAOs[0]);
                glGenBuffers(1, &gCubeV3BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeV3BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_VERTEX);
                glGenBuffers(1, &gCubeC4BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeC4BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_colors), cube_vertex_colors, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_COLOUR, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_COLOUR);
                glGenBuffers(1, &gCubeEABO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gCubeEABO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_faces), cube_faces, GL_STATIC_DRAW);
                glBindVertexArray(gCubeVAOs[1]);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeV3BO);
                glVertexAttribPointer(ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_VERTEX);
                glGenBuffers(1, &gCubeCb4BO);
                glBindBuffer(GL_ARRAY_BUFFER, gCubeCb4BO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_colors_black), cube_vertex_colors_black, GL_STATIC_DRAW);
                glVertexAttribPointer(ATTRIBUTE_COLOUR, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
                glEnableVertexAttribArray(ATTRIBUTE_COLOUR);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gCubeEABO);
            }

            glBindVertexArray(gCubeVAOs[0]);
#ifdef DEBUG
            if (!arglGLValidateProgram(program)) {
                ARLOGe("drawCube() Error: shader program %d validation failed.\n", program);
                return;
            }
#endif
            for (i = 0; i < 6; i++) {
                glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_SHORT, (void *)(i*4*sizeof(GLushort)));
                //ARLOGi("drawElements Faces\n");
            }
            glBindVertexArray(gCubeVAOs[1]);
            for (i = 0; i < 6; i++) {
                glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (void *)(i*4*sizeof(GLushort)));
                //ARLOGi("drawElements arestas\n");
            }
            glBindVertexArray(0);
        }
    }
#endif // HAVE_GLES2 || HAVE_GL3
}


static void drawAxis(float viewProjection[16], float pose[16])
{
#if HAVE_GLES2 || HAVE_GL3
    float modelViewProjection[16];
#endif

#if HAVE_GLES2 || HAVE_GL3
    if (drawAPI == ARG_API_GLES2 || drawAPI == ARG_API_GL3) {
        //ARLOGi("Draw API 2 ou 3\n");
        mtxLoadMatrixf(modelViewProjection, viewProjection);
        mtxMultMatrixf(modelViewProjection, pose);
        //mtxScalef(modelViewProjection, 15.0f, 15.0f, 15.0f);
        glUniformMatrix4fv(uniforms[UNIFORM_MODELVIEW_PROJECTION_MATRIX], 1, GL_FALSE, modelViewProjection);

        if (drawAPI == ARG_API_GL3) {
            //ARLOGi("Draw API 3\n");

            float vertices[] = { //6 vertices X, Y, Z, R, G, B
                    0.0f, 0.0f, 0.0f, /* */ 1.0f, 0.0f, 0.0f, /* */ 80.0f,  0.0f,  0.0f, /* */ 1.0f, 0.0f, 0.0f, /* */
                    0.0f, 0.0f, 0.0f, /* */ 0.0f, 1.0f, 0.0f, /* */ 0.0f,  80.0f,  0.0f, /* */ 0.0f, 1.0f, 0.0f, /* */
                    0.0f, 0.0f, 0.0f, /* */ 0.0f, 0.0f, 1.0f, /* */ 0.0f,  0.0f,  80.0f, /* */ /* */ 0.0f, 0.0f, 1.0f
            };

            glGenBuffers(1, &axesVBO); // Generate 1 buffer
            glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
            glVertexAttribPointer(ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), NULL);
            glEnableVertexAttribArray(ATTRIBUTE_VERTEX);
            glEnableVertexAttribArray(ATTRIBUTE_COLOUR);
            glVertexAttribPointer(ATTRIBUTE_COLOUR, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
            glLineWidth(4.0f);
            glDrawArrays(GL_LINES, 0, 6);

#ifdef DEBUG
            if (!arglGLValidateProgram(program)) {
                ARLOGe("drawCube() Error: shader program %d validation failed.\n", program);
                return;
            }
#endif
            glBindVertexArray(0);

        }
    }
#endif // HAVE_GLES2 || HAVE_GL3
}

void drawTexConfig(GLuint texture)
{
    float viewProjection[16];
    float left, right, bottom, top;

    glViewport(gViewport[0], gViewport[1], gViewport[2], gViewport[3]);

    if (!programTex) {
        GLuint vertShader = 0, fragShader = 0;
        // A simple shader pair which accepts just a vertex position and colour, no lighting.
        const char vertShaderStringGL3[] =
                "#version 330 core\n"
                "in vec3 vert;\n"
                "out vec2 fragTexCoord;\n"
                "void main()\n"
                "{\n"
                "gl_Position = vec4(vert,1);\n"
                "fragTexCoord = (vec2( vert.x, vert.y )+vec2(1,1))/2.0;\n"
                "}\n";
        const char fragShaderStringGL3[] =
                "#version 330 core\n"
                "uniform sampler2D tex;\n"
                "in vec2 fragTexCoord;\n"
                "out vec4 finalColor;\n"
                "void main()\n"
                "{\n"
                "finalColor = texture(tex, fragTexCoord);\n"
                "}\n";

        if (programTex) arglGLDestroyShaders(0, 0, programTex);
        programTex = glCreateProgram();
        if (!programTex) {
            ARLOGe("draw: Error creating shader program.\n");
            return;
        }

        if (!arglGLCompileShaderFromString(&vertShader, GL_VERTEX_SHADER, vertShaderStringGL3)) {
            ARLOGe("draw: Error compiling vertex shader.\n");
            arglGLDestroyShaders(vertShader, fragShader, programTex);
            programTex = 0;
            return;
        }
        if (!arglGLCompileShaderFromString(&fragShader, GL_FRAGMENT_SHADER, fragShaderStringGL3)) {
            ARLOGe("draw: Error compiling fragment shader.\n");
            arglGLDestroyShaders(vertShader, fragShader, programTex);
            programTex = 0;
            return;
        }
        glAttachShader(programTex, vertShader);
        glAttachShader(programTex, fragShader);

        glBindAttribLocation(programTex, ATTRIBUTE_VERTEX_TEX, "vert");
        glBindAttribLocation(programTex, ATTRIBUTE_TEXTURE_TEX, "vertTexCoord");
        if (!arglGLLinkProgram(programTex)) {
            ARLOGe("draw: Error linking shader program.\n");
            arglGLDestroyShaders(vertShader, fragShader, programTex);
            programTex = 0;
            return;
        }
        arglGLDestroyShaders(vertShader, fragShader, 0); // After linking, shader objects can be deleted.

        // Retrieve linked uniform locations.
        glUniform1i(glGetUniformLocation(programTex, "tex"), 0);

    }
    glUseProgram(programTex);

    mtxLoadIdentityf(gProjectionGeneric);
    mtxLoadIdentityf(gViewGeneric);

    mtxLoadMatrixf(viewProjection, gProjectionGeneric);
    mtxMultMatrixf(viewProjection, gViewGeneric);

    drawTex(viewProjection, texture, &(gModelPoses[0][0]));
}

void drawTex(float viewProjection[16], GLuint texture, float pose[16]) {

    float modelViewProjection[16];

    mtxLoadMatrixf(modelViewProjection, viewProjection);
    glUniformMatrix4fv(uniformsGeneric[UNIFORM_MVP_MATRIX_GENERIC], 1, GL_FALSE, modelViewProjection);

    float vertices[] = {
            // positions          // texture coords
            1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
            1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
            -1.1f, -1.1f, 0.0f,   0.0f, 0.0f,   // bottom left
            -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left
    };

//    float vertices[] = {
//            // positions          // texture coords
//            1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
//            1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
//            -1.0f,  1.0f, 0.0f,   0.0f, 1.0f,    // top left
//
//            1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
//            -1.1f, -1.1f, 0.0f,   0.0f, 0.0f,   // bottom left
//            -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left
//    };

    unsigned int indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    //glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    //glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(ATTRIBUTE_VERTEX_TEX, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(ATTRIBUTE_VERTEX_TEX);
    // color attribute
    glVertexAttribPointer(ATTRIBUTE_TEXTURE_TEX, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(ATTRIBUTE_TEXTURE_TEX);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    // glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, texture);
    //glBindVertexArray(VAO);

    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}