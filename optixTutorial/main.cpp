//
//  main.cpp
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef DEBUG
#    include <unistd.h>
#    include <sys/param.h>
#endif
#include <string>

#include <ARX/ARController.h>
#include <ARX/ARUtil/time.h>

#include <GL/glut.h>
#include <GL/gl.h>

#include "draw.h"

const char *vconf = NULL;
const char *cpara = NULL;

static int contextWidth = 0;
static int contextHeight = 0;
static bool contextWasUpdated = true;
static int32_t viewport[4];
static float projection[16];

static ARController* arController = NULL;
static ARG_API drawAPI = ARG_API_GL3;

static long gFrameNo = 0;

struct marker {
    const char *name;
    float height;
};
static const struct marker markers[] = {
        {"hiro.patt", 57.0},
        {"kanji.patt", 57.0}
};
static const int markerCount = (sizeof(markers)/sizeof(markers[0]));

// Add trackables.
int markerIDs[markerCount];
int markerModelIDs[markerCount];

int mainWindow, subWindow1,subWindow2,subWindow3;
char str[256];
float invOut[16];

static void usage(char *com);
static void quit(int rc);
static void reshape(int w, int h);
static void display(void);
static void displayMain(void);
static void displayWorld(void);
static void keyboard(unsigned char key, int x, int y);
static void init();
void showString(std::string str);
bool gluInvertMatrix(float m[16]);

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize (800, 600);
    mainWindow = glutCreateWindow ("artoolkitX axis view");
    glutDisplayFunc(displayMain);
    glutKeyboardFunc(keyboard);
    init ();

    subWindow1 = glutCreateSubWindow(mainWindow, 0,0,800, 600);
    glutDisplayFunc(display);

//    subWindow2 = glutCreateSubWindow(mainWindow, 0,480,640, 480);
//    glutDisplayFunc(displayWorld);


    glutMainLoop();
    return 0;
}

static void init(){
    int w = 800, h = 600;
    reshape(w, h);

    // Initialise the ARController.
    arController = new ARController();
    if (!arController->initialiseBase()) {
        ARLOGe("Error initialising ARController.\n");
        quit(-1);
    }

#ifdef DEBUG
    arLogLevel = AR_LOG_LEVEL_DEBUG;
#endif

#ifdef DEBUG
    char buf[MAXPATHLEN];
    ARLOGd("CWD is '%s'.\n", getcwd(buf, sizeof(buf)));
#endif
    char *resourcesDir = arUtilGetResourcesDirectoryPath(AR_UTIL_RESOURCES_DIRECTORY_BEHAVIOR_BEST);
    for (int i = 0; i < markerCount; i++) {
        std::string markerConfig = "single;" + std::string(resourcesDir) + '/' + markers[i].name + ';' + std::to_string(markers[i].height);
        markerIDs[i] = arController->addTrackable(markerConfig);
        if (markerIDs[i] == -1) {
            ARLOGe("Error adding marker.\n");
            quit(-1);
        }
    }
    arController->getSquareTracker()->setPatternDetectionMode(AR_TEMPLATE_MATCHING_MONO);
    arController->getSquareTracker()->setThresholdMode(AR_LABELING_THRESH_MODE_AUTO_BRACKETING);

#ifdef DEBUG
    ARLOGd("vconf is '%s'.\n", vconf);
#endif
}

static void display(void)
{
    glutSetWindow(subWindow1);

    // Start tracking.
    arController->startRunning(vconf, cpara, NULL, 0);
    // Main loop.
    bool done = false;
    while (!done) {
        bool gotFrame = arController->capture();
        if (!gotFrame) {
            arUtilSleep(1);
        } else {
            //ARLOGi("Got frame %ld.\n", gFrameNo);
            gFrameNo++;

            if (!arController->update()) {
                ARLOGe("Error in ARController::update().\n");
                quit(-1);
            }

            if (contextWasUpdated) {
                if (!arController->drawVideoInit(0)) {
                    ARLOGe("Error in ARController::drawVideoInit().\n");
                    quit(-1);
                }
                if (!arController->drawVideoSettings(0, contextWidth, contextHeight, false, false, false,
                                                     ARVideoView::HorizontalAlignment::H_ALIGN_CENTRE,
                                                     ARVideoView::VerticalAlignment::V_ALIGN_CENTRE,
                                                     ARVideoView::ScalingMode::SCALE_MODE_FIT, viewport)) {
                    ARLOGe("Error in ARController::drawVideoSettings().\n");
                    quit(-1);
                }
                drawSetup(drawAPI, false, false, false);
                //ARLOGd("Viewport: %d %d %d %d", viewport[0], viewport[1], viewport[2], viewport[3]);
                drawSetViewport(viewport);
                ARdouble projectionARD[16];
                arController->projectionMatrix(0, 0.1f, 10000.0f, projectionARD);
                for (int i = 0; i < 16; i++) projection[i] = (float) projectionARD[i];
                drawSetCamera(projection, NULL);

                for (int i = 0; i < markerCount; i++) {
                    markerModelIDs[i] = drawLoadModel(NULL);
                }
                contextWasUpdated = false;
            }

            // Clear the context.
            glClearColor(0.0, 0.0, 0.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Display the current video frame to the current OpenGL context.
            arController->drawVideo(0);
            //ARLOGi("Passou no drawVideo. \n");

            // Look for trackables, and draw on each found one.
            for (int i = 0; i < markerCount; i++) {

                // Find the trackable for the given trackable ID.
                ARTrackable *marker = arController->findTrackable(markerIDs[i]);
                float view[16];
                if (marker->visible) {
                    //arUtilPrintMtx16(marker->transformationMatrix);
                    //ARLOGi("\n \n");
                    for (int i = 0; i < 16; i++){
                        view[i] = (float) marker->transformationMatrix[i];
                        //ARLOGi("View %d: %0.3f  \n", i, view[i]);
                    }
                }
                //Linearização por coluna
                //sprintf(str, "Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", view[12], view[13], view[14], view[15]);
                //ARLOGd("Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", view[12], view[13], view[14], view[15]);
                if(gluInvertMatrix(view)){
                    //for (int i = 0; i < 16; i++){
                        //ARLOGi("Inv %d: %.3f  \n", i, invOut[i]);
                    //}
                    sprintf(str, "Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", invOut[12], invOut[13], invOut[14], invOut[15]);
                }
                //ARLOGi("%s", str);
                drawSetModel(markerModelIDs[i], marker->visible, view, invOut);
                showString( str );
            }

            draw();
            glutSwapBuffers();
        }
    }
}

void showString(std::string str){
    int   i;

    for (i = 0; i < (int)str.length(); i++)
    {
        if(str[i] != '\n' )
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, str[i]);
        else
        {
            glRasterPos2i(0.0, 2.5);
        }
    }
}

bool gluInvertMatrix(float m[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
             m[4]  * m[11] * m[14] +
             m[8]  * m[6]  * m[15] -
             m[8]  * m[7]  * m[14] -
             m[12] * m[6]  * m[11] +
             m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
              m[4]  * m[10] * m[13] +
              m[8]  * m[5] * m[14] -
              m[8]  * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
             m[1]  * m[11] * m[14] +
             m[9]  * m[2] * m[15] -
             m[9]  * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
             m[0]  * m[11] * m[13] +
             m[8]  * m[1] * m[15] -
             m[8]  * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
             m[0]  * m[7] * m[14] +
             m[4]  * m[2] * m[15] -
             m[4]  * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
              m[0]  * m[6] * m[13] +
              m[4]  * m[1] * m[14] -
              m[4]  * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

static void displayMain(void){
    glutSetWindow(mainWindow);
    glClearColor(0.0, 1.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutSwapBuffers();
}

static void displayWorld(void){
    glutSetWindow(subWindow2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //showString( str );
    glutSwapBuffers();
}

static void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            quit(0);
            exit(0);
            break;
    }
}


static void usage(char *com)
{
    ARPRINT("Usage: %s [options]\n", com);
    ARPRINT("Options:\n");
    ARPRINT("  --vconf <video parameter for the camera>\n");
    ARPRINT("  --cpara <camera parameter file for the camera>\n");
    ARPRINT("  --version: Print artoolkitX version and exit.\n");
    ARPRINT("  -loglevel=l: Set the log level to l, where l is one of DEBUG INFO WARN ERROR.\n");
    ARPRINT("  -h -help --help: show this message\n");
    exit(0);
}

static void quit(int rc)
{
    drawCleanup();
    if (arController) {
        arController->drawVideoFinal(0);
        arController->shutdown();
        delete arController;
    }
    exit(rc);
}

static void reshape(int w, int h)
{
    contextWidth = w;
    contextHeight = h;
    ARLOGd("Resized to %dx%d.\n", w, h);
    contextWasUpdated = true;
}