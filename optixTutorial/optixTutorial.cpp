#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-narrowing-conversions"
/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
//  tutorial
//
//-----------------------------------------------------------------------------

// 0 - normal shader
// 1 - lambertian
// 2 - specular
// 3 - shadows
// 4 - reflections
// 5 - miss
// 6 - schlick
// 7 - procedural texture on floor
// 8 - LGRustyMetal
// 9 - intersection
// 10 - anyhit
// 11 - camera



#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include "random.h"
#include <Arcball.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef DEBUG
#  ifdef _WIN32
#    define MAXPATHLEN MAX_PATH
#    include <direct.h>               // _getcwd()
#    define getcwd _getcwd
#  else
#    include <unistd.h>
#    include <sys/param.h>
#  endif
#    include <unistd.h>
#    include <sys/param.h>
#endif
#include <ARX/ARController.h>
#include <ARX/ARUtil/time.h>
#include <GL/gl.h>
#include "draw.h"
#include <ARX/ARG/mtx.h>
#include <optixu/optixu_quaternion.h>
#include <OptiXMesh.h>

#if ARX_TARGET_PLATFORM_WINDOWS
const char *vconf = "-module=WinMF -format=BGRA";
#else
const char *vconf = "-width=1280 -height=720";
#endif
const char *cpara = NULL;

#define ar 1

using namespace optix;

const char* const SAMPLE_NAME = "optixTutorial";

static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 1280u;
uint32_t     height = 720u;

std::string  texture_path;
const char*  tutorial_ptx;
int          tutorial_number = 4;

bool   m_interop = false;
GLuint m_pbo;
GLuint m_tex;
bool  use_tri_api = true;
bool  ignore_mats = false;

Buffer m_buffer;

// Viewport size
int m_width = 1280;
int m_height = 720;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
float3       camera_eyeOld;
Matrix4x4    camera_rotate;
sutil::Arcball arcball;
bool  camera_dirty = false;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

//--AR
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

char str[512];
float invOut[16];

Group m_top_object;
std::string mesh_teapotBody = std::string(sutil::samplesDir()) + "/data/teapot_body.ply";
std::string mesh_teapotLid = std::string(sutil::samplesDir()) + "/data/teapot_lid.ply";
std::string mesh_cubeMagenta = std::string(sutil::samplesDir()) + "/data/cubemagenta.ply";
std::string mesh_cubeYellow = std::string(sutil::samplesDir()) + "/data/cubeyellow.obj";
std::string mesh_happyBuddah = std::string(sutil::samplesDir()) + "/data/buddah.ply";
std::string mesh_lucy = std::string(sutil::samplesDir()) + "/data/Alucy.obj";
std::string mesh_dragon = std::string(sutil::samplesDir()) + "/data/dragon.ply";
std::string mesh_bunny = std::string(sutil::samplesDir()) + "/data/bunny.obj";
std::string mesh_cbRG = std::string(sutil::samplesDir()) + "/data/CornellBox-Empty-RG.obj";
std::string mesh_cbWhite = std::string(sutil::samplesDir()) + "/data/CornellBox-Empty-White.obj";
optix::Aabb  aabb;

int scene = 4;

Program pgram_intersection = 0;
Program pgram_bounding_box = 0;
Program diffuse_ch = 0;
Program diffuse_ah = 0;

Program phong_ch = 0;
Program phong_ah = 0;

Matrix4x4 teapotPose;
Matrix4x4 cubeMagentaPose;
Matrix4x4 cubeYellowPose;
Matrix4x4 happyBuddahPose;
Matrix4x4 lucyPose;
Matrix4x4 dragonPose;
Matrix4x4 bunnyPose;
Matrix4x4 cbRGPose;
Matrix4x4 cbWhitePose;

Transform teapotT;
Transform cubeMagentaT;
Transform cubeYellowT;
Transform happyBuddahT;
Transform lucyT;
Transform dragonT;
Transform cbRGT;
Transform cbWhiteT;
Transform bunnyT;
Transform cornellPose;
Transform spherePose;

float transformMat[16];
float scaleMat[16];

std::ofstream outFile;
std::ifstream inFile;

float deltaX = 25.0f;
float deltaY = -60.0f;
//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void updateCamera();


void glutInitialize( int* argc, char** argv );
void glutRun();
void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );

//--AR
static void quit(int rc);
static void reshape(int w, int h);
static void init();
void showString(std::string str);
bool gluInvertMatrix(float m[16]);
static void displayOnce(void);
bool distanceBigger(float3 P, float3 Q);

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

bool distanceBigger(float3 P, float3 Q){
    float threshold = 1.0f;

    float distance = sqrt(pow(P.x - Q.x, 2.0)+ pow(P.y - Q.y, 2.0) + pow(P.z - Q.z, 2.0));
    //printf("d = %f\n", distance);
    return distance > threshold;
}


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}

static std::string ptxPath( const std::string& cuda_file )
{
    return
            std::string(sutil::samplesPTXDir()) +
            "/" + std::string(SAMPLE_NAME) + "_generated_" +
            cuda_file +
            ".ptx";
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 4640 );
    if( tutorial_number < 8 )
        context->setMaxTraceDepth( 15 );
    else
        context->setMaxTraceDepth( 15 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 100 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["importance_cutoff"]->setFloat( 0.01f );
    context["ambient_light_color"]->setFloat( 0.8f, 0.8f, 0.8f );

    // OptiX buffer initialization:
    m_buffer = (m_interop) ? context->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_pbo)
                           : context->createBuffer(RT_BUFFER_OUTPUT);
    m_buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4); // BGRA8
    m_buffer->setSize(width, height);
    context["output_buffer"]->set(m_buffer);

    // Ray generation program
    const std::string camera_name = tutorial_number >= 11 ? "env_camera" : "pinhole_camera";
    Program ray_gen_program = context->createProgramFromPTXString( tutorial_ptx, camera_name );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( tutorial_ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    const std::string miss_name = tutorial_number >= 5 ? "envmap_miss" : "miss";
    context->setMissProgram( 0, context->createProgramFromPTXString( tutorial_ptx, miss_name ) );
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath = texture_path + "/" + std::string( "environment.hdr" );
    context["envmap"]->setTextureSampler( sutil::loadTexture( context, texpath, default_color) );
    context["bg_color"]->setFloat( make_float3( 1.0f, 0.0f, 0.0f ) );

    // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].

    const int tex_width  = 64;
    const int tex_height = 64;
    const int tex_depth  = 64;
    Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
    float *tex_data = (float *) noiseBuffer->map();

    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height * tex_depth;  i > 0; i--) {
        // One channel 3D noise in [0.0, 1.0] range.
        *tex_data++ = rand_range(0.0f, 1.0f);
    }
    noiseBuffer->unmap();


    // Noise texture sampler
    TextureSampler noiseSampler = context->createTextureSampler();

    noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
    noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
    noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    noiseSampler->setMaxAnisotropy(1.0f);
    noiseSampler->setBuffer(noiseBuffer);

    context["noise_texture"]->setTextureSampler(noiseSampler);
}


void setMaterial(GeometryInstance& gi, Material material)
{
    gi->addMaterial(material);
}


Material createMaterial(const float3& color){

    Material diffuse = context->createMaterial();
    diffuse->setClosestHitProgram( 0, phong_ch );
    diffuse->setAnyHitProgram( 1, phong_ah );
    diffuse["Kd"]->setFloat( color);
    diffuse["Ka"]->setFloat( color);
    diffuse["Ks"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["Kd2"]->setFloat( color);
    diffuse["Ka2"]->setFloat( 0.5f, 0.5f, 0.5f);
    diffuse["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
    diffuse["phong_exp"]->setFloat( 64.0f );
    diffuse["phong_exp2"]->setFloat( 64.0f );
    diffuse["Kr"]->setFloat( 0.0f, 0.0f, 0.0f);
    diffuse["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f);

    return diffuse;
}


GeometryInstance createParallelogram(float x0, float x1, float y0, float y1, optix::Material material,
                                     float scale )
{
    // Set up parallelogram programs
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    pgram_bounding_box = context->createProgramFromPTXString( ptx, "bounds" );
    pgram_intersection = context->createProgramFromPTXString( ptx, "intersect" );

    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( pgram_intersection );
    parallelogram->setBoundingBoxProgram( pgram_bounding_box );

    const float extent = scale*fmaxf( abs(x1-x0), abs(y1-y0) );
    const float3 anchor = make_float3( (x0+x1)/2 - 0.5f*extent, std::min(y0, y1) - 0.001f*abs(y1-y0), (y0+y1)/2 - 0.5f*extent );
    float3 v1 = make_float3( 0.0f, 0.0f, extent );
    float3 v2 = make_float3( extent, 0.0f, 0.0f );
    const float3 normal = normalize( cross( v1, v2 ) );
    float d = dot( normal, anchor );
    v1 *= 1.0f / dot( v1, v1 );
    v2 *= 1.0f / dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    optix::GeometryInstance instance = context->createGeometryInstance( parallelogram, &material, &material + 1 );
    return instance;
}

GeometryInstance loadMesh(const std::string& filename, Material mat, bool ignore)
{
    //const char *ptx = sutil::getPtxString( SAMPLE_NAME, "glass.cu" );

    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = use_tri_api;
    mesh.ignore_mats = false;
    if(ignore) {

        mesh.material = mat;
        //    mesh.closest_hit = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
        //    mesh.any_hit = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
        //
        const char *ptx = sutil::getPtxString(SAMPLE_NAME, "triangle_mesh.cu");
        mesh.intersection = context->createProgramFromPTXString(ptx, "mesh_intersect_refine");
        mesh.bounds = context->createProgramFromPTXString(ptx, "mesh_bounds");

    }
    loadMesh(filename, mesh);
    aabb.set(mesh.bbox_min, mesh.bbox_max);

    return mesh.geom_instance;
}

Material createDiffuseMaterial()
{
    const std::string ptx_path = ptxPath( "diffuse.cu" );
    Program ch_program = context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" );

    Material material = context->createMaterial();
    material->setClosestHitProgram( 0, ch_program );

    const std::string texture_filename = std::string( sutil::samplesDir() ) + "/data/teste.ppm";
    material["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    material["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    return material;
}

void createGeometry()
{
    m_top_object = context->createGroup();
    m_top_object->setAcceleration( context->createAcceleration("Trbvh"));

    // Floor geometry
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = make_float3( -5.5f, -5.5f,  0.0f);
    float3 v1 = make_float3( 11.0f, 0.0f, 0.0f );
    float3 v2 = make_float3( 0.0f, 11.0f, 0.0f );
    float3 normal = cross( v2, v1 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    //Top geometry
    Geometry top = context->createGeometry();
    top->setPrimitiveCount( 1u );
    top->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    top->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    anchor = make_float3( -5.5f, -5.5f,  11.0f);
    v1 = make_float3( 11.0f, 0.0f, 0.0f );
    v2 = make_float3( 0.0f, 11.0f, 0.0f );
    normal = cross( v2, v1 );
    normal = normalize( normal );
    d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    plane = make_float4( normal, d );
    top["plane"]->setFloat( plane );
    top["v1"]->setFloat( v1 );
    top["v2"]->setFloat( v2 );
    top["anchor"]->setFloat( anchor );

    // Front geometry
    Geometry back = context->createGeometry();
    back->setPrimitiveCount( 1u );
    back->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    back->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    anchor = make_float3( 5.5f, 5.5f,  0.0f);
    v1 = make_float3( -11.0f, 0.0f, 0.0f );
    v2 = make_float3( 0.0f, 0.0f, 11.0f );
    normal = cross( v2, v1 );
    normal = normalize( normal );
    d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    plane = make_float4( normal, d );
    back["plane"]->setFloat( plane );
    back["v1"]->setFloat( v1 );
    back["v2"]->setFloat( v2 );
    back["anchor"]->setFloat( anchor );

    // Back geometry
    Geometry front = context->createGeometry();
    front->setPrimitiveCount( 1u );
    front->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    front->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    anchor = make_float3( -5.5f, -5.5f,  11.0f);
    v1 = make_float3( 11.0f, 0.0f, 0.0f );
    v2 = make_float3( 0.0f, 0.0f, -11.0f );
    normal = cross( v2, v1 );
    normal = normalize( normal );
    d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    plane = make_float4( normal, d );
    front["plane"]->setFloat( plane );
    front["v1"]->setFloat( v1 );
    front["v2"]->setFloat( v2 );
    front["anchor"]->setFloat( anchor );

    // Left geometry
    Geometry left = context->createGeometry();
    left->setPrimitiveCount( 1u );
    left->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    left->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    anchor = make_float3( -5.5f, -5.5f,0.0f);
    v1 = make_float3( 0.0f, 11.0f, 0.0f );
    v2 = make_float3( 0.0f, 0.0f, 11.0f );
    normal = cross( v2, v1 );
    normal = normalize( normal );
    d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    plane = make_float4( normal, d );
    left["plane"]->setFloat( plane );
    left["v1"]->setFloat( v1 );
    left["v2"]->setFloat( v2 );
    left["anchor"]->setFloat( anchor );

    // Right geometry
    Geometry right = context->createGeometry();
    right->setPrimitiveCount( 1u );
    right->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    right->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    anchor = make_float3( 5.5f, -5.5f,0.0f);
    v1 = make_float3( 0.0f, 11.0f, 0.0f );
    v2 = make_float3( 0.0f, 0.0f, 11.0f );
    normal = cross( v2, v1 );
    normal = normalize( normal );
    d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    plane = make_float4( normal, d );
    right["plane"]->setFloat( plane );
    right["v1"]->setFloat( v1 );
    right["v2"]->setFloat( v2 );
    right["anchor"]->setFloat( anchor );

    // Glass material for solid objects
    ptx = sutil::getPtxString( SAMPLE_NAME, "tutorial10.cu" );
    Program glass_chSolid = context->createProgramFromPTXString( ptx, "glass_closest_hit_radiance" );
    Program glass_ahSolid = context->createProgramFromPTXString( ptx, "glass_any_hit_shadow" );
    Material glass_solid = context->createMaterial();
    glass_solid->setClosestHitProgram( 0, glass_chSolid );
    glass_solid->setAnyHitProgram( 1, glass_ahSolid );
    glass_solid["importance_cutoff"]->setFloat( 1e-2f );
    glass_solid["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
    glass_solid["fresnel_exponent"]->setFloat( 3.0f );
    glass_solid["fresnel_minimum"]->setFloat( 0.1f );
    glass_solid["fresnel_maximum"]->setFloat( 1.0f );
    glass_solid["refraction_index"]->setFloat( 1.4f );
    glass_solid["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_solid["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_solid["refraction_maxdepth"]->setInt( 100 );
    glass_solid["reflection_maxdepth"]->setInt( 100 );
    float3 extinctionSolid = make_float3(.80f, .89f, .75f);
    glass_solid["extinction_constant"]->setFloat( log(extinctionSolid.x), log(extinctionSolid.y), log(extinctionSolid.z) );
    glass_solid["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );

    tutorial_number = 10;

    // Materials
    std::string box_chname;
    if(tutorial_number >= 8){
        box_chname = "box_closest_hit_radiance";
    } else if(tutorial_number >= 3){
        box_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        box_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        box_chname = "closest_hit_radiance1";
    } else {
        box_chname = "closest_hit_radiance0";
    }

    std::string texture_filename = std::string( sutil::samplesDir() ) + "/data/mesa.ppm";

    Material box_matl = context->createMaterial();
    Program box_ch = context->createProgramFromPTXString( ptx, box_chname.c_str() );
    box_matl->setClosestHitProgram( 0, box_ch );
    if( tutorial_number >= 3) {
        Program box_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
        box_matl->setAnyHitProgram( 1, box_ah );
    }
    box_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
    box_matl["Kd"]->setFloat( 0.6f, 0.7f, 0.8f );
    box_matl["Ks"]->setFloat( 0.8f, 0.9f, 0.8f );
//    box_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
//    box_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
//    box_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    box_matl["phong_exp"]->setFloat( 88 );
    box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );
    box_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    box_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    tutorial_number = 3;
    ptx = sutil::getPtxString( SAMPLE_NAME, "tutorial3.cu" );

    std::string floor_chname;
    if(tutorial_number >= 7){
        floor_chname = "floor_closest_hit_radiance";
    } else if(tutorial_number >= 6){
        floor_chname = "floor_closest_hit_radiance5";
    } else if(tutorial_number >= 4){
        floor_chname = "floor_closest_hit_radiance4";
    } else if(tutorial_number >= 3){
        floor_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        floor_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        floor_chname = "closest_hit_radiance1";
    } else {
        floor_chname = "closest_hit_radiance0";
    }

    texture_filename = std::string( sutil::samplesDir() ) + "/data/base.ppm";

    Material floor_matl = context->createMaterial();
    Program floor_ch = context->createProgramFromPTXString( ptx, floor_chname );
    floor_matl->setClosestHitProgram( 0, floor_ch );
    Program floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    floor_matl->setAnyHitProgram( 1, floor_ah );
    floor_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    floor_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    floor_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    floor_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    floor_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    floor_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    floor_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    texture_filename = std::string( sutil::samplesDir() ) + "/data/topo_claro.ppm";

    Material top_matl = context->createMaterial();
    top_matl->setClosestHitProgram( 0, floor_ch );
    top_matl->setAnyHitProgram( 1, floor_ah );
    top_matl["Ka"]->setFloat(  0.55f, 0.55f, 0.55f );
    top_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    top_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    top_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    top_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    top_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    top_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    top_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    texture_filename = std::string( sutil::samplesDir() ) + "/data/lateraisupside.ppm";

    Material backObj_matl = context->createMaterial();
    backObj_matl->setClosestHitProgram( 0, floor_ch );
    backObj_matl->setAnyHitProgram( 1, floor_ah );
    backObj_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
    backObj_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    backObj_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    backObj_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    backObj_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    backObj_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    backObj_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    backObj_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    texture_filename = std::string( sutil::samplesDir() ) + "/data/lateraisdownside.ppm";

    Material front_matl = context->createMaterial();
    front_matl->setClosestHitProgram( 0, floor_ch );
    front_matl->setAnyHitProgram( 1, floor_ah );
    front_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
    front_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    front_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    front_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    front_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    front_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    front_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    front_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    texture_filename = std::string( sutil::samplesDir() ) + "/data/lateraisupside.ppm";

    Material left_matl = context->createMaterial();
    left_matl->setClosestHitProgram( 0, floor_ch );
    left_matl->setAnyHitProgram( 1, floor_ah );
    left_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
    left_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    left_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    left_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    left_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    left_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    left_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    left_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    texture_filename = std::string( sutil::samplesDir() ) + "/data/globo.ppm";

    Material right_matl = context->createMaterial();
    right_matl->setClosestHitProgram( 0, floor_ch );
    right_matl->setAnyHitProgram( 1, floor_ah );
    right_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
    right_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f);
    right_matl["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
    right_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
    right_matl["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
    right_matl["phong_exp"]->setFloat( 1 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.01f );
    right_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    right_matl["Kd_map_scale"]->setFloat( make_float2( 0.6f, 0.6f) );

    ptx = sutil::getPtxString( SAMPLE_NAME, "tutorial6.cu" );

//    Material metal_matl = context->createMaterial();
//    floor_ch = context->createProgramFromPTXString( ptx, "floor_closest_hit_radiance5" );
//    metal_matl->setClosestHitProgram( 0, floor_ch );
//    floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
//    metal_matl->setAnyHitProgram( 1, floor_ah );
//    metal_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f);
//    metal_matl["Kd"]->setFloat( 0.7f, 0.7f, 0.7f );
//    metal_matl["Ks"]->setFloat( 0.1f, 0.1f, 0.1f);
////    metal_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
////    metal_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
////    metal_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
//    metal_matl["reflectivity"]->setFloat( 1.0f, 1.0f, 1.0f );
//    metal_matl["reflectivity_n"]->setFloat( 1.0f, 1.0f, 1.0f );
//    metal_matl["phong_exp"]->setFloat( 88 );
//    metal_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
//    metal_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

//    //BUNNY
//    Material metal_matl = context->createMaterial();
//    floor_ch = context->createProgramFromPTXString( ptx, "floor_closest_hit_radiance5" );
//    metal_matl->setClosestHitProgram( 0, floor_ch );
//    floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
//    metal_matl->setAnyHitProgram( 1, floor_ah );
//    metal_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
//    metal_matl["Kd"]->setFloat( 0.8f, 0.1f, 0.1f );
//    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
//    metal_matl["reflectivity"]->setFloat( 0.0f,  0.0f,  0.0f );
//    metal_matl["reflectivity_n"]->setFloat( 0.5f,  0.3f,  0.3f );
//    metal_matl["phong_exp"]->setFloat( 64 );
//    metal_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
//    metal_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    //DRAGON
    Material metal_matl = context->createMaterial();
    floor_ch = context->createProgramFromPTXString( ptx, "floor_closest_hit_radiance5" );
    metal_matl->setClosestHitProgram( 0, floor_ch );
    floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
    metal_matl->setAnyHitProgram( 1, floor_ah );
    metal_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
    metal_matl["Kd"]->setFloat( 0.6f, 0.9f, 0.6f );
    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    metal_matl["reflectivity"]->setFloat( 0.0f,  0.0f,  0.0f );
    metal_matl["reflectivity_n"]->setFloat( 0.5f,  0.6f,  0.5f );
    metal_matl["phong_exp"]->setFloat( 256);
    metal_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
    metal_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

//    //LUCY
//    Material metal_matl = context->createMaterial();
//    floor_ch = context->createProgramFromPTXString( ptx, "floor_closest_hit_radiance5" );
//    metal_matl->setClosestHitProgram( 0, floor_ch );
//    floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
//    metal_matl->setAnyHitProgram( 1, floor_ah );
//    metal_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f);
//    metal_matl["Kd"]->setFloat( 0.1f, 0.6f, 0.7f );
//    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f);
//    metal_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
//    metal_matl["reflectivity_n"]->setFloat( 0.3f, 0.7f, 0.9f );
//    metal_matl["phong_exp"]->setFloat( 256 );
//    metal_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
//    metal_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

//    //BUDDAH
//    Material metal_matl = context->createMaterial();
//    floor_ch = context->createProgramFromPTXString( ptx, "floor_closest_hit_radiance5" );
//    metal_matl->setClosestHitProgram( 0, floor_ch );
//    floor_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );
//    metal_matl->setAnyHitProgram( 1, floor_ah );
//    metal_matl["Ka"]->setFloat( 0.1f, 0.1f, 0.1f );
//    metal_matl["Kd"]->setFloat( 0.9f, 0.6f, 0.4f );
//    metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
//    metal_matl["reflectivity"]->setFloat( 0.0f,  0.0f,  0.0f );
//    metal_matl["reflectivity_n"]->setFloat( 0.7f,  0.4f,  0.2f );
//    metal_matl["phong_exp"]->setFloat( 256 );
//    metal_matl["Kd_map"]->setTextureSampler( sutil::loadTexture( context, texture_filename, optix::make_float3( 1.0f ) ) );
//    metal_matl["Kd_map_scale"]->setFloat( make_float2( 1.0f, 1.0f) );

    std::vector<GeometryInstance> gis;
    GeometryGroup obj1, obj2, obj3, obj4;

    scene = 7;

    switch(scene){
        case 7: //Lucy and Cornell White
        {
//            {obj1 = context->createGeometryGroup();
//                obj1->addChild(loadMesh(mesh_lucy, metal_matl, true));
//                obj1->setAcceleration(context->createAcceleration("Trbvh"));
//
//                lucyPose = Matrix4x4::translate(make_float3(0.0f, 0.0f, -2.0f));
//                lucyPose = lucyPose * Matrix4x4::scale(make_float3(0.10, 0.10, 0.10));
//                lucyPose = lucyPose * Matrix4x4::rotate(M_PI/2, make_float3(0.0f, 0.0f, 1.0f));
//                lucyPose = lucyPose * Matrix4x4::rotate(M_PI / 2, make_float3(1.0f, 0.0f, 0.0f));
//
//                lucyT = context->createTransform();
//                lucyT->setMatrix(false, lucyPose.getData(), 0 );
//            }

            {obj1 = context->createGeometryGroup();
                obj1->addChild(loadMesh(mesh_dragon, metal_matl, true));
                obj1->setAcceleration(context->createAcceleration("Trbvh"));

                dragonPose = Matrix4x4::translate(make_float3(deltaX, deltaY, 29.0f));
                dragonPose = dragonPose * Matrix4x4::scale(make_float3(110.0, 110.0, 110.0));
                //dragonPose = dragonPose * Matrix4x4::rotate(2.61799, make_float3(0.0f, 0.0f, 1.0f));
                dragonPose = dragonPose * Matrix4x4::rotate(-M_PI/3, make_float3(0.0f, 0.0f, 1.0f));

                dragonT = context->createTransform();
                dragonT->setMatrix(false, dragonPose.getData(), 0 );
            }

//            {   obj1 = context->createGeometryGroup();
//                obj1->addChild(loadMesh(mesh_bunny, metal_matl, true));
//                obj1->setAcceleration(context->createAcceleration("Trbvh"));
//
//                bunnyPose = Matrix4x4::translate(make_float3(deltaX, deltaY, 0.0f));
//                bunnyPose = bunnyPose * Matrix4x4::scale(make_float3(65.0, 65.0, 65.0));
//                bunnyPose = bunnyPose * Matrix4x4::rotate(M_PI/10, make_float3(0.0f, 0.0f, 1.0f));
//                bunnyPose = bunnyPose * Matrix4x4::rotate(M_PI_2, make_float3(1.0f, 0.0f, 0.0f));
//
//                bunnyT = context->createTransform();
//                bunnyT->setMatrix(false, bunnyPose.getData(), 0 );
//            }

//            {   obj1 = context->createGeometryGroup();
//                obj1->addChild(loadMesh(mesh_happyBuddah, metal_matl, true));
//                obj1->setAcceleration(context->createAcceleration("Trbvh"));
//
//                happyBuddahPose = Matrix4x4::translate(make_float3(deltaX, deltaY, 50.0f));
//                happyBuddahPose = happyBuddahPose * Matrix4x4::scale(make_float3(100.0, 100, 100));
//                happyBuddahPose = happyBuddahPose * Matrix4x4::rotate(-M_PI/2, make_float3(0.0f, 0.0f, 1.0f));
//                happyBuddahPose = happyBuddahPose * Matrix4x4::rotate(-M_PI/3, make_float3(0.0f, 0.0f, 1.0f));
//
//                happyBuddahT = context->createTransform();
//                happyBuddahT->setMatrix(false, happyBuddahPose.getData(), 0 );
//            }

            {obj2 = context->createGeometryGroup();
                obj2->addChild(loadMesh(mesh_cbWhite, floor_matl, true));
                obj2->setAcceleration(context->createAcceleration("Trbvh"));

                cbWhitePose = Matrix4x4::translate(make_float3(deltaX, deltaY, 0.0f));
                cbWhitePose = cbWhitePose * Matrix4x4::scale(make_float3(20.0f, 20.0f, 20.0f));

                cbWhiteT = context->createTransform();
                cbWhiteT->setMatrix(false, cbWhitePose.getData(), 0 );
            }

                gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
                gis.push_back( context->createGeometryInstance( top, &top_matl, &top_matl+1 ) );
                gis.push_back( context->createGeometryInstance( back, &backObj_matl, &backObj_matl+1 ) );
                gis.push_back( context->createGeometryInstance( front, &front_matl, &front_matl+1 ) );
                gis.push_back( context->createGeometryInstance( left, &left_matl, &left_matl+1 ) );
                gis.push_back( context->createGeometryInstance( right, &right_matl, &right_matl+1 ) );
                obj3 = context->createGeometryGroup();
                obj3->setChildCount( static_cast<unsigned int>(gis.size()) );
                obj3->setChild( 0, gis[0] );
                obj3->setChild( 1, gis[1] );
                obj3->setChild( 2, gis[2] );
                obj3->setChild( 3, gis[3] );
                obj3->setChild( 4, gis[4] );
                obj3->setChild( 5, gis[5] );
                obj3->setAcceleration( context->createAcceleration("Trbvh") );


            //lucyT->setChild(obj1);
            cbWhiteT->setChild(obj3);
            dragonT->setChild(obj1);
            //bunnyT->setChild(obj1);
            //happyBuddahT->setChild(obj1);

            //m_top_object->addChild( lucyT );
            m_top_object->addChild( cbWhiteT );
            m_top_object->addChild( dragonT );
            //m_top_object->addChild( bunnyT );
            //m_top_object->addChild( happyBuddahT );

            context["top_object"]->set( m_top_object );
            context["top_shadower"]->set( m_top_object );
            break;
        }

    }
}


void setupCamera()
{

//    float scalef = 1.0f;
//    GLdouble m[16];
//    GLdouble eyepos[3], lookat[3], up[3];
//
//// See detection loop in Idle() in simpleLite.c for context of the
//    line below.
//            arGetTransMat(&(marker_info[k]), patt_centre, patt_width, patt_trans);
//
//// Make patt_trans into a standard OpenGL HCT matrix (N.B.:column-
//    major).
//    arglCameraView(patt_trans, m, scalef);
//
//// This treats the marker as lying in the x-y plane, with the +z axis
//    pointing towards the observer.
//            eyepos[0] = m[12]; eyepos[1] = m[13]; eyepos[2] = m[14];
//    lookat[0] = eyepos[0] - m[8]; lookat[1] = eyepos[1] - m[9]; lookat[2]
//                                                                        = eyepos[2] - m[10];
//    up[0] = m[4]; up[1] = m[5]; up[2] = m[6];

    camera_eye    = make_float3( invOut[12], invOut[13], invOut[14] );
    camera_lookat = make_float3( camera_eye.x - invOut[8], camera_eye.y - invOut[9],  camera_eye.z - invOut[10] );
    camera_up     = make_float3( invOut[4], invOut[5], invOut[6]);

    camera_rotate  = Matrix4x4::identity();
    camera_dirty = true;
}



void setupLights()
{
    //BUNNY
    BasicLight lights[] = {
            { make_float3( deltaX, deltaY , 218.0f ), make_float3( 0.5f, 0.5f, 0.5f ), 1 }
    };

//    BasicLight lights[] = {
//            { make_float3( 0.0f, -40.0f , 150.0f ), make_float3( 0.8f, 0.8f, 0.8f ), 1 }
//    };

//    BasicLight lights[] = {
//            { make_float3( -2.0f, 0.0f , 109.0f ), make_float3( 0.5f, 0.5f, 0.5f ), 1 }
//    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}

void updateCamera()
{
    const float vfov  = 45.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    camera_dirty = false;
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
//    outFile.open("camPoseDragon.txt", std::ios::out);  // abre o arquivo para escrita
//    if (! outFile)
//    {
//        std::cout << "Arquivo camPoseDragon.txt nao pode ser aberto" << std::endl;
//        abort();
//    }

//   inFile.open("camPoseDragon.txt", std::ios::in);    // abre o arquivo para leitura
//   if (! inFile)
//   {
//       std::cout << "Arquivo camPoseDragon.txt nao pode ser aberto" << std::endl;
//       abort();
//   }

    //Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    if (m_interop)
    {
        glGenBuffers(1, &m_pbo);
        if(m_pbo != 0){ // Buffer size must be > 0 or OptiX can't create a buffer from it.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(unsigned char) * 4, nullptr, GL_STREAM_READ); // BRGA8
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }else{
            ARLOGe("m_pbo tem tamanho zero");
        }
    }
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

    glGenTextures(1, &m_tex);
    if(m_tex != 0){
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);

        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
    }else{
        ARLOGe("m_tex tem tamanho zero");
    }

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    displayOnce();

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

//    outFile << camera_eye.x << " " << camera_eye.y << " " << camera_eye.z << " " << camera_lookat.x << " " << camera_lookat.y << " " << camera_lookat.z << " " << camera_up.x << " " << camera_up.y << " " << camera_up.z << " " << "\n";

//    if(inFile >> camera_eye.x >> camera_eye.y >> camera_eye.z >> camera_lookat.x >> camera_lookat.y >> camera_lookat.z >> camera_up.x >> camera_up.y >> camera_up.z){
//        camera_dirty = true;
//    } else{
//        inFile.close();
//    }
    float3 eyeTemp = make_float3(invOut[12], invOut[13], invOut[14]);
    if(distanceBigger(camera_eyeOld, eyeTemp)){
        camera_eyeOld = eyeTemp;
        camera_eye    = make_float3( invOut[12], invOut[13], invOut[14] );
        camera_lookat = make_float3( camera_eye.x - invOut[8], camera_eye.y - invOut[9],  camera_eye.z - invOut[10] );
        camera_up     = make_float3( invOut[4], invOut[5], invOut[6]);
        camera_dirty = true;
        //outFile << camera_eye.x << " " << camera_eye.y << " " << camera_eye.z << " " << camera_lookat.x << " " << camera_lookat.y << " " << camera_lookat.z << " " << camera_up.x << " " << camera_up.y << " " << camera_up.z << " " << "\n";
    }

    if( camera_dirty ) {
        updateCamera();
    }

    context->launch( 0, width, height );

    // Update the OpenGL texture with the results:
    if (m_interop)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer->getGLBOId());

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr); // BGRA8
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        void const* data = m_buffer->map(0, RT_BUFFER_MAP_READ );
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei) width, (GLsizei) height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data); // BGRA8
        m_buffer->unmap();
    }

    RTsize elmt_size = m_buffer->getElementSize();
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    drawTexConfig(m_tex);
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            drawCleanup();
            if (arController) {
                arController->drawVideoFinal(0);
                arController->shutdown();
                delete arController;
            }
            destroyContext();
            exit(0);
        }
        case GLUT_KEY_UP :
            camera_eye.y += 1.0f;
            camera_dirty = true;
            break;
        case GLUT_KEY_DOWN :
            camera_eye.y -= 1.0f;
            camera_dirty = true;
            break;
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_eye - camera_lookat)*scale;
        camera_dirty = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
        camera_dirty = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// AR
//
//------------------------------------------------------------------------------

static void init(){
#  if ARX_TARGET_PLATFORM_MACOS
    vconf = "-format=BGRA";
#  endif

    reshape(m_width, m_height);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
    //char *resourcesDir = arUtilGetResourcesDirectoryPath(AR_UTIL_RESOURCES_DIRECTORY_BEHAVIOR_BEST);
    char *resourcesDir = "../";
    ARLOGd("Resources are in'%s'.\n", resourcesDir);
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
    arController->startRunning(vconf, cpara, NULL, 0);
}


static void displayOnce(void)
{

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
//#ifndef ar
            //Display the current video frame to the current OpenGL context.
            arController->drawVideo(0);
//#endif

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
                    }
                }
                //ARLOGd("MK: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", view[12], view[13], view[14], view[15]);
                //sprintf(str, "Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", view[12], view[13], view[14], view[15]);
                if(gluInvertMatrix(view)){
                    //arUtilPrintMtx16(marker->transformationMatrix);
                    //ARLOGi("--- \n");
                    sprintf(str, "Cam Pos: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", invOut[12], invOut[13], invOut[14], invOut[15]);
                    //ARLOGd("Cam: x: %3.1f  y: %3.1f  z: %3.1f w: %3.1f \n", invOut[12], invOut[13], invOut[14], invOut[15]);
                }
                //drawSetModel(markerModelIDs[i], marker->visible, view, invOut);
                //showString( str );
            }
//#ifndef ar
            //draw();
//#endif
            done = true;
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


static void quit(int rc)
{
    drawCleanup();
    if (arController) {
        arController->drawVideoFinal(0);
        arController->shutdown();
        delete arController;
    }
    outFile.close();
    exit(rc);
}


static void reshape(int w, int h)
{
    contextWidth = w;
    contextHeight = h;
    ARLOGd("Resized to %dx%d.\n", w, h);
    contextWasUpdated = true;
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
              "App Options:\n"
              "  -h | --help         Print this usage message and exit.\n"
              "  -f | --file         Save single frame to file and exit.\n"
              "  -n | --nopbo        Disable GL interop for display buffer.\n"
              "  -T | --tutorial-number <num>              Specify tutorial number\n"
              "  -t | --texture-path <path>                Specify path to texture directory\n"
              "App Keystrokes:\n"
              "  qVec  Quit\n"
              "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
              << std::endl;

    exit(1);
}


int main( int argc, char** argv )
{
    std::string out_file;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if ( arg == "-f" || arg == "--file" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            m_interop = false;
        }
        else if ( arg == "-t" || arg == "--texture-path" )
        {
            if ( i == argc-1 ) {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            texture_path = argv[++i];
        }
        else if ( arg == "-T" || arg == "--tutorial-number" )
        {
            if ( i == argc-1 ) {
                printUsageAndExit( argv[0] );
            }
            tutorial_number = atoi(argv[++i]);
            if ( tutorial_number < 0 || tutorial_number > 11 ) {
                std::cerr << "Tutorial number (" << tutorial_number << ") is out of range [0..11]\n";
                printUsageAndExit( argv[0] );
            }
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( texture_path.empty() ) {
        texture_path = std::string( sutil::samplesDir() ) + "/data";
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif
        // load the ptx source associated with tutorial number
        std::stringstream ss;
        ss << "tutorial" << tutorial_number << ".cu";
        std::string tutorial_ptx_path = ss.str();
        tutorial_ptx = sutil::getPtxString( SAMPLE_NAME, tutorial_ptx_path.c_str() );

        init();
        displayOnce();
        displayOnce();

        createContext();
        createGeometry();
        camera_eyeOld = make_float3( invOut[12], invOut[13], invOut[14] );
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}


#pragma clang diagnostic pop