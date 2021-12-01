# Install script for directory: /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/../SDK")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so.1.0.6"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/libARX.so.1.0.6"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/libARX.so.1"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/libARX.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so.1.0.6"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libARX.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/depends/linux/lib:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX" TYPE FILE FILES
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARX_c.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARController.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackable.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARPattern.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackableMultiSquareAuto.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackableMultiSquare.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackableNFT.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackable2d.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackableSquare.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTracker.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackerVideo.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackerNFT.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTracker2d.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARTrackerSquare.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/Error.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/Platform.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARVideoSource.h"
    "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/include/ARX/ARVideoView.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR/include/ARX/AR/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/ar.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/arConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/arFilterTransMat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/arImageProc.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/arMulti.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/icp.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/icpCalib.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/icpCore.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/matrix.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/param.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR/include/ARX/AR/paramGL.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARVideo" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARVideo/include/ARX/ARVideo/video.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARVideo" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARVideo/include/ARX/ARVideo/videoConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARVideo" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARVideo/include/ARX/ARVideo/videoLuma.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARVideo" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARVideo/include/ARX/ARVideo/videoRGBA.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/types.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/log.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/profile.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/thread_sub.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/system.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/android.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/time.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/file_utils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARUtil" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARUtil/include/ARX/ARUtil/image_utils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARG" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARG/include/ARX/ARG/arg.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARG" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARG/include/ARX/ARG/mtx.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARG" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARG/include/ARX/ARG/glStateCache2.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/ARG" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/ARG/include/ARX/ARG/shader_gl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/coord.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/featureSet.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/imageFormat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/imageSet.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/marker.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/searchPoint.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/template.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/tracking.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/AR2" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/include/ARX/AR2/util.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/KPM" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/KPM/include/ARX/KPM/kpm.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/KPM" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/KPM/include/ARX/KPM/kpmType.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ARX/OCVT" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/OCVT/include/ARX/OCVT/PlanarTracker.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ARX/ARX.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ARX/ARX.cmake"
         "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/CMakeFiles/Export/lib/ARX/ARX.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ARX/ARX-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ARX/ARX.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ARX" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/CMakeFiles/Export/lib/ARX/ARX.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ARX" TYPE FILE FILES "/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/CMakeFiles/Export/lib/ARX/ARX-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/ARVideo/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/ARUtil/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/ARG/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/KPM/cmake_install.cmake")
  include("/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/OCVT/cmake_install.cmake")

endif()

