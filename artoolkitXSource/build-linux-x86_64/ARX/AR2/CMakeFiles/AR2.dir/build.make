# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64

# Include any dependencies generated for this target.
include ARX/AR2/CMakeFiles/AR2.dir/depend.make

# Include the progress variables for this target.
include ARX/AR2/CMakeFiles/AR2.dir/progress.make

# Include the compile flags for this target's objects.
include ARX/AR2/CMakeFiles/AR2.dir/flags.make

ARX/AR2/CMakeFiles/AR2.dir/coord.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/coord.c.o: ../ARX/AR2/coord.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object ARX/AR2/CMakeFiles/AR2.dir/coord.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/coord.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/coord.c

ARX/AR2/CMakeFiles/AR2.dir/coord.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/coord.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/coord.c > CMakeFiles/AR2.dir/coord.c.i

ARX/AR2/CMakeFiles/AR2.dir/coord.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/coord.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/coord.c -o CMakeFiles/AR2.dir/coord.c.s

ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/coord.c.o


ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o: ../ARX/AR2/featureMap.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/featureMap.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureMap.c

ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/featureMap.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureMap.c > CMakeFiles/AR2.dir/featureMap.c.i

ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/featureMap.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureMap.c -o CMakeFiles/AR2.dir/featureMap.c.s

ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o


ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o: ../ARX/AR2/featureSet.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/featureSet.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureSet.c

ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/featureSet.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureSet.c > CMakeFiles/AR2.dir/featureSet.c.i

ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/featureSet.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/featureSet.c -o CMakeFiles/AR2.dir/featureSet.c.s

ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o


ARX/AR2/CMakeFiles/AR2.dir/handle.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/handle.c.o: ../ARX/AR2/handle.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object ARX/AR2/CMakeFiles/AR2.dir/handle.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/handle.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/handle.c

ARX/AR2/CMakeFiles/AR2.dir/handle.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/handle.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/handle.c > CMakeFiles/AR2.dir/handle.c.i

ARX/AR2/CMakeFiles/AR2.dir/handle.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/handle.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/handle.c -o CMakeFiles/AR2.dir/handle.c.s

ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/handle.c.o


ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o: ../ARX/AR2/imageSet.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/imageSet.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/imageSet.c

ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/imageSet.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/imageSet.c > CMakeFiles/AR2.dir/imageSet.c.i

ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/imageSet.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/imageSet.c -o CMakeFiles/AR2.dir/imageSet.c.s

ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o


ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o: ../ARX/AR2/jpeg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/jpeg.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/jpeg.c

ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/jpeg.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/jpeg.c > CMakeFiles/AR2.dir/jpeg.c.i

ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/jpeg.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/jpeg.c -o CMakeFiles/AR2.dir/jpeg.c.s

ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o


ARX/AR2/CMakeFiles/AR2.dir/marker.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/marker.c.o: ../ARX/AR2/marker.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object ARX/AR2/CMakeFiles/AR2.dir/marker.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/marker.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/marker.c

ARX/AR2/CMakeFiles/AR2.dir/marker.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/marker.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/marker.c > CMakeFiles/AR2.dir/marker.c.i

ARX/AR2/CMakeFiles/AR2.dir/marker.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/marker.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/marker.c -o CMakeFiles/AR2.dir/marker.c.s

ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/marker.c.o


ARX/AR2/CMakeFiles/AR2.dir/matching.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/matching.c.o: ../ARX/AR2/matching.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object ARX/AR2/CMakeFiles/AR2.dir/matching.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/matching.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching.c

ARX/AR2/CMakeFiles/AR2.dir/matching.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/matching.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching.c > CMakeFiles/AR2.dir/matching.c.i

ARX/AR2/CMakeFiles/AR2.dir/matching.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/matching.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching.c -o CMakeFiles/AR2.dir/matching.c.s

ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/matching.c.o


ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o: ../ARX/AR2/matching2.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/matching2.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching2.c

ARX/AR2/CMakeFiles/AR2.dir/matching2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/matching2.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching2.c > CMakeFiles/AR2.dir/matching2.c.i

ARX/AR2/CMakeFiles/AR2.dir/matching2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/matching2.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/matching2.c -o CMakeFiles/AR2.dir/matching2.c.s

ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o


ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o: ../ARX/AR2/searchPoint.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/searchPoint.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/searchPoint.c

ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/searchPoint.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/searchPoint.c > CMakeFiles/AR2.dir/searchPoint.c.i

ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/searchPoint.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/searchPoint.c -o CMakeFiles/AR2.dir/searchPoint.c.s

ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o


ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o: ../ARX/AR2/selectTemplate.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/selectTemplate.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/selectTemplate.c

ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/selectTemplate.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/selectTemplate.c > CMakeFiles/AR2.dir/selectTemplate.c.i

ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/selectTemplate.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/selectTemplate.c -o CMakeFiles/AR2.dir/selectTemplate.c.s

ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o


ARX/AR2/CMakeFiles/AR2.dir/surface.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/surface.c.o: ../ARX/AR2/surface.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object ARX/AR2/CMakeFiles/AR2.dir/surface.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/surface.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/surface.c

ARX/AR2/CMakeFiles/AR2.dir/surface.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/surface.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/surface.c > CMakeFiles/AR2.dir/surface.c.i

ARX/AR2/CMakeFiles/AR2.dir/surface.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/surface.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/surface.c -o CMakeFiles/AR2.dir/surface.c.s

ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/surface.c.o


ARX/AR2/CMakeFiles/AR2.dir/template.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/template.c.o: ../ARX/AR2/template.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building C object ARX/AR2/CMakeFiles/AR2.dir/template.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/template.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/template.c

ARX/AR2/CMakeFiles/AR2.dir/template.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/template.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/template.c > CMakeFiles/AR2.dir/template.c.i

ARX/AR2/CMakeFiles/AR2.dir/template.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/template.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/template.c -o CMakeFiles/AR2.dir/template.c.s

ARX/AR2/CMakeFiles/AR2.dir/template.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/template.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/template.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/template.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/template.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/template.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/template.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/template.c.o


ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o: ../ARX/AR2/tracking.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building C object ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/tracking.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking.c

ARX/AR2/CMakeFiles/AR2.dir/tracking.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/tracking.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking.c > CMakeFiles/AR2.dir/tracking.c.i

ARX/AR2/CMakeFiles/AR2.dir/tracking.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/tracking.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking.c -o CMakeFiles/AR2.dir/tracking.c.s

ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o


ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o: ../ARX/AR2/tracking2d.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building C object ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/tracking2d.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking2d.c

ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/tracking2d.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking2d.c > CMakeFiles/AR2.dir/tracking2d.c.i

ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/tracking2d.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/tracking2d.c -o CMakeFiles/AR2.dir/tracking2d.c.s

ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o


ARX/AR2/CMakeFiles/AR2.dir/util.c.o: ARX/AR2/CMakeFiles/AR2.dir/flags.make
ARX/AR2/CMakeFiles/AR2.dir/util.c.o: ../ARX/AR2/util.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building C object ARX/AR2/CMakeFiles/AR2.dir/util.c.o"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/AR2.dir/util.c.o   -c /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/util.c

ARX/AR2/CMakeFiles/AR2.dir/util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/AR2.dir/util.c.i"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/util.c > CMakeFiles/AR2.dir/util.c.i

ARX/AR2/CMakeFiles/AR2.dir/util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/AR2.dir/util.c.s"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2/util.c -o CMakeFiles/AR2.dir/util.c.s

ARX/AR2/CMakeFiles/AR2.dir/util.c.o.requires:

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/util.c.o.requires

ARX/AR2/CMakeFiles/AR2.dir/util.c.o.provides: ARX/AR2/CMakeFiles/AR2.dir/util.c.o.requires
	$(MAKE) -f ARX/AR2/CMakeFiles/AR2.dir/build.make ARX/AR2/CMakeFiles/AR2.dir/util.c.o.provides.build
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/util.c.o.provides

ARX/AR2/CMakeFiles/AR2.dir/util.c.o.provides.build: ARX/AR2/CMakeFiles/AR2.dir/util.c.o


# Object files for target AR2
AR2_OBJECTS = \
"CMakeFiles/AR2.dir/coord.c.o" \
"CMakeFiles/AR2.dir/featureMap.c.o" \
"CMakeFiles/AR2.dir/featureSet.c.o" \
"CMakeFiles/AR2.dir/handle.c.o" \
"CMakeFiles/AR2.dir/imageSet.c.o" \
"CMakeFiles/AR2.dir/jpeg.c.o" \
"CMakeFiles/AR2.dir/marker.c.o" \
"CMakeFiles/AR2.dir/matching.c.o" \
"CMakeFiles/AR2.dir/matching2.c.o" \
"CMakeFiles/AR2.dir/searchPoint.c.o" \
"CMakeFiles/AR2.dir/selectTemplate.c.o" \
"CMakeFiles/AR2.dir/surface.c.o" \
"CMakeFiles/AR2.dir/template.c.o" \
"CMakeFiles/AR2.dir/tracking.c.o" \
"CMakeFiles/AR2.dir/tracking2d.c.o" \
"CMakeFiles/AR2.dir/util.c.o"

# External object files for target AR2
AR2_EXTERNAL_OBJECTS =

ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/coord.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/handle.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/marker.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/matching.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/surface.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/template.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/util.c.o
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/build.make
ARX/AR2/libAR2.a: ARX/AR2/CMakeFiles/AR2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Linking C static library libAR2.a"
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && $(CMAKE_COMMAND) -P CMakeFiles/AR2.dir/cmake_clean_target.cmake
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/AR2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ARX/AR2/CMakeFiles/AR2.dir/build: ARX/AR2/libAR2.a

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/build

ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/coord.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/featureMap.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/featureSet.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/handle.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/imageSet.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/jpeg.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/marker.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/matching.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/matching2.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/searchPoint.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/selectTemplate.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/surface.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/template.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/tracking.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/tracking2d.c.o.requires
ARX/AR2/CMakeFiles/AR2.dir/requires: ARX/AR2/CMakeFiles/AR2.dir/util.c.o.requires

.PHONY : ARX/AR2/CMakeFiles/AR2.dir/requires

ARX/AR2/CMakeFiles/AR2.dir/clean:
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 && $(CMAKE_COMMAND) -P CMakeFiles/AR2.dir/cmake_clean.cmake
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/clean

ARX/AR2/CMakeFiles/AR2.dir/depend:
	cd /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/ARX/AR2 /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64 /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2 /home/lidiane/Documentos/ARRay-Tracing-master/artoolkitXSource/build-linux-x86_64/ARX/AR2/CMakeFiles/AR2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ARX/AR2/CMakeFiles/AR2.dir/depend

