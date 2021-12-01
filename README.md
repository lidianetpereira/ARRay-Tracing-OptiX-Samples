# ARRay-Tracing-OptiX-Samples
This repository presents an utilization example of the ARRay - Tracing middleware integrating the artoolkitX with the OptiX.

### Requirements:
<ol>
<li> Download and install NVIDIA Optix SDK 6.5 following the instruction from https://developer.nvidia.com/designworks/optix/download
<li> Install the artoolkitX prerequisites shown in https://github.com/artoolkitx/artoolkitx/wiki/Building-artoolkitX-prerequisites
</ol>

### Instructions:
These instructions were tested on Ubuntu 18.04.5 LTS, other distros will may require some adaptations.<br>
<ol>
<li> Clone or download this repository to a BASE_FOLDER of your choice.<br>
<li> Go to folder "artoolkitXSource" and run the comand "./build.sh linux" as superuser.<br>
<li> Return to the BASE_FOLDER and run "ccmake ."<br>
  <ol>
  <li> Press "c" to configure.<br>
  <li> Press "g" to generate.<br>
  </ol>
<li>4- Run the command "make".<br>
</ol>

The compiled exemple will be into BASE_FOLDER/bin.
Run the sample and point your webcam to one of the markers from file [markers.pdf](markers.pdf).