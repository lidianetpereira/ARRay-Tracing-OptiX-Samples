#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ARX" for configuration "Debug"
set_property(TARGET ARX APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ARX PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libARX.so.1.0.6"
  IMPORTED_SONAME_DEBUG "libARX.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS ARX )
list(APPEND _IMPORT_CHECK_FILES_FOR_ARX "${_IMPORT_PREFIX}/lib/libARX.so.1.0.6" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
