# add_subdirectory(spout) # add Spout

set(USER_CODE_FILES
  # My codes 
  spout_osc.cpp
  SpoutInput.cpp
  TrackerPostProcessing.cpp
  # Spout Libraries
  spout/src/SpoutCopy.cpp
  spout/src/SpoutDirectX.cpp
  spout/src/SpoutGLDXinterop.cpp
  spout/src/SpoutGLextensions.cpp
  spout/src/SpoutMemoryShare.cpp
  spout/src/SpoutReceiver.cpp
  spout/src/SpoutSDK.cpp
  spout/src/SpoutSender.cpp
  spout/src/SpoutSenderMemory.cpp
  spout/src/SpoutSenderNames.cpp
  spout/src/SpoutSharedMemory.cpp
)

set(EXE_NAME "UserCustomCode")

message(STATUS "Adding Example ${EXE_NAME}")
link_directories(
  # spout/lib
  glfw/lib
)
add_executable(${EXE_NAME} ${USER_CODE_FILES})
target_include_directories(${EXE_NAME} PUBLIC 
  spout/include
  glfw/include
)
# target_link_libraries(${EXE_NAME} openpose ${examples_3rdparty_libraries} Spout)
target_link_libraries(${EXE_NAME}
  openpose
  ${examples_3rdparty_libraries}
  glfw3 glfw3dll)

if (WIN32) # Visual studio setup
  set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "User Code")
  configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${VCXPROJ_FILE_GPU_MODE}.vcxproj.user
      ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
  # Properties->General->Output Directory
  set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
endif (WIN32)
