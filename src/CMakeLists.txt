add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry.cpp
        ../include/myslam/optimize_end.h optimize_end.cpp)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)