"""
Metric-space person re-identification.

The package is deliberately import-light: nothing is re-exported here, so
importing `reid` does not drag in cv2, torch or onnxruntime. Import the module
you need directly.

    scene_geometry    ground plane and lens model recovered from pedestrians
    scene_depth       occluder map and stature field over that plane
    reachability      geodesic "could they have walked there" prior
    motion_prior      learned (cell, heading) movement model
    trajectory_stitch min-cost flow over the tracklet graph
    ghost_pool        rebinding of tracks that left and came back
    person_db         gallery that survives across sessions
    face_anchor       pose-gated, confirm-only face check

scene_depth depends on scene_geometry; everything else is independent and
takes the scene model as a plain argument.
"""
