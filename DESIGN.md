# Extending FlightLab into a calibrated multi-angle flight analyzer

## Present implementation

Live: `VideoCapture → consecutive resized frames → cut check → NanoTrack / wide-object CSRT → clean object overlays`. Optional background correction and calibration feed a **separate graph window**, and measurement failure does not halt object tracking.

Prepared review: `reviewed role/keyframe selections → forward + backward NanoTrack → endpoint shape constraints → explicit occlusion gaps → full video with audio + separate graphs + frame-level data`. This workflow is assisted and reviewable; it does not claim automatic pilot-role identification. Low-response interpolation and forward/backward disagreement are flagged in the data.

The Tk event loop remains separate from a single worker so decoding/tracking/detection do not require concurrent mutations of the analysis engine. The worker processes consecutive frames. The viewer displays the result when it is ready. Each segment owns its coordinate reference, scale, objects, timing factor, and history. A scene cut or broken camera-transform chain cannot silently continue producing new calibrated coordinates.

Recognition and tracking are separate: a detector suggests a class and initial box; the tracker follows the selected visual region. The model does not infer a person's identity and does not understand the custom craft's components. User labels such as `Pilot` supply role information.

## Geometry needed for better distance and height

1. **Scene survey:** obtain multiple fixed landmarks with known 3D locations, especially platform corners, actual launch edge, and water level. Confirm the stated 30-foot height and 20-foot wingspan dimensions and uncertainty. Do not use the reported 42 feet as both calibration and validation.
2. **Lens calibration:** obtain each original camera's intrinsic matrix and distortion parameters. Use original unedited footage when available; broadcast crop, digital stabilization, and zoom can change effective intrinsics. A checkerboard/ChArUco recording at the relevant zoom is useful for future flights.
3. **Fixed-camera ground geometry:** at least four non-collinear correspondences on the water/ground plane can define a planar homography. It can locate a surface impact point but cannot recover airborne height by itself. A separate vertical constraint or second view is required.
4. **Synchronization:** align simultaneous camera views using frame timestamps and a common visible event; refine offsets using shared motion. Replays with unknown time warping cannot simply be treated as synchronized camera streams. Audio can help only when its delay/editing is understood.
5. **Camera pose:** estimate per-camera extrinsics from surveyed landmarks. For moving/zooming cameras, solve pose and effective intrinsics per frame or keyframe with well-distributed fixed references. Validate held-out landmark reprojection errors.
6. **Correspondence and triangulation:** match the same physical aircraft landmark or pilot point across views, undistort observations, and triangulate using calibrated projection matrices. A bounding-box center can move due to rotation and is an unsuitable precise 3D landmark. Use wing roots/tips, helmet landmarks, or trained keypoints with explicit visibility flags.
7. **Measurements:** establish the actual takeoff and impact events; compute horizontal displacement from the defined platform edge to pilot impact, airborne path, height above the water plane, and elapsed physical time. Distinguish aircraft, pilot, and detached components throughout.
8. **Uncertainty:** propagate landmark-picking, calibration, camera-pose, timing, and detection uncertainties with Monte Carlo trials or covariance estimates. Report a range, not excessive precision. Validate against known lengths and held-out events before comparing with 42 feet.

## Tracking and recognition upgrades

- Annotate aircraft, pilot, wing panels, and components in a representative collection of launch, flight, and splash frames. Obtain held-out clips from different angles for evaluation.
- Add a modern detector/instance segmentation model with separate class-confidence and track-quality scores. Support user-trained weights and maintain their license information.
- Associate detections with motion/appearance cues, retain explicit lost/occluded states, and offer manual keyframe corrections with retracking. Re-identification should remain reviewable because splashes and debris make associations ambiguous.
- Use foreground masks when estimating camera motion and keep background landmarks distributed across the image. Avoid smooth water and fixed broadcast graphics. Similarity/affine compensation is a preview aid, not a substitute for a calibrated camera model.
- Estimate rotation from a coherent object coordinate frame or multiple keypoints; an axis-aligned tracking box does not measure roll, pitch, or yaw.

## Product milestones

1. Reliable per-shot review: current prototype, then persisted sessions, editable corrections, interval selection, and annotated-video export from stored trajectories.
2. Single fixed-camera calibrated analysis: lens correction, explicit plane mappings, launch/impact tagging, and uncertainty reporting.
3. Multi-angle synchronized analysis: offset timeline, cross-view landmarks, camera poses, triangulated 3D trajectory, and reprojection review.
4. Higher-throughput/live ingestion: benchmark a GPU detector and lightweight tracker, support webcams/capture devices, and separate an optional low-latency preview from complete frame-by-frame analysis. Never silently drop measurement samples to meet a playback target.

The current footage can support useful review and approximate planar tracking. It does not, by itself, supply all the calibration and synchronization data needed to promise an accurate 3D reconstruction.
