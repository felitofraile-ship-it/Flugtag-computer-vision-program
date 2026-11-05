# Flugtag distance and height estimates

Both graphs show the **craft's forwardmost visible edge** and the **pilot's hip / lower torso**. This replaces the initial craft-center convention after the user clarified that the front is the intended reference.

The visible craft front projects to approximately **37–38 feet** during the impact splash. The selected last visible point yields about 37.4 feet; it was selected from the image, not adjusted to force the user's suggested range. Splash then hides it, so its curve ends there. The exact water-entry instant and final resting range of that edge are unresolved. The earlier **31–32 feet** referred to the central wing/seat region reaching water, not the front or back. That comparison remains in calibration_and_checks.json.

The pilot's endpoint is **42 feet by calibration**, using the user's reported result. The platform is **30 feet tall**, also supplied by the user. These are approximate 2-D estimates, not a surveyed 3-D trajectory.

## Reading the graphs

- Distance is horizontal range from the launch edge, not cumulative path length. The craft front is already beyond the edge at time zero, while the pilot's hip starts slightly behind it.
- The selected craft extremity is on the wing: the forwardmost visible outline in this side-view projection, not a verified aerodynamic nose or a reconstructed 3-D forward coordinate. Rotation and spanwise depth affect its apparent range.
- Height uses that same craft edge and the pilot's hip/lower torso. The craft curve ends above zero because its last partly visible point is still elevated in this model. No hidden descent is invented to bring it to zero.
- Time zero is the estimated takeoff frame at about 10.20 seconds in the original video. Pilot entry occurs about 2.2 seconds later. The original side-view sequence is assumed normal speed; replays are not used as a flight clock.
- Solid lines connect reviewed points. Dashed lines mark splash-affected positions and interpolation. The pilot's late height rise corresponds to the visible tumble, but its amplitude is uncertain. The hip's estimated water entry precedes disappearance of the feet.
- Both reference points begin above the 30-foot deck because the launch apparatus supports them above it. Later floating debris is excluded.

## Method and limitations

Points were manually selected and checked against annotated frames at approximately 0.2-second intervals, with extra impact samples. Adjacent-frame SIFT background matches and RANSAC homographies compensate for camera pan/zoom, mapping frames into the 11-second view. Moving tracked objects and broadcast overlays were masked. Chained transforms have 154–666 accepted matches per step and median pairwise reprojection errors below 1.2 pixels. These are registration diagnostics, not physical-distance error bars.

In that stabilized 960 × 540 reference view, the selected launch corner is (326, 112), and its approximate waterline base is (231, 467). Their vector represents 30 vertical feet. The pilot's estimated entry point at source coordinates (748.5, 297), near 12.4 seconds, is stabilized. The vector from the platform waterline base to that point represents 42 horizontal feet. Stabilized observations are resolved against these two basis vectors. Pilot entry is 42 feet and zero height by construction.

This assumes a vertical flight plane with locally affine scale after camera stabilization. Perspective, travel toward/away from the camera, wing spanwise depth, registration drift, approximate corner selection, waves, changing pose, and splash limit accuracy. There is no validated numerical confidence interval; 37–38 feet is a rounded descriptive estimate, not a confidence interval. CSV decimals are for reproducibility, not precision claims. The event's reported 42 feet may use a scoring convention differing slightly from the selected hip point.

The approximately 20-foot wingspan is recorded as context but is not used as a constant pixel scale: the wing rotates, is foreshortened, and extends outside the assumed vertical flight plane. These measurements do not uniquely determine full camera calibration or a 3-D trajectory.

The registration method follows [OpenCV's homography documentation](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html). Planar image mapping alone does not establish exact metric scale for a moving 3-D object.

## Files

- distance_vs_time.png and height_vs_time.png: graphs, with matching SVG vector versions.
- Flugtag_distance_and_height.pdf: both graphs, one per page.
- distance_height_data.csv: plotted values, timestamps, source coordinates, and visibility flags.
- craft_front_points.json: reviewed craft-edge points in 640 × 360 coordinates.
- reviewed_points.json: pilot and original craft-center points in 640 × 360 coordinates.
- camera_transforms.json: transforms into the 960 × 540 reference view.
- calibration_and_checks.json: calibration, endpoints, center comparison, and basic checks.
- make_graphs.py: regenerate using Python, NumPy, and Matplotlib.

Source: IEEE_FIU_FlugtagFlight - Trim.mp4, supplied by the user.
