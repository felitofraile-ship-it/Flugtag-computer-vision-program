# Original local-video validation

This records validation on the author-provided clip. The videos and frame-level review outputs described here are not included in the public source distribution. See `measurements/README.md` for the later calibrated graph example.

## Automated checks

All **15 regression tests pass**. They cover tracking on a moving synthetic target, correction without changing identity, continued tracking after measurement-reference failure, top-view overlays without trajectory lines, scale/vertical geometry, missing measurements, exports, cut ownership, clipped boxes, and the clip-specific wing-color alignment.

The real Tk application was exercised programmatically with this video: load the prepared two-object overhead setup, open the separate graph window, correct the pilot while preserving its ID, step asynchronously, and continue tracking with an invalid measurement reference. Both aircraft and pilot remained tracked. Native-window screenshot verification was not completed; the earlier Computer Use capture timed out. The actual rendered tracking-video frames and the exported graph image were visually inspected.

A 60-frame live overhead benchmark with two objects processed at approximately **22.5 FPS** on this computer. This is one tested workload, not a guaranteed 30-FPS claim. Class detection and additional tracks cost more processing time. Rendered review videos play at the source frame rate.

## Supplied-video review

- Entire source: **1,625 frames**, approximately **54.142 seconds**.
- **135 selection/visibility keyframes**, including explicit hidden intervals, are stored in the editable review project.
- **2,711 object observations** after trimming observations to their shot boundaries; there are no duplicate `(frame, object ID)` entries.
- 2,686 observations are marked visible, 24 are explicitly marked guided estimates from reviewed-keyframe interpolation, and one is a low-response lost observation with no box.
- Aircraft, pilot, and launch-cart roles were manually identified from the footage. Neural tracking fills between reviewed selections, with endpoint shape constraints and a blue/pink-wing appearance cue for the shaky water-level view. These counts are not an independently scored tracking accuracy benchmark.
- Reviewed outputs include rear approach, side flight/recovery, the platform cutaway, water-level recovery, overhead replay, frontal replay, and intervening broadcast transitions.
- No pilot identity is assigned to the ambiguous group of swimmers in the water-level recovery view. Explicit splash-hidden intervals remain unboxed until a visible observation reacquires the pilot.
- Mid-interval visual checks caught oversized wing boxes, a fast-pan pilot offset, missed visible pilot intervals, and water-camera shake. The corresponding constraints and keyframes were refined before final rendering.

## Delivered media

All four MP4 files completed full FFmpeg video/audio decoding with **no reported errors**. They contain audio and use H.264 video at 1280×784, approximately 30.014 FPS. The bottom 64 pixels hold the shot name and method/time information, leaving the original picture area free of measurement readouts.

| File | Frames |
|---|---:|
| Flugtag_tracking_full.mp4 | 1,625 |
| Side_view.mp4 | 164 |
| Overhead_view.mp4 | 226 |
| Front_view.mp4 | 320 |

The CSV and JSON observation counts agree; all boxes are inside their recorded 960×540 coordinate space. `reviewed_flight/validation.json` contains the machine-readable media and observation checks.

The separate graph export uses image pixels and video time, including camera motion. This work improves object review/tracking; it does not independently measure the reported 42-foot distance or validate true 3D height.
