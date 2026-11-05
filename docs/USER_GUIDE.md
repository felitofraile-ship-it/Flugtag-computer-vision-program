# FlightLab 2 — object tracking first

The current version prioritizes identifying and following the aircraft, pilot, and launch cart. The overhead video has boxes only. The side video adds short image-trajectory lines. Measurements and motion graphs are separate.

## Generate reviewed tracking results

The source video and rendered footage are not distributed in the public repository. Supply your own matching clip as `media/flight.mp4` and run the review command below to generate these files in `reviewed_flight/`:

- **Flugtag_tracking_full.mp4** — the entire 54-second clip, including camera changes, with original audio and tracking overlays.
- **Side_view.mp4** — launch, descent, impact, and early recovery.
- **Overhead_view.mp4** — aircraft and pilot tracked separately through the overhead replay; launch cart has its own ID when visible.
- **Front_view.mp4** — frontal launch and impact replay.
- **motion_graphs.png / .pdf** — separate horizontal/vertical image-position graphs for aircraft and pilot, split by angle.
- **review_tracks.csv / .json** — frame-level boxes, track IDs, tracking responses, methods, and review flags.

The roles `Aircraft`, `Pilot`, and `Launch cart` were identified from the supplied footage and manually selected at review keyframes. Bidirectional NanoTrack follows the subject between those points. Reviewed endpoint box shapes constrain drift, especially for wide wings. Where neural tracking has low response inside a visible interval, interpolation is explicitly labeled in the data. This is **keyframe-assisted tracking**, not a claim of fully automatic role recognition.

The platform cutaway does not show the aircraft/pilot. The water-level recovery shot does not confidently establish which swimmer is the pilot. Those shots are retained without assigning an unsupported pilot identity. Explicitly hidden intervals have no pilot box; tracks are restarted from reviewed observations when the pilot is visible again. A partial-body box tracks the visible portion, not an inferred body center.

## Use the live desktop app

1. Place the original matching clip at `media/flight.mp4` and double-click **Open Prepared Demo.cmd**. It opens the overhead view with **two separate tracks: Aircraft and Pilot**. Press **Play**. This mode computes tracking as it plays, independently of the prepared review.
2. Use **Start FlightLab.cmd** and **Open video** for a fresh session or another clip.
3. Pause on a clear frame. Give an object a label and **Draw box**. Repeat for the pilot, aircraft, or another component.
4. **Identify objects** uses the bundled YOLOX detector to suggest common classes. It can recognize people, boats, and other trained classes, but does not know who has the role of pilot or recognize this custom craft reliably. For a role such as Pilot, select the correct subject and label it yourself.
5. Choose **Top view** for boxes only or **Side view** for boxes plus short trails. `Tracking` is also boxes only. The detector's analysis region can exclude broadcast graphics and the picture-in-picture.
6. If a box drifts, pause, select its row, click **Correct selected box**, and redraw it. **The same track ID is retained.** Correction events are saved in exported JSON. **Stop selected** freezes that endpoint.
7. Optionally enable **Refresh recognized people every 15 frames** after first clicking Identify objects. It associates nearby overlapping person detections with existing person/pilot tracks; it does not guarantee identity through crowd crossings or long occlusion. This costs CPU time.

NanoTrack handles people and compact objects. CSRT handles very wide selections because those are a poor match for NanoTrack's square search crop. Up to eight active tracks are supported. Low-response/lost objects stay unboxed. Track response is not an identity confidence or calibrated probability. Shape changes and splashes still benefit from correction.

The app shows processed FPS. It processes consecutive frames rather than silently skipping measurement frames, so actual playback speed depends on CPU load. Full 30-FPS live performance is not guaranteed. The prepared MP4s play normally because their tracking is already computed.

The tested two-object overhead segment processed at about 22.5 FPS on this computer. See VALIDATION.md for the checks and their limits.

## Measurements are separate

Click **Measurements & graphs**. Select an object in the main tracking table. The graph window offers image horizontal/vertical position, planar displacement, calibrated center height, and planar speed. Calibration and timing controls are on its second tab.

The 20-foot wingspan and 30-foot platform height remain available as scale presets. The reported 42-foot pilot landing distance is an independent reference, not a fitted tracking result. Mark a known reference and, for height, a zero level/vertical reference. Set a verified replay timing factor before tracking if physical speed is needed.

**Camera-calibration failure no longer stops object tracking.** Calibrated measurements are withheld when their reference is invalid, while image-coordinate plots remain available. Camera correction and calibration lines are hidden from the main video unless **Show selection / calibration guides** is enabled.

The review renderer's `motion_graphs` outputs use **image pixels and video time**, including camera motion. The separate `measurements/` example uses approximate feet; see its README for assumptions. They do not claim calibrated flight distance, true altitude, or physical replay speed. Gaps indicate missing observations. Real-world values remain approximate planar measurements when calibrated; 3D reconstruction is not implemented.

## Save and export

- **Export CSV + report + image** saves a timestamped segment report, trajectory CSV, correction events, and overlay image.
- **Record overlay MP4** records the live app's overlays as you process frames; that live export is silent. The separately prepared full review retains the source audio.
- Starting a new segment, jumping, opening another video, or closing archives existing tracks under `runs/`.
- A cut stops live tracking for a fresh selection; known shots are already separated in the prepared full-video review.

## Reproduce or change the prepared review

`flugtag_review_keyframes.json` contains the editable, reviewed selections. Its coordinates use a 640×360 image. Role IDs are 1 = aircraft, 2 = pilot, 3 = launch cart. A null box explicitly marks a hidden interval; the next visible box reacquires the subject.

```powershell
python -m pip install -r requirements.txt
python review_video.py --project flugtag_review_keyframes.json --output reviewed_flight
```

The review caches 960×540 frames under `work/flightlab_frames` in the working directory to avoid repeatedly decoding the source. It writes a silent intermediate under `work/review_render`, encodes the final videos with bundled FFmpeg from `imageio-ffmpeg`, and plots the separate graphs with Matplotlib. Video analysis is local; this program does not upload video. To change video paths, edit the project JSON's `video` value.

The water-level shot additionally uses a blue/pink-wing color cue to follow this particular craft through camera shake. It is a clip-specific appearance cue, not a general aircraft classifier. The CSV/JSON method field records its use.

## Installation and source

Use Python 3.11–3.13 with Tkinter. The Windows launcher creates `.venv` on first launch; initial dependency installation uses PyPI. Model files are bundled, and subsequent analysis is local/offline. The original source clip is referenced by path rather than duplicated in the package.

- `app.py`: tracking-first desktop interface, corrections, separate graph window.
- `engine.py`: independent object tracking and measurement state.
- `tracking.py`: NanoTrack / CSRT selection and YOLOX class detection.
- `review_video.py`: bidirectional, keyframe-assisted full-video processing and exports.
- `test_engine.py`: regression tests; run `python -m unittest -v`.
- `DESIGN.md`: plan for future calibrated multi-view measurement.
- `measurements/`: reproducible approximate distance/height graphs; the pilot endpoint uses 42 ft as a calibration input.

Model references and licenses: [NanoTrack](https://github.com/HonglinChu/NanoTrack), [OpenCV TrackerNano](https://docs.opencv.org/4.11.0/d5/d9b/structcv_1_1TrackerNano_1_1Params.html), and [OpenCV Zoo YOLOX](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox). Corresponding licenses are included in `models/`. MobileNet-SSD remains available only as a fallback for distributions without the YOLOX model.
