"""Reproduce the approximate Flugtag plots from reviewed video points.

Run: python make_graphs.py
Requires numpy and matplotlib. Raw reference points are in reviewed_points.json;
camera transforms map the 960x540 source frames to the 11-second reference view.
This is a calibrated 2-D approximation, not a 3-D reconstruction.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
FPS = 30.01366776255
CAMERA = {r["time"]: r for r in json.loads((ROOT / "camera_transforms.json").read_text())}
POINTS = json.loads((ROOT / "reviewed_points.json").read_text())
PLATFORM_BASE = np.array([231., 467.])
PLATFORM_TOP = np.array([326., 112.])
PLATFORM_HEIGHT_FT = 30.
PILOT_RANGE_FT = 42.
T0 = CAMERA[10.2]["frame"] / FPS


def stabilize(video_time, point):
    # Reviewed coordinates were picked on 640x360 frames; H uses 960x540.
    p = np.r_[np.asarray(point, dtype=float) * 1.5, 1.]
    q = np.asarray(CAMERA[video_time]["H"]) @ p
    return q[:2] / q[2]


LANDING = stabilize(12.4, [499., 198.])
# Assumed flight plane: deck is 30 ft above its waterline; landing is 42 ft
# horizontally from that launch edge. A single video does not verify this plane.
BASIS = np.column_stack(((LANDING - PLATFORM_BASE) / PILOT_RANGE_FT,
                         (PLATFORM_TOP - PLATFORM_BASE) / PLATFORM_HEIGHT_FT))


def make_row(t, object_name, point):
    stabilized = stabilize(t, point)
    distance, height = np.linalg.solve(BASIS, stabilized - PLATFORM_BASE)
    video_time = CAMERA[t]["frame"] / FPS
    return dict(object=object_name, video_time_s=video_time,
                time_after_takeoff_s=video_time - T0,
                distance_ft=float(distance), height_ft=float(height),
                source_frame=CAMERA[t]["frame"],
                source_x_px=point[0] * 1.5, source_y_px=point[1] * 1.5,
                stabilized_x_px=float(stabilized[0]),
                stabilized_y_px=float(stabilized[1]),
                status="reviewed_video_point" if t <= 11.5 else "splash_affected_estimate")


pilot, craft_raw = [], []
for time, craft_point, pilot_point in POINTS:
    if craft_point is not None:
        craft_raw.append(make_row(time, "Craft", craft_point))
    if pilot_point is not None:
        pilot.append(make_row(time, "Pilot", pilot_point))

# End the craft curve at the inferred wing-center water crossing. Do not plot
# negative physical heights or invent a stationary craft during the pilot fall.
craft = []
for row in craft_raw:
    if row["height_ft"] >= 0:
        craft.append(row.copy())
    else:
        previous = craft[-1]
        fraction = previous["height_ft"] / (previous["height_ft"] - row["height_ft"])
        endpoint = {"object": "Craft", "source_frame": "",
                    "source_x_px": "", "source_y_px": "",
                    "status": "water_entry_interpolated"}
        for key in ("video_time_s", "time_after_takeoff_s", "distance_ft", "height_ft",
                    "stabilized_x_px", "stabilized_y_px"):
            endpoint[key] = previous[key] + fraction * (row[key] - previous[key])
        endpoint["height_ft"] = 0.
        craft.append(endpoint)
        break

pilot[-1]["distance_ft"] = PILOT_RANGE_FT
pilot[-1]["height_ft"] = 0.
pilot[-1]["status"] = "reported_42ft_calibration_at_estimated_water_entry"

# Use the front edge following the user's clarified range convention. Preserve
# the center comparison; do not invent a water crossing after splash hides it.
craft_center = craft
craft = [make_row(t, "Craft front edge", p) for t, p in
         json.loads((ROOT / "craft_front_points.json").read_text())]
craft[-1]["status"] = "last_visible_front_edge_partial_splash"

# Calibration and finite-output checks, not assertions of physical accuracy.
assert np.allclose(np.linalg.solve(BASIS, PLATFORM_TOP - PLATFORM_BASE), [0, 30])
assert np.allclose(np.linalg.solve(BASIS, LANDING - PLATFORM_BASE), [42, 0])
for series in (pilot, craft):
    assert all(np.isfinite([r["distance_ft"], r["height_ft"]]).all() for r in series)
    assert all(b["time_after_takeoff_s"] > a["time_after_takeoff_s"] for a, b in zip(series, series[1:]))
    assert all(r["height_ft"] >= 0 for r in series)
assert craft[-1]["time_after_takeoff_s"] < pilot[-1]["time_after_takeoff_s"]

with (ROOT / "distance_height_data.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(pilot[0]))
    writer.writeheader()
    for r in sorted(craft + pilot, key=lambda r: (r["time_after_takeoff_s"], r["object"])):
        writer.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})

COLORS = {"Craft": "#2563EB", "Pilot": "#D7770B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.labelcolor": "#24334A", "text.color": "#18283D",
                     "xtick.color": "#48556A", "ytick.color": "#48556A",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#C6CED8", "axes.linewidth": .8,
                     "savefig.facecolor": "white"})


def draw_graph(quantity):
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=180)
    fig.subplots_adjust(left=.10, right=.96, bottom=.27, top=.80)
    fig.text(.10, .94, "FLUGTAG  /  SIDE-VIEW RECONSTRUCTION", fontsize=11,
             color="#627189", weight="bold")
    title = "Distance vs time" if quantity == "distance_ft" else "Height vs time"
    fig.text(.10, .88, title, fontsize=25, weight="bold")
    fig.text(.10, .835, "Craft front edge and pilot • approximate side-view reconstruction", color="#627189", fontsize=11)
    for name, series in [("Craft", craft), ("Pilot", pilot)]:
        # Solid through the final clear, seated observation; dashed during splash.
        clear_end = max(i for i, r in enumerate(series) if r["status"] == "reviewed_video_point")
        for part, style in [(series[:clear_end + 1], "-"), (series[clear_end:], (0, (5, 3)))]:
            ax.plot([r["time_after_takeoff_s"] for r in part], [r[quantity] for r in part],
                    linestyle=style, color=COLORS[name], lw=2.6, marker="o", ms=4.4,
                    markeredgecolor="white", markeredgewidth=.7, zorder=4)
        ax.scatter(series[-1]["time_after_takeoff_s"], series[-1][quantity],
                   s=72, color=COLORS[name], edgecolors="white", linewidths=1.2, zorder=6)

    ax.set_xlim(-.05, 2.48)
    ax.set_xticks(np.arange(0, 2.41, .3))
    ax.set_xlabel("Time after takeoff (seconds)", labelpad=12)
    ax.grid(axis="both", color="#E7ECF2", linewidth=.7, zorder=0)
    ax.tick_params(length=0, pad=8)
    handles = [Line2D([], [], color=COLORS[n], lw=2.6,
                     label="Craft — front edge" if n == "Craft" else "Pilot") for n in ("Craft", "Pilot")]
    handles.append(Line2D([], [], color="#6C788A", linestyle=(0, (5, 3)), lw=2, label="Splash / estimated"))
    ax.legend(handles=handles, loc="upper left" if quantity == "distance_ft" else "upper right",
              frameon=False, fontsize=10, handlelength=2.3)

    if quantity == "distance_ft":
        ax.set_ylim(-6, 48)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("Horizontal distance from launch edge (feet)", labelpad=12)
        ax.axhline(0, color="#9BA8BA", lw=1)
        ax.annotate("Pilot: 42 ft\nreported landing distance", (pilot[-1]["time_after_takeoff_s"], 42),
                    xytext=(1.57, 45), fontsize=11, color=COLORS["Pilot"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["Pilot"], lw=1), ha="left")
        ax.annotate("Craft front: ≈37–38 ft\nlast visible in impact splash",
                    (craft[-1]["time_after_takeoff_s"], craft[-1]["distance_ft"]),
                    xytext=(1.53, 28), fontsize=10, color=COLORS["Craft"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["Craft"], lw=1))
        note = "Pilot endpoint fixed at the reported 42 ft; craft front estimated from the visible edge.\nThe craft line ends when splash obscures that edge. This is horizontal range, not path length."
    else:
        ax.set_ylim(-2, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("Height above water (feet)", labelpad=12)
        ax.axhline(0, color="#9BA8BA", lw=1.1)
        ax.axhline(30, color="#B8C2D0", lw=1, linestyle=(0, (3, 4)))
        ax.text(.88, 30.9, "Launch deck: 30 ft", color="#778396", fontsize=10)
        ax.annotate("Front edge obscured\nby splash after this point", (craft[-1]["time_after_takeoff_s"], craft[-1]["height_ft"]),
                    xytext=(.52, 6.8), fontsize=10, color=COLORS["Craft"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["Craft"], lw=1))
        ax.annotate("Pilot enters water", (pilot[-1]["time_after_takeoff_s"], 0),
                    xytext=(1.90, 8), fontsize=10, color=COLORS["Pilot"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["Pilot"], lw=1))
        note = "Reference points: forwardmost visible craft edge and pilot hip / lower torso.\nCraft height stops at its last visible point; a zero-height landing is not inferred through splash."
    fig.text(.10, .105, note, fontsize=10, color="#526176", linespacing=1.65)
    fig.text(.10, .045, "Time zero ≈ 10.20 s in the original video • Camera motion corrected • Approximate 2-D scale",
             fontsize=9, color="#7C8798")
    stem = "distance_vs_time" if quantity == "distance_ft" else "height_vs_time"
    fig.savefig(ROOT / f"{stem}.png", dpi=180)
    fig.savefig(ROOT / f"{stem}.svg")
    return fig


with PdfPages(ROOT / "Flugtag_distance_and_height.pdf") as pdf:
    for quantity in ("distance_ft", "height_ft"):
        figure = draw_graph(quantity)
        pdf.savefig(figure)
        plt.close(figure)

summary = {"fps": FPS, "time_zero_video_s": T0,
           "platform_top_reference_px": PLATFORM_TOP.tolist(),
           "platform_base_reference_px": PLATFORM_BASE.tolist(),
           "pilot_landing_reference_px": LANDING.tolist(),
           "plane_basis_pixels_per_foot": BASIS.tolist(),
           "craft_front_last_visible": craft[-1], "pilot_water_entry": pilot[-1],
           "craft_center_water_entry_for_comparison": craft_center[-1],
           "checks": "Finite values, increasing time, nonnegative plotted heights, and 30/42 ft calibration anchors passed."}
(ROOT / "calibration_and_checks.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"craft_range_ft": craft[-1]["distance_ft"],
                  "craft_time_s": craft[-1]["time_after_takeoff_s"],
                  "pilot_range_ft": pilot[-1]["distance_ft"],
                  "pilot_time_s": pilot[-1]["time_after_takeoff_s"]}, indent=2))
