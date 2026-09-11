"""Render the obj2mjcf-converted asset offscreen: turntable hero + physics-drop video."""

import os
import subprocess
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from PIL import Image

from obj2mjcf import convert

OUT = Path("/home/robot/workspace/46-marine/artifacts/reports/obj2mjcf-usd-media")
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/o2m_render")
WORK.mkdir(exist_ok=True)

# 1) Convert the asset (MJCF + physics USD), the real pipeline output.
obj = WORK / "groups.obj"
obj.write_bytes(Path("tests/groups.obj").read_bytes())
out = convert(
    obj, export=("mjcf", "usd"), usd_physics=True, add_free_joint=True,
    density=300.0, usd_binary=False, validate=True,
)
print("converted:", {k: [p.name for p in v] for k, v in out.items()})

# 2) Wrapper scene: floor + sky + light, including the generated asset MJCF.
scene_xml = """
<mujoco model="obj2mjcf_demo">
  <compiler meshdir="groups" angle="radian"/>
  <include file="groups/groups.xml"/>
  <statistic center="0 0 1.8" extent="11"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <rgba haze="0.16 0.20 0.26 1"/>
    <global azimuth="135" elevation="-22" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.32 0.42" rgb2="0.05 0.07 0.10"
             width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.28"
             rgb2="0.27 0.30 0.35" width="512" height="512" mark="edge" markrgb="0.4 0.4 0.45"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light pos="4 -4 8" dir="-0.4 0.4 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid"/>
  </worldbody>
  <keyframe>
    <key name="drop" qpos="0 0 6.5 0.924 0.383 0 0"/>
  </keyframe>
</mujoco>
"""
scene_path = out["mjcf"][0].parent.parent / "scene.xml"
scene_path.write_text(scene_xml)
model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
data = mujoco.MjData(model)
W, H = 1280, 720
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 1.8]


def save_mp4(frames, name, fps):
    fdir = WORK / name
    fdir.mkdir(exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(fdir / f"{i:04d}.png")
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{fdir}/%04d.png",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         mp4.as_posix()],
        check=True, capture_output=True,
    )
    print("wrote", mp4, mp4.stat().st_size // 1024, "KB")


# 3) Turntable hero (static asset resting on the floor).
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
cam.distance = 16.0
cam.elevation = -18
turn = []
for i in range(72):
    cam.azimuth = 360.0 * i / 72
    renderer.update_scene(data, camera=cam)
    turn.append(renderer.render())
Image.fromarray(turn[10]).save(OUT / "hero.png")
save_mp4(turn, "turntable", 24)
print("hero.png", (OUT / "hero.png").stat().st_size // 1024, "KB")

# 4) Physics drop (real simulation of the free-joint asset settling on the floor).
mujoco.mj_resetDataKeyframe(model, data, 0)
cam.distance = 18.0
cam.elevation = -16
drop = []
steps_per_frame = max(1, int((1.0 / 30) / model.opt.timestep))
for f in range(120):
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)
    cam.azimuth = 130 + 0.25 * f
    renderer.update_scene(data, camera=cam)
    drop.append(renderer.render())
save_mp4(drop, "physics_drop", 30)
print("DONE")
