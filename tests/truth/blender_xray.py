# Blender truth for `tests/xray.rs`: Workbench (Solid) with X-ray, rendered
# by Blender 5.0 itself. Two walls face a perspective camera — a red one in
# front, a blue one behind it — lit by the default studio rig, Standard view
# transform. Each case is an X-ray alpha (1 is X-ray off); the pixels are
# where only the front wall, both walls, only the back wall and nothing is
# seen. Run with bpy 5.0 (`pip install bpy`), headless:
#
#   python3 tests/truth/blender_xray.py tests/truth/xray.txt
import bpy, sys

W, H = 80, 60
# (x, y) pixels, top row 0: the front wall alone, both, the back wall alone, the world.
SPOTS = {"front": (17, 30), "both": (40, 30), "back": (60, 30), "world": (2, 2)}


def wall(name, x, z, y, color):
    (x0, x1), (z0, z1) = x, z
    mesh = bpy.data.meshes.new(name)
    # Wound so the normal faces the camera (−Y).
    mesh.from_pydata([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, mesh)
    ob.color = color
    bpy.context.scene.collection.objects.link(ob)
    return ob


out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else sys.argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
front = wall("Front", (-2.5, 1.0), (-1.5, 1.5), 0.0, (0.8, 0.2, 0.2, 1.0))
back = wall("Back", (-1.0, 3.0), (-1.2, 1.2), 2.0, (0.2, 0.3, 0.8, 1.0))
normal = front.data.polygons[0].normal
assert normal.y < -0.99, normal
cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
cam.location = (0.0, -10.0, 0.0)
cam.rotation_euler = (1.5707963267948966, 0.0, 0.0)
cam.data.lens, cam.data.sensor_width, cam.data.sensor_fit = 50.0, 36.0, "HORIZONTAL"
scene.collection.objects.link(cam)
scene.camera = cam
scene.world = bpy.data.worlds.new("World")
scene.world.color = (0.05, 0.05, 0.05)
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
scene.display.render_aa = "OFF"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
shading = scene.display.shading
shading.light, shading.color_type = "STUDIO", "OBJECT"
shading.show_specular_highlight = True
shading.show_shadows = shading.show_cavity = shading.show_object_outline = False
lines = ["# Blender %s Workbench, X-ray (blender_xray.py): alpha spot r g b" % bpy.app.version_string]
for alpha in (1.0, 0.5, 0.2):
    shading.show_xray = alpha < 1.0
    shading.xray_alpha = alpha
    scene.render.filepath = "/tmp/phiray_xray_%d.png" % int(alpha * 10)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(scene.render.filepath)
    px = img.pixels[:]
    for name, (x, y) in SPOTS.items():
        i = ((H - 1 - y) * W + x) * 4
        rgb = [round(px[i + k] * 255) for k in range(3)]
        lines.append("%.1f %s %d %d %d" % (alpha, name, *rgb))
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
