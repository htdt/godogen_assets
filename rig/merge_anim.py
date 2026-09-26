"""
Attach the animation from Make-It-Animatable's FBX to its rigged rest.glb and export an animated GLB.

    asset-blender rig/merge_anim.py -- rest.glb anim.fbx out.glb

MIA's own GLB (made with FBX2glTF) loses the metallic/roughness map and renders glossy; rest.glb keeps the PBR
material. Both files come from the same Blender scene, so bone names and rest poses match and the action can be
reused directly; only root-motion translation is rescaled if the two armatures use different units.
"""
import bpy
import sys

argv = sys.argv[sys.argv.index('--') + 1:]
rest_glb, anim_fbx, out = argv
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=rest_glb, disable_bone_shape=True)
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
before = set(bpy.context.scene.objects)
bpy.ops.import_scene.fbx(filepath=anim_fbx)
fbx_objs = set(bpy.context.scene.objects) - before
src = next(o for o in fbx_objs if o.type == 'ARMATURE')
action = src.animation_data.action


def hips_len(a):
    # head-to-head distance (FBX has no bone tails, so bone lengths are not comparable between importers)
    bones = {b.name.split(':')[-1]: b for b in a.data.bones}
    return (bones['Neck'].head_local - bones['Hips'].head_local).length


# location keys are in armature-data units; both armatures come from the same scene (same world size), so the
# ratio of a bone's data-space length converts source units to target units
ratio = hips_len(arm) / hips_len(src)
for fc in action.fcurves:
    if fc.data_path.endswith('.location'):
        for kp in fc.keyframe_points:
            kp.co[1] *= ratio
            kp.handle_left[1] *= ratio
            kp.handle_right[1] *= ratio
for o in fbx_objs:
    bpy.data.objects.remove(o)
arm.animation_data_create()
arm.animation_data.action = action
if hasattr(action, 'slots') and action.slots and arm.animation_data.action_slot is None:
    arm.animation_data.action_slot = action.slots[0]  # Blender 4.4+: the slot was created for the FBX armature
f0, f1 = action.frame_range
bpy.context.scene.frame_start, bpy.context.scene.frame_end = int(f0), int(f1)
bpy.ops.export_scene.gltf(filepath=out, export_animations=True, export_animation_mode='ACTIONS')
print('WROTE', out, 'frames', int(f0), int(f1), 'root scale', round(ratio, 4))

if 'asset_result' in globals():
    asset_result({'output': out})
