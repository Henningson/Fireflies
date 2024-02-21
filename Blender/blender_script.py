import sys
import subprocess
import os
import platform
import bpy
import bmesh
import yaml
import numpy as np
import cv2
import math

from mathutils import Matrix


# FROM MITSUBA BLENDER
# https://github.com/mitsuba-renderer/mitsuba-blender/blob/master/mitsuba-blender/io/exporter/export_context.py
def transform_matrix(matrix):
    """
    Apply coordinate shift and convert to a mitsuba Transform 4f
    """
    from mitsuba import ScalarTransform4f

    if len(matrix) == 4:
        mat = Matrix() @ matrix
    else:  # 3x3
        mat = matrix.to_4x4()
    return ScalarTransform4f(list([list(x) for x in mat]))


def exportDeformableMesh(path, mesh):
    mesh_path = os.path.join(path, mesh.name)
    try:
        os.mkdir(mesh_path)
    except:
        print("Path {0} does already exist.".format(mesh_path))

    bpy.ops.export_scene.obj(
        filepath=mesh_path + "/" + mesh.name + ".obj",
        use_selection=True,
        use_animation=True,
        axis_forward="Y",
        axis_up="-Z",
    )


def exportCurve(path, curve):
    mesh_path = os.path.join(path, curve.name)
    try:
        os.mkdir(mesh_path)
    except:
        print("Path {0} does already exist.".format(mesh_path))

    bpy.ops.export_scene.obj(
        filepath=mesh_path + "/" + curve.name + ".obj",
        use_selection=True,
        use_animation=False,
        use_nurbs=True,
        axis_forward="Y",
        axis_up="-Z",
    )


def deselectObjects():
    for obj in bpy.data.objects:
        obj.select_set(False)


def mat4x4ToMitsuba(mat):
    init_rot = Matrix.Rotation(np.pi, 4, "Y")
    coordinate_shift = Matrix()
    coordinate_shift[1][1] = 0.0
    coordinate_shift[2][2] = 0.0
    coordinate_shift[2][1] = -1.0
    coordinate_shift[1][2] = 1.0

    return coordinate_shift @ mat @ init_rot


def generateSceneConstraints(context, mitsuba_path):
    last_active_obj = context.view_layer.objects.active

    bpy.context.view_layer.objects.active = None

    base_path = mitsuba_path
    # Check if Firefly is included in path
    if "Firefly" not in base_path:
        base_path = os.path.join(base_path, "Firefly")

    # Try to generate Firefly folderp
    try:
        os.mkdir(base_path)
    except:
        print("Folder {0} exists already.".format(base_path))

    # Deselect all objects
    for obj in bpy.data.objects:
        obj.select_set(False)

    # Iterate over all objects
    for obj in bpy.data.objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        obj_config = {}
        obj_config["animated"] = False
        obj_config["randomizable"] = True
        obj_config["is_relative"] = False
        obj_config["parent_name"] = None
        print(obj.name)

        # Coordinate Shift!
        mitsuba_world_transform = np.array(
            obj.matrix_local
        )  # np.array(mat4x4ToMitsuba(obj.matrix_local))
        print(mitsuba_world_transform)
        obj_config["to_world"] = mitsuba_world_transform.tolist()
        print(obj.name)

        # Check if the mesh is animated, if yes export per Frame OBJ Files
        if obj.type == "MESH":
            if is_deformable(context.scene, obj):
                obj_config["animated"] = True
                print(obj.name + " is animated")
                exportDeformableMesh(base_path, obj)
            else:
                print(obj.name + " is not animated")

        print(obj.type)

        # Gotta fix this
        if obj.type == "CURVE" or obj.type == "SURFACE":
            print("EXPORTING CURVE\n" * 10)
            exportCurve(base_path, obj)

        if obj.parent is not None:
            obj_config["is_relative"] = True
            obj_config["parent_name"] = obj.parent.name
            # obj_config = constraintDictFromRelativeObject(obj_config, obj)
        # else:
        obj_config = constraintDictFromObject(obj_config, obj)

        # Save as yaml file
        path = os.path.join(base_path, obj.name + ".yaml")
        saveToYAML(path, obj_config)
        obj.select_set(False)

    # Hack projector and projector texture into scene xml.
    hacky_xml_append(
        mitsuba_path, math.degrees(bpy.data.objects["Projector"].data.angle)
    )
    # Generate asset folder and initial texture for projector
    generate_init_texture(base_path)

    # Set original active objects back as active obj
    context.view_layer.objects.active = last_active_obj


def constraintDictFromRelativeObject(constraint_dict, obj):
    ############# IMPORTANT ###############
    # CONSTRAINTS ARE ALWAYS RELATIVE TO THE OBJECTS WORLD POSITION AND ROTATION!

    rotationConstraint = getBaseConstraint(0.0, 0.0, 0.0)
    translateConstraint = getBaseConstraint(0.0, 0.0, 0.0)
    scaleConstraint = getBaseConstraint(1.0, 1.0, 1.0)

    constraint_dict["translation"] = translateConstraint
    constraint_dict["rotation"] = rotationConstraint
    constraint_dict["scale"] = scaleConstraint

    return constraint_dict


def constraintDictFromObject(constraint_dict, obj):
    mitsuba_local = obj.matrix_local  # mat4x4ToMitsuba(obj.matrix_local)
    local_rotation = mitsuba_local.to_euler()
    local_translation = mitsuba_local.to_translation()
    local_scale = mitsuba_local.to_scale()

    ############# IMPORTANT ###############
    # CONSTRAINTS ARE ALWAYS RELATIVE TO THE OBJECTS WORLD POSITION AND ROTATION!
    rotationConstraint = getBaseConstraint(0.0, 0.0, 0.0)
    translateConstraint = getBaseConstraint(0.0, 0.0, 0.0)
    scaleConstraint = getBaseConstraint(1.0, 1.0, 1.0)

    # Check if any constraint exists and update the base constraints
    for constraint in obj.constraints:
        if isinstance(constraint, bpy.types.LimitLocationConstraint):
            translateConstraint = getConstraintValues(constraint)
            constraint_dict["randomizable"] = True

        if isinstance(constraint, bpy.types.LimitRotationConstraint):
            rotationConstraint = getConstraintValues(constraint)
            constraint_dict["randomizable"] = True

        if isinstance(constraint, bpy.types.LimitScaleConstraint):
            scaleConstraint = getConstraintValues(constraint)
            constraint_dict["randomizable"] = True

    constraint_dict["translation"] = translateConstraint
    constraint_dict["rotation"] = rotationConstraint
    constraint_dict["scale"] = scaleConstraint

    return constraint_dict


def saveToYAML(path, dict) -> None:
    print(path)
    print(dict)
    with open(path, "w+") as outfile:
        yaml.dump(dict, outfile)


def getBaseConstraint(val0, val1, val2):
    limits = {}
    limits["min_x"] = val0
    limits["max_x"] = val0
    limits["min_y"] = val1
    limits["max_y"] = val1
    limits["min_z"] = val2
    limits["max_z"] = val2

    return limits


def getConstraintValues(constraint):
    limits = {}

    ############# IMPORTANT ###############
    # Y- AND Z-AXIS ARE SWAPPED IN MITSUBA!
    # Z ALSO LOOKS INTO NEGATIVE DIRECTION!
    limits["min_x"] = constraint.min_x
    limits["max_x"] = constraint.max_x
    limits["min_z"] = -constraint.min_y
    limits["max_z"] = -constraint.max_y
    limits["min_y"] = constraint.min_z
    limits["max_y"] = constraint.max_z

    return limits


def is_deformable(scene, obj):
    if scene.frame_start == scene.frame_end:
        return False

    # get the object's evaluated dependency graph:

    scene.frame_set(scene.frame_start)
    depgraph = bpy.context.evaluated_depsgraph_get()
    obj = obj.evaluated_get(depgraph)

    if obj.rigid_body is not None:
        return True

    if obj.soft_body is not None:
        return True

    if "Cloth" in obj.modifiers.keys():
        return True

    return False


def hacky_xml_append(base_path, projector_fov):
    temp_lines = None
    with open(os.path.join(base_path, "scene.xml"), "r+") as f:
        temp_lines = f.readlines()

    temp_lines.insert(-1, '    <texture type="bitmap" id="tex">\n')
    temp_lines.insert(
        -1, '        <string name="filename" value="Firefly/assets/init_tex.png"/>\n'
    )
    temp_lines.insert(-1, "    </texture>\n")
    temp_lines.insert(-1, "\n")
    temp_lines.insert(-1, '    <emitter type="projector">\n')
    temp_lines.insert(-1, '        <ref name="irradiance" id="tex"/>\n')
    temp_lines.insert(
        -1, '        <float name="fov" value="{0}"/>\n'.format(projector_fov)
    )
    temp_lines.insert(
        -1, '        <float name="scale" value="1000.0"/>\n'
    )  # Transform is read from yaml file anyway and is not of interest.
    temp_lines.insert(-1, '        <transform name="to_world">\n')
    temp_lines.insert(-1, '            <lookat origin="0, 0, 4"\n')
    temp_lines.insert(-1, '                    target="0, 0, 0"\n')
    temp_lines.insert(-1, '                    up="0, 1, 0"/>\n')
    temp_lines.insert(-1, "        </transform>\n")
    temp_lines.insert(-1, "    </emitter>\n")

    with open(os.path.join(base_path, "scene.xml"), "w+") as f:
        f.writelines(temp_lines)


# THIS NEEDS TO BE HACKED INTO THE XML!

# INIT TEXTURE
# <texture type="bitmap" id="tex">
#    <string name="filename" value="assets/noise_texture.png"/>
# </texture>


# PROJECTOR
# <emitter type="projector">
#    <ref name="irradiance" id="tex"/>
#    <float name="fov" value="34.0"/>
#    <float name="scale" value="1000.0"/>
#    <transform name="to_world">
#        <lookat origin="0, 0, 4"
#                target="0, 0, 0"
#                up="0, 1, 0"/>
#    </transform>
# </emitter>


def generate_init_texture(base_path: str) -> None:
    assets_path = os.path.join(base_path, "assets")
    try:
        os.mkdir(assets_path)
    except:
        print("Path {0} does already exist.".format(assets_path))

    print(os.path.join(assets_path, "init_tex.png"))
    im = np.random.randint(
        0,
        256,
        size=(
            bpy.data.scenes[0].render.resolution_y,
            bpy.data.scenes[0].render.resolution_x,
        ),
        dtype=np.uint8,
    )
    print(cv2.imwrite(os.path.join(assets_path, "init_tex.png"), im))


#
#
#
#
#
#
#
#
#
#


if __name__ == "__main__":
    obj = bpy.context.active_object
    scene = bpy.context.scene

    generateSceneConstraints(
        bpy.context, "/home/nu94waro/Documents/Vocalfold/DSLPO/scenes/TestbedWithBG/"
    )

# if is_deformable(scene, obj):
#    bpy.ops.export_scene.obj(filepath=file_path, use_selection=True, use_animation=True)
