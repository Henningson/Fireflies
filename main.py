

#@dr.wrap_ad(source="torch", target="drjit")
def render_scene(scene, spp=256, seed=1):
    # TODO: Implement me
    return None

def render_depth_map(scene, spp=256, seed=1):
    # TODO: Implement me
    return None

def get_gt_surface(scene):
    #TODO: Implement me
    return None

def randomize_scene(scene, scene_args, objs):
    # TODO: Implement me
    return None

def triangulate_surface(laser, render, camera):
    # TODO: Implement me
    return None


def compute_depth_map(imageA, imageB, calibration_data):
    # TODO: Implement me
    return None




def convert_to_image(render):
    # TODO: Implement me
    return None

def l2_loss(a, b):
    return None


def main():
    # Allow argument parsing here.
    # We can easily reuse the previous argparser here.


    # Load initial stuff
    scene = None
    scene_args = None
    objects = None
    proj_pattern = None
    calibration_data = None
    iterations = 100

    # Initialize _MITSUBA_ optimizer here, since the item we want to optimize is a mitsuba object.
    optimizer = None

    # Using depth map and block matching
    for epoch in range(iterations):
        rand_scene = randomize_scene(scene)
        render = render_scene(scene)
        gt_depth_map = render_depth_map(scene)
        image = convert_to_image(render)
        estim_depth_map = compute_depth_map(image, proj_pattern, calibration_data)
        loss = l2_loss(gt_depth_map, estim_depth_map)
        loss.backward()

    # Using laser and direct transform, omitting correspondence estimation
    for epoch in range(iterations):
        rand_scene = randomize_scene(scene)
        render = render_scene(scene)
        gt_surface = get_gt_surface(scene)
        



    pass






if __name__ == "__main__":
    main()







