import os


# python main.py --scene_path "scenes/Vocalfold" --pattern_initialization "RANDOM" --n_beams 75
def generate_general_string(scene_path, pattern, n_beams, lr_laser):
    return f'python main.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --n_beams {n_beams} --lr_laser {lr_laser};'


# python main.py --scene_path "scenes/Vocalfold" --pattern_initialization "SMARTY" --n_beams 75 --max_smarty_radius 35.0
def generate_smarty_string(scene_path, pattern, smarty_max_radius, lr_laser):
    return f'python main.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --n_beams {int(smarty_max_radius)} --smarty_max_radius {smarty_max_radius} --lr_laser {lr_laser};'


def generate_flame_shape_string(scene_path, pattern, lr_laser, n_beams):
    return f'python shapemodel_reconstruction_2.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --lr_laser {lr_laser} --n_beams {n_beams};'


def generate_flame_smarty_string(
    scene_path, pattern, lr_laser, n_beams, smarty_min, smarty_max
):
    return f'python shapemodel_reconstruction_2.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --n_beams {n_beams} --smarty_min_radius {smarty_min} --smarty_max_radius {smarty_max} --lr_laser {lr_laser};'


def generate_flame_eval_string(scene_path, checkpoint_path):
    return f'python shapemodel_reconstruction_2.py --scene_path "{scene_path}" --checkpoint_path "{checkpoint_path}" --eval;'


if __name__ == "__main__":
    GRID_PATTERN = "GRID"
    POISSON_PATTERN = "POISSON"
    SMARTY_PATTERN = "SMARTY"
    UNIFORM_PATTERN = "RANDOM"

    scene_path = "scenes/FlameShape"
    laser_lr_off = "0.00"
    laser_lr_on = "0.001"

    print(generate_smarty_string("scenes/Vocalfold", "SMARTY", 27.5, 27))
    exit()
    N_BEAMS = [16, 25, 36, 49, 64, 81]
    SMARTY_INTERVALS = [
        (35.0, 50.0),
        (30.0, 45.0),
        (25.0, 40.0),
        (15.0, 32.5),
        (12.5, 29.0),
        (10.0, 25.0),
    ]

    PATTERNS = [UNIFORM_PATTERN]
    LASER_OFF_ON = [laser_lr_off, laser_lr_on]
    checkpoint_paths = [
        "2024-03-06-23:09:37_GRID_2500_100_0.0",
        "2024-03-06-23:16:21_GRID_2500_100_0.001",
        "2024-03-06-23:23:06_POISSON_2500_100_0.0",
        "2024-03-06-23:29:52_POISSON_2500_100_0.001",
        "2024-03-06-23:36:37_SMARTY_2500_100_0.0",
        "2024-03-06-23:43:24_SMARTY_2500_100_0.001",
        "2024-03-06-23:50:10_RANDOM_2500_100_0.0",
        "2024-03-06-23:56:56_RANDOM_2500_100_0.001",
    ]

    streng = ""
    count = 0

    """
    for laser_status in LASER_OFF_ON:
        for i in range(len(N_BEAMS)):
            streng += generate_flame_smarty_string(
                scene_path,
                SMARTY_PATTERN,
                laser_status,
                N_BEAMS[i],
                SMARTY_INTERVALS[i][0],
                SMARTY_INTERVALS[i][1],
            )
    """

    for pattern in PATTERNS:
        for laser_status in LASER_OFF_ON:
            for n_beams in N_BEAMS:
                streng += generate_flame_shape_string(
                    scene_path, pattern, laser_status, n_beams
                )
                count += 1
    streng += ""
    print(f"Time Estimate: {count * 7 / 60:.5f} Hours")
    print()
    print(streng)

    exit()

    for checkpoint_path in checkpoint_paths:
        eval_string = generate_flame_eval_string(
            scene_path, os.path.join(scene_path, "optim", checkpoint_path)
        )
        streng += eval_string
        print(eval_string)

    exit()

    print(streng)

    exit()
    blue_noise_points = [100, 150, 200, 250, 300, 350, 400]
    grid_points = [100, 144, 196, 256, 324, 361, 400]
    random_points = [100, 150, 200, 250, 300, 350, 400]
    smarty_radii = [15.0, 20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]

    vf_path = "scenes/Vocalfold"
    colon_path = "scenes/RealColon"

    streng = ""
    count = 0

    ### VOCALFOLD STRINGS
    # for points in blue_noise_points:
    #     streng += generate_general_string(
    #         vf_path, POISSON_PATTERN, points, laser_lr_off
    #     )
    #     count += 1

    for points in blue_noise_points:
        streng += generate_general_string(vf_path, POISSON_PATTERN, points, laser_lr_on)
        count += 1

    # for points in grid_points:
    #     streng += generate_general_string(vf_path, GRID_PATTERN, points, laser_lr_off)
    #     count += 1

    for points in grid_points:
        streng += generate_general_string(vf_path, GRID_PATTERN, points, laser_lr_on)
        count += 1

    # for points in random_points:
    #     streng += generate_general_string(
    #         vf_path, UNIFORM_PATTERN, points, laser_lr_off
    #     )
    #     count += 1

    for points in random_points:
        streng += generate_general_string(vf_path, UNIFORM_PATTERN, points, laser_lr_on)
        count += 1

    # for radii in smarty_radii:
    #     streng += generate_smarty_string(vf_path, SMARTY_PATTERN, radii, laser_lr_off)
    #     count += 1

    for radii in smarty_radii:
        streng += generate_smarty_string(vf_path, SMARTY_PATTERN, radii, laser_lr_on)
        count += 1

    ### COLON STRINGS
    for points in blue_noise_points:
        streng += generate_general_string(
            colon_path, POISSON_PATTERN, points, laser_lr_off
        )
        count += 1

    for points in blue_noise_points:
        streng += generate_general_string(
            colon_path, POISSON_PATTERN, points, laser_lr_on
        )
        count += 1

    for points in grid_points:
        streng += generate_general_string(
            colon_path, GRID_PATTERN, points, laser_lr_off
        )
        count += 1

    for points in grid_points:
        streng += generate_general_string(colon_path, GRID_PATTERN, points, laser_lr_on)
        count += 1

    for points in random_points:
        streng += generate_general_string(
            colon_path, UNIFORM_PATTERN, points, laser_lr_off
        )
        count += 1

    for points in random_points:
        streng += generate_general_string(
            colon_path, UNIFORM_PATTERN, points, laser_lr_on
        )
        count += 1

    for radii in smarty_radii:
        streng += generate_smarty_string(
            colon_path, SMARTY_PATTERN, radii, laser_lr_off
        )
        count += 1

    for radii in smarty_radii:
        streng += generate_smarty_string(colon_path, SMARTY_PATTERN, radii, laser_lr_on)
        count += 1

    print(count)
    print()
    print()
    print()
    print(streng)
