# python main.py --scene_path "scenes/Vocalfold" --pattern_initialization "RANDOM" --n_beams 75
def generate_general_string(scene_path, pattern, n_beams, lr_laser):
    return f'python main.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --n_beams {n_beams} --lr_laser {lr_laser};'


# python main.py --scene_path "scenes/Vocalfold" --pattern_initialization "SMARTY" --n_beams 75 --max_smarty_radius 35.0
def generate_smarty_string(scene_path, pattern, smarty_max_radius, lr_laser):
    return f'python main.py --scene_path "{scene_path}" --pattern_initialization "{pattern}" --n_beams {int(smarty_max_radius)} --smarty_max_radius {smarty_max_radius} --lr_laser {lr_laser};'


if __name__ == "__main__":
    blue_noise_points = [100, 150, 200, 250, 300, 350, 400]
    grid_points = [100, 144, 196, 256, 324, 361, 400]
    random_points = [100, 150, 200, 250, 300, 350, 400]
    smarty_radii = [15.0, 20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]

    vf_path = "scenes/Vocalfold"
    colon_path = "scenes/RealColon"

    laser_lr_off = "0.00"
    laser_lr_on = "0.001"

    GRID_PATTERN = "GRID"
    POISSON_PATTERN = "POISSON"
    SMARTY_PATTERN = "SMARTY"
    UNIFORM_PATTERN = "RANDOM"

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
