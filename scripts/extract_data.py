import os
import h5py
import numpy as np


def _extract_image(image):
    # conversion adapted from here:
    # https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
    rgb_color = np.asarray(image["dataset"], dtype=np.float32)
    render_entity_id = np.asarray(image["dataset"], dtype=np.int32)
    # assert (render_entity_id != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = 90  # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = render_entity_id != -1

    if np.all(valid_mask):
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :,
                                                                                   2]  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_color_tm = np.clip(rgb_color_tm, 0, 1)
    return rgb_color_tm


def _new_path(path):
    new_path, file_name = os.path.split(path)
    new_path = new_path.replace("HyperSim_Data", "HyperSim_Data_extracted")
    os.makedirs(new_path, exist_ok=True)
    new_path = os.path.join(new_path, file_name.replace(".hdf5", ".npy"))
    return new_path


def _image(img_path):
    with h5py.File(img_path, "r") as file:
        image_np = _extract_image(file)
    new_path = _new_path(img_path)
    np.save(new_path, image_np)
    return new_path


def _depth(depth_path):
    with h5py.File(depth_path, "r") as file:
        depth_np = np.asarray(file["dataset"], dtype=np.float32)
    new_path = _new_path(depth_path)
    np.save(new_path, depth_np)


def _seg(seg_path):
    with h5py.File(seg_path, "r") as file:
        seg_np = np.asarray(file["dataset"], dtype=np.float32)
    new_path = _new_path(seg_path)
    np.save(new_path, seg_np)


def _extract_data(path):
    with open(path, "r") as file:
        lines = file.readlines()

    new_path = path.replace("imgPath.txt", "img_path_extracted.txt")

    with open(new_path, "w") as new_file:
        for line in lines:
            img_path = line.strip()
            depth_path = img_path.replace('/image/', '/depth/').replace('_final_hdf5', '_geometry_hdf5').replace(
                'color.hdf5', 'depth_meters.hdf5')
            seg_path = img_path.replace('/image/', '/semantic/').replace('_final_hdf5', '_geometry_hdf5').replace(
                'color.hdf5', 'semantic.hdf5')

            new_path = _image(img_path)
            _depth(depth_path)
            _seg(seg_path)
            print(new_path, file=new_file)


def main():
    train_path = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/train_imgPath.txt"
    val_path = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/val_imgPath.txt"
    test_path = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/test_imgPath.txt"

    _extract_data(train_path)
    _extract_data(val_path)
    _extract_data(test_path)

if __name__ == '__main__':
    main()
