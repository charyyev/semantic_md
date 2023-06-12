# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# IMPORTANT:  This Apple software is supplied to you by Apple
# Inc. ("Apple") in consideration of your agreement to the following
# terms, and your use, installation, modification or redistribution of
# this Apple software constitutes acceptance of these terms.  If you do
# not agree with these terms, please do not use, install, modify or
# redistribute this Apple software.
#
# In consideration of your agreement to abide by the following terms, and
# subject to these terms, Apple grants you a personal, non-exclusive
# license, under Apple's copyrights in this original Apple software (the
# "Apple Software"), to use, reproduce, modify and redistribute the Apple
# Software, with or without modifications, in source and/or binary forms;
# provided that if you redistribute the Apple Software in its entirety and
# without modifications, you must retain this notice and the following
# text and disclaimers in all such redistributions of the Apple Software.
# Neither the name, trademarks, service marks or logos of Apple Inc. may
# be used to endorse or promote products derived from the Apple Software
# without specific prior written permission from Apple.  Except as
# expressly stated in this notice, no other rights or licenses, express or
# implied, are granted by Apple herein, including but not limited to any
# patent rights that may be infringed by your derivative works or by other
# works in which the Apple Software may be incorporated.
#
# The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
# MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
# THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
# OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
#
# IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
# MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
# AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
# STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# -------------------------------------------------------------------------------
# SOFTWARE DISTRIBUTED IN THIS REPOSITORY:
#
# This software includes a number of subcomponents with separate copyright
# notices and license terms - please see the file ACKNOWLEDGEMENTS.txt
# -------------------------------------------------------------------------------

import os

import numpy as np

import h5py


def _extract_image(image):
    # conversion adapted from here:
    # https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
    # see license above
    rgb_color = np.asarray(image["dataset"], dtype=np.float32)
    render_entity_id = np.asarray(image["dataset"], dtype=np.int32)
    # assert (render_entity_id != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = render_entity_id != -1

    if np.all(valid_mask):
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = (
            0.3 * rgb_color[:, :, 0]
            + 0.59 * rgb_color[:, :, 1]
            + 0.11 * rgb_color[:, :, 2]
        )  # "CCIR601 YIQ" method for computing brightness
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

            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

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
    with open(path, "r", encoding="UTF-8") as file:
        lines = file.readlines()

    _, file_name = os.path.split(path)
    file_name = file_name.replace("imgPath.txt", "img_path_extracted.txt")
    new_dir = "./source/datasets/paths/"
    new_txt_path = os.path.join(new_dir, file_name)

    try:
        with open(new_txt_path, "w", encoding="UTF-8") as new_file:
            for idx, line in enumerate(lines):
                print(idx, line)
                img_path = line.strip()
                depth_path = (
                    img_path.replace("/image/", "/depth/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.hdf5", "depth_meters.hdf5")
                )
                seg_path = (
                    img_path.replace("/image/", "/semantic/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.hdf5", "semantic.hdf5")
                )

                new_path = _image(img_path)
                _depth(depth_path)
                _seg(seg_path)
                print(new_path, file=new_file)
    except FileNotFoundError as e:
        print(e)


def main():
    train_path = "./source/datasets/paths/train_imgPath.txt"
    val_path = "./source/datasets/paths/val_imgPath.txt"
    test_path = "./source/datasets/paths/test_imgPath.txt"

    _extract_data(train_path)
    print("Train done")
    _extract_data(val_path)
    print("Val done")
    _extract_data(test_path)
    print("Done")


if __name__ == "__main__":
    main()
