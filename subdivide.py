import pathlib
import time
import os
import math
import yaml
import warnings
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndi
from utils import BASE_PROJECT_DIR


def reg_method(im2reg, shift):
    reg = ndi.shift(im2reg, shift)
    return reg


def register_and_crop(img_list):
    """
    img_list: list of filenames
    """
    reg_img = list()
    shifts = np.empty(shape=(len(img_list), 2))
    im_ref = skio.imread(img_list[0])
    for idx, fn in enumerate(img_list):
        im2reg = skio.imread(fn)
        shift, error, diffphase = phase_cross_correlation(
            im_ref, im2reg, upsample_factor=10
        )
        print(f"IR {idx+2},Detected pixel offset (y, x): {shift}")
        shifts[idx, :] = shift
        reg_img.append(reg_method(im2reg, shift))

    dx_tl = np.max(np.maximum(shifts[:, 1], 0))
    dy_tl = np.max(np.maximum(shifts[:, 0], 0))

    dx_br = np.min(np.minimum(shifts[:, 1], 0))
    dy_br = np.min(np.minimum(shifts[:, 0], 0))
    print(f"tl:= dx {dx_tl}, dy {dy_tl}")
    print(f"br:= dx {dx_br}, dy {dy_br}")

    tl = np.ceil([dy_tl, dx_tl]).astype(int)
    br = np.floor([im_ref.shape[0] + dy_br, im_ref.shape[1] + dx_br]).astype(int)
    print(f"new topleft: y {tl[0]} x {tl[1]}")
    print(f"new bottomright: y {br[0]} x {br[1]}")

    crop_shift = list()
    for idx, img in enumerate(reg_img):
        crop_shift.append(img[tl[0] : br[0], tl[1] : br[1]])

    return crop_shift


def anglerfish_chop(anglerfish_dir_path: pathlib.Path, tile_list: list, size: int = 40):
    """Splits an anglerfish simulated images and associated data to (size x size) tiles.

    Args:
        anglerfish_dir_path (pathlib.Path): path to the top level folder that contains simulated images
        tile_list (list[int]): list of tile numbers to split i.e. [1,3,7]
        size (int, optional): size of the chopped tile. Defaults to 40.
    """
    wavelengths = [473, 561, 647, 750]
    num_rounds = 8

    # Set-up the directories
    t = time.strftime("%Y%m%d-%H-%M-%S")
    savepath = BASE_PROJECT_DIR / "results" / f"anglerfish_chopped_{t}" / "1"
    save_dirs = {wv: savepath / f"{wv}nm, Raw" for wv in wavelengths}
    img_dirs = {wv: anglerfish_dir_path / "1" / f"{wv}nm, Raw" for wv in wavelengths}

    # Create the save directories
    savepath.mkdir(parents=True, exist_ok=True)
    for wv in wavelengths:
        save_dirs[wv].mkdir(parents=True, exist_ok=True)
    groundtruth_path = savepath / "groundtruths"
    groundtruth_path.mkdir(parents=True, exist_ok=True)

    # Break up the tile to pieces of (size x size)
    print("Splitting tiles")
    tile_positions = []
    tile_cnt_start = 1
    for tile_num in tile_list:
        print(f"Tile: {tile_num}")
        for round_num in range(1, num_rounds + 1):
            print(f"Rounds: {round_num}")
            for wv in wavelengths:
                org_img = skio.imread(
                    str(img_dirs[wv] / f"merFISH_{round_num:02}_{tile_num:03}_01.TIFF")
                )

                # Compute image values
                mean, stddev = np.mean(org_img), np.std(org_img)
                im_max, im_min = np.max(org_img), np.min(org_img)

                r, c, _ = org_img.shape
                r_cnt, c_cnt = math.floor(r / size), math.floor(c / size)
                tile_cnt = tile_cnt_start
                for i in range(r_cnt):
                    for j in range(c_cnt):
                        img = org_img[
                            i * size : (i + 1) * size, j * size : (j + 1) * size
                        ]
                        skio.imsave(
                            str(
                                save_dirs[wv]
                                / f"merFISH_{round_num:02}_{tile_cnt:03}_01.TIFF"
                            ),
                            img,
                        )
                        if wv == 647 or wv == 750:
                            tile_positions.append(
                                [
                                    tile_cnt,
                                    tile_num,
                                    round_num,
                                    wv,
                                    i,
                                    j,
                                    int(im_max),
                                    int(im_min),
                                    f"{mean:5f}",
                                    f"{stddev:5f}",
                                ]
                            )
                        tile_cnt += 1
        tile_cnt_start = tile_cnt

    # Chop the groundtruths
    print("Splitting ground truths")
    gt_cnt = 1
    for tile_num in tile_list:
        print(f"Tile: {tile_num}")
        groundtruth = pd.read_csv(
            anglerfish_dir_path / "1" / "groundtruths" / f"groundtruth_{tile_num}.csv"
        )
        chopped_groundtruths = [
            np.empty(shape=(0, len(groundtruth.columns))) for _ in range(r_cnt * c_cnt)
        ]

        for _, row in groundtruth.iterrows():
            c_mul, c_rem = divmod(row["column"], size)
            r_mul, r_rem = divmod(row["row"], size)
            if c_mul < c_cnt and r_mul < r_cnt:
                row["column"] = c_rem
                row["row"] = r_rem
                row["pixel_index_num"] = r_rem * size + c_rem
                chopped_groundtruths[r_mul * c_cnt + c_mul] = np.vstack(
                    (chopped_groundtruths[r_mul * c_cnt + c_mul], row.to_numpy())
                )

        for gt in chopped_groundtruths:
            gt = pd.DataFrame(
                gt,
                columns=[
                    "photoncount",
                    "pixel_index_num",
                    "column",
                    "row",
                    "z",
                    "column_shift",
                    "row_shift",
                    "z_shift",
                    "genes",
                    "gene_id",
                    "barcode",
                    "mapped_barcode",
                    "is_bit_drop",
                    "is_bit_add",
                ],
            )
            gt.to_csv(groundtruth_path / f"groundtruth_{gt_cnt}.csv")
            gt_cnt += 1

    # Save the file denoting the orginial position of the chopped tiles
    tile_positions = pd.DataFrame(
        tile_positions,
        columns=[
            "tile id",
            "org tile id",
            "round num",
            "wavelength",
            "row num",
            "col num",
            "max pixel value",
            "min pixel value",
            "mean",
            "standard dev",
        ],
    )
    tile_positions.to_csv(savepath / "tile_positions.csv", index=False)

    # Save info regarding run
    config = {
        "source data path": str(anglerfish_dir_path),
        "chopped tiles": tile_list,
        "chopped tile count": int(len(tile_positions) / (num_rounds * 2)),
    }
    with open(str(savepath / "config.yml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    print("Done: image chopping complete")


def merfish_chop(
    merfish_dir_path: pathlib.Path,
    tile_num: int,
    zstack_num: int,
    num_rounds: int,
    size: int = 40,
):
    """

    Note: this method is specific to the merfish protocol used at Aparicio Lab

    Args:
        merfish_dir_path (pathlib.Path): path to the merfish top-level directiory
        tile_num (int): specify which tile to process
        zstack_num (int): specify which z stack to process
        num_rounds (int): total number of imaging rounds
        size (int): (size x size) of the chopped tiles
    """

    wavelengths = [473, 561, 647, 750]
    img_dirs = {wv: merfish_dir_path / "1" / "1" / f"{wv}nm, Raw" for wv in wavelengths}

    shifts = np.empty(shape=(num_rounds, 2))
    ref_img = skio.imread(
        str(img_dirs[561] / f"merFISH_01_{tile_num:03}_{zstack_num:02}.TIFF")
    )
    for idx in range(1, num_rounds + 1):
        reg_img = skio.imread(
            str(img_dirs[561] / f"merFISH_{idx:02}_{tile_num:03}_{zstack_num:02}.TIFF")
        )

        shift, _, _ = phase_cross_correlation(ref_img, reg_img, upsample_factor=10)
        print(f"IR {idx+2},Detected pixel offset (y, x): {shift}")
        shifts[idx - 1, :] = shift

    dx_tl = np.max(np.maximum(shifts[:, 1], 0))
    dy_tl = np.max(np.maximum(shifts[:, 0], 0))

    dx_br = np.min(np.minimum(shifts[:, 1], 0))
    dy_br = np.min(np.minimum(shifts[:, 0], 0))
    print(f"tl:= dx {dx_tl}, dy {dy_tl}")
    print(f"br:= dx {dx_br}, dy {dy_br}")

    tl = np.ceil([dy_tl, dx_tl]).astype(int)
    br = np.floor([ref_img.shape[0] + dy_br, ref_img.shape[1] + dx_br]).astype(int)
    print(f"new topleft: y {tl[0]} x {tl[1]}")
    print(f"new bottomright: y {br[0]} x {br[1]}")

    # Make the directory structure for the chopped images
    t = time.strftime("%Y%m%d-%H-%M-%S")
    savepath = BASE_PROJECT_DIR / "results" / f"merfish_{t}"
    save_dirs = {wv: savepath / "1" / f"{wv}nm, Raw" for wv in wavelengths}
    savepath.mkdir(parents=True, exist_ok=True)
    for wv in wavelengths:
        save_dirs[wv].mkdir(parents=True, exist_ok=True)

    tile_positions = []

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}...")
        for wv in wavelengths:
            img = skio.imread(
                str(
                    img_dirs[wv]
                    / f"merFISH_{round_num:02}_{tile_num:03}_{zstack_num:02}.TIFF"
                )
            )
            reg_img = reg_method(img, shifts[round_num - 1])
            crop_img = reg_img[tl[0] : br[0], tl[1] : br[1]]

            # Compute image values
            mean, stddev = np.mean(img), np.std(img)
            im_max, im_min = np.max(img), np.min(img)

            # Now just need to split it into smaller pieces then save
            r, c = crop_img.shape
            r_cnt, c_cnt = math.floor(r / size), math.floor(c / size)
            tile_cnt = 1
            for i in range(r_cnt):
                for j in range(c_cnt):
                    img = crop_img[i * size : (i + 1) * size, j * size : (j + 1) * size]
                    skio.imsave(
                        str(
                            save_dirs[wv]
                            / f"merFISH_{round_num:02}_{tile_cnt:03}_{zstack_num:02}.TIFF"
                        ),
                        img,
                    )
                    if wv == 647 or wv == 750:
                        tile_positions.append(
                            [
                                tile_cnt,
                                round_num,
                                wv,
                                i,
                                j,
                                int(im_max),
                                int(im_min),
                                f"{mean:5f}",
                                f"{stddev:5f}",
                            ]
                        )
                    tile_cnt += 1

    # Save the file denoting the orginial position of the chopped tiles
    tile_positions = pd.DataFrame(
        tile_positions,
        columns=[
            "tile id",
            "round num",
            "wavelength",
            "row num",
            "col num",
            "max pixel value",
            "min pixel value",
            "mean",
            "standard dev",
        ],
    )
    tile_positions.to_csv(savepath / "tile_positions.csv", index=False)

    # Save info regarding run
    config = {
        "source data path": str(merfish_dir_path),
        "chopped tile number": tile_num,
        "chopped tile count": tile_positions.shape[0],
        "tile size": size,
    }
    with open(str(savepath / "config.yml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    print("DONE: image chopping completed")


def merfish_chop_cambridge(
    merfish_dir_path: pathlib.Path,
    tile_num: int,
    zstack_num: int,
    num_rounds: int,
    size: int = 40,
    overlap: int = 12,
):
    """

    Note: this method is specific to the merfish protocol used at Cambridge

    Args:
        merfish_dir_path (pathlib.Path): path to the merfish top-level directiory
        tile_num (int): specify which tile to process
        zstack_num (int): specify which z stack to process
        num_rounds (int): total number of imaging rounds
        size (int): (size x size) of the chopped tiles
    """

    # Changes these values depending on the experimental set-up
    wavelengths = [488, 568, 650]
    img_dirs = {wv: merfish_dir_path / f"{wv}nm, Raw" for wv in wavelengths}

    shifts = np.empty(shape=(num_rounds, 2))
    ref_img = skio.imread(str(img_dirs[488] / f"merFISH_00_{tile_num:03}_{zstack_num:02}.tiff"))
    for idx in range(num_rounds):
        reg_img = skio.imread(
            str(img_dirs[488] / f"merFISH_{idx:02}_{tile_num:03}_{zstack_num:02}.tiff")
        )

        shift, _, _ = phase_cross_correlation(ref_img, reg_img, upsample_factor=10)
        print(f"IR {idx+1},Detected pixel offset (y, x): {shift}")
        shifts[idx, :] = shift

    dx_tl = np.max(np.maximum(shifts[:, 1], 0))
    dy_tl = np.max(np.maximum(shifts[:, 0], 0))

    dx_br = np.min(np.minimum(shifts[:, 1], 0))
    dy_br = np.min(np.minimum(shifts[:, 0], 0))
    print(f"tl:= dx {dx_tl}, dy {dy_tl}")
    print(f"br:= dx {dx_br}, dy {dy_br}")

    tl = np.ceil([dy_tl, dx_tl]).astype(int)
    br = np.floor([ref_img.shape[0] + dy_br, ref_img.shape[1] + dx_br]).astype(int)
    print(f"new topleft: y {tl[0]} x {tl[1]}")
    print(f"new bottomright: y {br[0]} x {br[1]}")

    # Make the directory structure for the chopped images
    t = time.strftime("%Y%m%d-%H-%M-%S")
    
    savepath = pathlib.Path(os.path.join(r"F:\merfish_cambridge_data\merfish_cambridge_fov_slice_specific_3", f"merfish_cambridge_clahe_{tile_num:03}_{zstack_num:02}_chopped"))
    
    save_dirs = {wv: savepath / "1" / f"{wv}nm, Raw" for wv in wavelengths if wv != 561}
    savepath.mkdir(parents=True, exist_ok=True)
    for _, dir_path in save_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)

    tile_positions = []
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}...")
        # Loop through all the non-fiducial channels
        for wv in [i for i in wavelengths if i != 561]:
            img = skio.imread(
                str(
                    img_dirs[wv]
                    / f"merFISH_{round_num:02}_{tile_num:03}_{zstack_num:02}.tiff"
                )
            )
            reg_img = reg_method(img, shifts[round_num])
            crop_img = reg_img[tl[0] : br[0], tl[1] : br[1]]

            # Compute image values
            mean, stddev = np.mean(img), np.std(img)
            im_max, im_min = np.max(img), np.min(img)

            # Now just need to split it into smaller pieces then save
            r, c = crop_img.shape
            r_cnt = math.floor((r - overlap) / (size - overlap))
            c_cnt = math.floor((c - overlap) / (size - overlap))

            tile_cnt = 1
            for i in range(r_cnt):
                for j in range(c_cnt):
                    img = crop_img[
                        i * (size - overlap) : i * (size - overlap) + size,
                        j * (size - overlap) : j * (size - overlap) + size,
                    ]
                    skio.imsave(
                        str(
                            save_dirs[wv]
                            / f"merFISH_{round_num:02}_{tile_cnt:03}_{zstack_num:02}.tiff"
                        ),
                        img,
                    )
                    tile_positions.append(
                        [
                            tile_cnt,
                            round_num,
                            wv,
                            i,
                            j,
                            i * (size - overlap),
                            j * (size - overlap),
                            int(im_max),
                            int(im_min),
                            f"{mean:5f}",
                            f"{stddev:5f}",
                        ]
                    )
                    tile_cnt += 1

    # Save the file denoting the orginial position of the chopped tiles
    tile_positions = pd.DataFrame(
        tile_positions,
        columns=[
            "tile id",  # 1 indexed
            "round num",
            "wavelength",
            "row num",
            "col num",
            "top-left row",
            "top-left col",
            "max pixel value",
            "min pixel value",
            "mean",
            "standard dev",
        ],
    )
    tile_positions.to_csv(savepath / "tile_positions.csv", index=False)

    # Save info regarding run
    config = {
        "source data path": str(merfish_dir_path),
        "chopped tile number": tile_num,
        "chopped tile count": tile_positions.shape[0],
        "tile size": size,
        "tile overlap": overlap,
    }
    with open(str(savepath / "config.yml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    print("DONE: image chopping completed")


if __name__ == "__main__":

    # merfish_folder = pathlib.Path("E://MERFISH RUNS//XP1565 20220223 merfish 4T1 C1E1")
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore") # suppress warnings
    #     merfish_chop(merfish_folder, 1, 1, 8)

    #anglerfish_folder = BASE_PROJECT_DIR / "results/merfish_photons6k_scr2_clahe"
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")  # suppress warnings
    #    anglerfish_chop(anglerfish_folder, list(np.arange(1,11)), 40)

    # Run "merfish_chop_cambridge"
    merfish_dir_path = pathlib.Path(r"F:\merfish_cambridge_data\merfish_cambridge_reorganized_and_clahe_2")
    
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")  # suppress warnings
        
        fov = list(np.arange(0,69))
        
        zstack = list(np.arange(1,11))
        
        for f in fov:
            
            for z in zstack:
            
                merfish_chop_cambridge(merfish_dir_path, f, z, 8, 410, 10)
