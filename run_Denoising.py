import os
import numpy as np
import time

# Denoising dependencies
from trefide.pmd import batch_decompose, \
    batch_recompose, \
    overlapping_batch_decompose, \
    overlapping_batch_recompose, \
    determine_thresholds
import argparse
import sys
import tifffile as tiff
from trefide.reformat import overlapping_component_reformat

# Plotting & Video Rendering Dependencies
import funimag

import matplotlib.pyplot as plt
from trefide.plot import pixelwise_ranks

from skimage import io


def run_Denoising(filename, start_frame=0, num_frames=1000, block_height =20, block_width=20, d_sub = 2,t_sub=2):
    mov = io.imread(filename).transpose([1, 2, 0])[:, :, int(start_frame):int(start_frame) + int(num_frames)]
    mov = np.asarray(mov, order='C', dtype=np.float64)
    print('Movie shape:' + str(mov.shape))

    fov_height, fov_width, real_num_frames = mov.shape

    # default parameters
    block_height = int(block_height)
    block_width = int(block_width)
    num_frames = int(num_frames)
    d_sub = int(d_sub)
    t_sub = int(t_sub)

    # Maximum of rank 50 blocks (safeguard to terminate early if this is hit)
    max_components = 50

    # Enable Decimation
    max_iters_main = 10
    max_iters_init = 40
    #d_sub = 2
    #t_sub = 2

    # Defaults
    consec_failures = 3
    tol = 0.0005

    # Set Blocksize Parameters
    #block_height = 20
    #block_width = 20
    overlapping = True

    print('Determine thresholds....')
    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, int(num_frames)),
                                                           (block_height, block_width),
                                                           consec_failures, max_iters_main,
                                                           max_iters_init, tol,
                                                           d_sub, t_sub, 5, False)

    print('Decompose the movie....')
    # Blockwise Parallel, Single Tiling
    if not overlapping:
        spatial_components, \
        temporal_components, \
        block_ranks, \
        block_indices = batch_decompose(fov_height, fov_width, num_frames,
                                        mov, block_height, block_width,
                                        max_components, consec_failures,
                                        max_iters_main, max_iters_init, tol,
                                        d_sub, t_sub)

    # Blockwise Parallel, 4x Overlapping Tiling
    else:
        spatial_components, \
        temporal_components, \
        block_ranks, \
        block_indices, \
        block_weights = overlapping_batch_decompose(fov_height, fov_width, int(num_frames),
                                                    mov, block_height, block_width,
                                                    spatial_thresh, temporal_thresh,
                                                    max_components, consec_failures,
                                                    max_iters_main, max_iters_init, tol,
                                                    d_sub=int(d_sub), t_sub=int(t_sub))

    print('Recompose the movie....')
    # Single Tiling (No need for reqweighting)
    if not overlapping:
        mov_denoised = np.asarray(batch_recompose(spatial_components,
                                                  temporal_components,
                                                  block_ranks,
                                                  block_indices))
    # Overlapping Tilings With Reweighting
    else:
        mov_denoised = np.asarray(overlapping_batch_recompose(fov_height, fov_width, num_frames,
                                                              block_height, block_width,
                                                              spatial_components,
                                                              temporal_components,
                                                              block_ranks,
                                                              block_indices,
                                                              block_weights))

    print('Denoising finished. Saving files...')
    # write files
    write_path = os.path.dirname(filename)

    U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
                                          block_height, block_width,
                                          spatial_components,
                                          temporal_components,
                                          block_ranks,
                                          block_indices,
                                          block_weights)

    basename = os.path.basename(filename)
    prjname = os.path.splitext(basename)[0] + '_s'+ str(start_frame)+'_n'+str(num_frames)

    if not os.path.exists(os.path.join(write_path, prjname)):
        os.makedirs(os.path.join(write_path, prjname))

    np.savez_compressed(os.path.join(write_path, prjname, "demixing_results.npz"), U, V, block_ranks, block_height, block_width)

    # save images
    mov_1 = np.asarray(mov, order='F', dtype=np.float64)
    # print(mov_1.shape)
    # mov_1 = np.reshape(mov_1, ((mov.shape[2], mov.shape[0], mov.shape[1])), order='C')
    # print(mov_1.shape)

    if not os.path.exists(os.path.join(write_path, prjname, "mov")):
        os.makedirs(os.path.join(write_path, prjname, "mov"))

    if not os.path.exists(os.path.join(write_path, prjname, "mov_denoised")):
        os.makedirs(os.path.join(write_path, prjname, "mov_denoised"))

    for t in range(mov_1.shape[2]):
        tiff.imsave(os.path.join(write_path, prjname, "mov", "img_%03d.tif" % (t)), mov_1[:, :, t].astype('float32'),
                    imagej=True)
    # # cv2.imwrite(os.path.join(ext, "mov.tif"), mov_1)

    mov_denoised_1 = np.asarray(mov_denoised, order=None, dtype=np.float64)

    for t in range(mov_denoised_1.shape[2]):
        tiff.imsave(os.path.join(write_path, prjname, "mov_denoised", "img_%03d.tif" % (t)),
                    mov_denoised_1[:, :, t].astype('float32'),
                    imagej=True)

    print('Finished! >> ' + os.path.join(write_path, prjname))


def main():
    parser = argparse.ArgumentParser(description='Run denoising.')
    parser.add_argument('--filename', dest='filename', nargs='?',
                        help='Filename',action="store")

    #filename, start_frame=0, num_frames=1000, fov_height=0, fov_width=0
    parser.add_argument('--start_frame', dest='start_frame', nargs='?',
                        help='start_frame',action="store",default=0)
    parser.add_argument('--num_frames', dest='num_frames', nargs='?',
                        help='num_frames', action="store",default=1000)
    parser.add_argument('--block_height', dest='block_height', nargs='?',
                        help='block_height', action="store")
    parser.add_argument('--block_width', dest='block_width', nargs='?',
                        help='block_width', action="store")
    parser.add_argument('--d_sub', dest='d_sub', nargs='?',
                        help='d_sub', action="store")
    parser.add_argument('--t_sub', dest='t_sub', nargs='?',
                        help='t_sub', action="store")

    args = parser.parse_args()

    if args.filename is not None:
        filename = args.filename
    else:
        sys.exit("Filename cannot be none.")

    if args.start_frame is not None:
        start_frame = args.start_frame
    else:
        start_frame =0
    if args.num_frames is not None:
        num_frames = args.num_frames
    else:
        num_frames = 1000
        print("Frame number is not set, default 1000")

    if args.block_height is not None:
        block_height = args.block_height
    else:
        block_height = 20
        print("block_height is not set, default 20")
    if args.block_width is not None:
        block_width = args.block_width
    else:
        block_width = 20
        print("block_width is not set, default 20")
    if args.d_sub is not None:
        d_sub = args.d_sub
    else:
        d_sub = 1
        print("d_sub is not set, default 1")
    if args.t_sub is not None:
        t_sub = args.t_sub
    else:
        t_sub = 1
        print("t_sub is not set, default 1")

    start = time.time()
    run_Denoising(filename, start_frame, num_frames, block_height, block_width, d_sub,t_sub)
    end = time.time()
    print("Time used: " + str(end - start) + " seconds")

if __name__ == "__main__":
    # execute only if run as a script
    main()