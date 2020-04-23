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
import h5py
import scipy.io as sciio


def run_Denoising(mov, block_height =128, block_width=128, d_sub = 2,t_sub=2):

    mov = np.asarray(mov, order='C', dtype=np.float64)
    print('Movie shape:' + str(mov.shape))

    fov_height, fov_width, num_frames = mov.shape

    # default parameters
    block_height = int(block_height)
    block_width = int(block_width)

    d_sub = int(d_sub)
    t_sub = int(t_sub)

    # Maximum of rank 50 blocks (safeguard to terminate early if this is hit)
    max_components = 10

    # Enable Decimation
    max_iters_main = 10
    max_iters_init = 40
    d_sub = 2
    t_sub = 2

    # Defaults
    consec_failures = 3
    tol = 0.00005

    # Set Blocksize Parameters
    #block_height = 20
    #block_width = 20
    overlapping = True

    print('Determine thresholds....')
    #spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, int(num_frames)),
    #                                                       (block_height, block_width),
     #                                                      consec_failures, max_iters_main,
    #                                                       max_iters_init, tol,
     #                                                      d_sub, t_sub, 5, False)

    #print(spatial_thresh)
    #print(temporal_thresh)
    spatial_thresh = 5
    temporal_thresh = 5

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


    #U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
    #                                      block_height, block_width,
     #                                    spatial_components,
      #                                    temporal_components,
       #                                   block_ranks,
        #                                  block_indices,
         #                                 block_weights)


    return mov_denoised


def main():
    parser = argparse.ArgumentParser(description='Run denoising.')
    parser.add_argument('--filename', dest='filename', nargs='?',
                        help='Filename',action="store")

    #filename, start_frame=0, num_frames=1000, fov_height=0, fov_width=0
    parser.add_argument('--start_frame', dest='start_frame', nargs='?',
                        help='start_frame',action="store",default=0)
    parser.add_argument('--num_frames', dest='num_frames', nargs='?',
                        help='num_frames', action="store",default=5000)
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
        block_height = 128
        print("block_height is not set, default 128")
    if args.block_width is not None:
        block_width = args.block_width
    else:
        block_width = 128
        print("block_width is not set, default 128")
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

    f = h5py.File(filename, 'r')
    data = f['Data/Images']

    basename = os.path.basename(filename)
    prjname = os.path.splitext(basename)[0]
    write_path = os.path.join(os.path.dirname(filename), prjname+"_denoising")


    #preview the 1st 1000 frames
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    preview = True
    if preview:
        if not os.path.exists(os.path.join(write_path, "mov")):
            os.makedirs(os.path.join(write_path, "mov"))

        if not os.path.exists(os.path.join(write_path, "mov_denoised")):
            os.makedirs(os.path.join(write_path, "mov_denoised"))

    datashape = data.shape

    data_denoised = np.zeros(datashape)

    for x in range(int(np.ceil(datashape[0]/num_frames))):
        start_frame = x*num_frames
        mov = data[int(start_frame):min(datashape[0],int(start_frame) + int(num_frames)),:,:]
        mov = mov.transpose([1, 2, 0])
        mov_denoised = run_Denoising(mov, block_height, block_width, d_sub,t_sub)

        if preview & x==0:
            mov_1 = np.asarray(mov, order='F', dtype=np.float64)
            for t in range(mov_1.shape[2]):
                tiff.imsave(os.path.join(write_path, "mov", "img_%03d.tif" % (t)), mov_1[:, :, t].astype('float32'),
                            imagej=True)
            mov_denoised_1 = np.asarray(mov_denoised, order=None, dtype=np.float64)
            for t in range(mov_denoised_1.shape[2]):
                tiff.imsave(os.path.join(write_path, "mov_denoised", "img_%03d.tif" % (t)),
                            mov_denoised_1[:, :, t].astype('float32'),
                            imagej=True)
            del mov_1
            del mov_denoised_1
            # save to h5 file

        data_denoised[int(start_frame):min(datashape[0],int(start_frame) + int(num_frames)),:,:] = np.transpose(mov_denoised, (2,0,1))
        print(str(min(datashape[0],int(start_frame) + int(num_frames))) + '/' + str(datashape[0]) + ' processed')

    del mov
    del mov_denoised
    del data

    f.close()
    foutput = h5py.File(os.path.join(write_path, prjname + "_denoised.hdf5"), "w")
    foutput.create_dataset('/Data/Images', data=data_denoised.astype(float), dtype='f', compression="gzip", compression_opts=9)
    foutput.close()

    print('Finished! >> ' + os.path.join(write_path, prjname))
    end = time.time()
    print("Time used: " + str(end - start) + " seconds")

if __name__ == "__main__":
    # execute only if run as a script
    #main()
    main()