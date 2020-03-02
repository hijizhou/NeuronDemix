
Functional Imaging Compression, Denoising and Demixing
======================================================

For now, a pipeline to compress, denoise, and demix several types of functional imaging recordings is presented here. This repo includes:

- complete codes forked from [funimag](https://github.com/paninski-lab/funimag) and [TreFiDe](https://github.com/ikinsella/trefide).
- easier setup and usage codes
- compiled docker image, available at https://hub.docker.com/r/lijz/neurondemix.
      
Reference
---------

If you use this code please cite the paper:

E. Kelly Buchanan, Ian Kinsella, Ding Zhou, Rong Zhu, Pengcheng Zhou, Felipe Gerhard, John Ferrante, Ying Ma, Sharon Kim, Mohammed Shaik, Yajie Liang, Rongwen Lu, Jacob Reimer, Paul Fahey, Taliah Muhammad, Graham Dempsey, Elizabeth Hillman, Na Ji, Andreas Tolias, Liam Paninski
bioRxiv 334706; doi: https://doi.org/10.1101/334706 



## Start with
### Pull this image
` docker pull lijz/neurondemix`

### Run image and mount local folder
` docker run -it -v /Users/lijz/Data:/mnt/data lijz/neurondemix`

### Run Jupyter Notebook
` docker run -it -p 8888:8888 lijz/neurondemix`
` jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root`

## Main function
### Denoising
`python run_Denoising.py  --filename=/mnt/data/data.tif --num_frames=5000`

This operation will process the data and save the results (both denoised movies and demixed components) in the folder

#### Key parameters
- _filename_: image file, needs to be in the mounted folder, for now, only tif files are supported
- _num_frames_: the number of frames to be processed, by default 1000
- _block_height_, _block_width_: block height and width, used for large image to get blocks
- _d_sub_, _t_sub_: the subsampling factor in the spatial and temporal directions


