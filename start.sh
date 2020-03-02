
#conda create -n neurondemix python=3.6 trefide=1.2 -c jw3132 -c conda-forge
pip install scikit-image
conda install -c cvxgrp cvxpy -y

pip install tifffile
pip install opencv-python
pip install Cython
pip install jupyter
pip install .



#cd trefide
#sudo chmod 777 ~/.bashrc
#echo 'export TREFIDE=$(pwd)' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src"' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/proxtv"' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/glmgen/lib""' >> ~/.bashrc

#cd src/
#make

#cd ../
#LDSHARED="icc -shared" CC=icc CXX=icpc pip install $(pwd)

#cd demos
#pip install --upgrade pip
#pip install jupyter

#python3 -m ipykernel install --user

#jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root