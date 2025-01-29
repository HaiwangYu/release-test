source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone.sh; setup uboonecode v10_03_01 -q e26:prof;

#Ben
setup SparseConvNet v01_00_00
path-prepend /cvmfs/uboone.opensciencegrid.org/products/SparseConvNet/v01_00_00/venv/lib/python3.9/site-packages/ PYTHONPATH
# path-prepend /cvmfs/uboone.opensciencegrid.org/products/SparseConvNet/v01_00_00/SparseConvNet/ PYTHONPATH
path-prepend /cvmfs/uboone.opensciencegrid.org/products/SparseConvNet/v01_00_00/SparseConvNet/build/lib.linux-x86_64-3.9/ PYTHONPATH

#Wenqiang
source /exp/uboone/data/users/wenqiang/mcc10/SparseConvNet/venv/bin/activate
