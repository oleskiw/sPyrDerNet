#setup script for prince machines and running steerable pyramid pytorch code
cd ~
module purge
module load python3/intel/3.6.3
#setup a python virtualenv
#you can setup the virtualenv using virtualenv [name of folder for virtualenv i.e. "pytorchenv"]
virtualenv pytorchenv
#activate virtualenv using 
source pytorchenv/bin/activate
#pip install the following packages
pip install numpy scipy matplotlib h5py jupyter torch==0.4.0 scikit-learn scikit-image seaborn







