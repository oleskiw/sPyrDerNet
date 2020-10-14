# sPyrDerNet
Steerable Pyramid Derivative Network

First setup your machine (this is specifically for the prince NYU HPC but it should translate to most unix clusters):

<pre>
#setup script for prince machines and running steerable pyramid pytorch code

cd ~
module purge
module load cuda/8.0.44

#setup a python virtualenv
#you can setup the virtualenv using virtualenv (name of folder for virtualenv i.e. "pytorchenv")
virtualenv pytorchenv

#activate virtualenv using 
source pytorchenv/bin/activate

#pip install the following packages
pip install numpy scipy matplotlib h5py jupyter torch==0.4.0 scikit-learn
</pre>

Next test the pytorch steerable pyramid implementation:

<pre>
cd models
python steerable.py
</pre>

