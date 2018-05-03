#setup script for prince machines and running steerable pyramid pytorch code
cd ~
module purge
module load cuda/8.0.44 pytorch/python3.6/0.3.0_4

#setup a python virtualenv
#you can setup the virtualenv using virtualenv [name of folder for virtualenv i.e. "pytorchenv"]
virtualenv pytorchenv
#activate virtualenv using 
source pytorchenv/bin/activate
#pip install the following packages
pip install numpy, scipy, matplotlib, h5py, cffi==1.11.4, jupyter

#clone the pytorch_fft repo and install: 
git clone https://github.com/locuslab/pytorch_fft.git
cd pytorch_fft 
python setup.py install

#the pytorch_fft package has some bugs right now so you need to replace the files that are created in setup
#with the pre-built files provided in the sPyrDerNet git repo "pytorch_fft_built.zip"

cd ~/sPyrDerNet/ #replace with your directory for this repo
unzip pytorch_fft_built.zip -d ~/pytorchenv/lib/python3.6/site-packages/pytorch_fft-0.14-py3.6-linux-x86_64.egg/pytorch_fft

cd ~/sPyrDerNet





