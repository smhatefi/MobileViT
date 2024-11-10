import os
import subprocess

def downloadImageNet():
    os.system("wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate -P /res")
    os.system("wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate -P /res")

    #os.system("tar -xf ILSVRC2012_img_val.tar")
    #os.system("tar -xf ILSVRC2012_devkit_t12.tar.gz")
    subprocess.call(['sh', './extract_ILSVRC.sh'])