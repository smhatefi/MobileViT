import os

def downloadImageNet():
    if not(os.path.exists("./res/ILSVRC2012_devkit_t12.tar.gz")):
        os.system("wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate -P ./res")
    
    if not(os.path.exists("./res/ILSVRC2012_img_val.tar")):
        os.system("wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate -P ./res")