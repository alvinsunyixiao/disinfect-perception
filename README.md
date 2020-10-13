# Setting up Training (assuming that dependencies have been installed)

Note: all links need to be accessed with Google @ Illinois email

1. Get ImageNet pretrained model from https://download.pytorch.org/models/resnet18-5c106cde.pth
2. Create a new directory /data under root fs
3. Download ADE20K dataset here: https://drive.google.com/file/d/1CiSu6m8-rS_3XFi7lEwWr-E3ruYZN5HX/view?usp=sharing
4. Uncompress the downloaded ADE20K dataset as /data/ADE20K\_2016\_07\_26/
5. Download Custom Disinfect dataset here: https://drive.google.com/file/d/18koQ-VT7dtF2vsTmqojPN7jPM05t_Uxo/view?usp=sharing
6. Uncompress the downloaded Disinfection dataset as /data/hospital\_images/
7. run
```
python3 segmentation/train.py -p segmentation/params.py --logdir ./log/ --pretrained /path/to/downloaded/imagenet/pretrained/resnet18-5c106cde.pth
```

Trained weights can be downloaded here: https://drive.google.com/file/d/13r32_6Ku24lIpiHpnwASkAKEp6WiVlWX/view?usp=sharing
