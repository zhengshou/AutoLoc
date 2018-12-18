# AutoLoc

### Citing
If you find AutoLoc or pre-extracted features useful, please consider citing:

    @InProceedings{zheng_eccv18_autoloc,
      title={AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos},
      author={Shou, Zheng and Gao, Hang and Zhang, Lei and Miyazawa, Kazuyuki and Chang, Shih-Fu},
      booktitle = {ECCV},
      year={2018}
    }

### Installation
This code has been tested with NVIDIA Titan X GPU of 12GB memory, CUDA 8.0, caffe. Please use "Issues" to ask questions or report bugs. Thanks.

### Feature extraction

If you are willing to extract features by yourself, note that for experiments on AN v1.2, please refer to https://github.com/wanglimin/UntrimmedNet/issues/16 to find UntrimmedNet pre-trained model on train set. Models released here http://mmlab.siat.ac.cn/untrimmednet_model/ were trained on train+val set.

Alternatively, we release our extracted features for public downloading.

TH'14: [OneDrive](https://1drv.ms/u/s!ArlzSZKcWKazgogA3Vr1Yacs8i9QTA), [BaiduCloud To-do](); AN v1.2: [OneDrive](), [BaiduCloud To-do]() 

Details of the above features:

0. put `TH14.tar.gz` under `./data/TH14/`; `cd ./data/TH14/`; `tar -xzvf TH14.tar.gz -C ./`; features are contained in the `att_unfused/` folder. In `att_unfused/`:
    - Each video
    - all
1. put `AN.tar.gz` under `./data/AN/`; `cd ./data/AN/`; `tar -xzvf AN.tar.gz -C ./`; features are contained in the `att_unfused_onlytrain/` folder
2. We follow UntrimmedNet to extract feature every 15 frames.
