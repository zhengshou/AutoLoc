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
This code has been tested with NVIDIA Titan X GPU of 12GB memory, Ubuntu 14.04, CUDA 8.0, CUDNN v5, caffe python interface (not included in this repo yet, please set up first), python2, python libs specified in requirements.txt. Please use "Issues" to ask questions or report bugs. Thanks.

### Feature extraction

If you are willing to extract features by yourself, note that for experiments on AN v1.2, please refer to https://github.com/wanglimin/UntrimmedNet/issues/16 to find UntrimmedNet pre-trained model on train set. Models released here http://mmlab.siat.ac.cn/untrimmednet_model/ were trained on train+val set.

Alternatively, we release our extracted features for public downloading.

TH'14: [OneDrive](https://1drv.ms/u/s!ArlzSZKcWKazgogA3Vr1Yacs8i9QTA), [BaiduCloud](https://pan.baidu.com/s/1ussSNpI1nsXc7TpuT7inVA); AN v1.2: [OneDrive split00](), [OneDrive split01](), [BaiduCloud split00](), [BaiduCloud split01]() 

Details of the above features:

0. Put `TH14.tar.gz` under `./data/TH14/`; `cd ./data/TH14/`; `tar -xzvf TH14.tar.gz -C ./`; features are contained in the `att_unfused/` folder. In `att_unfused/`:
    - Each video has one individual tsv file storing features.
    - Each row in tsv represents features of one whole video. Each row sequentially contains: video id, class label, feature, class activation, attention score.
    - For feature, class activations, attention score, values are separated by `;` and data corresponding to each frame are concatenated sequentially.
    - For feature, the data field has dimension `num of frames (T) x dim of feature (2048)`. T is the total number of frames of the whole video. For the 2048-dim vector, the first half 1024 is corresponding to the temporal CNN and the second half 1024 is spatial.
    - Likewise, class activation has dimension `num of frames (T) x 2 x num of classes (K)`. The first half K is temporal (score after softmax over classes) and the second half is spatial. Attention score has dimension `num of frames (T) x 2 (temporal attention, spatial attention)`.
    - All videos within each subset have stacked into one whole file, i.e. `test.all.tsv` and `val.all.tsv`.
1. For AN:
    - Due to the storage limit, I have first splited `AN.tar.gz` into `AN.tar.gz.split00` and `AN.tar.gz.split01` using `split -d -b 19G AN.tar.gz AN.tar.gz.split`. Thus after downloading the above splited files, the user needs to first merge them via `cat AN.tar.gz.split00 AN.tar.gz.split01 > AN.tar.gz`: 
    - Put `AN.tar.gz` under `./data/AN/`; `cd ./data/AN/`; `tar -xzvf AN.tar.gz -C ./`; features are contained in the `att_unfused_onlytrain/` folder
2. We follow UntrimmedNet to extract feature every 15 frames.

### Overview of AutoLoc code structure

0. `data/`: contains TH'14 and AN datasets
1. `exp/`: scripts for running experiments and our trained model and testing results etc. 
2. `lib/`: core code of AutoLoc.
3. `tools/`: tools for launching AutoLoc and evaluating results. Note that `proc_net.py` takes a whole set of videos as input and thus usually is used for training. `proc_prll_net.py` can launch multiple concurrent processes that each of them only takes one single video as input and thus usually is used for testing so that making predictions for each video in parallel to speedup.

### Running AutoLoc
Under the root directory of this repo, run `python exp/TH14/run.py` for experiments on TH'14 and `python exp/AN/run.py` for experiments on AN v1.2. These scripts run training first and then run testing.
