name: "autoloc-3conv"
layer {
  name: 'data'
  type: 'Python'
  top: 'feature'
  top: 'v_info'
  top: 'label'
  top: 'heatmap'
  top: 'att'
  top: 'videoid'
  python_param {
    module: 'layers.tsv_data'
    layer: 'TSVVideoDataLayer'
  }
}

# 1. add a bbox regression layer
# 2. add a single channel heatmap for the positive class
# 3. add a loss layer, which calculates the inner-outer contrastive loss for each bbox

layer {
  name: "bbox_conv1/3x3"
  type: "Convolution"
  bottom: "feature"
  top: "bbox_conv1/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
  propagate_down: false
}
layer {
  name: "bbox_bn1"
  type: "BatchNorm"
  bottom: "bbox_conv1/3x3"
  top: "bbox_conv1/3x3"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "bbox_scale1"
  type: "Scale"
  bottom: "bbox_conv1/3x3"
  top: "bbox_conv1/3x3"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "bbox_relu1/3x3"
  type: "ReLU"
  bottom: "bbox_conv1/3x3"
  top: "bbox_conv1/3x3"
  relu_param {
    negative_slope: 0.1
  }
}

layer {
  name: "bbox_conv2/3x3"
  type: "Convolution"
  bottom: "bbox_conv1/3x3"
  top: "bbox_conv2/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "bbox_bn2"
  type: "BatchNorm"
  bottom: "bbox_conv2/3x3"
  top: "bbox_conv2/3x3"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "bbox_scale2"
  type: "Scale"
  bottom: "bbox_conv2/3x3"
  top: "bbox_conv2/3x3"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "bbox_relu2/3x3"
  type: "ReLU"
  bottom: "bbox_conv2/3x3"
  top: "bbox_conv2/3x3"
  relu_param {
    negative_slope: 0.1
  }
}

layer {
  name: "bbox_conv3/3x3"
  type: "Convolution"
  bottom: "bbox_conv2/3x3"
  top: "bbox_conv3/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}

layer {
  name: "bbox_bn3"
  type: "BatchNorm"
  bottom: "bbox_conv3/3x3"
  top: "bbox_conv3/3x3"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}

layer {
  name: "bbox_scale3"
  type: "Scale"
  bottom: "bbox_conv3/3x3"
  top: "bbox_conv3/3x3"
  scale_param {
    bias_term: true
  }
}

layer {
  name: "bbox_relu3/3x3"
  type: "ReLU"
  bottom: "bbox_conv3/3x3"
  top: "bbox_conv3/3x3"
  relu_param {
    negative_slope: 0.1
  }
}

layer {
  name: "bbox_pred"
  type: "Convolution"
  bottom: "bbox_conv3/3x3"
  top: "bbox_pred"
  param {
    lr_mult: 1
    decay_mult: 2
  }
  param {
    lr_mult: 1
    decay_mult: 0
  }
  convolution_param {
    num_output: <num_bbox_pred_regout> # x, w
    kernel_size: 1 pad: 0 stride: 1
    #bias_term: false
    #weight_filler { type: "msra" }
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer {
  name: 'bbox_pred_inner'
  type: 'Python'
  bottom: 'bbox_pred'
  top: 'bbox_pred_inner'
  python_param {
    module: 'layers.bbox_transform'
    layer: 'BboxTransformLayer'
  }
}
layer {
  name: 'bbox_pred_outer'
  type: 'Python'
  bottom: 'bbox_pred_inner'
  top: 'bbox_pred_outer'
  python_param {
    module: 'layers.bbox_transform'
    layer: 'BboxInflateRatioMinLayer'
  }
}

layer {
  name: 'bbox_pred_inner_clip'
  type: 'Python'
  bottom: 'bbox_pred_inner'
  bottom: 'v_info'
  top: 'bbox_pred_inner_clip'
  python_param {
    module: 'layers.bbox_transform'
    layer: 'BboxClipKeepGradMgLayer'
    param_str: '{"margin": <inner_margin>}'
  }
}
layer {
  name: 'bbox_pred_outer_clip'
  type: 'Python'
  bottom: 'bbox_pred_outer'
  bottom: 'v_info'
  top: 'bbox_pred_outer_clip'
  python_param {
    module: 'layers.bbox_transform'
    layer: 'BboxClipKeepGradMgLayer'
    param_str: '{"margin": <outer_margin>}'
  }
}

layer {
  name: 'oic_loss'
  type: 'Python'
  bottom: 'label'
  bottom: 'heatmap'
  bottom: 'att'
  bottom: 'videoid'
  bottom: 'bbox_pred_inner_clip'
  bottom: 'bbox_pred_outer_clip'
  top: 'oic_loss'
  top: 'oic_score'
  top: 'bbox_pred_rslt'
  python_param {
    module: 'layers.oic_loss'
    layer: 'OuterInnerContrastiveLossLayer'
  }
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: true # only use inner backprop
  propagate_down: true # change to use outer backprop as well
  # set loss weight so Caffe knows this is a loss layer. since PythonLayer inherits directly from Layer, this isn't automatically exposed to Caffe
  loss_weight: 1
  loss_weight: 0
  loss_weight: 0
}

layer {
  name: 'silence_oic'
  type: 'Silence'
  bottom: 'oic_score'
}
layer {
  name: 'silence_oic'
  type: 'Silence'
  bottom: 'bbox_pred_rslt'
}
