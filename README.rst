
ATU Public Pre-Trained Models Zoo
=========================
Here, we give the full list of publicly pre-trained models supported by the `Hailo Model Zoo <https://github.com/hailo-ai/hailo_model_zoo>`_ .
These modules have undergone quantization processing and are suitable for the I.MX8MP platform (Vivante VIP8000) and NXP i.MX93 (Arm Ethos-U65).

* All models were **Post-training quantization** using Tensorflow v2.5 , ONNX v10.0, Vela compiler v3.5. 
* Or You can using `COLAB <https://colab.research.google.com/drive/13KJtrcxHVDW_dMaSIL-3rOL5k75oLQNX?usp=sharing>`_ again.

* Supported tasks:

  * `Classification`_
  * `Object Detection`_
  * `Segmentation`_
  * `Segmentation ( Instance )`_
  * `Pose Estimation`_
  * `Face Detection`_
  * `Depth Estimation`_
  * `Facial Landmark Detection`_
  * `Person Re-ID`_
  * `Super Resolution`_
  * `Face Recognition`_
  * `Person Attribute`_
  * `Face Attribute`_
  * `Zero-shot Classification`_
  * `Low Light Enhancement`_
  * `Image Denoising`_
  * `Hand Landmark detection`_


.. _Classification:

Classification
--------------

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - efficientnet_l
     - 80.46
     - 79.36
     - 300x300x3
     - 10.55
     - 19.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_L/efficientnet-edgetpu-L_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_L/efficientnet-edgetpu-L_quant_vela.tflite>`_
     - `link <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_L/efficientnet-edgetpu-L_quant.tflite>`_
   * - efficientnet_M
     - 78.91
     - 78.63
     - 240x240x3
     - 6.87
     - 7.32
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_M/efficientnet-edgetpu-M_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_M/efficientnet-edgetpu-M_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_s
     - 77.64
     - 77.32
     - 224x224x3
     - 5.41
     - 4.72
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_S/efficientnet-edgetpu-S_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_S/efficientnet-edgetpu-S_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_lite0
     - 74.99
     - 73.81
     - 224x224x3
     - 4.63
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_
     - 
     - 
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_lite1
     - 76.68
     - 76.21
     - 240x240x3
     - 5.39
     - 1.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite1/efficientnet-lite1-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite1/efficientnet-lite1-int8_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_lite2
     - 77.45
     - 76.74
     - 260x260x3
     - 6.06
     - 1.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite2/efficientnet-lite2-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite2/efficientnet-lite2-int8_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_lite3
     - 79.29
     - 78.33
     - 280x280x3
     - 8.16
     - 2.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite3/efficientnet-lite3-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite3/efficientnet-lite3-int8_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - efficientnet_lite4
     - 80.79
     - 80.47
     - 300x300x3
     - 12.95
     - 5.10
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite4/efficientnet-lite4-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/efficientnet_lite4/efficientnet-lite4-int8_vela.tflite>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
   * - hardnet39ds
     - 73.43
     - 72.92
     - 224x224x3
     - 3.48
     - 0.86
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/hardnet39ds/hardnet39ds_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/hardnet39ds/hardnet39ds_quant_vela.tflite>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
   * - hardnet68
     - 75.47
     - 75.04
     - 224x224x3
     - 17.56
     - 8.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/hardnet68/hardnet68_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/hardnet68/hardnet68_quant_vela.tflite>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
   * - inception_v1
     - 69.74
     - 69.54
     - 224x224x3
     - 6.62
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v1/inception_v1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v1/inception_v1_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - inception_v2
     - 73.9
     - None
     - 224x224x3
     - 56
     - None
     - `download <http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v2/inception_v2_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v2/inception_v2_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - inception_v3
     - 78.0
     - None
     - 224x224x3
     - 24
     - None
     - `download <http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v3/inception_v3_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v3/inception_v3_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - inception_v4
     - 80.2
     - None
     - 224x224x3
     - 43
     - None
     - `download <http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v4/inception_v4_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/inception_v4/inception_v4_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v1
     - 70.97
     - 70.26
     - 224x224x3
     - 4.22
     - 1.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v1/mobilenet_v1_0.75_224_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v1/mobilenet_v1_0.75_224_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v2_1.0
     - 71.78
     - 71.0
     - 224x224x3
     - 3.49
     - 0.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2021-07-11/mobilenet_v2_1.0.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_1_0/mobilenet_v2_1_0_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_1_0/mobilenet_v2_1_0_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v2_1.4
     - 74.18
     - 73.18
     - 224x224x3
     - 6.09
     - 1.18
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_1_4/mobilenet_v2_1_4_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_1_4/mobilenet_v2_1_4_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v2_edgetpu
     - None
     - None
     - 224x224x3
     - 6.09
     - None
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_edgetpu/Mobilenet-edgetpu-v2_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_edgetpu/Mobilenet-edgetpu-v2_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v3
     - 72.21
     - 71.73
     - 224x224x3
     - 4.07
     - 2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v3/v3-small_224_1.0_uint8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v3/v3-small_224_1.0_uint8_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
   * - mobilenet_v3_edgetpu
     - None
     - None
     - 224x224x3
     - 4.07
     - None
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_edgetpu/Mobilenet-edgetpu-v2_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v2_edgetpu/Mobilenet-edgetpu-v2_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
   * - mobilenet_v3_large_minimalistic
     - 72.11
     - 70.96
     - 224x224x3
     - 3.91
     - 0.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v3_large_minimalistic/v3-large_224_1.0_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/mobilenet_v3_large_minimalistic/v3-large_224_1.0_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
   * - regnetx_1.6gf
     - 77.05
     - 76.75
     - 224x224x3
     - 9.17
     - 3.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx-1.6gf/RegNet16GF_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx-1.6gf/RegNet16GF_quant_vela.tflite>`_
     - `link <https://github.com/facebookresearch/pycls>`_
   * - regnetx_200mf
     - 70.38
     - 69.52
     - 224x224x3
     - 6.09
     - 0.59
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx200mf/regnetx200mf_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx200mf/regnetx200mf_quant_vela.tflite>`_
     - `link <https://github.com/facebookresearch/pycls>`_
   * - regnetx_800mf
     - 75.16
     - 74.84
     - 224x224x3
     - 7.24
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx800mf/RegNetx800MF_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/regnetx800mf/RegNetx800MF_quant_vela.tflite>`_
     - `link <https://github.com/facebookresearch/pycls>`_
   * - repvgg_a1
     - 74.4
     - 73.61
     - 224x224x3
     - 12.79
     - 4.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/repvgg_a1/repvgg_a1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/repvgg_a1/repvgg_a1_quant_vela.tflite>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
   * - repvgg_a2
     - 76.52
     - 75.08
     - 224x224x3
     - 25.5
     - 10.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/repvgg_a2/repvgg_a2_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/repvgg_a2/repvgg_a2_quant_vela.tflite>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
   * - resmlp12_relu
     - 75.26
     - 74.32
     - 224x224x3
     - 15.77
     - 6.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resmlp12_relu/resmlp_12_224_bn_relu_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resmlp12_relu/resmlp_12_224_bn_relu_quant_vela.tflite>`_
     - `link <https://github.com/rwightman/pytorch-image-models/>`_
   * - resnet_v1_18
     - 71.26
     - 71.06
     - 224x224x3
     - 11.68
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v1_18/resnet_v1_18_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v1_18/resnet_v1_18_quant_vela.tflite>`_
     - `link <https://github.com/yhhhli/BRECQ>`_
   * - resnet_v1_34
     - 72.7
     - 72.14
     - 224x224x3
     - 21.79
     - 7.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2021-07-11/resnet_v1_34.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v1_34/resnet_v1_34_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v1_34/resnet_v1_34_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
   * - resnet_v1_50 
     - 75.12
     - 74.47
     - 224x224x3
     - 25.53
     - 6.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2021-07-11/resnet_v1_50.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v1_50/resnet_v1_50_qunat.tflite>`_
     - 
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
   * - resnet_v2_18 
     - None
     - None
     - 224x224x3
     - None
     - 6None
     - 
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v2_18/resnet_v2_18_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnet_v2_18/resnet_v2_18_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
   * - resnext26_32x4d
     - 76.18
     - 75.78
     - 224x224x3
     - 15.37
     - 4.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnext26_32x4d/resnext26_32x4d_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnext26_32x4d/resnext26_32x4d_quant_vela.tflite>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
   * - resnext50_32x4d
     - 79.31
     - 78.11
     - 224x224x3
     - 24.99
     - 8.48
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2021-07-11/resnext50_32x4d.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnext50_32x4d/resnext50_32x4d_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/resnext50_32x4d/resnext50_32x4d_quant_vela.tflite>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
   * - shufflenet_g8_w1
     - 66.30
     - 65.44
     - 224x224x3
     - 2.46
     - 0.18
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/shufflenet_g8_w1/pretrained/2021-07-11/shufflenet_g8_w1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/shufflenet/shufflenet_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/shufflenet/shufflenet_quant_vela.tflite>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
   * - squeezenet_v1.1
     - 59.85
     - 59.4
     - 224x224x3
     - 1.24
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/squeezenet_v1_1/squeezenet_v1_1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Classification/squeezenet_v1_1/squeezenet_v1_1_quant_vela.tflite>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
   * - vit_base_bn
     - 79.98
     - 78.88
     - 224x224x3
     - 86.5
     - 34.25
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2023-01-25/vit_base.zip>`_
     - 
     - 
     - `link <https://github.com/rwightman/pytorch-image-models>`_
   * - vit_small_bn
     - 78.12
     - 77.02
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2022-08-08/vit_small.zip>`_
     - 
     -  
     - `link <https://github.com/rwightman/pytorch-image-models>`_
   * - vit_tiny_bn
     - 68.95
     - 66.75
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2023-08-29/vit_tiny_bn.zip>`_
     - 
     - 
     - `link <https://github.com/rwightman/pytorch-image-models>`_

.. _Object Detection:

Object Detection
----------------

.. list-table::
   :widths: 33 8 7 12 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - centernet_resnet_v1_18_postprocess
     - 26.3
     - 23.31
     - 512x512x3
     - 14.22
     - 31.26
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2023-07-18/centernet_resnet_v1_18.zip>`_
     - 
     - 
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
   * - centernet_resnet_v1_50_postprocess
     - 31.78
     - 29.64
     - 512x512x3
     - 30.07
     - 56.92
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_50_postprocess/pretrained/2023-07-18/centernet_resnet_v1_50_postprocess.zip>`_
     - 
     - 
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
   * - damoyolo_tinynasL20_T
     - 42.8
     - 42.0
     - 640x640x3
     - 11.35
     - 18.06
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL20_T/damoyolo_tinynasL20_T_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL20_T/damoyolo_tinynasL20_T_quant_vela.tflite>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
   * - damoyolo_tinynasL25_S
     - 46.53
     - 46.04
     - 640x640x3
     - 16.25
     - 37.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL25_S/pretrained/2022-12-19/damoyolo_tinynasL25_S.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL25_S/damoyolo_tinynasL25_S_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL25_S/damoyolo_tinynasL25_S_quant_vela.tflite>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
   * - damoyolo_tinynasL35_M
     - 49.7
     - 47.23
     - 640x640x3
     - 33.98
     - 61.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL35_M/pretrained/2022-12-19/damoyolo_tinynasL35_M.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL35_M/damoyolo_tinynasL35_M_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/damoyolo_tinynasL35_M/damoyolo_tinynasL35_M_quant_vela.tflite>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
   * - detr_resnet_v1_18_bn
     - 33.91
     - 30.56
     - 800x800x3
     - 32.42
     - 59.16
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/detr/detr_r18/detr_resnet_v1_18/2022-09-18/detr_resnet_v1_18_bn.zip>`_
     - 
     - 
     - `link <https://github.com/facebookresearch/detr>`_
   * - efficientdet_lite0
     - 27.32
     - 26.49
     - 320x320x3
     - 3.56
     - 1.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite0/pretrained/2023-04-25/efficientdet-lite0.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet_lite0/efficientdet-lite0_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet_lite0/efficientdet-lite0_quant_vela.tflite>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
   * - efficientdet_lite1
     - 32.27
     - 31.72
     - 384x384x3
     - 4.73
     - 4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite1/pretrained/2023-04-25/efficientdet-lite1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet-lite1/efficientdet-lite1-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet-lite1/efficientdet-lite1-int8_vela.tflite>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
   * - efficientdet_lite2
     - 35.95
     - 34.67
     - 448x448x3
     - 5.93
     - 6.84
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite2/pretrained/2023-04-25/efficientdet-lite2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet-lite2/efficientdet-lite2-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/efficientdet-lite2/efficientdet-lite2-int8_vela.tflite>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
   * - nanodet_repvgg  
     - 29.3
     - 28.53
     - 416x416x3
     - 6.74
     - 11.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg/pretrained/2022-02-07/nanodet.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/nanodet_repvgg/nanodet_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/nanodet_repvgg/nanodet_quant_vela.tflite>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
   * - nanodet_repvgg_a12
     - 33.73
     - 31.93
     - 640x640x3
     - 5.13
     - 28.23
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a12/pretrained/2023-05-31/nanodet_repvgg_a12_640x640.zip>`_
     - 
     - 
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - nanodet_repvgg_a1_640
     - 33.28
     - 32.88
     - 640x640x3
     - 10.79
     - 42.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a1_640/pretrained/2022-07-19/nanodet_repvgg_a1_640.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/nanodet_repvgg_a1_640/nanodet_repvgg_a1_640_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/nanodet_repvgg_a1_640/nanodet_repvgg_a1_640_quant_vela.tflite>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
   * - ssd_mobiledet_dsp
     - 28.9
     - 28.17
     - 320x320x3
     - 7.07
     - 2.83
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobiledet_dsp/pretrained/2021-07-11/ssd_mobiledet_dsp.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobiledet_dsp/ssd_mobiledet_dsp_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobiledet_dsp/ssd_mobiledet_dsp_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
   * - ssd_mobilenet_v1 
     - 23.19
     - 22.29
     - 300x300x3
     - 6.79
     - 2.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/2023-07-18/ssd_mobilenet_v1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobilenet_v1/ssd_mobilenet_v1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobilenet_v1/ssd_mobilenet_v1_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
   * - ssd_mobilenet_v2
     - 24.15
     - 22.94
     - 300x300x3
     - 4.46
     - 1.52
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v2/pretrained/2023-03-16/ssd_mobilenet_v2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobilenet_v2/ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/ssd_mobilenet_v2/ssdlite_mobilenet_v2_coco_300_full_integer_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
   * - tiny_yolov3
     - 14.66
     - 13.61
     - 416x416x3
     - 8.85
     - 5.58
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov3/pretrained/2021-07-11/tiny_yolov3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3-tiny/yolov3-416_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3-tiny/yolov3-416_quant_vela.tflite>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
   * - tiny_yolov4
     - 19.18
     - 17.73
     - 416x416x3
     - 6.05
     - 6.92
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov4/pretrained/2023-07-18/tiny_yolov4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov4-tiny/yolov4-tiny_OmniXR_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov4-tiny/yolov4-tiny_OmniXR_quant_vela.tflite>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
   * - yolov3 
     - 38.42
     - 37.32
     - 608x608x3
     - 68.79
     - 158.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3/pretrained/2021-08-16/yolov3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3/yolov3_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3/yolov3_quant_vela.tflite>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
   * - yolov3_416
     - 37.73
     - 36.08
     - 416x416x3
     - 61.92
     - 65.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_416/pretrained/2021-08-16/yolov3_416.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3_416/yolov3-416_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov3_416/yolov3-416_quant_vela.tflite>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
   * - yolov3_gluon 
     - 37.28
     - 35.64
     - 608x608x3
     - 68.79
     - 140.69
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon/pretrained/2023-07-18/yolov3_gluon.zip>`_
     -
     -
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
   * - yolov3_gluon_416
     - 36.27
     - 34.92
     - 416x416x3
     - 61.92
     - 65.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon_416/pretrained/2023-07-18/yolov3_gluon_416.zip>`_
     -
     -
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
   * - yolov4_leaky
     - 42.37
     - 41.08
     - 512x512x3
     - 64.33
     - 91.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov4/pretrained/2022-03-17/yolov4.zip>`_
     -
     -
     - `link <https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo>`_
   * - yolov5l
     - 46.01
     - 44.01
     - 640x640x3
     - 48.54
     - 60.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5l_spp/pretrained/2022-02-03/yolov5l.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5l/yolov5l-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5l/yolov5l-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5m
     - 42.59
     - 41.09
     - 640x640x3
     - 21.78
     - 52.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_spp/pretrained/2023-04-25/yolov5m.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m/yolov5m-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m/yolov5m-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5m6_6.1
     - 50.67
     - 48.74
     - 1280x1280x3
     - 35.70
     - 200.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m6_6.1/pretrained/2023-04-25/yolov5m6.zip>`_
     - 
     - 
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_
   * - yolov5m_6.1
     - 44.8
     - 43.36
     - 640x640x3
     - 21.17
     - 48.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_6.1/pretrained/2023-04-25/yolov5m_6.1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m6_6/yolov5m6_6-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m6_6/yolov5m6_6-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_
   * - yolov5m_wo_spp
     - 43.06
     - 40.71
     - 640x640x3
     - 22.67
     - 41.67
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2023-04-25/yolov5m_wo_spp.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m_wo_spp/yolov5m_wo_spp_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5m_wo_spp/yolov5m_wo_spp_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5s
     - 35.33
     - 33.98
     - 640x640x3
     - 7.46
     - 17.44
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2023-04-25/yolov5s.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s/yolov5s-int8_.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s/yolov5s-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5s_256
     - 35.33
     - 33.98
     - 640x640x3
     - 7.46
     - 17.44
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2023-04-25/yolov5s.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s_256/yolov5s_256-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s_256/yolov5s_256-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5s_c3tr
     - 37.13
     - 35.33
     - 640x640x3
     - 10.29
     - 17.02
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_c3tr/pretrained/2023-04-25/yolov5s_c3tr.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s_c3tr/yolov5s_c3tr_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5s_c3tr/yolov5s_c3tr_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/tree/v6.0>`_
   * - yolov5xs_wo_spp
     - 33.18
     - 32.2
     - 512x512x3
     - 7.85
     - 11.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2023-04-25/yolov5xs.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5xs_wo_spp/yolov5xs_wo_spp_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5xs_wo_spp/yolov5xs_wo_spp_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov5xs_wo_spp_nms_core
     - 32.57
     - 31.06
     - 512x512x3
     - 7.85
     - 11.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2022-05-10/yolov5xs_wo_spp_nms.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5xs_wo_spp_nms/yolov5xs_wo_spp_nms_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov5xs_wo_spp_nms/yolov5xs_wo_spp_nms_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
   * - yolov6n
     - 34.28
     - 31.78
     - 640x640x3
     - 4.32
     - 4.65
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n/pretrained/2023-05-31/yolov6n.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov6n/yolov6n_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov6n/yolov6n_quant_vela.tflite>`_
     - `link <https://github.com/meituan/YOLOv6/releases/tag/0.1.0>`_
   * - yolov6n_0.2.1
     - 35.16
     - 33.21
     - 640x640x3
     - 4.33
     - 11.06
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n_0.2.1/pretrained/2023-04-17/yolov6n_0.2.1.zip>`_
     - 
     - 
     - `link <https://github.com/meituan/YOLOv6/releases/tag/0.2.1>`_
   * - yolov7
     - 50.59
     - 47.8
     - 640x640x3
     - 36.91
     - 104.68
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7/pretrained/2023-04-25/yolov7.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov7/yolov7_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov7/yolov7_quant_vela.tflite>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
   * - yolov7_tiny
     - 37.07
     - 35.97
     - 640x640x3
     - 6.22
     - 13.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7_tiny/pretrained/2023-04-25/yolov7_tiny.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov7_tiny/yolov7_tiny_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov7_tiny/yolov7_tiny_quant_vela.tflite>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
   * - yolov7e6
     - 55.37
     - 53.17
     - 1280x1280x3
     - 97.20
     - 515.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7e6/pretrained/2023-04-25/yolov7-e6.zip>`_
     -  Large
     -  Large
     - `link <https://github.com/WongKinYiu/yolov7>`_
   * - yolov8l
     - 52.61
     - 51.95
     - 640x640x3
     - 43.7
     - 165.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8l/2023-02-02/yolov8l.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8l/yolov8l_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8l/yolov8l_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8m
     - 50.08
     - 48.83
     - 640x640x3
     - 25.9
     - 78.93
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8m/2023-02-02/yolov8m.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8m/yolov8m_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8m/yolov8m_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8n
     - 37.23
     - 36.23
     - 640x640x3
     - 3.2
     - 8.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8n/2023-01-30/yolov8n.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8n/yolov8n_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8n/yolov8n_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8n_256
     - 37.23
     - 36.23
     - 640x640x3
     - 3.2
     - 8.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8n/2023-01-30/yolov8n.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8n_256/yolov8n_integer_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8n_256/yolov8n_integer_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8s
     - 44.75
     - 44.15
     - 640x640x3
     - 11.2
     - 28.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8s/2023-02-02/yolov8s.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8s/yolov8s_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8s/yolov8s_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8s
     - 44.75
     - 44.15
     - 640x640x3
     - 11.2
     - 28.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8s/2023-02-02/yolov8s.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8s_256/yolov8s_integer_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolov8s_256/yolov8s_integer_quant_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8x
     - 53.61
     - 52.21
     - 640x640x3
     - 68.2
     - 258
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8x/2023-02-02/yolov8x.zip>`_
     - 
     - 
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolox_l_leaky
     - 48.69
     - 46.71
     - 640x640x3
     - 54.17
     - 155.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_l_leaky/pretrained/2023-05-31/yolox_l_leaky.zip>`_
     - 
     - 
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - yolox_s_leaky
     - 38.12
     - 37.27
     - 640x640x3
     - 8.96
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2023-05-31/yolox_s_leaky.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_s_leaky/yolox_s_leaky_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_s_leaky/yolox_s_leaky_quant_vela.tflite>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - yolox_s_wide_leaky
     - 42.4
     - 40.97
     - 640x640x3
     - 20.12
     - 59.46
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_wide_leaky/pretrained/2023-05-31/yolox_s_wide_leaky.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_s_wide_leaky/yolox_s_wide_leaky_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_s_wide_leaky/yolox_s_wide_leaky_quant_vela.tflite>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - yolox_tiny
     - 32.64
     - 30.92
     - 416x416x3
     - 5.05
     - 6.44
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox/yolox_tiny/pretrained/2023-05-31/yolox_tiny.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_tiny/yolox_tiny_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_tiny/yolox_tiny_quant_vela.tflite>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - yolox_tiny_leaky
     - 30.26
     - 29.64
     - 416x416x3
     - 5.05
     - 3.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_tiny_leaky/pretrained/2021-08-12/yolox_tiny_leaky.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_tiny_leaky/yolox_tiny_leaky_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ObjectDetection/yolox_tiny_leaky/yolox_tiny_leaky_quant_vela.tflite>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
   * - ssd_mobilenet_v1_visdrone
     - 2.37
     - 2.22
     - 300x300x3
     - 5.64
     - 2.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Visdrone/ssd/ssd_mobilenet_v1_visdrone/pretrained/2023-07-18/ssd_mobilenet_v1_visdrone.zip>`_
     - 
     - 
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_

.. _Semantic Segmentation:

Segmentation
---------------------

.. list-table::
   :widths: 31 7 9 12 9 8 9 8 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - fcn16_resnet_v1_18 
     - 66.83
     - 66.57
     - 1024x1920x3
     - 11.19
     - 71.26
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn16_resnet_v1_18/pretrained/2022-02-07/fcn16_resnet_v1_18.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/fcn16_resnet_v1_18/fcn16_resnet_v1_18_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/fcn16_resnet_v1_18/fcn16_resnet_v1_18_quant_vela.tflite>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
   * - fcn8_resnet_v1_18 
     - 69.41
     - 69.21
     - 1024x1920x3
     - 11.20
     - 142.49
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2023-06-22/fcn8_resnet_v1_18.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/fcn8_resnet_v1_18/fcn8_resnet18_fhd_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/fcn8_resnet_v1_18/fcn8_resnet18_fhd_quant_vela.tflite>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
   * - fcn8_resnet_v1_22 
     - 67.55
     - 67.39
     - 1024x1920x3
     - 15.12
     - 150.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_22/pretrained/2021-07-11/fcn8_resnet_v1_22.zip>`_
     - 
     - 
     - `link <https://cv.gluon.ai/model_zoo/segmentation.html>`_
   * - stdc1 
     - 74.57
     - 73.57
     - 1024x1920x3
     - 8.27
     - 126.47
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2023-06-12/stdc1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/stdc1/stdc1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/stdc1/stdc1_quant_vela.tflite>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
   * - unet_mobilenet_v2
     - 77.32
     - 77.02
     - 256x256x3
     - 10.08
     - 28.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2022-02-03/unet_mobilenet_v2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/unet_mobilenet_v2/unet_mobilenet_v2_quant.tflite>`_
     - 
     - `link <https://www.tensorflow.org/tutorials/images/segmentation>`_
   * - deeplab_v3_mobilenet_v2
     - 76.05
     - 74.8
     - 513x513x3
     - 2.10
     - 17.65
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2023-08-22/deeplab_v3_mobilenet_v2_dilation.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/deeplab_v3_mobilenet_v2/model_full_integer_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/deeplab_v3_mobilenet_v2/model_full_integer_quant_vela.tflite>`_
     - `link <https://github.com/bonlime/keras-deeplab-v3-plus>`_
   * - deeplab_v3_mobilenet_v2_wo_dilation
     - 71.46
     - 71.11
     - 513x513x3
     - 2.10
     - 3.21
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2021-08-12/deeplab_v3_mobilenet_v2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/deeplab_v3_mobilenet_v2_wo_dilation/edgetpu_deeplab_slim_257_os16_full_integer_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/deeplab_v3_mobilenet_v2_wo_dilation/edgetpu_deeplab_slim_257_os16_full_integer_quant_vela.tflite>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/deeplab>`_



.. Segmentation ( Instance ):

Segmentation ( Instance )
---------------------

.. list-table::
   :widths: 34 7 7 11 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - yolact_regnetx_1.6gf
     - 27.57
     - 27.27
     - 512x512x3
     - 30.09
     - 125.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolact_regnetx_1_6gf/yolact_regnetx_1_6gf_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolact_regnetx_1_6gf/yolact_regnetx_1_6gf_quant_vela.tflite>`_
     - `link <https://github.com/dbolya/yolact>`_
   * - yolact_regnetx_800mf
     - 25.61
     - 25.5
     - 512x512x3
     - 28.3
     - 116.75
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolact_regnetx_800mf/yolact_regnetx_800mf_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolact_regnetx_800mf/yolact_regnetx_800mf_quant_vela.tflite>`_
     - `link <https://github.com/dbolya/yolact>`_
   * - yolov5l_seg
     - 39.78
     - 39.09
     - 640x640x3
     - 47.89
     - 147.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5s_seg/yolov5s-seg-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5s_seg/yolov5s-seg-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5>`_
   * - yolov5m_seg
     - 37.05
     - 36.32
     - 640x640x3
     - 32.60
     - 70.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5m_seg/yolov5m-seg-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5m_seg/yolov5m-seg-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5>`_
   * - yolov5n_seg  |star|
     - 23.35
     - 22.24
     - 640x640x3
     - 1.99
     - 7.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5n_seg/yolov5n-seg-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5n_seg/yolov5n-seg-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5>`_
   * - yolov5s_seg
     - 31.57
     - 30.49
     - 640x640x3
     - 7.61
     - 26.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5s_seg/yolov5s-seg-int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov5s_seg/yolov5s-seg-int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/yolov5>`_
   * - yolov8m_seg
     - 40.6
     - 39.88
     - 640x640x3
     - 27.3
     - 104.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8m_seg/yolov8m-seg_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8m_seg/yolov8m-seg_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8n_seg
     - 30.32
     - 29.68
     - 640x640x3
     - 3.4
     - 12.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8n_seg/yolov8n-seg_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8n_seg/yolov8n-seg_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - yolov8s_seg
     - 36.63
     - 35.8
     - 640x640x3
     - 11.8
     - 40.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8s_seg/yolov8s-seg_int8.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/Segmentation/yolov8s_seg/yolov8s-seg_int8_vela.tflite>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
   * - stereonet
     - 91.79
     - 89.14
     - 368X1232X3, 368X1232X3
     - 5.91
     - 126.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip>`_
     - 
     - 
     - `link <https://github.com/nivosco/StereoNet>`_


.. _Pose Estimation:

Pose Estimation
---------------

.. list-table::
   :widths: 24 8 9 18 9 8 9 8 7 7
   :header-rows: 1

   * - Network Name
     - AP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - centerpose_regnetx_1.6gf_fpn  
     - 53.54
     - 52.84
     - 640x640x3
     - 14.28
     - 57.19
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_1.6gf_fpn/pretrained/2022-03-23/centerpose_regnetx_1.6gf_fpn.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_regnetx_1.6gf_fpn/centerpose_regnetx_1.6gf_fpn_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_regnetx_1.6gf_fpn/centerpose_regnetx_1.6gf_fpn_quant_vela.tflite>`_
     - `link <https://github.com/tensorboy/centerpose>`_
   * - centerpose_regnetx_800mf
     - 44.07
     - 42.87
     - 512x512x3
     - 12.31
     - 86.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_regnetx_800mf/centerpose_regnet_800_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_regnetx_800mf/centerpose_regnet_800_quant_vela.tflite>`_
     - `link <https://github.com/tensorboy/centerpose>`_
   * - centerpose_repvgg_a0
     - 39.17
     - 36.97
     - 416x416x3
     - 11.71
     - 24.76
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_repvgg_a0/pretrained/2021-09-26/centerpose_repvgg_a0.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_repvgg_a0/centerpose_repvgg_a0_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/centerpose_repvgg_a0/centerpose_repvgg_a0_quant_vela.tflite>`_
     - `link <https://github.com/tensorboy/centerpose>`_
   * - mspn_regnetx_800mf  
     - 70.8
     - 70.3
     - 256x192x3
     - 7.17
     - 2.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/mspn_regnetx_800mf.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/mspn_regnetx_800mf/mspn_regnetx_800mf_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PoseEstimation/mspn_regnetx_800mf/mspn_regnetx_800mf_quant_vela.tflite>`_
     - `link <https://github.com/open-mmlab/mmpose>`_
   * - vit_pose_small_bn
     - 72.01
     - 70.51
     - 256x192x3
     - 24.32
     - 10.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small_bn/pretrained/2023-07-20/vit_pose_small_bn.zip>`_
     - 
     - 
     - `link <https://github.com/ViTAE-Transformer/ViTPose>`_


.. _Face Detection:

Face Detection
--------------

.. list-table::
   :widths: 24 7 12 11 9 8 8 8 7 7 
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - lightface_slim 
     - 39.7
     - 39.22
     - 240x320x3
     - 0.26
     - 0.08
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/lightface_slim/lightface_slim_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/lightface_slim/lightface_slim_quant_vela.tflite>`_
     - `link <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
   * - retinaface_mobilenet_v1 
     - 81.27
     - 81.17
     - 736x1280x3
     - 3.49
     - 25.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_
     - 
     - 
     - `link <https://github.com/biubug6/Pytorch_Retinaface>`_
   * - scrfd_10g
     - 82.13
     - 82.03
     - 640x640x3
     - 4.23
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/scrfd_10g/scrfd_10g.tflite>`_
     - 
     - `link <https://github.com/deepinsight/insightface>`_
   * - scrfd_2.5g
     - 76.59
     - 76.32
     - 640x640x3
     - 0.82
     - 6.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/scrfd_2_5g/scrfd_2_5g.tflite>`_
     - 
     - `link <https://github.com/deepinsight/insightface>`_
   * - scrfd_500m
     - 68.98
     - 68.88
     - 640x640x3
     - 0.63
     - 1.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/scrfd_500m/scrfd_500m_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceDetection/scrfd_500m/scrfd_500m_quant_vela.tflite>`_
     - `link <https://github.com/deepinsight/insightface>`_


.. _Depth Estimation:

Depth Estimation
----------------

.. list-table::
   :widths: 34 7 7 11 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - RMSE
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - fast_depth
     - 0.6
     - 0.62
     - 224x224x3
     - 1.35
     - 0.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/DepthEstimation/fast_depth/fastdepth_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/DepthEstimation/fast_depth/fastdepth_quant_vela.tflite>`_
     - `link <https://github.com/dwofk/fast-depth>`_
   * - scdepthv3
     - 0.48
     - 0.51
     - 256x320x3
     - 14.8
     - 10.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/scdepthv3/pretrained/2023-07-20/scdepthv3.zip>`_
     - 
     -
     - `link <https://github.com/JiawangBian/sc_depth_pl/>`_
     

.. _Facial Landmark Detection:

Facial Landmark Detection
-------------------------
.. list-table::
   :widths: 28 8 8 16 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - NME
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - tddfa_mobilenet_v1  |star|
     - 3.68
     - 4.05
     - 120x120x3
     - 3.26
     - 0.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tddfa_mobilenet_v1.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FacialLandmark/tddfa_mobilenet_v1/tddfa_mobilenet_v1_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FacialLandmark/tddfa_mobilenet_v1/tddfa_mobilenet_v1_quant_vela.tflite>`_
     - `link <https://github.com/cleardusk/3DDFA_V2>`_


.. _Person Re-ID:

Person Re-ID
------------

.. list-table::
   :widths: 28 8 9 13 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - rank1
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - osnet_x1_0
     - 94.43
     - 93.53
     - 256x128x3
     - 2.19
     - 1.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/osnet_x1_0/osnet_x1_0_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/osnet_x1_0/osnet_x1_0_quant_vela.tflite>`_
     - `link <https://github.com/KaiyangZhou/deep-person-reid>`_
   * - repvgg_a0_person_reid_512 
     - 89.9
     - 89.3
     - 256x128x3
     - 7.68
     - 1.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/repvgg_a0_person_reid_512/repvgg_a0_person_reid_512_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/repvgg_a0_person_reid_512/repvgg_a0_person_reid_512_quant_vela.tflite>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
   * - repvgg_a0_person_reid_2048  
     - 90.02
     - 88.77
     - 256x128x3
     - 9.65
     - 0.89
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_2048/2022-04-18/repvgg_a0_person_reid_2048.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/repvgg_a0_person_reid_2048/repvgg_a0_person_reid_2048_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonReID/repvgg_a0_person_reid_2048/repvgg_a0_person_reid_2048_quant_vela.tflite>`_
     - `link <https://github.com/KaiyangZhou/deep-person-reid>`_


.. _Super Resolution:

Super Resolution
----------------

.. list-table::
   :widths: 32 8 7 11 9 8 8 8 7 7 
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - espcn_x2
     - 31.4
     - 30.3
     - 156x240x1
     - 0.02
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x2/2022-08-02/espcn_x2.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x2/espcn_x2_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x2/espcn_x2_quant_vela.tflite>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
   * - espcn_x3
     - 28.41
     - 28.06
     - 104x160x1
     - 0.02
     - 0.76
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x3/2022-08-02/espcn_x3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x3/espcn_x3_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x3/espcn_x3_quant_vela.tflite>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
   * - espcn_x4
     - 26.83
     - 26.58
     - 78x120x1
     - 0.02
     - 0.46
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x4/2022-08-02/espcn_x4.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x4/espcn_x4_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/SuperResolution/espcn_x4/espcn_x4_quant_vela.tflite>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_

.. _Face Recognition:

Face Recognition
----------------

.. list-table::
   :widths: 12 7 12 14 9 8 10 8 7 7 
   :header-rows: 1

   * - Network Name
     - lfw verification accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - arcface_mobilefacenet
     - 99.43
     - 99.41
     - 112x112x3
     - 2.04
     - 0.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceRecognition/arcface_mobilefacenet/arcface_mobilefacenet_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceRecognition/arcface_mobilefacenet/arcface_mobilefacenet_quant_vela.tflite>`_
     - `link <https://github.com/deepinsight/insightface>`_
   * - arcface_r50
     - 99.72
     - 99.71
     - 112x112x3
     - 31.0
     - 12.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_r50/pretrained/2022-08-24/arcface_r50.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceRecognition/arcface_r50/arcface_r50_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceRecognition/arcface_r50/arcface_r50_quant_vela.tflite>`_
     - `link <https://github.com/deepinsight/insightface>`_

.. _Person Attribute:

Person Attribute
----------------

.. list-table::
   :widths: 24 14 12 14 9 8 10 8 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - person_attr_resnet_v1_18
     - 82.5
     - 82.61
     - 224x224x3
     - 11.19
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonAttribute/person_attr_resnet_v1_18/person_attr_resnet_v1_18_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/PersonAttribute/person_attr_resnet_v1_18/person_attr_resnet_v1_18_quant_vela.tflite>`_
     - `link <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_

.. _Face Attribute:

Face Attribute
--------------

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - face_attr_resnet_v1_18
     - 81.19
     - 81.09
     - 218x178x3
     - 11.74
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceAttribute/face_attr_resnet_v1_18/face_attr_resnet_v1_18_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/FaceAttribute/face_attr_resnet_v1_18/face_attr_resnet_v1_18_quant_vela.tflite>`_
     - `link <https://github.com/d-li14/face-attribute-prediction>`_


.. _Zero-shot Classification:

Zero-shot Classification
------------------------

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - clip_resnet_50
     - 42.07
     - 38.57
     - 224x224x3
     - 38.72
     - 11.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50/pretrained/2023-03-09/clip_resnet_50.zip>`_
     -
     -
     - `link <https://github.com/openai/CLIP>`_
   * - clip_resnet_50x4
     - 50.31
     - 48.34
     - 288x288x3
     - 87.0
     - 41.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50x4/pretrained/2023-03-09/clip_resnet_50x4.zip>`_
     -
     -
     - `link <https://github.com/openai/CLIP>`_


.. _Low Light Enhancement:

Low Light Enhancement
---------------------

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - zero_dce
     - 16.23
     - 16.24
     - 400x600x3
     - 0.21
     - 38.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce/pretrained/2023-04-23/zero_dce.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/LowLightEnhancement/zero_dce/zero_dce_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/LowLightEnhancement/zero_dce/zero_dce_quant_vela.tflite>`_
     - `link <Internal>`_
   * - zero_dce_pp
     - 15.95
     - 15.82
     - 400x600x3
     - 0.02
     - 4.84
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce_pp/pretrained/2023-07-03/zero_dce_pp.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/LowLightEnhancement/zero_dce_pp/zero_dce_pp_quant.tflite>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/LowLightEnhancement/zero_dce_pp/zero_dce_pp_quant_vela.tflite>`_
     - `link <Internal>`_

.. _Image Denoising:

Image Denoising
---------------

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - dncnn3
     - 31.46
     - 31.26
     - 321x481x1
     - 0.66
     - 205.26
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn3/2023-06-15/dncnn3.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ImageDenoising/dncnn3/dncnn3_quant.tflite>`_ 
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ImageDenoising/dncnn3/dncnn3_quant_vela.tflite>`_
     - `link <https://github.com/cszn/KAIR>`_
   * - dncnn_color_blind
     - 33.87
     - 32.97
     - 321x481x3
     - 0.66
     - 205.97
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn_color_blind/2023-06-25/dncnn_color_blind.zip>`_
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ImageDenoising/dncnn_color_blind/dncnn_color_blind_quant.tflite>`_ 
     - `download <https://github.com/weilly0912/ATU_Model_Zoo/blob/main/ImageDenoising/dncnn_color_blind/dncnn_color_blind_quant_vela.tflite>`_
     - `link <https://github.com/cszn/KAIR>`_

.. _Hand Landmark detection:

Hand Landmark detection
-----------------------
.. list-table::
   :header-rows: 1

   * - Network Name
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Quant
     - Quant(Vela)
     - Source
   * - hand_landmark_lite
     - 224x224x3
     - 1.01
     - 0.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2023-07-18/hand_landmark_lite.zip>`_
     - 
     - 
     - `link <https://github.com/google/mediapipe>`_

