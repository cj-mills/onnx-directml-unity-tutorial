# ONNX Runtime → DirectML → Unity → Tutorial
This tutorial covers creating an object detection plugin for a [Unity](https://unity.com/) game engine project using [ONNX Runtime](https://onnxruntime.ai/docs/) and [DirectML](https://docs.microsoft.com/en-us/windows/ai/directml/dml).



https://user-images.githubusercontent.com/9126128/185470552-bd485f66-5181-4767-b5c4-4e38d04d896d.mp4



## Training Code

| Jupyter Notebook                                             | Colab                                                        | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kaggle&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GitHub Repository](https://github.com/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID-Colab.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/cj-mills/icevision-openvino-unity-tutorial/blob/main/notebooks/Icevision-YOLOX-to-OpenVINO-Tutorial-HaGRID-Kaggle.ipynb) |

> **Note:** Training on the free GPU tier for Google Colab takes approximately 11 minutes per epoch, while training on the free GPU tier for Kaggle Notebooks takes around 15 minutes per epoch.



## Kaggle Datasets

* [HaGRID Sample 30k 384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)
* [HaGRID Sample 120k 384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-120k-384p)


<details><summary><h2>Reference Images</h2></summary><br/>

| Class    | Image                                              |
| --------- | ------------------------------------------------------------ |
| call    | ![call](./images/call.jpg) |
| dislike         | ![dislike](./images/dislike.jpg) |
| fist    | ![ fist](./images/fist.jpg) |
| four         | ![four](./images/four.jpg) |
| like         | ![ like](./images/like.jpg) |
| mute         | ![ mute](./images/mute.jpg) |
| ok    | ![ ok](./images/ok.jpg) |
| one         | ![ one](./images/one.jpg) |
| palm         | ![ palm](./images/palm.jpg) |
| peace         | ![peace](./images/peace.jpg) |
| peace_inverted         | ![peace_inverted](./images/peace_inverted.jpg) |
| rock         | ![rock](./images/rock.jpg) |
| stop         | ![stop](./images/stop.jpg) |
| stop_inverted         | ![stop_inverted](./images/stop_inverted.jpg) |
| three         | ![three](./images/three.jpg) |
| three2         | ![three2](./images/three2.jpg) |
| two_up         | ![ two_up](./images/two_up.jpg) |
| two_up_inverted         | ![two_up_inverted](./images/two_up_inverted.jpg) |
</details>





## Tutorial Links

[Training Tutorial](https://christianjmills.com/posts/icevision-openvino-unity-tutorial/part-1/): covers training and exporting the model

[Part 1](https://christianjmills.com/posts/onnx-directml-unity-tutorial/part-1/): covers creating a dynamic link library (DLL) file in Visual Studio to perform inference with ONNX Runtime and DirectML

[Part 2](https://christianjmills.com/posts/onnx-directml-unity-tutorial/part-2/): covers performing object detection in a Unity project with ONNX Runtime and DirectML
