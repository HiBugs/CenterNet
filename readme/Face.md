# Face

> 介绍CenterNet人脸检测项目如何使用。

## 1 安装环境

> 与CenterNet一致，参考[INSTALL.md](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md)

## 2 数据准备

### 2.1 数据下载

> * 下载 [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html)数据集
> * 下载[RetinaFace](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA)提供的标注集，包含bounding boxes 和 five facial landmarks

### 2.2 数据处理

> 切换到`src/tools/widerface/`目录下，
>
> * 修改**get_widerface_landmark.py**中行末位置的**txt_path**和**save_path**路径，分别用于训练和验证数据，然后运行。
>
> * 修改**keypoints2coco.py.py**中行末位置的**img_path**、**txt_path**和**save_path**路径，分别用于训练和验证数据，然后运行。

### 2.3 数据目录

数据集最终形成如下的目录。

> ```
> WiderFace
>     |-- |-- WIDER_train
>         |   |-- images
>         |   |   |-- xxxxx.jpg
>         |   |   |-- ...
>         |   |-- keypoints_train.json
>         |-- WIDER_val
>         |   ...
>         |-- WIDER_test
>         |   ...
> ```

## 3 训练

opts选项按需要自己设置，运行类似如下命令

```shell
python main.py multi_pose --exp_id widerface --arch dla_34 --dataset face
```

## 4 评估

这里只评估了bbox, 未对landmark进行评估。

### 4.1 COCOAPI评估

```shell
python test.py multi_pose --exp_id widerface --arch dla_34 --dataset face --load_model ../exp/multi_pose/widerface/model_best.pth --keep_res
```

### 4.2 WIDERFACE TOOL评估

```shell
python eval_widerface.py multi_pose --exp_id widerface --arch dla_34 --dataset face --load_model ../exp/multi_pose/widerface/model_best.pth --keep_res
```

## 5 测试

```shell
python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --arch dla_34 --load_model ../models/xxx.pth
```

