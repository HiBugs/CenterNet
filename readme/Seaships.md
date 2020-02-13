# SeaShips

> 介绍CenterNet船只检测项目如何使用。

## 1 安装环境

> 与CenterNet一致，参考[INSTALL.md](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md)

## 2 数据准备

### 2.1 数据下载

> * 下载 [SeaShips](http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips%287000%29.zip)数据集

### 2.2 数据处理

> 切换到`src/tools/`目录下，
>
> * 修改**convert_seaships_to_coco.py.py**中行末位置的**xml_path**、**split_path**和**json_file**路径，分别用于训练、验证和测试数据，然后运行。
> * 原始数据集划分数量为train1750, val1750, test3500；由于数量关系，这里我简单的将train和test数据集互换。可以再重新划分如5000-1000-1000等。

### 2.3 数据目录

数据集形成如下的目录。

> ```
> SeaShips
>     |-- |-- JPEGImages
>         |   |-- xxxxx.jpg
>         |   |-- ...
>         |-- CocoAnnotations
>         |   |-- train.json
>         |   |-- val.json
>         |   |-- test.json
>    ```

## 3 训练

opts选项按需要自己设置，运行类似如下命令：

```shell
python main.py ctdet --exp_id seaships_dla --arch dla_34 --dataset seaships
```

## 4 评估

```shell
python test.py ctdet --exp_id seaships_dla --arch dla_34 --dataset seaships --load_model ../exp/ctdet/seaships_dla/model_best.pth --keep_res
```

## 5 测试

```shell
python demo.py ctdet --demo /path/to/image/or/folder/or/video/or/webcam --arch dla_34 --load_model ../models/xxx.pth
```
