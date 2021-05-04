
从拼音到汉字。


### 使用：

`cd pinyin*`

#### 转换：

`python ./src/convert.py --input ./data/input.txt --output ./data/output.txt`

#### 训练：

`python ./src/train.py`. 随便改变`train.py`用的数据。
这个程序会在./data/里面造成新的start, emissions 和 transitions 的概率。
