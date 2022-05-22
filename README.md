从拼音到汉字。

### 需求分析

使用基于人工智能的模型，实现一个拼音到汉字的转换程序。
输入：几个拼音
需要输出：适合的汉字

### 项目说明

#### 马尔可夫模型训练

##### 目标

有三种我们要知道的可能性：
- Emission Probabilities / 发射概率:   获取从拼音(观察) 到汉字 (隐含状态)的可能性
    - `{'我': {'wo':1.0}, '了':{'liao':0.5, 'le':0.5}}`
- Transition Probabilities / 转移概率: 获取一个汉子被另一个汉子跟踪的可能性
    - `{'我': {'们':0.3, '是':0.8}, '了': {}}`
- Start Probabilities / 开始概率: 	   获取以汉字开头的序列的可能性
    - `{'我':0.006, '了':0.00001}`

##### 实现

首先，我从文档中提取了汉字。因为没有提供拼音，我需要使用外部工具提取拼音。这是因为在解码过程中，我们需要知道一个拼音能代表哪些汉字。我决定使用叫 pypinyin 的一个 PythonPackage。然后，我简单地统计了每个汉字的出现次数。最后，我将计数更改为概率。

#### Viterbi解码

##### 目标

根据一个拼音序列，用计算效率高的方式找到最可能的汉字序列。

##### 实现

对于输入序列中的每个拼音，我首先检索出各自的概率。对于每个可能的汉字，我计算其概率。在动态字典中，我一直保存当前和上一个状态的分数。去下一个状态时，程序就会删除状态两个字段之前的分数。最后，得出得分最高的隐含状态序列。

### 难度分析

我的中文没有那么好，所以有可能我的看法有错。可是我一个人觉得这个问题比较难，因为中文大概有60个拼音，但是10，000个汉字。所以平均每个拼音有16个对应的汉字。

### 数据

数据都是从sina下载的。`data/2016-11.txt`是训练数据的一部分。我还使用了4个类似的文件，但是由于文件大小的限制，没有包括它们。
测试数据是我自己写的。`data/input.txt`是比较简单。`data/input_hard.txt`是非常难 （我一个人也不知道那些拼音的汉字)。

### 改进思想

#### 增长数据 

用的数据只是 1000MB，所以像《似懂非懂》，《海市蜃楼》这样的词可能一次也没有出现在数据中。为了解决这个问题需要更多数据！

#### 马尔可夫性质

马尔可夫性就是下个隐含状态仅依赖于当前的状态。这样我们的计算更容易，但是质量也变得更差。《海市蜃楼》的《楼》显然不仅要看《蜃》，还要看前面的字。解决这个问题有两种方案：第一个就是看更多的过去的隐含状态。可以用一个二阶或者三阶的马尔可夫模型。第二种选择是改变模式。通过深度学习方法我们可以看每个状态模型。当然深度学习也会需要更多的计算量，但一般来说，它比马尔科夫模型更有效率。

### 操作手册

`cd pinyin*`

#### 转换：

`python ./src/convert.py --input ./data/input.txt --output ./data/output.txt`
可以改变`input.txt`的内容。

#### 训练：

需要先安装好：`pip install -r requirements.txt` (只有一个package)
训练：`python ./src/train.py`. 可以随便改变`train.py`用的数据。
这个程序会在./data/里面造成新的start, emissions 和 transitions 的概率。
