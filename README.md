# 2026 寒假交接
### 內容


| notable file        | descricption |
| --------------- | -------- |
| pg1_50_epoch.pt |   tracking weight file       |
| label_tool.py                |   a toolkit for you to label a video        |
|   setup_zones_4.py               |   a function to label region of interest and table.       |
|    velocity_rec4.py             |    a implementation of tracknetV2 with ROI       |
| requirements.txt            | all library for this project     |

### 攝影機桌球桌相對位置
![image](https://hackmd.io/_uploads/HJHMUhzVZx.png)


### 方法
1. ```enviornment setup and activate environment```
2. ```python setup_zones_4.py <source.MP4>```
3. ```python  velocity_rec4.py     --source <source.MP4>      --weights  <weight.pt> --project <place to save result>```

### enviornment setup :

 - python 3.9.23
##### steps:
 - **Open your terminal or command prompt**
 - **Navigate to the project directory** where the requirements.txt file is located.
 - **Activate a virtual environment**
 - **Run the installation command :**
 ```pip install -r requirements.txt```

### 使用方式:

##### setup_zones_4.py
```用法 : python setup_zone.py 影片/圖片路徑```
```輸出 : table_zone.json , speed_zone.json ```
##### velocity_rec4.py
```
usage:    velocity_rec4.py 
          --source  <source.MP4> 
          --weights <weight.pt>
          --project <place to save result>
output: a video will be save at <place to save result>

```
##### label_tool.py
```
a toolkit for you to label a video or 
you could use it as a frame by frame video viewer.

usage:    label_tool.py 

```

