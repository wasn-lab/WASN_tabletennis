
## TracknetV2 應用於實際場域情形
### 相機位置:
![image](https://github.com/wasn-lab/WASN_tabletennis/blob/main/images/camera_setting_formosa_univ.png)

### 設備 :
 - Window10 64bits with Nvidia 2060 GPU
 - Sony FDR AX43 handycam lock on the wall

### Run TracknetV2
```
python setup_zones_4.py <source.MP4>
python velocity_rec4.py --source <source.MP4> --weights <weight.pt> --project <place to save result>
```
##### 上述操作完成後 result(csv and output_video)會被保存在 < place to save result >內
###### 如需詳細內容，可參考 https://hackmd.io/M9QlVYDLT1qtWV2vaROphA
### Analysis TracknetV2 result
```
原始逐 frame CSV (由velocity_rec4.py生成的)
        ↓
get_max_speed.py
→ 每次揮拍的 max_speed
         ↓
tools/label_tool.py
→ 肉眼逐frame檢查每次揮拍有沒有抓取正確
        ↓
add_median_speed.py
→ 加入 median / mean 統計列
        ↓
draw_plot.py
→ 繪製速度變化圖
```
###### 如需詳細內容，可參考 https://hackmd.io/@9EFGM0VJSuut8PiUEwtTpg/Bk4IaEYV-g

### nvidia_toolkit , pytorch_version  安裝說明
![image](https://github.com/wasn-lab/WASN_tabletennis/blob/main/images/hardware_info.PNG)
- 輸入 nvidia-smi ，查看自己驅動的版本。
  ![image](https://github.com/wasn-lab/WASN_tabletennis/blob/main/images/nvidia_smi_result.PNG)
 - 前往 https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 保證自己的驅動版本數大於需求，
 - 像我這裏的驅動版本數為560.94，滿足要求。
 ![image](https://github.com/wasn-lab/WASN_tabletennis/blob/main/images/nvidia_driver_manual.PNG)
 -  如果發現自己的版本數不滿足要求，前往https://www.nvidia.com/en-us/drivers/ 更新
 -  由於官網的stable version 需python 3.10 因此我採用以下指令安裝pytorch
 -  ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126```
:::
 
### 環境安裝(CONDA)

```
conda create -n <proj_name> Python=3.9.23
conda activate <proj_name>
pip install -r requirements.txt
git clone  https://github.com/wasn-lab/WASN_tabletennis.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
simple gpu test:
```python gpu_test.py```

### 檔案結構 : 
```
│  add_median_speed.py
│  draw_plot.py
│  get_max_speed.py
│  gpu_test.py
│  max_speed_analysis.py
│  pg1_50_epoch.pt
│  requirements.txt
│  setup_zones.py
│  velocity_rec4.py
│  visual_roi.py
│
├─data
│  └─data_example
│          ex.mp4
│          ex_roi_vis.jpg
│          speed_zone.json
│          table_zone.json
│
├─models
│  │  tracknet.py
│
├─runs
│
├─tools
│      check_labels.py
│      Frame_Generator.py
│      Frame_Generator_batch.py
│      Frame_Generator_rally.py
│      handle_Darklabel.py
│      handle_tracknet_dataset.py
│      kalman_track_ball.py
│      kalman_track_ball_2.py
│      kalman_track_ball_3.py
│      label_tool.py
│
└─utils
    │  augmentations.py
    │  dataloaders.py
    │  general.py
    
    
```



