# EllipssianNet

## 0. Pre-Requisite
```bash
conda env create -f environment.yml
conda activate ellipssianNet
```

## 1. Run (Inference)
```bash 
python train.py --img_path "C:/your_img_path" 
```
or
```bash
bash run.sh 
```

## 2. Create Dataset
```bash 
python create_dataset.py --save_path "C:/your_dataset_path"
```
or
```bash
bash create_dataset.sh
```

<div align="center">

<table>
  <tr>
    <td align="center"><img src="created_data_img/voronoi_000070.png" width="200"/></td>
    <td align="center"><img src="created_data_img/edges_000070.png" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Voronoi diagram (used as GT)</td>
    <td align="center">Edges of the diagram</td>
  </tr>
</table>
</div>


<div align="center">
<table>
  <tr>
    <td align="center"><img src="created_data_img/gradient_000070.png" width="200"/></td>
    <td align="center"><img src="created_data_img/center_000070.png" width="200"/></td>
    <td align="center"><img src="created_data_img/ellipssian_000070.png" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Gradient map (used as GT)</td>
    <td align="center">Center probability map (used as GT)</td>
    <td align="center">Rendered Ellipssians</td>
  </tr>
</table>
</div>


### 2.1 Basic parameters
 - --batch 
   - Number of dataset to be created (100 by default)
   - ``python create_dataset.py --batch 200 ``  
 - --render 
   - Visualize(or not) the created dataset (True by default)
   - ``python create_dataset.py --render True ``  
   - ``python create_dataset.py --render False ``  
 - --save_path
   - Path to store the created dataset (Blank by default - does not save) 
   - ``python create_dataset.py --save_path "C:/your_path" `` 
### 2.2 Batch command parameters
 - --iteration
   - Considers the code is running with nth iteration (0 by default)
   - Indices of dataset are computed with this parameter.
 - --begin_batch
   - Beginning index number (0 by default)  

### 2.3 Usage in .sh file
 - The following command runs the code 10 (0-9) iterations. In total, dataset size of 2000 is to be created.
   ```
   for (( x=0; x<=9; x++ ))
   do
      iter=$((x+1))
      echo "Running iteration $iter/3"
      python create_dataset.py --iteration $x --batch 200 
   done
   ```

 - The following command creates dataset size of 190 (200 - 10), beginning with 10, ending with 199.
    ```
       python create_dataset.py --batch 200 --begin_batch 10
    ```
---
### 2.4 For whom may be curious
 - The following command creates dataset size of 1900 ((200 - 10) * 10).
 - The indices of dataset will be 10-199, 210-399, 410-599, etc. 
   ```
   for (( x=0; x<=9; x++ ))
   do
      iter=$((x+1))
      echo "Running iteration $iter/3"
      python create_dataset.py --iteration $x --batch 200 --begin_batch 10 
   done
   ```
   
---

## 3. Train
```bash
python train.py  --dataset_path "C:/your_path" --chkpoint_save_path "C:/your_path" --epoch 100 --dataset_num 50000 --batch_size 10
```
