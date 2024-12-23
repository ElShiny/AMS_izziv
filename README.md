# AMS IZZIV - not so final report
Name, Surname: Matej Šajn

Model: Trans Match
```bash
```

Location: Its here?

## Challenge
This is a fork of original TransMatch model from:
[here](https://github.com/tzayuan/TransMatch_TMI)

## Results
Results on school data are inconclusive. Model cannot converge... yet

Results of the original dataset LPBA40 were as described in the original paper. 
Model converged with DICE score 0.611 using image decimation. will update with results later.

## Auto setup script
This project is equipped with a automatic setup script **setup.sh** that will prepare *almost* everything for you.
Check the **Releases** tab.
The script downloads this repository, installs training datasets and builds a docker image just for you.
You can also do everything manually, but I cannot guarantee statisfying results.

The downloaded datasets are LPBA40, dataset from original repo, and ThoraxCBCT, recieved from our school.

### Installing with setup script
Run following command
```bash
wget https://github.com/ElShiny/AMS_izziv/releases/download/Release/setup.sh && chmod u+x setup.sh
```
```bash
. setup.sh
```
Do not use / in `. setup.sh` as enviroment variables will not export correctly.
If you want to use WandB input the API key when prompted. This will create a temporary enviroment variable which you can pass to a docker container.

## Docker Information
If you used the setup script you can skip this command. This creates a new docker image.
Also useful if you want to rebuild image with updated code
```bash
docker build -t transmatch .
```
Docker container has 3 possible mounted volumes. `/app/data` mounts to your data folder with images. `/app/Checkpoint` mounts to folder where it can periodically save snapshots of trained network. `/app/out_fields` mounts to folder for outputted deformation fields. Only used when testing the model.<br/>
If you use Nvidia GPU include `--runtime=nvidia`. Training script is `Train.py` and testing is `Infer.py`. If you want interactive shell use `-it`. To use WandB KEY add `-e WANDB_API_KEY`<br/>

Example commands:
```bash
docker run --runtime=nvidia -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py
```
```bash
docker run --runtime=nvidia -e WANDB_API_KEY -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py
```

## Data Preparation
Data should be prepared in the same manner as in this dataset:<br/>
https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg/download/ThoraxCBCT_OncoRegRelease_06_12_23.zip <br/>
All the images should be in the same folder. Their relations and uses should be configured in a dataset configuration JSON file. Labels should be in a separated folder and their names must be the same as the pictures they represent. Image masks should also follow the label rules.

## Train Commands
### Running with ThoraxCBCT dataset
To run docker with ThoraxCBCT dataset simply run:
```bash
docker run --runtime=nvidia -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py --image_size 160 160 160 --window_size 5 5 5 --downsample True
```
Or to run it with WandB:
```bash
docker run --runtime=nvidia -it -e WANDB_API_KEY -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py --image_size 160 160 160 --window_size 5 5 5 --downsample True 
```

### Running with LPBA40 dataset
To run docker with LPBA40 dataset run:
```bash
docker run -it -v ./data:/app/data -v ./output:/app/Checkpoint --runtime=nvidia transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label --DICE_lst 21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43  44  45  46  47  48  49  50  61 62 63  64  65  66  67  68  81  82  83  84  85  86  87  88  89  90  91  92  101  102  121  122  161  162 163  164  165  166
```
Or to run it with WandB:
```bash
docker run --runtime=nvidia -it -e WANDB_API_KEY -v ./data:/app/data -v ./output:/app/Checkpoint  transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label --DICE_lst 21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43  44  45  46  47  48  49  50  61 62 63  64  65  66  67  68  81  82  83  84  85  86  87  88  89  90  91  92  101  102  121  122  161  162 163  164  165  166
```

Using --DICE_lst is not necessary. It is used for DICE score calculation. Useful for keeping track of the model training.<br/>

If the --image_size matches the size of input images then --downsample True is not needed.<br/>
If you want to process decimated images for faster training, set --image_size to preffered size and use --downsample True<br/>
Window size is cruical. Every image axis should be divisible by 32. ex: image of size (96, 128, 256) should use --window_size (3, 4, 8)<br/>
Image size of 160, 160, 192 is about as much as RTX 4060 can manage. Some code optimisation is necessary.

## Test Commands
### Testing with ThoraxCBCT dataset
Similar as train command, added volume mount for output deformation fields. Use the same image and window sizes you used in training.
```bash
docker run -it -v ./data:/app/data -v ./output:/app/Checkpoint -v ./out_fields:/app/out_fields  --runtime=nvidia transmatch python Infer.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --model_save_dir /app/Checkpoint/dsc0.5975epoch011.pth.tar
```

### Testing with LPBA40 dataset
```bash
docker run --runtime=nvidia -it -e WANDB_API_KEY -v ./data:/app/data -v ./output:/app/Checkpoint -v ./out_fields:/app/out_fields transmatch python Infer.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label
```