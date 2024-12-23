# AMS IZZIV - not so final report
Welcome to the TransMatch repository for medical image registration. 

## Previous work
This is a fork of original TransMatch model from: [here](https://github.com/tzayuan/TransMatch_TMI)<br/>And the original [paper](https://ieeexplore.ieee.org/abstract/document/10158729/)


## Results
Results are shown for the original dataset LPBA40 with 500 epochs and a learning rate of 0.0004. Optimization is performed with adaptive moment estimation "Adam". Loss function consisted of MSE and LNCC with weights of 0.04 and 4.

Results visualisation is done with Weights and Biases module. 
![training](/images/LPBA40_run.PNG)

Results are consistent with ones in the original paper. Model converged with final DICE score of 0.66.

With the ThoraxCBCT dataset the results were inconclusive. After a lot of tweaking parameters I still could not get the model to converge. Below are results of multiple runs.
![training_Thorax](/images/ThoraxCBCT_run.PNG)



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
To remove containers and images use:
```bash
docker container rm transmatch
```
```bash
docker rmi transmatch
```

Docker container has 3 possible mounted volumes.<br/>
`/app/data` mounts to your data folder with images.<br/>
`/app/Checkpoint` mounts to folder where it can periodically save snapshots of trained network.<br/>
`/app/out_fields` mounts to folder for outputted deformation fields. Only used when testing the model.<br/>
If you use Nvidia GPU include `--runtime=nvidia`. Training script is `Train.py` and testing is `Infer.py`. If you want interactive shell use `-it`. To use WandB KEY add `-e WANDB_API_KEY`<br/>

Example commands:
```bash
docker run --runtime=nvidia --name transmatch -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py
```
```bash
docker run --runtime=nvidia --name transmatch -e WANDB_API_KEY -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py
```

## Data Preparation
Data should be prepared in the same manner as in this dataset:<br/>
https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg/download/ThoraxCBCT_OncoRegRelease_06_12_23.zip <br/>
All the images should be in the same folder. Their relations and uses should be configured in a dataset configuration JSON file. Labels should be in a separated folder and their names must be the same as the pictures they represent. Image masks should also follow the label rules.

## Train Commands
### Running with ThoraxCBCT dataset
To run docker with ThoraxCBCT dataset simply run:
```bash
docker run --runtime=nvidia --name transmatch -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True
```
Or to run it with WandB:
```bash
docker run --runtime=nvidia --name transmatch -it -e WANDB_API_KEY -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True 
```

### Running with LPBA40 dataset
To run docker with LPBA40 dataset run:
```bash
docker run --runtime=nvidia --name transmatch -it -v ./data:/app/data -v ./output:/app/Checkpoint transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label --DICE_lst 21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43  44  45  46  47  48  49  50  61 62 63  64  65  66  67  68  81  82  83  84  85  86  87  88  89  90  91  92  101  102  121  122  161  162 163  164  165  166
```
Or to run it with WandB:
```bash
docker run --runtime=nvidia --name transmatch -it -e WANDB_API_KEY -v ./data:/app/data -v ./output:/app/Checkpoint  transmatch python Train.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label --DICE_lst 21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43  44  45  46  47  48  49  50  61 62 63  64  65  66  67  68  81  82  83  84  85  86  87  88  89  90  91  92  101  102  121  122  161  162 163  164  165  166
```

Using --DICE_lst is not necessary. It is used for DICE score calculation. Useful for keeping track of the model training.<br/>

If the --image_size matches the size of input images then --downsample True is not needed.<br/>
If you want to process decimated images for faster training, set --image_size to preffered size and use --downsample True<br/>
Window size is cruical. Every image axis should be divisible by 32.<br/>
Ex: image of size (96, 128, 256) should use `--window_size 3 4 8`<br/>
Image size of 160, 160, 192 is about as much as RTX 4060 can manage. Some code optimisation is necessary.

## Test Commands
### Testing with ThoraxCBCT dataset
Similar as train command, added volume mount for output deformation fields. Use the same image and window sizes you used in training.
```bash
docker run --runtime=nvidia --name transmatch -it -v ./data:/app/data -v ./output:/app/Checkpoint -v ./out_fields:/app/out_fields transmatch python Infer.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --model_save_dir /app/Checkpoint/[MODEL_NAME]
```

### Testing with LPBA40 dataset
```bash
docker run --runtime=nvidia --name transmatch -it -v ./data:/app/data -v ./output:/app/Checkpoint -v ./out_fields:/app/out_fields transmatch python Infer.py --image_size 96 96 96 --window_size 3 3 3 --downsample True --dataset_cfg /app/data/LPBA40/dataset.json --train_dir /app/data/LPBA40/train --label_dir /app/data/LPBA40/label --model_save_dir /app/Checkpoint/[MODEL_NAME]
```

Calculated deformation fields will be in `out_fields`

## Configuration variables
`--gpu`         Set to preffered GPU if you have multiple.<br/>
`--model`       Voxelmorph 1 or 2.<br/>
`--lr`          Set learning rate.<br/>
`--epochs`      Set the number of epochs to run.<br/>
`--sim_loss`    Image similarity loss algorithm. mse or ncc.<br/>
`--alpha`       Regularisation parameter.<br/>
`--n_save_iter` Sets how frequently to save the model. Every n-th epoch.<br/>
`--image_size`  Sets the wanted image size. Ex: `--image_size 96 96 96`<br/>
`--window_size` Sets the window size.  Ex: `--window_size 3 3 3`<br/>
`--partial_results` Shows warped images and labels with current model. Only use for testing code.<br/>
`--downsample`  If you use smaller image size than input images you must set this to True.<br/>
`--DICE_lst`    Used for inputting custom label list for other datasets. Ex: `--DICE_lst 1 2 3 4`<br/>
`--norm`        Set this if you want to use normalisation. Supports minmax and meanstd normalisation.<br/>
`--label_dir`   Label data directory.<br/>
`--dataset_cfg` Dataset configuration JSON file.<br/>
`--model_save_dir`  Directory to save the model.<br/>
`--train_dir`   Directory to images.<br/>
`--mask_dir`    If you want to use masks add a maks directory link.<br/>
`--result_dir`  Directory for saving output deformation fields.<br/>

For more information check `utils/config.py`
