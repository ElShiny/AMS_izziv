# AMS IZZIV - not so final report
Name, Surname: Matej Šajn

Model: Trans Match

Location: Its here?

## Results
Results on school data are inconclusive. Model cannot converge.

Results of the original dataset LPBA40 were as described in the original paper. 
Model converged with DICE score 0.611. will add updated results later.


## Docker Information
To setup and build docker:
```bash
    git clone https://github.com/ElShiny/AMS_izziv.git
    cd AMS_izziv
    docker build -t YOUR_NAME .
```

Docker file will automatically download CBCT images into /app/data.
To run docker with downloaded images simply run:
```bash
    docker run -it --runtime=nvidia matejs python Train.py --image_size 160 160 160 --window_size 5 5 5 --downsample True
```
If the --image_size matches the size of input images then --downsample True is not needed
If you want to process decimated images for faster training, set --image_size to preffered size and use --downsample True

Window size is cruical. Every image axis must be divisible by 32. ex: image of size (96, 128, 256) should use --window_size (3, 4, 8)


## Data Preparation
Data should be prepared in the same manner as in this dataset https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg/download/ThoraxCBCT_OncoRegRelease_06_12_23.zip 
Will elaborate later.

## Train Commands
TODO tomorrow

## Test Commands
TODO
