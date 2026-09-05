# Installing additional requirements

## Convert YOLO annotation to COCO annotation

Refer to <https://github.com/SamSamhuns/ml_data_processing/tree/master/annotation_format_conv>

## Docker build and run

### Docker Build

```shell
t=yolov5_adversarial:latest && docker build --build-arg UID=$(id -u) -f adv_patch_gen/Dockerfile -t $t .
```

### Docker Run

Create shared volumes in host system before starting containers

```shell
mkdir -p "$(pwd)"/data
```

```shell
t=yolov5_adversarial:latest && docker run -ti --rm --gpus device=0 -v "$(pwd)"/data:/home/user1/app/data -v "$(pwd)"/runs:/home/user1/app/runs $t bash
```

#### Notes on requirements for pycocotools setup:

If the `pip install -r requirements.txt` is not working.

```bash
pip install Cython
sudo apt-get install python3.8-dev # or any desired python version
pip install pycocotools            # https://github.com/ppwwyyxx/cocoapi
# pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI  # depreciated
```
