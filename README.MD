# LiftFormer

## Train

Download Human3.6M 3D pose data and extract to data/h36m

run `python datasets/process.py` to process dataset.

`python main.py --log_path runs/1/ --train_batchsize 1536 --test_batchsize 1024`

## Eval

`python main.py --load runs/best.ckpt --test --test_batchsize 1024`

Download `best.cpkt`: https://drive.google.com/file/d/1o3ZbeDh3O2hdkTydUXqd8J62WfO9Lib1/view?usp=sharing

## MPJPE Result

MPJPE = 39.325274987216694

| Directions | Discussion | Eating  | Greeting | Phoning | Photo   | Posing  | Purchases | Sitting | SittingDown | Smoking | Waiting | WalkDog | Walking | WalkTogether |
| ---------- | ---------- | ------- | -------- | ------- | ------- | ------- | --------- | ------- | ----------- | ------- | ------- | ------- | ------- | ------------ |
| 31.3420    | 39.1426    | 34.4639 | 35.2884  | 38.6983 | 51.0125 | 37.0098 | 33.9597   | 53.0999 | 52.6290     | 37.3606 | 43.8358 | 38.8236 | 31.0698 | 32.1434      |
