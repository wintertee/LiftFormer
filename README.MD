# LiftFormer

## Train

Download Human3.6M 3D pose data and extract to data/h36m

run `python datasets/process.py` to process dataset.

`python main.py --log_path runs/1/ --train_batchsize 1536 --test_batchsize 1024`

## Eval

`python main.py --load runs/best.ckpt --test --test_batchsize 1024`

Download `best.cpkt`: https://drive.google.com/file/d/1o3ZbeDh3O2hdkTydUXqd8J62WfO9Lib1/view?usp=sharing

## MPJPE Result

MPJPE = 39.3252 mm

| Directions | Discussion | Eating  | Greeting | Phoning | Photo   | Posing  | Purchases | Sitting | SittingDown | Smoking | Waiting | WalkDog | Walking | WalkTogether |
| ---------- | ---------- | ------- | -------- | ------- | ------- | ------- | --------- | ------- | ----------- | ------- | ------- | ------- | ------- | ------------ |
| 31.3420    | 39.1426    | 34.4639 | 35.2884  | 38.6983 | 51.0125 | 37.0098 | 33.9597   | 53.0999 | 52.6290     | 37.3606 | 43.8358 | 38.8236 | 31.0698 | 32.1434      |

## Params

```bash
$python main.py -h
usage: main.py [-h] [--data_path DATA_PATH] [--log_path LOG_PATH]
               [--load LOAD] [--test] [--d_model D_MODEL]
               [--receptive_field RECEPTIVE_FIELD] [--n_layers N_LAYERS]
               [--n_head N_HEAD] [--d_in D_IN] [--d_out D_OUT]
               [--d_inner D_INNER] [--d_k D_K] [--d_v D_V] [--dropout DROPOUT]
               [--pre_LN] [--no-pre_LN] [--use_noam] [--no-use_noam]
               [--ReduceLROnPlateau] [--no-ReduceLROnPlateau] [--lr LR]
               [--epochs EPOCHS] [--train_batchsize TRAIN_BATCHSIZE]
               [--test_batchsize TEST_BATCHSIZE] [--num_workers NUM_WORKERS]
               [--pin_memory] [--no-pin_memory] [--log_img_freq LOG_IMG_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to dataset
  --log_path LOG_PATH
  --load LOAD           path to load a pretrained checkpoint
  --test                test
  --d_model D_MODEL     demension in tranformer model
  --receptive_field RECEPTIVE_FIELD
                        receptive field of network
  --n_layers N_LAYERS   number of transformer encoder
  --n_head N_HEAD       number of head in attention module
  --d_in D_IN           demension of input, should be 2 * number of joints
  --d_out D_OUT         demension of output, should be 3 * number of joints
  --d_inner D_INNER     demension of feed-forward network module
  --d_k D_K             demension of key matrix
  --d_v D_V             demension of value matrix
  --dropout DROPOUT     probability of dropout
  --pre_LN              use pre layer normalisation (default)
  --no-pre_LN           not use pre layer normalisation
  --use_noam            use Noam learning rate warm up
  --no-use_noam         not use Noam learning rate warm up (default)
  --ReduceLROnPlateau   use ReduceLROnPlateau learning rate scheduler
                        (default)
  --no-ReduceLROnPlateau
                        not use ReduceLROnPlateau learning rate scheduler
  --lr LR               learning rate
  --epochs EPOCHS       number of epochs
  --train_batchsize TRAIN_BATCHSIZE
                        batch size used for train
  --test_batchsize TEST_BATCHSIZE
                        batch size for used test
  --num_workers NUM_WORKERS
                        subprocesses to use for data loading
  --pin_memory          use pin_memory in DataLoader (default)
  --no-pin_memory       not use pin_memory in DataLoader
  --log_img_freq LOG_IMG_FREQ
                        frequency for logging image in tensorboard (steps)
```
