# Prepare Dataset

use Owlii to generate the training dataset.

- modify the file path at line 17 and line 75 in `Dataset/generate_dataset.py`.
- build `Dataset/train` manually.
- modify and run `Dataset/generate_dataset.py` to generate the training dataset under `Dataset/train/`.

# Prepare Thirdparty Software

- build `./Thirdparty` manually.
- add pc_error_d to `./Thirdparty`, and modify the filepath in line 94 of `utils.py`.


# Usage

## Training
- use `python train.py --help` to see the help doc.
- modify `train_config.cfg` for different configurations, or use command line configuration directly.
- modify the file path at line 8 in `Model/model.py`.
- run `python train.py --config=train_config.cfg` for trainig.
- checkpoints will be at `./Checkpoints`.

## Testing
- modify `test_config.cfg`
- run `python test.py --config=test_config.cfg` for testing.
- results will be stored at `./Results/`.

# TODO List
- improve the performance.