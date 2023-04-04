# Question Generator using Text2Text Transformers for the German language

[![Explore on Huggingface](https://img.shields.io/badge/%F0%9F%A4%97-Explore%20on%20Huggingface-yellow?style=for-the-badge)](https://huggingface.co/GiantTreeG)
[![License](https://img.shields.io/github/license/GiantTreeLP/german-trivia-qg?style=for-the-badge)](LICENSE)

This repository contains code for training and evaluating a text-to-text transformer model for question generation.  
The models are trained on the [deepset/germanquad](https://huggingface.co/datasets/deepset/germanquad) dataset.

## Model

The model is based on the [mT5](https://huggingface.co/docs/transformers/model_doc/mt5) model.

These models are fine-tuned on the aforementioned datasets for the amount of epochs specified in
the [script_config.json](script_config.json) file.

## Training

The training is done using the [run-qg.py](run-qg.py) script.
The script can be run with the following command:

```bash
python run-qg.py script_config.json
```

The script can be configured using the [script_config.json](script_config.json) file.

## Evaluation

The evaluation is done using the [run-qg.py](run-qg.py) script as well.
For this, change the configuration to include the attribute `"do_eval": true` and run the script.
Feel free to set `"do_train": false` as well, if you don't want to train the model.

## Inference

The inference is done using the [run-qg.py](run-qg.py) script as well.
For this, change the configuration to include the attribute `"do_predict": true` and run the script.
Feel free to set `"do_train": false` and `"do_eval": false` as well, if you don't want to train or evaluate the model.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Huggingface](https://huggingface.co/) for providing the [datasets](https://huggingface.co/docs/datasets/index),
  [transformers](https://huggingface.co/docs/transformers/index)
  and [evaluate](https://huggingface.co/docs/evaluate/index) libraries
* [deepset](https://www.deepset.ai/) for providing
  the [German Question Answering Dataset (GermanQuAD)](https://huggingface.co/datasets/deepset/germanquad)

## Contact

If you have any questions or suggestions, feel free to raise an issue.
