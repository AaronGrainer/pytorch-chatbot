# Pytorch Chatbot

A simple example on implementing a seq2seq bidirectional GRU encoder-decoder network using pytorch for creating a chatbot. The cornell movie-dialogs corpus dataset is used for training: 

* 220,579 conversational exchanges between 10,292 pairs of movie characters
* 9,035 characters from 617 movies
* 304,713 total utterances

The dataset is preprocessed into a suitable format for training. 

This seq2seq network uses Luong's Global Attention to improve training by paying attention only to certain parts of the encoder output sequence.  

## Getting Started

### Prerequisites

The program requires the following dependencies.  


* python 3.6
* pytorch
* CUDA (for using GPU)


### Installing

1. Download movie corpus data from [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), create a new folder called 'data' in the current directory and place the dataset in that folder. 

2. Create new virtual environment (recommended)

```
conda create -n pytorch-chatbot python=3.6
```

3. Install required python packages

```
pip install -r requirements.txt
```

## Running

### Training and Testing

To simultaneously train and evaluate the model, simply run `main.py`. 

The model is saved at certain training intervals. By saving all the metadata of the model-in-training, the model can continue training midway through or be loaded for evaluation. You can also chat with the trained chatbot by uncommenting "evaluateInput". 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The Pytorch Team

