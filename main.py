from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from helpers import printLines, normalizeString, filterPairs, indexesFromSentence
from process import Preprocess
from models import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os
from io import open
import itertools
import math


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), 
            len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


class Dataset:
    def __init__(self, corpus, corpus_name, datafile):
        self.corpus = corpus
        self.corpus_name = corpus_name
        self.datafile = datafile
  
    # Read query/response pairs and return a voc object
    def readVocs(self):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(self.datafile, encoding='utf-8').\
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Voc(self.corpus_name)
        return voc, pairs

    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(self):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs()
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = filterPairs(pairs, MAX_LENGTH)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs

    def trimRareWords(self, voc, pairs, MIN_COUNT):
        # Trim words used under the MIN_COUNT from the voc
        voc.trim(MIN_COUNT)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs

    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        indexes_batch = [indexesFromSentence(voc, sentence, EOS_token) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [indexesFromSentence(voc, sentence, EOS_token) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self, voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len


class Model:
    def __init__(self, batch2TrainData, teacher_forcing_ratio):
        self.batch2TrainData = batch2TrainData
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    def train(self, input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, 
              embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
      # Zero gradients
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()

      # Set device options
      input_variable = input_variable.to(device)
      lengths = lengths.to(device)
      target_variable = target_variable.to(device)
      mask = mask.to(device)

      # Initialize variables
      loss = 0
      print_losses = []
      n_totals = 0

      # Forward pass through encoder
      encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

      # Create initial decoder input (start with SOS tokens for each sentence)
      decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
      decoder_input = decoder_input.to(device)

      # Set initial decoder hidden state to the encoder's final hidden state
      decoder_hidden = encoder_hidden[:decoder.n_layers]

      # Determine if we are using teacher forcing this iteration
      use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

      # Forward batch of sequences through decoder one time step at a time
      if use_teacher_forcing:
          for t in range(max_target_len):
              decoder_output, decoder_hidden = decoder(
                  decoder_input, decoder_hidden, encoder_outputs
              )
              # Teacher forcing: next input is current target
              decoder_input = target_variable[t].view(1, -1)
              # Calculate and accumulate loss
              mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
              loss += mask_loss
              print_losses.append(mask_loss.item() * nTotal)
              n_totals += nTotal
      else:
          for t in range(max_target_len):
              decoder_output, decoder_hidden = decoder(
                  decoder_input, decoder_hidden, encoder_outputs
              )
              # No teacher forcing: next input is decoder's own current output
              _, topi = decoder_output.topk(1)
              decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
              decoder_input = decoder_input.to(device)
              # Calculate and accumulate loss
              mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
              loss += mask_loss
              print_losses.append(mask_loss.item() * nTotal)
              n_totals += nTotal

      # Perform backpropatation
      loss.backward()

      # Clip gradients: gradients are modified in place
      _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
      _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

      # Adjust model weights
      encoder_optimizer.step()
      decoder_optimizer.step()

      return sum(print_losses) / n_totals

    def trainIters(self, model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, 
                  encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, 
                  clip, corpus_name, loadFilename):
        # Load batches for each iteration
        training_batches = [self.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                            for _ in range(n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                        iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                    encoder_n_layers, decoder_n_layers, hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


    def evaluate(self, encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
        indexes_batch = [indexesFromSentence(voc, sentence, EOS_token)]

        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        
        tokens, scores = searcher(input_batch, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words


    def evaluateInput(self, encoder, decoder, searcher, voc):
        input_sentence = ''
        while(1):
            try:
                input_sentence = input('> ')
                if input_sentence == 'q' or input_sentence == 'quit': break
                input_sentence = normalizeString(input_sentence)
                output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")


def main():
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)

    printLines(os.path.join(corpus, "movie_lines.txt"))

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    linefile = os.path.join(corpus, "movie_lines.txt")
    conversationfile = os.path.join(corpus, "movie_conversations.txt")

    # Initialize lines dict, conversations list, and field ids
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    preprocess = Preprocess(datafile, linefile, conversationfile, MOVIE_LINES_FIELDS, MOVIE_CONVERSATIONS_FIELDS)
    preprocess.loadLines()
    preprocess.loadConversations()
    preprocess.writeCSV()

    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    dataset = Dataset(corpus, corpus_name, datafile)
    voc, pairs = dataset.loadPrepareData()
    
    # # Print some pairs to validate
    # print("\npairs:")
    # for pair in pairs[:10]:
    #   print(pair)

    # Trim voc and pairs
    pairs = dataset.trimRareWords(voc, pairs, MIN_COUNT)

    # Example for validation
    small_batch_size = 5
    batches = dataset.batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

  

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    #loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 1
    save_every = 500

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    model = Model(dataset.batch2TrainData, teacher_forcing_ratio)
    model.trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                     embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                     print_every, save_every, clip, corpus_name, loadFilename)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    # model.evaluateInput(encoder, decoder, searcher, voc)


if __name__ == "__main__":
    main()


