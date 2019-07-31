import unicodedata
import re

def printLines(file, n=10):
  with open(file, 'rb') as datafile:
    lines = datafile.readlines()
  for line in lines[:n]:
    print(line)


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
  )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  s = re.sub(r"\s+", r" ", s).strip()
  return s


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p, max_length):
  # Input sequences need to preserve the last word for EOS token
  return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


# Filter pairs using filterPair condition
def filterPairs(pairs, max_length):
  return [pair for pair in pairs if filterPair(pair, max_length)]


def indexesFromSentence(voc, sentence, eos_token):
  return [voc.word2index[word] for word in sentence.split(' ')] + [eos_token]
