import csv
import codecs

class Preprocess:
    def __init__(self, datafile, linefile, conversationfile, lineFields, conversationFields):
        self.datafile = datafile
        self.linefile = linefile
        self.conversationfile = conversationfile
        self.lineFields = lineFields
        self.conversationFields = conversationFields

    def loadLines(self):
        """Splits each line of the file into a dictionary of fields
        """
        print("\nProcessing corpus...")
        self.lines = {}
        with open(self.linefile, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(self.lineFields):
                    lineObj[field] = values[i]
                self.lines[lineObj['lineID']] = lineObj

    def loadConversations(self):
        """Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
        """
        print("\nLoading conversations...")
        self.conversations = []
        with open(self.conversationfile, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(self.conversationFields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = eval(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])
                self.conversations.append(convObj)

    def extractSentencePairs(self):
        """Extracts pairs of sentences from conversations
        """
        qa_pairs = []
        for conversation in self.conversations:
            # Iterate over all the lines of the conversation
            # We ignore the last line (no answer for it)
            for i in range(len(conversation["lines"]) - 1):
                    inputLine = conversation["lines"][i]["text"].strip()
                    targetLine = conversation["lines"][i+1]["text"].strip()
                    # Filter wrong samples (if one of the lists is empty)
                    if inputLine and targetLine:
                        qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    def writeCSV(self):
        """Write new csv file
        """
        print("\nWriting newly formatted file...")
        delimiter = '\t'
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))
        with open(self.datafile, 'w', encoding='utf-8', newline='') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter)
            for pair in self.extractSentencePairs():
                writer.writerow(pair)












