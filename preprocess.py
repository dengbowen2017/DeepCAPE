import os
import numpy as np
import hickle as hkl
from sklearn.model_selection import KFold

class Preprocessor:
    cell_name = 'epithelial_cell_of_esophagus'
    ratio = 5

    all_genome_sequences_file_path = './raw_data/Chromosomes/'
    all_negative_sample_indexes_file_path = './raw_data/res_complement.bed'
    all_positive_sample_indexes_file_path = './raw_data/epithelial_cell_of_esophagus_enhancers.bed'

    all_genome_sequences_hkl_file = './processed_data/all_genome_sequences.hkl'
    negative_sample_pool_file_path = './processed_data/negative_sample_pool/'

    all_genome_sequences = dict()
    all_negative_sample_indexes = list()
    selected_negative_sample_indexes = dict()

    def __init__(self):
        os.mkdir('./processed_data')
        os.mkdir('./processed_data/negative_sample_pool')
        os.mkdir('./processed_data/dataset')

        # init selected_negative_sample_indexes
        chrs = list(range(1, 23))
        chrs.extend(['X', 'Y'])
        keys = ['chr' + str(x) for x in chrs]
        for key in keys:
            self.selected_negative_sample_indexes[key] = list()

        # init all genome sequences
        if os.path.isfile(self.all_genome_sequences_hkl_file):
            print('Found the hkl file containing all the genome sequences')
            self.all_genome_sequences = hkl.load(self.all_genome_sequences_hkl_file)
        else:
            print('Loading all the genome sequences')
            for i in range(24):
                fa = open(self.all_genome_sequences_file_path + keys[i] + '.fa', 'r')
                sequence = fa.read().splitlines()[1:]
                fa.close()
                sequence = ''.join(sequence)
                self.all_genome_sequences[keys[i]] = sequence
            hkl.dump(self.all_genome_sequences, self.all_genome_sequences_hkl_file)

        # init all negative sample indexes
        fin = open(self.all_negative_sample_indexes_file_path, 'r')
        self.all_negative_sample_indexes = fin.readlines()
        fin.close()
        

    def checkSequence(self, chrkey:str, start:int, end:int):
        sequence = self.all_genome_sequences[chrkey][start:end]
        legal = ('n' not in sequence) and ('N' not in sequence)
        return sequence, legal
    
    
    def calculateGCContent(self, sequence:str, length:int):
        seq = sequence.upper()
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        return seq, gc_content
    

    def isNewNegativeSampleIndex(self, chrkey:str, start:int, end:int):
        for index in self.selected_negative_sample_indexes[chrkey]:
            if (start >= index[0] and start <= index[1]) or (end >= index[0] and end <= index[1]) or (end >= index[0] and start <= index[0]) or (end >= index[1] and start <= index[1]) : 
                return False
        return True        
    

    def generateNegativeSamplePool(self, length:int):
        file_path = self.negative_sample_pool_file_path + 'length_'+ str(length) + '.bed'
        if os.path.isfile(file_path):
            print('Found the file containing negative samples of length {0}'.format(str(length)))
            f = open(file_path, 'r')
            samples = f.readlines()
            f.close()
            return samples
        else:
            print('Generating negative sample pool of length {0}'.format(str(length)))
            samples = list()
            fout = open(file_path, 'w')
            for index in self.all_negative_sample_indexes:
                chrkey, start, end = index.strip().split('\t')
                start = int(start)
                end = int(end)
                orig_length = end - start
                if orig_length >= length:
                    for i in range(0, orig_length // length):
                        startpos = start + i * length
                        endpos = start + (i + 1) * length
                        sequence, legal = self.checkSequence(chrkey, startpos, endpos)
                        if legal:
                            seq, gc_content = self.calculateGCContent(sequence, length)
                            samples.append('%s\t%d\t%d\t%.3f\t.\t.\n' % (chrkey, startpos, endpos, gc_content))
            fout.writelines(samples)
            fout.close()
            return samples
        
        
    def generateSampleIndexes(self):
        file_path = './processed_data/%s_sample_indexes_ratio_%d.hkl' % (self.cell_name, self.ratio)

        if os.path.isfile(file_path):
            print("Found {0} sample indexes file".format(self.cell_name))
            pos_sample_indexes, neg_sample_indexes = hkl.load(file_path)
            print('Totally {0} positive samples and {1} negative samples'.format(len(pos_sample_indexes), len(neg_sample_indexes)))
            return pos_sample_indexes, neg_sample_indexes
        else:
            f = open(self.all_positive_sample_indexes_file_path, 'r')
            all_positive_sample_indexes = f.readlines()
            f.close()

            pos_sample_indexes = list()
            neg_sample_indexes = list()

            all_length = list()
            for index in all_positive_sample_indexes:
                _, start, end = index.split('\t')[0:3]
                all_length.append(int(end) - int(start))
            all_length = np.array(all_length)
            bot = np.percentile(all_length, 5)
            top = np.percentile(all_length, 95)

            for i, index in enumerate(all_positive_sample_indexes):
                chrkey, start, end = index.split('\t')[0:3]
                start = int(start)
                end = int(end)
                seq_length = end - start
                sequence, legal = self.checkSequence(chrkey, start, end)
                if legal and seq_length > bot and seq_length < top:
                    pos_sample_indexes.append(['%s_%d' % (self.cell_name, i), chrkey, start, end])
                    sequence, pos_gc = self.calculateGCContent(sequence, seq_length)

                    negative_sample_indexes = self.generateNegativeSamplePool(seq_length)
                    abs_gc = list()
                    for index in negative_sample_indexes:
                        neg_gc = index.split('\t')[3]
                        abs_gc.append(abs(pos_gc - float(neg_gc)))
                    sorted_indexes = np.argsort(np.array(abs_gc)).tolist()

                    count = 0
                    for i in sorted_indexes:
                        neg_chrkey, neg_start, neg_end = negative_sample_indexes[i].split('\t')[0:3]
                        neg_start = int(neg_start)
                        neg_end = int(neg_end)
                        if self.isNewNegativeSampleIndex(neg_chrkey, neg_start, neg_end):
                            count += 1
                            self.selected_negative_sample_indexes[neg_chrkey].append([neg_start, neg_end])
                            neg_sample_indexes.append(['neg_%d_%d' % (i, count), neg_chrkey, neg_start, neg_end])
                            if count == self.ratio:
                                break
                    if count != self.ratio:
                        print('Error: No enough negative samples')
                        return None

            print('Totally {0} enhancers and {1} negative samples.'.format(len(pos_sample_indexes), len(neg_sample_indexes)))
            hkl.dump([pos_sample_indexes, neg_sample_indexes], file_path)
            return pos_sample_indexes, neg_sample_indexes      


    def train_test_split(self, pos_sample_indexes, neg_sample_indexes):
        file_path = './processed_data/%s_sample_indexes_split_ratio_%d.hkl' % (self.cell_name, self.ratio)
        if os.path.isfile(file_path):
            print('Found {0} splitted indexes file'.format(self.cell_name))
            pos_allsplits, neg_allsplits = hkl.load(file_path)
            return pos_allsplits, neg_allsplits
        else:
            print('Splitting the indexes into train and test parts')
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            pos_sample_indexes_num = len(pos_sample_indexes)            
            pos_allsplits = list()
            neg_allsplits = list()
            for train, test in kf.split(range(pos_sample_indexes_num)):
                train_indexes = [pos_sample_indexes[i] for i in train]
                test_indexes = [pos_sample_indexes[i] for i in test]
                pos_allsplits.append((train_indexes, test_indexes))

                neg_train = list()
                for i in train:
                    neg_train = neg_train + list(range(i*self.ratio, (i+1)*self.ratio))
                neg_test = list()
                for i in test:
                    neg_test = neg_test + list(range(i*self.ratio, (i+1)*self.ratio))
                neg_train_indexes = [neg_sample_indexes[i] for i in neg_train]
                neg_test_indexes = [neg_sample_indexes[i] for i in neg_test]
                neg_allsplits.append((neg_train_indexes, neg_test_indexes))
            hkl.dump([pos_allsplits, neg_allsplits], file_path)
            return pos_allsplits, neg_allsplits


    def dataAugmentation(self, sequences, indexes, length):
        results = list()
        for index in indexes:
            sample_id, chrkey, startpos, endpos = index[0:4]
            startpos = int(startpos)
            endpos = int(endpos)
            orig_length = endpos - startpos

            if orig_length < length:
                for offset in range(0, length - orig_length + 1):
                    start = startpos - offset
                    end = start + length
                    sequence = sequences[chrkey][start:end]
                    if ('n' not in sequence) and ('N' not in sequence):
                        results.append([sample_id, chrkey, start, end, sequence])
            
            else:
                for offset in range(0, orig_length - length + 1):
                    start = startpos + offset
                    end = start + length
                    sequence = sequences[chrkey][start:end]
                    if ('n' not in sequence) and ('N' not in sequence):
                        results.append([sample_id, chrkey, start, end, sequence])

        print('Data augmentation: from {0} samples to {1} samples'.format(len(indexes), len(results)))
        return results