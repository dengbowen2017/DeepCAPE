import numpy as np
import hickle as hkl
import os

genome_data_file_path = './raw_data/Chromosomes/'
genome_negative_data_file_path = './raw_data/res_complement.bed'
genome_positive_data_file_path = './raw_data/epithelial_cell_of_esophagus_enhancers.bed'

loaded_genome_data_file_path = './processed_data/whole_sequences.hkl'
negative_sample_pool_file_path = './negative_sample_pool/'

chrs = list(range(1, 23))
chrs.extend(['X', 'Y'])
keys = ['chr' + str(x) for x in chrs]

all_negative_sample = dict()
for key in keys:
    all_negative_sample[key] = list()

def loadGenomeSequences():
    if os.path.isfile(loaded_genome_data_file_path):
        print('The whole genome sequences hkl file exists')
        sequences = hkl.load(loaded_genome_data_file_path)
    else:
        print('Loading the whole genome data')
        sequences = dict()
        for i in range(24):
            fa = open(genome_data_file_path + keys[i] + '.fa', 'r')
            sequence = fa.read().splitlines()[1:]
            fa.close()
            sequence = ''.join(sequence)
            sequences[keys[i]] = sequence
        hkl.dump(sequences, loaded_genome_data_file_path, 'w')

    return sequences

def generateNegativeSamplePool(sequences, length):
    file_path = negative_sample_pool_file_path + 'length_'+ str(length) + '.bed'
    if os.path.isfile(file_path):
        print('Found the file containing negative samples of length ' + str(length))
        f = open(file_path, 'r')
        samples = f.readlines()
        f.close()
        return samples
    else:
        print('Generating negative sample pool of length ' + str(length))
        samples = list()
        fin = open(genome_negative_data_file_path, 'r')
        fout = open(file_path, 'w')
        negative_sequences = fin.readlines()
        for line in negative_sequences:
            chrkey, start, end = line.strip().split('\t')
            start = int(start)
            end = int(end)
            orig_length = end - start
            if orig_length >= length:
                for i in range(0, orig_length % length):
                    startpos = start + i * length
                    endpos = start + (i + 1) * length
                    sequence = sequences[chrkey][startpos: endpos]
                    if ('n' not in sequence) and ('N' not in sequence):
                        sequence = sequence.upper()
                        gc_content = 1.0 * (sequence.count('G') + sequence.count('C')) / length
                        samples.append('%s\t%d\t%d\t%.3f\t.\t.\n' % (chrkey, startpos, endpos, gc_content))
        fout.writelines(samples)
        fin.close()
        fout.close()
        return samples
    
def IsNewNegativeSample(neg_chrkey, neg_start, neg_end):
    for item in all_negative_sample[neg_chrkey]:
        if (neg_start >= item[0] and neg_start <= item[1]) or (neg_end >= item[0] and neg_end <= item[1]) or (neg_end >= item[0] and neg_start <= item[0]) or (neg_end >= item[1] and neg_start <= item[1]) : 
            return False
        else:
            return True

def generateSampleIndexes(cell_name, sequences, ratio):
    file_path = './processed_data/%s_index_ratio%d.hkl' % (cell_name, ratio)

    f = open(genome_positive_data_file_path, 'r')
    positive_samples = f.readlines()
    f.close()

    pos_index = list()
    neg_index = list()

    lens = list()
    for sample in positive_samples:
        _, start, end = sample.split('\t')[0:3]
        lens.append(int(end) - int(start))
    lens = np.array(lens)
    bot = np.percentile(lens, 5)
    top = np.percentile(lens, 95)

    for i, sample in enumerate(positive_samples):
        chrkey, start, end = sample.split('\t')[0:3]
        start = int(start)
        end = int(end)
        sequence = sequences[chrkey][start: end]
        seq_length = end - start
        if ('n' not in sequence) and ('N' not in sequence) and seq_length > bot and seq_length < top:
            pos_index.append(['%s%05d' % (cell_name, i), chrkey, start, end])
            sequence = sequence.upper()
            pos_gc = 1.0 * (sequence.count('G') + sequence.count('C')) / seq_length
            negative_samples = generateNegativeSamplePool(sequences, seq_length)

            abs_gc = list()
            for neg_sample in negative_samples:
                neg_gc = neg_sample.split('\t')[3]
                abs_gc.append(abs(pos_gc - neg_gc))
            sorted_ind = np.argsort(np.array(abs_gc)).tolist()

            count = 0
            for ind in sorted_ind:
                neg_chrkey, neg_start, neg_end = negative_samples[ind].split('\t')[0:3]
                neg_start = int(neg_start)
                neg_end = int(neg_end)
                if IsNewNegativeSample(neg_chrkey, neg_start, neg_end):
                    count += 1
                    all_negative_sample[neg_chrkey].append([neg_start, neg_end])
                    neg_index.append(['neg%05d_%d' % (i, count), neg_chrkey, neg_start, neg_end])
                    if count == ratio:
                        break
            if count != ratio:
                 print('Error: No enough negative samples')

    print('Totally {0} enhancers and {1} negative samples.'.format(len(pos_index), len(neg_index)))
    hkl.dump([pos_index, neg_index], file_path)
    return [pos_index, neg_index]

def preprocessing():
    sequences = loadGenomeSequences()
    generateNegativeSamplePool(sequences)
    # generate positive samples

    # generate corresponding negative samples

# temp
if __name__ == "__main__":
    preprocessing()