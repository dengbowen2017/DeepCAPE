import numpy as np
import hickle as hkl
import os

genome_data_file_path = './raw_data/Chromosomes/'
genome_negative_data_file_path = './raw_data/res_complement.bed'

loaded_genome_data_file_path = './processed_data/whole_sequences.hkl'
generated_negative_data_file_path = './processed_data/negative_sample_pool.bed'

sample_length = 300

# load the whole genome sequence
def loadGenomeSequences():
    chrs = list(range(1, 23))
    chrs.extend(['X', 'Y'])
    keys = ['chr' + str(x) for x in chrs]

    if os.path.isfile(loaded_genome_data_file_path):
        print('Corresponding hkl file exists')
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

# load all the negative sample data
def generateNegativeSamplePool(sequences):
    print('Generating negative sample pools')
    fin = open(genome_negative_data_file_path, 'r')
    fout = open(generated_negative_data_file_path, 'w')

    negative_sequences = fin.readlines()

    for line in negative_sequences:
        chrkey, start, end = line.strip().split('\t')
        start = int(start)
        end = int(end)
        orig_length = end - start

        if orig_length >= sample_length:
            for i in range(0, orig_length % sample_length):
                startpos = start + i * sample_length
                endpos = start + (i + 1) * sample_length
                sequence = sequences[chrkey][startpos: endpos]
                if ('n' not in sequence) and ('N' not in sequence):
                    sequence = sequence.upper()
                    gc_content = 1.0 * (sequence.count('G') + sequence.count('C')) / sample_length
                    fout.write('%s\t%d\t%d\t%.3f\t.\t.\n' % (chrkey, startpos, endpos, gc_content))

    fin.close()
    fout.close()

def preprocessing():
    sequences = loadGenomeSequences()
    generateNegativeSamplePool(sequences)
    # generate positive samples

    # generate corresponding negative samples

# temp
if __name__ == "__main__":
    preprocessing()