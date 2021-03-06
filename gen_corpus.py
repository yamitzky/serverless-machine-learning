import glob
import random


if __name__ == '__main__':
    paths = glob.glob('./Reuters21578-Apte-115Cat/training/*/*')
    random.shuffle(paths)
    with open('corpus.txt', 'w') as w:
        for path in paths:
            split = path.split('/')
            category = split[-2]
            with open(path) as f:
                line = ' '.join(f.read().split()).lower()
                w.write('{}\t{}\n'.format(category, line))
