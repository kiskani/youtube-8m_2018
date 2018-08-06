import os
from collections import defaultdict, Counter

files       =  os.listdir('/hb/scratch/mkarimz1/yt8m/rank5-2017/output')
FILE_PATH   =  '/hb/scratch/mkarimz1/yt8m/rank5-2017/output/'

SIGFIGS = 6
w = 1 

def read_models(blend=None):
    if not blend:
        blend = defaultdict(Counter)
    for m in files:
        print(m)
        with open(os.path.join(FILE_PATH, m), 'r') as f:
            f.readline()
            for l in f:
                id, r = l.split(',')
                id, r = str(id), r.split(' ')
                n = len(r) // 2
                for i in range(0, n, 2):
                    k = int(r[i])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend

def write_models(blend, file_name, total_weight=len(files)):
    with open(os.path.join('/hb/scratch/mkarimz1/yt8m/rank5-2017/averaging_code',file_name), 'w') as f:
        f.write('VideoID,LabelConfidencePairs\n')
        for id, v in blend.items():
            l = ' '.join(['{} {:{}f}'.format(t[0]
                                            , float(t[1]) / 10 ** (SIGFIGS - 1) / total_weight
                                            , SIGFIGS) for t in v.most_common(20)])
            f.write(','.join([str(id), l + '\n']))
    return None

avg = read_models()
write_models(avg, 'combined_submission_2.csv')
