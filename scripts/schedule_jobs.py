import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/pytorch-fair-ranking/runs/train_sentiment_sse'
SCRIPT_ROOT='/scratch/cluster/pkar/pytorch-fair-ranking/scripts'

mapping_dict = {
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL',
    # Script parameters
    '__JOBNAME__': ['lr_2e-3', 'lr_6e-4', 'lr_2e-4'],
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/pytorch-fair-ranking',
    '__MODE__': 'train_sentiment',
    '__DATA_DIR__': '/scratch/cluster/pkar/pytorch-fair-ranking/data',
    '__NWORKERS__': '4',
    '__BSIZE__': '64',
    '__SHUFFLE__': 'True',
    '__ENC_ARCH__': 'sse',
    '__MAXLEN__': '60',
    '__DROPOUT_P__': '0.4',
    '__HIDDEN_SIZE__': '100',
    '__PRETRAINED_BASE__': 'none',
    '__OPTIM__': 'adam',
    '__LR__': ['2e-3', '6e-4', '2e-4'],
    '__WD__': '5e-4',
    '__MOMENTUM__': '0.9',
    '__EPOCHS__': '30',
    '__MAX_NORM__': '1',
    '__START_EPOCH__': '0',
    '__LOG_ITER__': '100',
    '__RESUME__': 'True',
    '__SEED__': '123',
}

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.items():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    job_name = mapping_dict['__JOBNAME__'][idx]
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, job_name)
    if os.path.isdir(mapping_dict['__LOGNAME__']):
        print ('Skipping job ', mapping_dict['__LOGNAME__'], ' directory exists')
        continue

    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.items():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
