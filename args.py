import argparse

def str2bool(t):
    if t.lower() in ['true', 't', '1']:
        return True
    else:
        return False

def get_args():

    parser = argparse.ArgumentParser(description='EE 381V: Learning to Rank for Fairness in Exposure')

    # general
    parser.add_argument('--mode', default='train_sentiment', type=str, help='mode of the python script')

    # DataLoader
    parser.add_argument('--data_dir', default='./data', type=str, help='root directory of the dataset')
    parser.add_argument('--nworkers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--bsize', default=64, type=int, help='mini-batch size')
    parser.add_argument('--shuffle', default='True', type=str2bool, help='shuffle the data?')

    # Model Parameters
    parser.add_argument('--enc_arch', default='bilstm', type=str, help='sentence encoder arch [bilstm | sse]')
    parser.add_argument('--maxlen', default=60, type=int, help='max length of the sentence')
    parser.add_argument('--dropout_p', default=0.1, type=float, help='dropout probability')
    parser.add_argument('--hidden_size', default=200, type=int, help='hidden layer size')
    parser.add_argument('--pretrained_base', default=None, type=str, help='Path to the pretrained sentiment analysis model')

    # Optimization Parameters
    parser.add_argument('--optim', default='adam', type=str, help='optimizer type')
    parser.add_argument('--lr', default=2e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--max_norm', default=1, type=float, help='max grad norm')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    # Other
    parser.add_argument('--save_path', default='./trained_models', type=str, help='directory where models are saved')
    parser.add_argument('--log_dir', default='./logs', type=str, help='directory where tensorboardX logs are saved')
    parser.add_argument('--log_iter', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume if previous checkpoint exists')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')

    args = parser.parse_args()
    args.glove_emb_file = 'glove.6B.{}d.txt'.format(args.hidden_size)

    return args
