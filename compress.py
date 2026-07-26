import ast
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import math
import shutil
import sys
import time
import logging
import argparse

import torch

import compress_model
import arithmeticcoding_fast
from utils import *

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Source file.')
    parser.add_argument('output', type=str, help='Compressed file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
    parser.add_argument('--tempdir', '-T', type=str, help='Temporary folder name.')
    parser.add_argument('--prefix', '-p', type=str, help='Prefixes of files')
    parser.add_argument('--index', '-i', type=str, help='Index of files')
    parser.add_argument('--batchsize', '-b', type=int, default=512, help='Sample size in one batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-7, help='Weight decay.')
    parser.add_argument('--timesteps', type=int, default=16, help='The number of history symbols')
    parser.add_argument('--vocab_dim', type=int, default=16, help='The dimension of vocab.')
    parser.add_argument('--vocab_size', type=int, default=64, help='The size of vocab.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='The dimension of hidden layer.')
    parser.add_argument('--ffn_dim', type=int, default=4096, help='The dimension of ffn layer.')
    parser.add_argument('--seed', type=int, default=0, help='Random seeds.')
    parser.add_argument('--layers', type=int, help='Num of layers')
    parser.add_argument('--ratio', type=float, default=0.05, help='Pretrain ratio.')
    args = parser.parse_args(argv)
    return args


def compress(args, temp_file, series, train_data, final):
    bs, ts = args.batchsize, args.timesteps
    f = [open(temp_file + '.' + str(i), 'wb') for i in range(bs)]

    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

    prob = np.ones(args.vocab_size) / args.vocab_size
    cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    iter_num = len(train_data) // bs
    ind = np.array(range(bs)) * iter_num
    iter_num -= ts
    for i in range(bs):
        for j in range(ts):
            enc[i].write(cumul, series[ind[i] + j])
    cumul_batch = np.zeros((bs, args.vocab_size + 1), dtype=np.uint64)
    model = compress_model.MixedModel(batchsize=args.batchsize, layers=args.layers, hidden_dim=args.hidden_dim,
                                      ffn_dim=args.ffn_dim, vocab_size=args.vocab_size,
                                      vocab_dim=args.vocab_dim, timesteps=ts).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for train_index in range(iter_num):
        model.train()
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).cuda().long()
        logits = model.forward(train_batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], train_batch[:, -1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        for i in range(bs):
            enc[i].write(cumul_batch[i, :], y[i])
        ind += 1
    logging.info('Compreesion finished.')

    for i in range(bs):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if final is not None:
        logging.info("last series")
        f = open(temp_file + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(args.vocab_size) / args.vocab_size
        cumul = np.zeros(args.vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(len(final)):
            enc.write(cumul, final[j])
        logging.info("Last encode part don't need inference.")

        enc.finish()
        bitout.close()
        f.close()
    return


def main(args):
    t1 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # Preserve a scheduler or caller supplied device mask.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.layers is None:
        args.layers = int(math.log2(args.timesteps) + 1)

    if not args.prefix:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]

    if not args.tempdir:
        args.tempdir = "{}_bs{}_ts{}_v{}_h{}_f{}_l{}".format(args.prefix, args.batchsize, args.timesteps, args.vocab_dim, args.hidden_dim, args.ffn_dim, args.layers)

    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    temp_file = args.tempdir + '/compressed_temp_file'

    series = np.load(args.input)

    with open('params_' + args.prefix, 'r') as f:
        params = ast.literal_eval(f.read())

    # skmer temporarily emits full k-mer dictionaries. The archive format uses
    # a fixed A=0, C=1, G=2, T=3 mapping, so only reconstruction state remains.
    params = {
        'length': len(series),
        'suffix': params.get('Write-Chars', ''),
    }
    with open('params_'+args.prefix, 'w') as f:
        f.write(str(params))
    f.close()

    train_data = strided_app(series, args.timesteps + 1, 1)

    total_num = len(train_data)
    if total_num % args.batchsize == 0:
        compress(args, temp_file, series, train_data, None)
    else:
        ini_num = total_num // args.batchsize * args.batchsize
        compress(args, temp_file, series[:ini_num + args.timesteps], train_data[:ini_num], series[ini_num:])

    # Combined compressed results
    f = open(args.output, 'wb')
    for i in range(args.batchsize):
        f_in = open(temp_file + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    if total_num % args.batchsize != 0:
        f_in = open(temp_file + '.last', 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f.close()

    total = 0
    for ff in os.listdir(args.tempdir):
        total += os.path.getsize(args.tempdir + '/' + ff)

    # Remove temp file
    shutil.rmtree(args.tempdir)
    t2 = time.time()
    f1_size, f2_size = os.stat(args.input).st_size, os.stat(args.output).st_size
    # logging.info('Compressed File Size: {} Bytes'.format(round(f2_size, 5)))
    # logging.info('Compression Ratio: {}'.format(round(f2_size / f1_size * 8, 5)))
    # logging.info('Compression Time: {} secs'.format(round(t2 - t1, 5)))
    # logging.info('Peak GPU memory usage: {} KBs'.format(torch.cuda.max_memory_allocated() // 1024))
    logging.info(
        'The params are:\nbatchsize\tlr\thidden_dim\tvocab_dim\tffn_dim\tlayers\ttimesteps\tvocab_size\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            args.batchsize, args.lr, args.hidden_dim, args.vocab_dim, args.ffn_dim, args.layers,
            args.timesteps, args.vocab_size))


def setupLogging(debug=False):
    logLevel = logging.DEBUG if debug else logging.INFO
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stderr, level=logLevel, format=logFormat)
    logging.info("Running %s" % " ".join(sys.argv))


def run(argv):
    setupLogging()
    args = parseArgs(argv)
    starttime = time.time()
    main(args)
    logging.info("Finished in %0.2f seconds." % (time.time() - starttime))


if __name__ == '__main__':
    run(sys.argv[1:])
