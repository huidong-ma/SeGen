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
    parser.add_argument('--tempfile', type=str, help='TemFile Folder .')
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
    # ===== sk-mer
    parser.add_argument('--K', type=int, default=3, help='K in skmer.')
    parser.add_argument('--S', type=int, default=3, help='S in skmer.')
    args = parser.parse_args(argv)
    return args

def decompress(args, temp_file, len_series, last):
    bs, ts = args.batchsize, args.timesteps
    vocab_size = args.vocab_size
    iter_num = (len_series - ts) // bs

    series_2d = np.zeros((bs, iter_num), dtype=np.uint8).astype('int')

    f = [open(temp_file + '.' + str(i), 'rb') for i in range(bs)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

    prob = np.ones(vocab_size) / vocab_size
    cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    # Decode first K symbols in each stream with uniform probabilities
    for i in range(bs):
        for j in range(min(ts, iter_num)):
            series_2d[i, j] = dec[i].read(cumul, vocab_size)

    cumul_batch = np.zeros((bs, vocab_size + 1), dtype=np.uint64)
    model = compress_model.MixedModel(batchsize=args.batchsize, layers=args.layers, hidden_dim=args.hidden_dim, ffn_dim=args.ffn_dim, vocab_size=vocab_size, vocab_dim=args.vocab_dim, timesteps=ts).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for train_index in range(iter_num - ts):
        model.train()
        train_batch = torch.LongTensor(series_2d[:, train_index:train_index + ts]).cuda()
        logits = model.forward(train_batch)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        # Decode with Arithmetic Encoder
        for i in range(bs):
            series_2d[i, train_index + ts] = dec[i].read(cumul_batch[i, :], vocab_size)

        logits = logits.transpose(1, 2)
        label = torch.from_numpy(series_2d[:, train_index + 1:train_index + ts + 1]).cuda()
        train_loss = F.cross_entropy(logits[:, :, -1], label[:, -1].long())
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    logging.info('Decompression finished.')

    fout = open(args.output, 'wb')
    fout.write(bytearray(series_2d.reshape(-1).tolist()))

    for i in range(bs):
        bitin[i].close()
        f[i].close()

    if last:
        series = np.zeros(last, dtype=np.uint8).astype('int')
        f = open(temp_file + '.last', 'rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(last):
            series[j] = dec.read(cumul, vocab_size)
        print("Last decode part don't need inference.")
        fout.write(bytearray(series.tolist()))
        bitin.close()
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

    if args.prefix is None:
        filename = os.path.basename(args.input)
        args.prefix = filename.split('.')[0]

    if not args.tempdir:
        args.tempdir = "{}_bs{}_ts{}_v{}_h{}_f{}_l{}".format(args.prefix, args.batchsize, args.timesteps, args.vocab_dim, args.hidden_dim, args.ffn_dim, args.layers)


    if os.path.exists(args.tempdir):
        shutil.rmtree(args.tempdir)
    os.mkdir(args.tempdir)
    temp_file = args.tempdir + '/compressed_temp_file'
    with open('params_' + args.prefix, 'r') as f:
        params = ast.literal_eval(f.read())

    len_series = params['length']
    suffix = params.get('suffix', '')
    f = open(args.input, 'rb')
    for i in range(args.batchsize):
        f_out = open(temp_file + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()

    f_out = open(temp_file + '.last', 'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    if (len_series - args.timesteps) % args.batchsize == 0:
        decompress(args, temp_file, len_series, 0)
    else:
        last_length = (len_series - args.timesteps) % args.batchsize + args.timesteps
        decompress(args, temp_file, len_series, last_length)
    shutil.rmtree(args.tempdir)

    # Reconstruction
    logging.info('Begin Reconstruction...')
    with open(args.output, 'rb') as f:
        series = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()

    re_start = time.time()
    reconstructed_series = recover_data(args.K, args.S, series, suffix)
    with open(args.output, "w") as file_w:
        file_w.write(reconstructed_series)
    file_w.close()

    t2 = time.time()
    print('Reconstruction Time: {} secs'.format(round(t2-re_start, 5)))
    print('Decompression Time: {} secs'.format(round(t2-t1, 5)))

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
