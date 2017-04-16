import logging
import os
import pickle
import glob
import model
import utils
import data
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

args = utils.get_arguments()

logger = logging.getLogger("model")
logger.setLevel(logging.INFO)
logger_fmt = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M")
if args.test:
    logger_handler = logging.StreamHandler(stream=sys.stdout)
else:
    date = time.strftime("%y-%m-%d", time.localtime())
    log_path = date + "_training_" + args.log_path
    logger_handler = logging.FileHandler(log_path, mode='w')
logger_handler.setFormatter(logger_fmt)
logger.addHandler(logger_handler)

if os.path.isfile(args.vocab) and os.path.isfile(args.wv_cache):
    with open(args.vocab, 'rb') as f:
        vocabulary = pickle.load(f)
    embeddings = torch.load(args.wv_cache)
else:
    vocabulary, embeddings = data.load_embeddings(args.embeddings,
                                                  args.wdim)
    with open(args.vocab, 'wb') as f:
        pickle.dump(vocabulary, f)
    torch.save(embeddings, args.wv_cache)

train_loader = torch.utils.data.DataLoader(data.SNLICorpus(
    args.train, vocabulary, pad=True, dependency=args.dependency),
    batch_size=args.batch_size, shuffle=True, num_workers=1,
    collate_fn=data.collate_transitions)
dev_loader = torch.utils.data.DataLoader(data.SNLICorpus(
    args.dev, vocabulary), batch_size=args.batch_size,
    collate_fn=data.collate_transitions)

# Set a fixed random seed
torch.manual_seed(42)
if args.model:
    network = torch.load(args.model)
else:
    if args.dependency:
        encoder = model.DependencyEncoder(args.edim,
                                          tracking_lstm=args.tracking,
                                          tracking_lstm_dim=args.tdim)
        save_prefix = args.save + "_dependency"
    else:
        encoder = model.StackEncoder(args.edim,
                                     tracking_lstm=args.tracking,
                                     tracking_lstm_dim=args.tdim)
        save_prefix = args.save + "_constituency"
    network = model.SPINNetwork(embeddings, encoder)

if args.test:
    assert args.model, "You need to provide a model to test."
    test_loader = torch.utils.data.DataLoader(data.SNLICorpus(
        args.test, vocabulary), batch_size=args.batch_size,
        collate_fn=utils.collate_transitions)
    test_loss, correct = model.test(network, test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    logger.log(network.__repr__())
    logger.log("Accuracy: %d/%d (%f), average loss: %f" %
               (correct, len(test_loader.dataset), test_accuracy,
                test_loss))
    sys.exit()

# Set up the training logger
training_logger = logging.getLogger("model.training")
print_handler = logging.StreamHandler(stream=sys.stdout)
print_fmt = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M")
print_handler.setFormatter(print_fmt)
training_logger.addHandler(print_handler)
training_logger.info(network.__repr__())
training_logger.info(
    "Batch size: %d, learning rate: %.2e, L2 regularization: %.2e" %
    (train_loader.batch_size, args.lr, args.l2))
training_logger.info("Started training.")

learning_rate = args.lr
best_dev_acc = 0
iteration = 0
parameters = filter(lambda p: p.requires_grad, network.parameters())
optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.l2)
for epoch in range(1, args.epochs + 1):
    no_batches = len(train_loader.dataset) // train_loader.batch_size
    correct_train = 0
    for batch, (prem, hypo, prem_trans,
                hypo_trans, target) in enumerate(train_loader):
        network.train()
        iteration += 1
        start_time = time.time()
        prem = Variable(prem)
        hypo = Variable(hypo)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        output = network(prem, hypo, prem_trans, hypo_trans)
        loss = F.nll_loss(output, target)
        _, pred = output.data.max(1)
        correct_train += pred.eq(target.data).sum()
        loss.backward()
        optimizer.step()
        exec_time = time.time() - start_time

        if iteration % 1000 == 0:
            new_lr = args.lr * (args.lr_decay ** (iteration /
                                                  args.lr_decay_every))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            training_logger.info(
                "Lowered learning rate to %.2e after %d iterations" %
                (new_lr, iteration))

        if iteration % 5000 == 0:
            dev_loss, correct_dev = model.test(network, dev_loader)
            dev_accuracy = correct_dev / len(dev_loader.dataset)
            training_logger.info(
                "Iteration %d: dev acc. %.4f, dev loss %.5f" %
                (iteration, dev_accuracy, dev_loss))
            if dev_accuracy > best_dev_acc:
                best_dev_acc = dev_accuracy
                save_suffix = "_devacc{:.4f}_iters{}.pt".format(
                    dev_accuracy, iteration)
                save_path = save_prefix + save_suffix
                # Remove old snapshots first
                for f in glob.glob(save_prefix + "*"):
                    os.remove(f)
                with open(save_path, 'wb') as f:
                    torch.save(network, f)
                training_logger.info(
                    "Saved new best model to %s" % (save_path))
        elif batch % args.log_interval == 0:
            training_logger.info(
                "Epoch %d, batch %d/%d, %.2f sec/batch. Loss: %.5f" %
                (epoch, batch, no_batches, exec_time, loss.data[0]))

    training_logger.info("Training acc. epoch %d: %d/%d (%.4f)" %
                         (epoch, correct_train, len(train_loader.dataset),
                          correct_train / len(train_loader.dataset)))

training_logger.info("Completed training.")
