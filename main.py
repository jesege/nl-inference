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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

args = utils.get_arguments()

logger = logging.getLogger("model")
logger.setLevel(logging.INFO)
logger_fmt = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M")
if args.log_path:
    date = time.strftime("%y-%m-%d", time.localtime())
    directory, log_name = os.path.split(args.log_path)
    log_name = date + "_training_" + log_name
    log_path = os.path.join(directory, log_name)
    logger_handler = logging.FileHandler(log_path, mode='w')
else:
    logger_handler = logging.StreamHandler(stream=sys.stdout)
logger_handler.setFormatter(logger_fmt)
logger.addHandler(logger_handler)
git_commit = utils.get_git_commit_hash()
logger.info("git commit %s" % git_commit)

if args.dependency:
    DEPENDENCY_TRANSITIONS = True
else:
    DEPENDENCY_TRANSITIONS = False

if args.vocab and os.path.isfile(args.vocab):
    with open(args.vocab, 'rb') as f:
        vocabulary = pickle.load(f)

if args.wv_cache and os.path.isfile(args.wv_cache):
    assert args.vocab, "Provide a vocabulary file to load cached embeddings."
    embeddings = torch.load(args.wv_cache)
elif args.embeddings:
    vocabulary, embeddings = data.load_embeddings(args.embeddings,
                                                  args.wdim)
    if args.wv_cache:
        torch.save(embeddings, args.wv_cache)
    if args.vocab:
        with open(args.vocab, 'wb') as f:
            pickle.dump(vocabulary, f)

# Set a fixed random seed
torch.manual_seed(42)
if args.dependency:
    encoder = model.DependencyEncoder(args.edim,
                                      tracking_lstm=args.tracking,
                                      tracking_lstm_dim=args.tdim)
    save_prefix = args.save + "_dependency"
elif args.lstm:
    encoder = model.LSTMEncoder(args.wdim, args.edim)
    save_prefix = args.save + "_lstm"
elif args.bow:
    encoder = model.BOWEncoder(args.edim)
    save_prefix = args.save + "_bow"
else:
    encoder = model.StackEncoder(args.edim,
                                 tracking_lstm=args.tracking,
                                 tracking_lstm_dim=args.tdim)
    save_prefix = args.save + "_constituency"

network = model.SPINNetwork(args.wdim, len(vocabulary), encoder)
if args.model:
    network.load_state_dict(torch.load(args.model))
if args.embeddings or args.wv_cache:
    network.word_embedding.weight = nn.Parameter(embeddings)

if args.test:
    assert args.model, "You need to provide a model to test."
    assert args.vocab, "You need to provide a vocabulary file."
    logger.info(network.__repr__())
    test_loader = torch.utils.data.DataLoader(data.SNLICorpus(
        args.test, vocabulary, dependency=DEPENDENCY_TRANSITIONS),
        batch_size=args.batch_size, collate_fn=data.test_collation)
    test_loss, correct, misclassified, conf_matrix = model.test(
        network, test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    logger.info("Accuracy: %.4f (%d/%d), average loss: %.5f" %
                (test_accuracy, correct, len(test_loader.dataset),
                 test_loss))
    logger.info("Confusion matrix: \n %s" % repr(conf_matrix))
    if args.misclass:
        utils.write_misclassifications(misclassified, args.misclass)
    sys.exit()

if args.training_cache:
    if os.path.isfile(args.training_cache):
        with open(args.training_cache, 'rb') as f:
            training_corpus = pickle.load(f)
    else:
        training_corpus = data.SNLICorpus(args.train, vocabulary,
                                          seq_length=50,
                                          dependency=DEPENDENCY_TRANSITIONS)
        with open(args.training_cache, 'wb') as f:
            pickle.dump(training_corpus, f)
else:
    training_corpus = data.SNLICorpus(args.train, vocabulary, seq_length=50,
                                      dependency=DEPENDENCY_TRANSITIONS)


# TODO: What to do about the very long sentences?
train_loader = torch.utils.data.DataLoader(
    training_corpus, batch_size=args.batch_size, shuffle=True,
    num_workers=1, collate_fn=data.collate_transitions)
dev_loader = torch.utils.data.DataLoader(data.SNLICorpus(
    args.dev, vocabulary, seq_length=50, dependency=DEPENDENCY_TRANSITIONS),
    batch_size=args.batch_size, collate_fn=data.collate_transitions)

# Set up the training logger
training_logger = logging.getLogger("model.training")
# If we have a log file 
if args.log_path:
    print_handler = logging.StreamHandler(stream=sys.stdout)
    print_fmt = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M")
    print_handler.setFormatter(print_fmt)
    training_logger.addHandler(print_handler)
training_logger.info(repr(network))
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
    for batch, (prem, hypo, prem_trans, hypo_trans, masks,
                target) in enumerate(train_loader):
        network.train()
        iteration += 1
        start_time = time.time()
        prem = Variable(prem)
        hypo = Variable(hypo)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        output = network(prem, hypo, prem_trans, hypo_trans, masks)
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
            dev_loss, correct_dev = model.validate(network, dev_loader)
            dev_accuracy = correct_dev / len(dev_loader.dataset)
            training_logger.info(
                "Dev acc. iteration %d: %.4f (loss: %.5f)" %
                (iteration, dev_accuracy, dev_loss))
            if dev_accuracy > best_dev_acc:
                best_dev_acc = dev_accuracy
                save_suffix = "_devacc{:.4f}_iters{}.pt".format(
                    dev_accuracy, iteration)
                save_path = save_prefix + save_suffix
                with open(save_path, 'wb') as f:
                    torch.save(network.state_dict(), f)
                # Remove old snapshots
                for f in glob.glob(save_prefix + "*"):
                    if f != save_path:
                        os.remove(f)
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
dev_loss, correct_dev = model.validate(network, dev_loader)
dev_accuracy = correct_dev / len(dev_loader.dataset)
training_logger.info("Final dev. accuracy: %.4f" % dev_accuracy)
if dev_accuracy > best_dev_acc:
    save_suffix = "_devacc{:.4f}_iters{}_final.pt".format(
        dev_accuracy, iteration)
    save_path = save_prefix + save_suffix
    with open(save_path, 'wb') as f:
        torch.save(network.state_dict(), f)
    training_logger.info("Saved final model to %s" % save_path)
