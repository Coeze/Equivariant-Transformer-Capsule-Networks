import os
from src.loss import DynamicRoutingLoss
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.datasets import DATASET_CONFIGS
from torch.utils.tensorboard import SummaryWriter
import time
import src.models as m

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def train(model, train_loader, val_loader, scheduler, optimizer, args):
    """
    Train the model on the training set.

    A checkpoint of the model is saved after each epoch
    and if the validation accuracy is improved upon,
    a separate ckpt is created for use on the test set.
    """
    
    # train_patience = 10 # Uncomment to use early stopping
    
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'logs'))
    if args.rank == 0 or args.rank == -1:
        print("\n[*] Train on {} samples, validate on {} samples".format(
            len(train_loader.dataset), len(val_loader.dataset))
        )
    best_valid_acc = 0 

    loss_fn = DynamicRoutingLoss()
    counter = 0

    for epoch in range(0, args.epochs):
        # get current lr
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            break

        if args.rank == 0 or args.rank == -1:
            print(
                    '\nEpoch: {}/{} - LR: {:.1e}'.format(epoch+1, args.epochs, lr)
            )

        # train for 1 epoch
        train_loss, train_acc = train_one_epoch(train_loader, loss_fn, model, epoch, optimizer, writer, args)


        # evaluate on validation set
        with torch.no_grad():
            valid_loss, valid_acc = validate(model, loss_fn, val_loader, epoch, writer, args)


        if args.rank == 0 or args.rank == -1:
            msg1 = "train loss: {:.3f} - train acc: {:.3f}"
            msg2 = " - val loss: {:.3f} - val acc: {:.3f}"

            is_best = valid_acc > best_valid_acc
            if is_best:
                counter = 0
                msg2 += " [*]"

            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                counter += 1
            '''
            if counter > train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            '''

            best_valid_acc = max(valid_acc, best_valid_acc)

            # Only save checkpoints on rank 0
            if args.rank == 0 or args.rank == -1:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_valid_acc': best_valid_acc
                }

                # Save the checkpoint to a file
                torch.save(checkpoint, f'{args.save_dir}/{args.model_name}/{args.dataset}_best_{args.num_caps}_{args.depth}.pth')

        # decay lr
        scheduler.step()

    if args.rank == 0 or args.rank == -1:
        print(best_valid_acc)
        writer.close()
        
    return model


def train_one_epoch(train_loader, loss_fn, model, current_epoch, optimizer, writer, args):
    """
    Train the model for 1 epoch of the training set.

    An epoch corresponds to one full pass through the entire
    training set in successive mini-batches.

    This is used by train() and should not be called manually.
    """
    model.train()

    # Replace AverageMeter with simple variables
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    tic = time.time()
    
    # Only use tqdm on rank 0 or when not using distributed training
    if args.rank == 0 or args.rank == -1:
        pbar = tqdm(total=len(train_loader.dataset))
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(args.device, non_blocking=True), y.cuda(args.device, non_blocking=True)
        b = x.shape[0]
        out = model(x)
        loss = loss_fn(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # Update running statistics
        total_loss += loss.data.item() * b
        total_acc += acc.data.item() * b
        total_samples += b
        
        # Calculate current average
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        # compute gradients and update SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Only update progress bar on rank 0 or when not using distributed training
        if args.rank == 0 or args.rank == -1:
            toc = time.time()
            pbar.set_description(
                (
                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                        (toc-tic), loss.data.item(), acc.data.item()
                    )
                )
            )
            pbar.update(b)

        if writer and (args.rank == 0 or args.rank == -1):
            iteration = current_epoch*len(train_loader) + i
            writer.add_scalar('train_loss', loss, iteration)
            writer.add_scalar('train_acc', acc, iteration)

    if args.rank == 0 or args.rank == -1:
        pbar.close()

    # If using distributed training, we need to synchronize the statistics across all processes
    if dist.is_initialized():
        # Create a tensor for each stat on the current device
        loss_tensor = torch.tensor([total_loss], device=args.device)
        acc_tensor = torch.tensor([total_acc], device=args.device)
        samples_tensor = torch.tensor([total_samples], device=args.device)
        
        # Sum across all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # Update variables with synchronized values
        total_loss = loss_tensor.item()
        total_acc = acc_tensor.item()
        total_samples = samples_tensor.item()

    # Calculate final averages
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    return avg_loss, avg_acc

def validate(model, loss_fn, valid_loader,current_epoch ,writer, args):
    """
    Evaluate the model on the validation set.
    """
    model.eval()

    # Replace AverageMeter with simple variables
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(valid_loader):
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)
        loss = loss_fn(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # Update running statistics
        batch_size = x.size()[0]
        total_loss += loss.data.item() * batch_size
        total_acc += acc.data.item() * batch_size
        total_samples += batch_size

    # If using distributed training, we need to synchronize the statistics across all processes
    if dist.is_initialized():
        # Create a tensor for each stat on the current device
        loss_tensor = torch.tensor([total_loss], device=args.device)
        acc_tensor = torch.tensor([total_acc], device=args.device)
        samples_tensor = torch.tensor([total_samples], device=args.device)
        
        # Sum across all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # Update variables with synchronized values
        total_loss = loss_tensor.item()
        total_acc = acc_tensor.item()
        total_samples = samples_tensor.item()

    # Calculate final averages
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    

    # log to tensorboard (only on rank 0 or when not using distributed training)
    if writer is not None and (args.rank == 0 or args.rank == -1):
        writer.add_scalar('valid_loss', avg_loss, current_epoch)
        writer.add_scalar('valid_acc', avg_acc, current_epoch)

    return avg_loss, avg_acc

def test(model_pth, test_loader, args):
    """
    Test the model on the held-out test data.
    This function should only be called at the very
    end once the model has finished training.
    """
    correct = 0

    device = torch.device('cuda')

    # load the best checkpoint
    model = m.ETCAPS(backbone=args.encoder, num_caps=args.num_caps, caps_size=4, depth=1, cfg_data=DATASET_CONFIGS[args.dataset]).to(device)
    checkpoint = torch.load(model_pth, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / len(test_loader.dataset)
    error = 100 - perc
   
    print(
        '[*] Test Acc: {}/{} (acc - {:.2f}% - err - {:.2f}%)'.format(
            correct, len(test_loader.dataset), perc, error)
    )
    return error
