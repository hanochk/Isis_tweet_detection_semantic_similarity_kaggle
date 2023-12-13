from data import *
import numpy as np
def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list


def train_model(model, dataloader: DataSet, criterion, optimizer, device, num_epochs: int = 1,
                            **kwargs):  ##(self, dataloader: Data, **kwargs):
    all_targets = list()
    all_predictions = list()
    max_iter = kwargs.pop('max_iterations', -1)
    all_total_loss = list()
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for iters, (embeds, labels) in enumerate(tqdm.tqdm(dataloader)):
            if (max_iter != -1 and iters > max_iter - 1):
                break

            optimizer.zero_grad()
            embeds = to_device(embeds, device)

            outputs = model(embeds)

            labels = to_device(labels, device).long()
            total_loss = criterion(outputs, labels.long())
            all_total_loss.append(total_loss)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * dataloader.batch_size
            # Training acc in batch
            predictions = torch.nn.functional.softmax(outputs, dim=1)
            all_predictions.append(predictions.detach().cpu().numpy())
            max_sim_ind = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            acc = np.where(max_sim_ind - labels.numpy() == 0)[0].shape[0] / labels.shape[0]
            print(acc)

#            all_predictions.append(max_sim_ind)
            all_targets.append(labels)
            print("Step loss {} running_loss {}".format(total_loss.item(), running_loss))
        epoch_loss = running_loss / len(dataloader.dataset)
        print(epoch_loss)
        # max_sim_ind = np.argmax(similarity.detach().cpu().numpy(), axis=1)
        # acc =  np.where(max_sim_ind-labels.numpy() == 0)[0].shape[0]/labels.shape[0]
        # print(acc)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    return all_targets, all_predictions, all_total_loss

def eval_model(model, dataloader: DataSet, criterion, device, **kwargs):
    all_targets = list()
    all_predictions = list()
    all_total_loss = list()

    model.eval()

    with torch.no_grad():
        for iters, (embeds, labels) in enumerate(tqdm.tqdm(dataloader)):
            embeds = to_device(embeds, device)
            # text is already tokenized  :text_inputs = to_device(clip.tokenize(["A photo of a {}".format(x) for x in text]), self.device) =>#
            outputs = model(embeds)
            labels = to_device(labels, device).long()
            total_loss = criterion(outputs, labels)

            predictions = torch.nn.functional.softmax(outputs, dim=1)
            all_predictions.append(predictions.detach().cpu().numpy())

            all_total_loss.append(total_loss)
            all_targets.append(labels)

        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        return all_targets, all_predictions, all_total_loss



"""
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


"""