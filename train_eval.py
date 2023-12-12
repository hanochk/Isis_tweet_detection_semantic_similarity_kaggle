from data import *
def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list


def train_model(model, dataloader: DataSet, loss_img, optimizer, device, num_epochs: int = 1,
                            **kwargs):  ##(self, dataloader: Data, **kwargs):
    all_targets = list()
    all_predictions = list()
    max_iter = kwargs.pop('max_iterations', -1)

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
            # text is already tokenized  :text_inputs = to_device(clip.tokenize(["A photo of a {}".format(x) for x in text]), self.device) =>#
            # if dataloader.dataset.is_classifier_uniq_cls:
            #     text_features = self.model.encode_text(to_device(dataloader.dataset.classifier_uniq_cls_tokens,
            #                                                      self.device))  # TBD : avoid re-encode pre-defined texts just read from local tensor
            # else:
            #     text_inputs = to_device(texts, self.device)
            #     text_features = self.model.encode_text(
            #         text_inputs)  # TBD : avoid re-encode pre-defined texts just read from local tensor

            outputs = model.forward(embeds)

            labels = to_device(labels, device).long()
            total_loss = loss_img(outputs, labels.long())
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * dataloader.batch_size
            # Training acc in batch
            predictions = torch.nn.functional.softmax(outputs, dim=1)
            all_predictions.append(predictions.detach().cpu().numpy())
            max_sim_ind = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            acc = np.where(max_sim_ind - labels.numpy() == 0)[0].shape[0] / labels.shape[0]
            print(acc)

            all_predictions.append(max_sim_ind)
            all_targets.append(labels)
            print("total_loss", total_loss)
            print("running_loss", running_loss)

        print(running_loss / len(dataloader.dataset))
        # max_sim_ind = np.argmax(similarity.detach().cpu().numpy(), axis=1)
        # acc =  np.where(max_sim_ind-labels.numpy() == 0)[0].shape[0]/labels.shape[0]
        # print(acc)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    return all_targets, all_predictions
