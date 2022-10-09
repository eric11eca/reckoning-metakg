import os
import torch
import higher

from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup


def run(args, logger):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_data = NLPFewshotGymMetaLearningData(
        logger, args, args.train_dir,
        tasks=train_tasks, data_type="train",
        is_training=True)
    dev_data = NLPFewshotGymMetaLearningData(
        logger, args, args.train_dir,
        tasks=DEFAULT_SPLIT["dev"],
        data_type="dev",
        is_training=False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key): value for key, value in state_dict.items()}
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.model)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_batch = 0
    global_step = 0
    train_losses = []
    dev_losses = []
    best_accuracy = -1.0
    stop_training = False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):
            global_batch += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch[0]]

            # train batch
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            # dev batch
            batch[4], batch[5] = trim_batch(batch[4], pad_token_id, batch[5])
            batch[6], batch[7] = trim_batch(batch[6], pad_token_id, batch[7])

            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # train
                train_loss = fmodel(
                    batch[0], batch[1], batch[2], batch[3], labels=batch[3])
                train_loss = train_loss[0]
                train_losses.append(train_loss.item())
                diffopt.step(train_loss)

                # dev
                dev_loss = fmodel(batch[4], batch[5],
                                  batch[6], batch[7], labels=batch[7])
                dev_loss = dev_loss[0]
                dev_losses.append(dev_loss.detach().cpu())
                dev_loss.backward()

                if global_batch % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enough gradients
                    scheduler.step()
                    model.zero_grad()

                    if global_step % args.eval_period == 0:
                        model.eval()
                        curr_em = inference(
                            model if args.n_gpu == 1 else model.module, dev_data)
                        logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em,
                            epoch))
                        logger.info("train loss: {}; dev loss: {}".format(
                            np.mean(train_losses), np.mean(dev_losses)))
                        train_losses = []
                        dev_losses = []
                    #     if best_accuracy < curr_em:
                    #         model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    #         torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    #         logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                    #                 (dev_data.metric, best_accuracy, curr_em, epoch, global_step))
                    #         best_accuracy = curr_em
                    #         wait_step = 0
                    #         stop_training = False
                    #     else:
                    #         wait_step += 1
                    #         if wait_step >= args.wait_step:
                    #             stop_training = True
                    #             break
                    #     model.train()

                if global_step >= args.total_steps:
                    stop_training = True
                    break

            if stop_training:
                break
