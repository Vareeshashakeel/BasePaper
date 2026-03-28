from Dataloader import dataloader
from VID_Trans_model import VID_Trans
from Loss_fun import make_loss

import random
import torch
import numpy as np
import os
import argparse
import time

from torch_ema import ExponentialMovingAverage
from torch.cuda import amp

from utility import AverageMeter, optimizer as make_optimizer, scheduler as make_scheduler
from torch.autograd import Variable


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()
    qf, q_pids, q_camids = [], [], []

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)

            b, s, c, h, w = imgs.size()
            features = model(imgs, pids, cam_label=camids)
            features = features.view(b, -1)
            features = torch.mean(features, 0)
            features = features.data.cpu()

            qf.append(features)
            q_pids.append(pids)
            q_camids.extend(camids)

        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)

            b, s, c, h, w = imgs.size()
            features = model(imgs, pids, cam_label=camids)
            features = features.view(b, -1)

            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)

            features = features.data.cpu()
            gf.append(features)
            g_pids.append(pids)
            g_camids.extend(camids)

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ---------- ")
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve r1:", cmc[0])

    return cmc[0], mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID Teacher")
    parser.add_argument("--Dataset_name", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str, help="pretrained ViT weight path")
    parser.add_argument("--output_dir", default="./output_camera_aware_teacher", type=str)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seq_len", default=4, type=int)
    parser.add_argument("--num_instances", default=4, type=int)
    parser.add_argument("--center_w", default=0.0005, type=float)
    parser.add_argument("--attn_w", default=1.0, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset_name = args.Dataset_name
    pretrainpath = args.model_path

    train_loader, num_query, num_classes, camera_num, view_num, q_val_set, g_val_set = dataloader(
        dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        num_instances=args.num_instances
    )

    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)

    loss_fun, center_criterion = make_loss(num_classes=num_classes)

    optimizer_center = None
    if args.center_w > 0:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    scaler = amp.GradScaler()

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    epochs = args.epochs

    model = model.to(device)
    center_criterion = center_criterion.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    best_rank1 = 0.0
    best_path = os.path.join(args.output_dir, f"{dataset_name}_camera_aware_best.pth")
    latest_path = os.path.join(args.output_dir, f"{dataset_name}_camera_aware_latest.pth")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loss weights -> center: {args.center_w}, attn: {args.attn_w}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        scheduler.step(epoch)
        model.train()

        for epoch_n, (img, pid, target_cam, labels2) in enumerate(train_loader):
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            labels2 = labels2.to(device)

            with amp.autocast(enabled=use_gpu):
                target_cam = target_cam.view(-1)

                score, feat, a_vals = model(img, pid, cam_label=target_cam)

                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()

                loss_id, center = loss_fun(score, feat, pid, target_cam)
                loss = loss_id + args.center_w * center + args.attn_w * attn_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)
                scaler.step(optimizer_center)

            scaler.update()
            ema.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if use_gpu:
                torch.cuda.synchronize()

            if (epoch_n + 1) % 50 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (epoch_n + 1), len(train_loader),
                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                ))

        if epoch % args.eval_every == 0:
            model.eval()
            cmc, mAP = test(model, q_val_set, g_val_set, use_gpu=use_gpu)
            print('CMC: %.4f, mAP : %.4f' % (cmc, mAP))

            torch.save(model.state_dict(), latest_path)

            if best_rank1 < cmc:
                best_rank1 = cmc
                torch.save(model.state_dict(), best_path)
                print(f"[OK] Saved best teacher checkpoint: {best_path}")

        print("Epoch {} finished in {:.1f}s".format(epoch, time.time() - start_time))