import torch as t
import time
import numpy as np
import os
import sys
def out2loss(opt, model, inputs, iscuda, nsp, cbs, y_n2, C, angles_list, lb,
             vox2mm, GT, loss_fn, moments, phase, dirname):
    if opt.inputt in ('img', 'f') and \
            not lb in ('pc+f', 'pose6', 'eul', 'orient') \
            and not opt.outputt == 'f_n':
        outputs, latent = model(inputs)
        # print('outputs', outputs.shape)
    elif opt.inputt in ('img', 'f') and not lb == 'pc+f' and \
            opt.outputt == 'f_n':
        outputs = model(inputs)
    # print('outputs.shape', outputs.shape)
    # if opt.measure_time:
    #     tf = time.time()
    # if opt.wandb and opt.measure_time:
    #     wandb.log({'model output time ' + phase: tf - ts})
    # if phase == 'train' and opt.measure_time:
    #     lt[1, i_batch] = tf - ts
    # if opt.measure_time:
    #     ts = time.time()

    # if mt:
    #     print('calculate loss', time.time())
    if opt.lb == 'pc' and opt.rotate_output and \
            opt.outputt != 'f_n':
        outputs_1 = t.reshape(outputs,
                              (outputs.shape[0], 3, nsp))
        outputs_1 = outputs_1.cuda() if iscuda else outputs_1
        outputs_2 = t.zeros(cbs, 3, nsp)
    elif all([opt.lb == 'pc' or opt.lb == 'f',
              opt.outputt == 'f_n',
              opt.merging == 'color']):
        # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
        outputs_2 = t.einsum('bh,bph->bp', outputs, y_n2[:cbs])
    elif all([opt.lb == 'pc' or opt.lb == 'f',
              opt.outputt == 'pc',
              opt.merging == 'color']):
        # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
        outputs_2 = outputs.reshape([outputs.shape[0], 3, -1])
    if opt.lb == 'pc' and opt.rotate_output and \
            opt.merging != 'color' and \
            opt.outputt != 'f_n':
        for i in range(cbs):
            outputs_2[i, :, :] = t.matmul(
                t.transpose(t.squeeze(
                    C[int(angles_list[i] / 10), :, :]), 0, 1),
                outputs_1[i, :, :])
    if opt.lb == 'pc' and opt.rotate_output:
        outputs_2 = outputs_2.cuda() if iscuda else outputs_2
    # if all([lb == 'f', opt.criterion == 'L1',
    #         opt.outputt != 'f_n']):
    #     loss = vox2mm*t.mean(
    #         t.abs(GT-\
    #             opt.pscale*t.norm(
    #             outputs_2.reshape([outputs_2.shape[0], 3, -1]),
    #             dim=1)))
    if all([lb == 'pc', opt.criterion == 'L1',
            opt.outputt != 'f_n', not opt.rotate_output,
            opt.merging == 'color']):
        loss = vox2mm * t.mean(
            t.abs(GT - opt.pscale * outputs_2))
    if all([lb == 'pc', opt.criterion == 'L1n',
            opt.outputt != 'f_n', not opt.rotate_output,
            opt.merging == 'color']):
        # print('before loss norm', t.abs(GT -opt.pscale * outputs_2).shape)
        # sys.exit()
        loss = vox2mm * t.mean(
            t.norm(GT - opt.pscale * outputs_2, dim=1))
    # TODO unwrap conditions for only use case
    if all([opt.inputt == 'img', opt.outputt == 'pose6', lb == 'pose6']):
        outputs = model(inputs)
        loss = t.mean(loss_fn(outputs, moments))
        # print('')
        outputs_2 = outputs
        latent = outputs
    if all([opt.inputt == 'img', opt.outputt in ('eul'),
            lb in ('eul')]):
        # print('switcher', 81, inputs.shape, inputs.is_cuda)
        outputs = model(inputs)
        loss = t.mean(loss_fn(outputs, GT))
        # print('')
        outputs_2 = outputs
        latent = outputs
    # if opt.measure_time:
    #     tf = time.time()
    # if opt.wandb and opt.measure_time:
    #     wandb.log({'loss calc time ' + phase: tf - ts})
    # if phase == 'train' and opt.measure_time:
    #     lt[2, i_batch] = tf - ts
    # if mt:
    #     print('make backward pass', time.time())
    # if all([lb == 'pc' or lb == 'f', opt.criterion == 'L1',
    #         opt.outputt == 'f_n']):
    #     loss = vox2mm*t.mean(
    #         t.abs(GT-outputs_2))
    # if all([lb == 'pc' or lb == 'f', opt.criterion == 'L2',
    #         opt.outputt == 'f_n']):
    #     loss = vox2mm*t.sqrt(t.sum(
    #         (GT-outputs_2)**2)/\
    #            (GT.shape[0]*GT.shape[1]*GT.shape[2]))
    if all([opt.inputt == 'img', opt.outputt in ('orient'),
            lb in ('orient'), phase == 'train']):
        # print(105)
        # print('switcher', 81, inputs.shape, inputs.is_cuda)
        outputs = model(inputs)
        loss = t.mean(loss_fn(outputs, GT))
        # print('')
        outputs_2 = 0
        latent = 0
    if all([opt.inputt == 'img', opt.outputt in ('orient'),
            lb in ('orient'), opt.aug_gt == 'orient', phase=='val']):
        outputs = model(inputs)
        # print(115)
        # gt4 = t.zeros(gt3.shape[0], 12, 3, 3)
        peridx = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                  [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        # TODO write in vectorize form instead of for loop
        lo = t.zeros(outputs.shape[0], 12, 9)
        # st = time.time()
        outputs = outputs.reshape(-1,3,3)
        for j in range(outputs.shape[0]):
            outputs2 = t.matmul(t.svd(outputs[j, :,:])[0], t.svd(outputs[j, :, :])[2].T)
            for i in range(12):
                outputs3 = (-1) ** (i % 2) * outputs2[:, peridx[i // 2, :]]
                # print(129, outputs3.shape, GT.shape)
                # gt4 = (-1) ** (i % 2) * GT.reshape(-1, 3, 3)[j, :, peridx[i // 2, :]]
                # print(loss_fn(gt4.reshape(-1, 9), outputs[j, :]).shape)
                # print(114, gt4.reshape(-1, 9).shape, outputs[j,:].shape)
                lo[j, i, :] = loss_fn(GT[j,:], outputs3.reshape(9))
        # np.savetxt(os.path.join(dirname, 'val_out'), outputs.detach().cpu().numpy(), delimiter=',')
        # sys.exit()
        loss = t.mean(t.amin(lo, dim=1))
        # loss = t.mean(t.min(loss_fn(outputs, GT), loss_fn(outputs, -GT)))
        outputs_2 = 0
        latent = 0
        # fi = time.time()
        print(140, fi-st)
    return loss, outputs, outputs_2, latent

#def plotitout(lossar):