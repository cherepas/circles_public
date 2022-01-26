import torch as t
def out2loss(opt, model, inputs, iscuda, nsp, cbs, y_n2, C, angles_list, lb,
             vox2mm, GT, loss_fn, moments):
    if (opt.inputt == 'img' or \
        opt.inputt == 'f') and \
            not (lb == 'pc+f' or lb == 'pose6') \
            and not opt.outputt == 'f_n':
        outputs, latent = model(inputs)
        # print('outputs', outputs.shape)
    elif (opt.inputt == 'img' or \
          opt.inputt == 'f') and \
            not lb == 'pc+f' and \
            (opt.outputt == 'f_n' or opt.outputt == 'pose6'):
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
              opt.merging_order == 'color_channel']):
        # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
        outputs_2 = t.einsum('bh,bph->bp', outputs, y_n2[:cbs])
    elif all([opt.lb == 'pc' or opt.lb == 'f',
              opt.outputt == 'pc',
              opt.merging_order == 'color_channel']):
        # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
        outputs_2 = outputs.reshape([outputs.shape[0], 3, -1])
    if opt.lb == 'pc' and opt.rotate_output and \
            opt.merging_order != 'color_channel' and \
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
            opt.merging_order == 'color_channel']):
        loss = vox2mm * t.mean(
            t.abs(GT - opt.pscale * outputs_2))
    if all([lb == 'pc', opt.criterion == 'L1n',
            opt.outputt != 'f_n', not opt.rotate_output,
            opt.merging_order == 'color_channel']):
        # print('before loss norm', t.abs(GT -opt.pscale * outputs_2).shape)
        # sys.exit()
        loss = vox2mm * t.mean(
            t.norm(GT - opt.pscale * outputs_2, dim=1))
    if all([lb == 'pose6', opt.outputt == 'pose6',
            opt.inputt == 'img']):
        # print('890', outputs.shape, moments.shape)
        # loss = t.sqrt(t.sum((outputs - moments)**2)/6)
        # print('890', (loss_fn(outputs, moments)).shape)
        loss = t.mean(loss_fn(outputs, moments))
    return loss, outputs_2, latent