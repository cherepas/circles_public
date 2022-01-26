if opt.lb == 'pc' and opt.rotate_output and \
        opt.outputt != 'f_n':
    outputs_1 = t.reshape(outputs,
                          (outputs.shape[0], 3, nsp))
    outputs_1 = outputs_1.cuda() if iscuda else outputs_1
    outputs_2 = t.zeros(cbs, 3, nsp)
elif all([opt.lb == 'pc' or opt.lb == 'f',
          opt.outputt == 'f_n',
          opt.merging_order == 'color_channel']):
    outputs_2 = t.einsum('bh,bph->bp', outputs, y_n2[:cbs])
elif all([opt.lb == 'pc' or opt.lb == 'f',
          opt.outputt == 'pc',
          opt.merging_order == 'color_channel']):
    outputs = t.cat(
        (outputs.reshape([outputs.shape[0], 3, -1]),
         t.ones(nsp).repeat(outputs.shape[0], 1, 1).cuda()),
        axis=1)
    mat = t.einsum('ij, kij-> kij',
                   basisminv, prmat[:, 0, :, :])
    outputs = t.einsum('ijk,ijn->ikn', outputs, mat)
    outputs_2 = (outputs / t.unsqueeze(outputs[:, 3], axis=1))[:, :, :3]
    outputs_2 = t.transpose(outputs_2, 1, 2)
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
if all([lb == 'f', opt.criterion == 'L1',
        opt.outputt != 'f_n']):
    loss = vox2mm * t.mean(
        t.abs(GT - \
              opt.pc_scale * t.norm(
            t.reshape(outputs_2, (-1, 3, nsp)),
            dim=1)))
if all([lb == 'pc', opt.criterion == 'L1',
        opt.outputt != 'f_n', not opt.rotate_output,
        opt.merging_order == 'color_channel']):
    loss = vox2mm * t.mean(
        t.abs(GT - opt.pc_scale * outputs_2))
if all([lb == 'pc' or lb == 'f', opt.criterion == 'L1',
        opt.outputt == 'f_n']):
    loss = vox2mm * t.mean(
        t.abs(GT - outputs_2))