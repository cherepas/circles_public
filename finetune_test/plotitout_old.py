log1 = all([rank == 0, epoch > epoch0, opt.save_output,
                    lb != 'f_n'])
                if epoch == epoch0+1 and opt.save_output and rank == 0 and\
                    phase == 'train':
                    original_stdout = sys.stdout
                    with open(jn(dirname,"opt.txt"), 'a') as f:
                        sys.stdout = f  # Change the standard output to the file we created.
                        print(opt)
                    with open(jn(dirname,"sys_argv.txt"), 'a') as f:
                        sys.stdout = f  # Change the standard output to the file we created.
                        print(sys.argv)
                    sys.stdout = original_stdout
                    # print('opt:\n',opt)
                    # print('sys.argv:\n',sys.argv)
                    # with open(jn(dirname,"job-parameters.txt"), "w") as f:
                    #     f.write(parameters_list)
                # if log1 and opt.minmax and not opt.outputt == 'f_n':
                #     ampl = opt.ampl
                #     o = np.multiply(
                #         outputs_2.detach().cpu().numpy(),
                #         tmean[3, :ampl] -
                #         tmean[2, :ampl]) + tmean[2, :ampl]
                #     gt = np.multiply(
                #         f_n.detach().cpu().numpy(),
                #         tmean[3, :ampl] -
                #         tmean[2, :ampl]) + tmean[2, :ampl]
                if all([log1, not opt.minmax,
                        not (opt.outputt == 'f_n' or opt.outputt == 'pose6')]):
                    o = outputs_2.detach().cpu().numpy()
                    # print('o stat1', getstat(o[0,2,:]))
                    # print('o shape = ', o.shape)
                elif log1 and not opt.minmax and opt.outputt == 'f_n':
                    o = (fn2p(y_n2[:cbs], outputs, dirs[:cbs],
                        nsp, vox2mm, iscuda)).detach().cpu().numpy()
                # print('o shape', o.shape)
                if all([rank == 0, opt.save_output, lb == 'pc' or lb == 'f', not opt.minmax,
                epoch > epoch0]):
                    o = o*opt.pscale*vox2mm
                    # print('o before showmanypoints', getstat(o[0,2,:]))
                if opt.merging_order == 'color_channel':
                    i0 = np.squeeze(inputs[0,0].detach().cpu().numpy())
                else:
                    i0 = np.squeeze(inputs[0].detach().cpu().numpy())
                # if all([rank == 0, opt.save_output, lb != 'f_n', not opt.minmax,
                #         epoch == epoch0+1, opt.merging_order != 'color_channel',
                #         opt.outputt != 'f_n']):
                #     gt.append(GT.detach().cpu().numpy())
                #     # print(gt[0].shape)
                #     gt0.append(gt[pcnt][0])
                #     # gt[pcnt] = np.reshape(gt[pcnt],(gt[pcnt].shape[0],-1))
                #     curangle = int(angles_list[0]/10)
                #     backproject(prmat[4*curangle:4*(curangle+1)],i0, gt0[pcnt],
                #         jn(dirname,phase))
                if all([rank == 0, opt.save_output, not(lb == 'f_n' or lb == 'pose6'), not opt.minmax,
                        epoch == epoch0+1, opt.merging_order == 'color_channel',
                        opt.outputt == 'f_n' or opt.outputt == 'pc']):
                    # print(GT.shape)

                    # gt.append(f2p(GT, dirs[:cbs], nsp, vox2mm).detach().cpu().numpy())
                    gt.append(GT.detach().cpu().numpy())
                    # gt.append(GT.reshape(GT.shape[0],-1))
                    gt0.append(gt[pcnt][0])
                    curangle = int(angles_list[0][0]/10)
                    # print('agnles shape', angles_list.shape)
                    # print('prmat shape', prmat.shape)

                    # print('gt stat = ', np.min(gt0[pcnt]), np.mean(gt0[pcnt]), np.max(gt0[pcnt]))
                    # print('gt0[pcnt].shape',gt0[pcnt].reshape(3,nsp).shape)
                    # print('gt stat', getstat(gt0[pcnt]))
                    # print('curangle', curangle)
                    # np.savetxt('C:/cherepashkin1/gt0', gt0[pcnt], delimiter=',')
                    # np.save('C:/cherepashkin1/prmat',prmat.detach().cpu().numpy())
                    # np.savetxt('C:/cherepashkin1/i0', i0, delimiter=',')
                    # print('before backproject', prmatw[index[0], curangle, :, :].shape, i0.shape, gt0[pcnt].shape)
                    backproject(prmatw[index[0], curangle, :, :].detach().cpu().numpy(),i0,
                        gt0[pcnt].reshape(3,nsp),
                        jn(dirname,phase), opt.rotate_output)
                    # sys.exit()
                # elif all([rank == 0, opt.save_output, lb != 'f_n', not opt.minmax,
                #         epoch == epoch0+1, opt.merging_order == 'color_channel',
                #         opt.outputt != 'f_n', nim>1]):
                #     gt.append(GT.detach().cpu().numpy())
                #     print('gt[0].shape', gt[0].shape)
                #     gt0.append(gt[pcnt][0])
                #     # gt[pcnt] = np.reshape(gt[pcnt],(gt[pcnt].shape[0],-1))
                #     curangle = int(angles_list[0][0]/10)
                #     backproject(prmat[4*curangle:4*(curangle+1)],i0,
                #         np.squeeze(f2p(np.expand_dims(gt0[pcnt],axis=0),
                #          dirs[0], nsp, vox2mm)),
                #         jn(dirname,phase))
                if log1 and phase == 'val' and opt.measure_time:
                    fig = plt.figure()
                    for i in range(4):
                        plt.plot(lt[i,:])
                        # plt.xlim(0,epoch*bs)
                    plt.savefig(dirname+'lt.png')
                    plt.close(fig)
                    np.savetxt(dirname+'lt',lt)
                if all([rank == 0, epoch == epoch0+1, opt.save_output,
                    not(lb == 'f_n' or lb == 'pose6')]):
                    savelist(pathes, jn(dirname,"pathes_"+phase+".txt"))
                    try:
                        savelist([str(int(i/10)) for i in angles_list.tolist()],
                            jn(dirname,"angles_"+phase+".txt"))
                    except:
                        savelist([str(int(i/10)) for i in\
                            t.flatten(angles_list).tolist()],
                            jn(dirname,"angles_"+phase+".txt"))
                if all([log1, not (lb == 'f_n' or lb == 'pose6'), epoch == epoch0+1]):
                    # try:
                    #     print(gt.shape)
                    # except:
                    #     print(gt[0].shape)
                    Path(jn(dirname,'netOutputs')).mkdir(parents=True, exist_ok=True)
                    np.savetxt(jn(dirname,'netOutputs','gt_'+phase),
                               np.reshape(gt[pcnt],(cbs,-1)), delimiter=',')
                    # print('inputs[0].shape=',inputs[0].shape)
                    np.savetxt(jn(dirname,'input_image_'+phase),i0,delimiter=',')
                if rank == 0 and epoch == epoch0+1 and opt.save_output:
                    for n in opt.netname:
                        shutil.copy(jn(homepath, 'circles/finetune_test/experiments',
                                                 opt.expnum, n+'.py'),
                                                    jn(dirname,n+'.py'))
                    shutil.copy(jn(homepath, finePath, 'main.py'),
                                jn(dirname, 'main.py'))
                    shutil.copy(jn(finePath, "transform"+opt.transappendix+".py"),
                                jn(dirname, "transform"+opt.transappendix+".py"))
                if all([rank == 0, lb == 'pc' or lb == 'f', opt.save_output, epoch > epoch0]):
                    oo = o[0]
                    # print('oo.shape',oo.shape)
                    oo = oo.reshape((3, nsp))
                # elif all([rank == 0, lb == 'f', opt.outputt == 'f_n', opt.save_output, epoch > epoch0]):
                #     oo = o[0]
                if all([rank == 0, not(lb == 'f_n' or lb == 'pose6'), opt.save_output, epoch > epoch0]):
                    # print('gt shape', gt[pcnt].shape)
                    # print('o shape', o.reshape((-1,3,nsp)).shape, gt[pcnt].shape)
                    # print('o stat', getstat(o.reshape((-1,3,nsp))[0,2,:]), getstat(vox2mm*gt[pcnt][0,2,:]))
                    showmanypoints(cbs,nim,o.reshape((-1,3,nsp)),vox2mm*gt[pcnt],
                        pathes,angles_list,phase,i_batch,cnt,jn(dirname, 'showPoints'),
                        opt.merging_order, vox2mm)
                    curloss[pcnt,epoch] = np.mean(np.abs(LA.norm(oo,axis=0)-\
                        LA.norm(gt0[pcnt],axis=0)))

                    print('curloss for %s phase for %d epoch = %f'\
                        %(phase,epoch,curloss[pcnt,epoch]))
                if rank == 0 and lb == 'pc' and\
                    opt.save_output and epoch > epoch0 and opt.wandb:
                        wandb.log({phase+"_points": wandb.Image(jn(dirname,'pc_'+\
                        phase+'_' +str(cnt).zfill(3) + '.png'))})
                if rank == 0 and not(lb == 'f_n' or lb == 'pose6') and\
                    opt.save_output and epoch > epoch0 and opt.wandb:
                    wandb.log({"point_cloud " + phase:\
                        wandb.Object3D(oo.transpose())})
                if rank == 0 and not(lb == 'f_n' or lb == 'pose6') and\
                    opt.save_output and epoch > epoch0 and opt.wandb and\
                     phase == 'val':
                        wandb.log({'train curloss': curloss[0,epoch],\
                         'val curloss': curloss[1,epoch]})
                if log1 and phase == 'val':
                    lossout('Average_loss_', 'Epoch', lossar, epoch,
                            lossoutdir , lb)
                    lossout('!Single_seed_loss_', 'Epoch', curloss, epoch, lossoutdir, lb)
                    #
                    # lossmb_eff = [np.array([lossmb[i], np.zeros(lossmb[i].shape)])\
                    #               for i in range(2)]
                    lossmb_eff = [np.array([lossmb[0], np.zeros(lossmb[0].shape)]),
                                  np.array([np.zeros(lossmb[1].shape), lossmb[1]])]
                    # print('1165', curloss.shape, lossmb_eff[0].shape, lossmb_eff)

                    # for q in range(2):
                    for pcnt_c, phase_c in enumerate(['train', 'val']):
                        lossout('Average_loss_minibatch_'+phase_c+'_', 'Iteration',
                                lossmb_eff[pcnt_c],
                                abs_batch_cnt[pcnt_c], lossoutdir, lb)

    #                 lossfig(jn(dirname,'learning_curve_'), lossar,
    #                  'Loss', 'Learning curve', (0,epoch), (0,1), lb)
    #                 logloss = np.ma.log10(lossar)
    #                 lossfig(jn(dirname,'learning_curve_'), logloss.filled(0),
    #                     'log10(Loss)', 'Learning curve', (0,epoch), (0,0), lb)
    #                 lossfig(jn(dirname,'curloss_'), curloss,
    #                  'Loss', 'Loss for single seed', (0,epoch), (0,1), lb)
    #                 # lossfig(dirname+'curloss_', np.abs(curloss),
    #                 #  'Abs Loss', 'Loss for single seed', (0,epoch), (0,0.2), lb)
    #                 logloss = np.ma.log10(np.abs(curloss))
    #                 lossfig(jn(dirname,'curloss_'), logloss.filled(0),
    #                     'log10(abs loss)', 'Loss for single seed', (0,epoch),
    #                      (0,0), lb)
    #     #                 lossars = np.array([np.trim_zeros(lossar[0,:]),
    # #                     np.trim_zeros(lossar[1,:]), ])
    #
    # #                 print(lossars)
                if (rank == 0 and
                        epoch > epoch0 and
                        opt.save_output and
                        not(lb == 'f_n' or lb == 'pose6')):
                    # print(o.shape)
                    # model.eval()
                    # with t.no_grad():
                    #     o3,latent3 = model(t.unsqueeze(inputs[0],axis=0))
                    #     o4,latent4 = model(inputs)
                    #     # t.save(model.state_dict(), jn(dirname,"model2_"+\
                    #     #     str(cnt).zfill(3)))
                    #     np.savetxt(jn(dirname,'o3_'+phase+'_' +\
                    #                str(cnt).zfill(3)), o3.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'latent3_'+phase+'_' +\
                    #                str(cnt).zfill(3)), latent3.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'o4_'+phase+'_' +\
                    #                str(cnt).zfill(3)), o4.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'latent4_'+phase+'_' +\
                    #                str(cnt).zfill(3)), latent4.cpu().detach().numpy(), delimiter=',')


                    np.savetxt(jn(dirname,'netOutputs','o_'+phase+'_' +\
                               str(cnt).zfill(3)), o.reshape(-1,3*nsp), delimiter=',')

                    # np.save(dirname + 'lossar.npy', lossar)
                    np.savetxt(jn(dirname,'lossar'), lossar, delimiter=',')
                    np.savetxt(jn(dirname,'curloss'), curloss, delimiter=',')
                if (rank == 0 and
                        epoch > 0 and
                        opt.save_output and
                        not(lb == 'f_n' or lb == 'pose6')):
                    t.save(model.state_dict(), jn(dirname,"model"))
                        # str(cnt).zfill(3))
                         # and epoch % opt.ufmodel == 0
                    if opt.wandb:
                        wandb.watch(model)
                if all([rank == 0, epoch > epoch0, opt.save_output,
                        not(lb == 'f_n' or lb == 'pose6')]):
                    np.savetxt(jn(dirname,'latent','latent_'+phase+'_' +\
                               str(cnt).zfill(3)), latent.detach().cpu().numpy(), delimiter=',')
                # GIF movie making
                # if rank == 0 and opt.save_output and\
                #     (lb == 'pc' or lb == 'pc+f') and epoch > epoch0:
                #     cip = []
                #     for root, directories, filenames in os.walk(dirname):
                #         for filename in filenames:
                #             if 'pc_'+phase in filename and not 'checkpoint' in filename:
                #                 cip.append(jn(root,filename))
                #     cip.sort()
                #     images = []
                #     for filename in cip:
                #         images.append(imageio.imread(filename))
                #     imageio.mimsave(jn(dirname, phase+'_movie.gif'), images, duration=0.5)
                losst = 0 if phase == 'train' else 1