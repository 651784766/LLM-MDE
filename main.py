import argparse
from peft import get_peft_model, LoraConfig
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import os, sys, time
from tensorboardX import SummaryWriter
import matplotlib as mpl
import PIL.Image as pil 
import matplotlib.cm as cm
from tools.utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from LanguageMono import Language4Depth

parser = argparse.ArgumentParser(description='Language for Depth.', fromfile_prefix_chars='@')


parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='Language4Depth')
parser.add_argument('--LoRA',                      type=str,   help='LoRA', required=False)
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
# Model
parser.add_argument('--height',                    type=int,   help='the height of input image', default=480)
parser.add_argument('--width',                     type=int,   help='the width of input image', default=640)
parser.add_argument('--rank',                      type=int,   help='the rank in low-rank adaption', default=2)
parser.add_argument('--alpha',                     type=int,   help='the alpha in low-rank adaption', default=2)  
parser.add_argument('--Microsoft_LoRA',            type=int,   help='the alpha in low-rank adaption', default=0)  
# Bert
parser.add_argument('--llm_model',                 type=str,   help='name of pre-trained language models', default='BERT')
parser.add_argument('--prompt_domain',             action='store_true', help='Set prompt domain')
parser.add_argument('--content',                   type=str,   help='customized prompt for dataset', default=' ')
parser.add_argument('--n_heads',                   type=int,   help='number of heads in attention mechanism', default=8)
parser.add_argument('--d_ff',                      type=int,   help='dimensions of FFN in Transformer-based model', default=768)
parser.add_argument('--llm_layers',                type=int,   help='layers of loaded pre-trained LLM', default=4)
parser.add_argument('--save_model',                action='store_true', help='If set, the model will be saved')

parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./models')

parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)
# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1) 
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:


    arg_filename_with_prefix = '@' + sys.argv[1]

    args = parser.parse_args([arg_filename_with_prefix])

else:

    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'NYU':
    from dataloaders.dataloader import NewDataLoader

# -----------------------------------------------------------------------------------------------------------------

def main_worker(args):
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prompt
    model = Language4Depth(args, prompt=True,LoRA=True).to(device)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))
    print("== Model Initialized")
    global_step = 0
    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.parameters()}],
                                lr=args.learning_rate)
    model_just_loaded = False
    model_save_path = f"./weight/test_model_121.pth"

    if args.checkpoint_path != '':

        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            loc = 'cuda:{}'.format(0)
            checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True
    dataloader_train = NewDataLoader(args, mode ='train')
    if args.save_model:
        dataloader_online_eval = NewDataLoader(args, mode='online_eval')
    

    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
    
    if args.do_online_eval:
        if args.eval_summary_directory != '':
            eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
        else:
            eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
        eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    # silog
    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    mse = nn.MSELoss()
    start_time = time.time()


    duration = 0
    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate
    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader_train.data)

    num_total_steps = args.num_epochs * steps_per_epoch

    epoch = global_step // steps_per_epoch
    total_time = 0.0
    val_losses_per_epoch = []

    lowest_loss = [float('inf')]
    lowest_avg_d3 = [float('inf')]
    while epoch < args.num_epochs:
        epoch_start_time = time.time()
        model.train()
        # 遍历dataloader
        for step, sample_batched in enumerate(dataloader_train.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            # 获取image和gt
            image = sample_batched['image'].to(device)
            depth_gt = sample_batched['depth'].to(device)  # bs,480,640,1
            depth_est = model(image)  # 16,1,640,480

            if args.dataset == 'nyu':
                depth_gt = depth_gt.permute(0, 3, 1, 2)
                mask = depth_gt > 0.1

            elif args.dataset == 'NYU':
                depth_gt = depth_gt.squeeze()
                depth_gt = depth_gt.permute(0, 3, 1, 2)  # 1,480,640,3 -> 1,3,480,640
                depth_gt = torch.mean(depth_gt, dim=1, keepdim=True) # 1,1,480,640
            else:
                mask = depth_gt > 1.0

            #loss = silog_criterion.forward(depth_est, depth_gt)  # no log
            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))  # no log
            #loss = mse(depth_est, depth_gt)
            loss.backward()


            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            print('Training---epoch={},step={},step_per={},step_total={}, lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
            os.makedirs('./output', exist_ok=True)
            for i in range(depth_est.size(0)): 

                os.makedirs('./output', exist_ok=True)
                disp_resized_np = depth_est[i].squeeze().detach().cpu().numpy()
                #disp_resized_np = depth_est[i].squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                im.save(f'./output/depth_epoch_{epoch}_index_{i}.png')

            #print('[epoch][s/s_per_e/gs][time]: [{}][{}/{}/{}][{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step,op_duration, current_lr, loss))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1

            duration += time.time() - before_op_time


            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:


                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                # print(print_string.format(examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))
                writer.add_scalar('silog_loss', loss, global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)
                writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)

                for i in range(num_log_images):
                #for i in range(min(num_log_images, depth_gt.size(0), depth_est.size(0), image.size(0))):
                    writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :].data), global_step)
                    writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est[i, :, :, :].data), global_step)
                    writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                writer.flush()

            model_just_loaded = False
            global_step += 1

        epoch_end_time = time.time()
        time_per_epoch = (epoch_end_time-epoch_start_time)/60

        total_time+=time_per_epoch 

        Remain_time = (time_per_epoch * (args.num_epochs-epoch))/60
        print('time_per_epoch={:.2f}m, predic_total_time={:.2f}h'.format(time_per_epoch,Remain_time))
        total_d1 = 0.0
        total_d2 = 0.0
        total_d3 = 0.0
        if args.save_model:
            # Validation
            model.eval() 
            with torch.no_grad():
                print('Validation')
                for step, sample_batched in enumerate(dataloader_online_eval.data):
                    image = sample_batched['image'].to(device)  # 1,3,480,640
                    depth_gt = sample_batched['depth'].to(device) # 1,480,640,1  

                    depth_gt = depth_gt.squeeze(3).unsqueeze(1)  # 1,1,480,640
                    depth_est = model(image)  # 1,1,480,640


                    if args.dataset == 'nyu':
                        mask = depth_gt > 0.1  # 1,480,640,1

                    elif args.dataset == 'NYU':
                        depth_est = depth_est.squeeze()
                        depth_gt = depth_gt.squeeze()
                        depth_gt = depth_gt.permute(2,0,1)
                        depth_gt = torch.mean(depth_gt, dim=0, keepdim=True)

                    else:
                        mask = depth_gt > 1.0 
                    #loss_val = silog_criterion.forward(depth_est, depth_gt)
                    loss_val = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
                    #loss_val = mse(depth_est, depth_gt)
                    metrics =compute_errors(depth_gt,depth_est)
                    total_d1 += metrics['d1']
                    total_d2 += metrics['d2']
                    total_d3 += metrics['d3']
                    val_losses_per_epoch.append(loss_val.item())
                num_samples = len(dataloader_online_eval.data)
                avg_d1 = total_d1 / num_samples
                avg_d2 = total_d2 / num_samples
                avg_d3 = total_d3 / num_samples
                average_val_loss = sum(val_losses_per_epoch) / len(val_losses_per_epoch)

                # 每个epoch 对比
                if average_val_loss < lowest_loss[0]:
                    lowest_loss[0] = average_val_loss

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': average_val_loss
                    },model_save_path)
                    print(f"lowest_loss={average_val_loss:.5f}--- model_save")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                # ES
                if no_improvement_count >= 5:
                    print(f'No improvement in validation loss for 5 epochs. Stopping training.')
                    break
                val_losses_per_epoch.clear()

            print(f"Average d1: {avg_d1:.5f}, Average d2: {avg_d2:.5f}, Average d3: {avg_d3:.5f}")     
    
        epoch += 1

def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1


    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)


    args_out_path = os.path.join('./weight')

    #command = 'copy "' + sys.argv[1] + '" "' + args_out_path + '"'
    timestamp = int(time.time())
    output_path = os.path.join(args_out_path, f'train_nyu_{timestamp}.txt')
    command = f'copy "{sys.argv[1]}" "{output_path}"'

    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
    torch.cuda.empty_cache()
    main_worker(args)
# tensorboard --logdir=./logs
if __name__ == '__main__':
    main()