import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
# import torch.utils.tensorboard as tensorboardX
from thop import profile
from thop import clever_format
import pdb
import numpy
import os
import skimage.io
import sys
import platform
from skimage.metrics import structural_similarity as SSIM


if platform.system() == 'Windows':
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    sys.path.append(r"/home/chenxu/我的坚果云/sourcecode/python/util")

import common_metrics
import common_brats


class NICE(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'NICE_light'
        else :
            self.model_name = 'NICE'

        self.result_dir = args.result_dir
        self.data_dir = args.data_dir
        self.checkpoint_dir = args.checkpoint_dir

        self.mini = args.mini
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        if args.mini:
            self.img_size //= 2

        self.do_validation = args.do_validation
        self.psnr_threshold = args.psnr_threshold

        if args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            self.device = torch.device("cuda")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device = torch.device("cpu")

        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        #print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,pin_memory=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,pin_memory=True)
        """

        dataset_s = common_brats.Dataset(self.data_dir, modality="t2", n_slices=self.img_ch)
        dataset_t = common_brats.Dataset(self.data_dir, modality="t1", n_slices=self.img_ch)
        self.dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        self.dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        if self.do_validation:
            self.val_data_t, self.val_data_s = common_brats.load_test_data(self.data_dir, "val")

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        _,_, _,  _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)


    def train(self):
        # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

        self.start_iter = 1
        if self.resume:
            params = torch.load(os.path.join(self.checkpoint_dir, 'params_latest.pt'))
            self.gen2B.load_state_dict(params['gen2B'])
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])
            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])
            self.start_iter = params['start_iter']+1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
            print("ok")
          

        # training loop
        """
        testnum = 4            
        for step in range(1, self.start_iter):
            if step % self.print_freq == 0:
                for _ in range(testnum):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()
        print("self.start_iter",self.start_iter)
        """

        print('training start !')
        start_time = time.time()
        best_psnr = 0
        for step in range(self.start_iter, self.iteration + 1):
            for batch_id, (data_s, data_t) in enumerate(zip(self.dataloader_s, self.dataloader_t)):
                if self.decay_flag and step > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))


                real_A = data_s["image"].to(self.device)
                real_B = data_t["image"].to(self.device)

                # Update D
                self.D_optim.zero_grad()

                real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(real_A)
                real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(real_B)

                fake_A2B = self.gen2B(real_A_z)
                fake_B2A = self.gen2A(real_B_z)

                fake_B2A = fake_B2A.detach()
                fake_A2B = fake_A2B.detach()

                fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
                fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)


                D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
                D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
                D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_optim.step()
                # writer.add_scalar('D/%s' % 'loss_A', D_loss_A.data.cpu().numpy(), global_step=step)
                # writer.add_scalar('D/%s' % 'loss_B', D_loss_B.data.cpu().numpy(), global_step=step)

                # Update G
                self.G_optim.zero_grad()

                _,  _,  _, _, real_A_z = self.disA(real_A)
                _,  _,  _, _, real_B_z = self.disB(real_B)

                fake_A2B = self.gen2B(real_A_z)
                fake_B2A = self.gen2A(real_B_z)

                fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(fake_B2A)
                fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(fake_A2B)

                fake_B2A2B = self.gen2B(fake_A_z)
                fake_A2B2A = self.gen2A(fake_B_z)


                G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

                G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
                G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

                G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

                fake_A2A = self.gen2A(real_A_z)
                fake_B2B = self.gen2B(real_B_z)

                G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2B, real_B)


                G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.step()
                # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)
                # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)

                # for name, param in self.gen2B.named_parameters():
                #     writer.add_histogram(name + "_gen2B", param.data.cpu().numpy(), global_step=step)

                # for name, param in self.gen2A.named_parameters():
                #     writer.add_histogram(name + "_gen2A", param.data.cpu().numpy(), global_step=step)

                # for name, param in self.disA.named_parameters():
                #     writer.add_histogram(name + "_disA", param.data.cpu().numpy(), global_step=step)

                # for name, param in self.disB.named_parameters():
                #     writer.add_histogram(name + "_disB", param.data.cpu().numpy(), global_step=step)


            if step % self.print_freq == 0:
                msg = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)
                msg += '  current D_learning rate:{}'.format(self.D_optim.param_groups[0]['lr'])
                msg += '  current G_learning rate:{}'.format(self.G_optim.param_groups[0]['lr'])

                if self.do_validation:
                    val_st_psnr, val_ts_psnr, val_st_list, val_ts_list = self.validate()

                    msg += "  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
                           (val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
                    gen_images_test = numpy.concatenate([self.val_data_s[0], val_st_list[0], val_ts_list[0], self.val_data_t[0]], 2)
                    gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
                    gen_images_test = common_brats.generate_display_image(gen_images_test, is_seg=False)

                    if self.checkpoint_dir:
                        try:
                            skimage.io.imsave(os.path.join(self.checkpoint_dir, "gen_images_test.jpg"), gen_images_test)
                        except:
                            pass

                    if val_ts_psnr.mean() > best_psnr:
                        best_psnr = val_ts_psnr.mean()

                        if best_psnr > self.psnr_threshold:
                            self.save("best")

                msg += "  best_ts_psnr:%f" % best_psnr

                print(msg)

        self.save("final")


    def save(self, tag):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        torch.save(params, os.path.join(self.checkpoint_dir, 'params_%s.pt' % tag))


    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(self.checkpoint_dir, 'params_final.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])

    def validate(self):
        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

        val_st_psnr = numpy.zeros((self.val_data_s.shape[0], 1), numpy.float32)
        val_ts_psnr = numpy.zeros((self.val_data_t.shape[0], 1), numpy.float32)
        val_st_list = []
        val_ts_list = []
        with torch.no_grad():
            for i in range(self.val_data_s.shape[0]):
                val_st = numpy.zeros(self.val_data_s.shape[1:], numpy.float32)
                val_ts = numpy.zeros(self.val_data_t.shape[1:], numpy.float32)
                used = numpy.zeros(self.val_data_s.shape[1:], numpy.float32)
                for j in range(self.val_data_s.shape[1] - self.img_ch + 1):
                    val_patch_s = torch.tensor(self.val_data_s[i:i + 1, j:j + self.img_ch, :, :], device=self.device)
                    val_patch_t = torch.tensor(self.val_data_t[i:i + 1, j:j + self.img_ch, :, :], device=self.device)

                    _, _, _, _, z_s = self.disA(val_patch_s)
                    _, _, _, _, z_t = self.disB(val_patch_t)

                    ret_st = self.gen2B(z_s)
                    ret_ts = self.gen2A(z_t)

                    val_st[j:j + self.img_ch, :, :] += ret_st[0].cpu().detach().numpy()[0]
                    val_ts[j:j + self.img_ch, :, :] += ret_ts[0].cpu().detach().numpy()[0]
                    used[j:j + self.img_ch, :, :] += 1

                assert used.min() > 0
                val_st /= used
                val_ts /= used

                st_psnr = common_metrics.psnr(val_st, self.val_data_t[i])
                ts_psnr = common_metrics.psnr(val_ts, self.val_data_s[i])

                val_st_psnr[i] = st_psnr
                val_ts_psnr[i] = ts_psnr
                val_st_list.append(val_st)
                val_ts_list.append(val_ts)

        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()
        return val_st_psnr, val_ts_psnr, val_st_list, val_ts_list

    def test(self):
        self.load()

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.gen2B.eval(), self.gen2A.eval()

        test_data_t, test_data_s = common_brats.load_test_data(self.data_dir, "test")

        test_st_psnr = numpy.zeros((len(test_data_s), 1), numpy.float32)
        test_ts_psnr = numpy.zeros((len(test_data_t), 1), numpy.float32)
        test_st_ssim = numpy.zeros((len(test_data_s), 1), numpy.float32)
        test_ts_ssim = numpy.zeros((len(test_data_t), 1), numpy.float32)
        test_st_mae = numpy.zeros((len(test_data_s), 1), numpy.float32)
        test_ts_mae = numpy.zeros((len(test_data_t), 1), numpy.float32)
        test_st_list = []
        test_ts_list = []
        with torch.no_grad():
            for i in range(len(test_data_s)):
                test_st = numpy.zeros(test_data_s[i].shape, numpy.float32)
                test_ts = numpy.zeros(test_data_t[i].shape, numpy.float32)
                used = numpy.zeros(test_data_s[i].shape, numpy.float32)
                for j in range(test_data_s[i].shape[0] - self.img_ch + 1):
                    test_patch_s = torch.tensor(numpy.expand_dims(test_data_s[i][j:j + self.img_ch, :, :], 0), device=self.device)
                    test_patch_t = torch.tensor(numpy.expand_dims(test_data_t[i][j:j + self.img_ch, :, :], 0), device=self.device)

                    _, _, _, _, z_s = self.disA(test_patch_s)
                    _, _, _, _, z_t = self.disB(test_patch_t)

                    ret_st = self.gen2B(z_s)
                    ret_ts = self.gen2A(z_t)

                    test_st[j:j + self.img_ch, :, :] += ret_st[0].cpu().detach().numpy()[0]
                    test_ts[j:j + self.img_ch, :, :] += ret_ts[0].cpu().detach().numpy()[0]
                    used[j:j + self.img_ch, :, :] += 1

                assert used.min() > 0
                test_st /= used
                test_ts /= used

                if self.result_dir:
                    common_brats.save_nii(test_ts, os.path.join(self.result_dir, "syn_%d.nii.gz" % i))

                st_psnr = common_metrics.psnr(test_st, test_data_t[i])
                ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])
                st_ssim = SSIM(test_st, test_data_t[i])
                ts_ssim = SSIM(test_ts, test_data_s[i])
                st_mae = abs(test_st - test_data_t[i])
                ts_mae = abs(test_ts - test_data_s[i])

                test_st_psnr[i] = st_psnr
                test_ts_psnr[i] = ts_psnr
                test_st_ssim[i] = st_ssim
                test_ts_ssim[i] = ts_ssim
                test_st_mae[i] = st_mae
                test_ts_mae[i] = ts_mae
                test_st_list.append(test_st)
                test_ts_list.append(test_ts)

        msg = "test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_st_mae:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f" % \
              (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(), test_st_mae.mean(), test_st_mae.std(),
               test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(), test_ts_mae.mean(), test_ts_mae.std())
        print(msg)

        if self.result_dir:
            with open(os.path.join(self.result_dir, "result.txt"), "w") as f:
                f.write(msg)

            numpy.save(os.path.join(self.result_dir, "st_psnr.npy"), test_st_psnr)
            numpy.save(os.path.join(self.result_dir, "ts_psnr.npy"), test_ts_psnr)
            numpy.save(os.path.join(self.result_dir, "st_ssim.npy"), test_st_ssim)
            numpy.save(os.path.join(self.result_dir, "ts_ssim.npy"), test_ts_ssim)
            numpy.save(os.path.join(self.result_dir, "st_mae.npy"), test_st_mae)
            numpy.save(os.path.join(self.result_dir, "ts_mae.npy"), test_ts_mae)
