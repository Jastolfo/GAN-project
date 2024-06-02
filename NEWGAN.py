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
from pidinet.edge_extractor import *
import os
import lpips
from focal_frequency_loss import FocalFrequencyLoss as FFL

def truncate(fake_B,a=127.5):#[-1,1]
    return ((fake_B+1)*a).int().float()/a-1

class NEWGAN(object) :
    def __init__(self, args):
        self.model_name = 'NEWGAN'

        self.result_dir = args.result_dir
        self.log_file = args.log_file
        self.dataset = args.dataset
        self.vgg_path = args.vgg_path

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
        self.edge_weight = args.edge_weight
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight
        self.trunc_weight = args.trunc_weight
        self.ffl_weight = args.ffl_weight

        """ Generator """
        self.n_res_2B = args.n_res_2B
        self.n_res_2A = args.n_res_2A
        self.asymmetric_cycle_mapping = args.asymmetric_cycle_mapping
        self.trunc_flag = args.trunc_flag
        self.trunc_a = args.trunc_a

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)
        print("# truncation magnitude : ", self.trunc_a)

        print()

        print("##### Generator #####")
        print("# 2B generator residual blocks : ", self.n_res_2B)
        print("# 2A generator residual blocks : ", self.n_res_2A)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)
        print("# edge_weight : ", self.edge_weight)
        print("# content_weight : ", self.content_weight)
        print("# style_weight : ", self.style_weight)
        print("# truncate_weight : ", self.trunc_weight)
        print("# focal_frequency_weight : ", self.ffl_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
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

        self.trainA = ImageFolder('../datasets/train_human', train_transform)
        self.trainB = ImageFolder('../datasets/train_anime', train_transform)
        self.testA = ImageFolder(os.path.join('../datasets/test_human'), test_transform)
        self.testB = ImageFolder(os.path.join('../datasets/test_anime'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,pin_memory=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,pin_memory=True)

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res_2B, img_size=self.img_size).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res_2A, img_size=self.img_size).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disEdgeA = PatchDiscriminator(input_nc=1, ndf=self.ch).to(self.device)
        self.disEdgeB = PatchDiscriminator(input_nc=1, ndf=self.ch).to(self.device)
        
        """ Define Edge Extractor """
        self.pidinet = get_model('/notebooks/NEW_GAN/pidinet/trained_models/table5_pidinet.pth').to(self.device)

        """ Define LPIPS"""
        self.lpips = lpips.LPIPS(net='alex', model_path='/notebooks/NEW_GAN/PerceptualSimilarity/alexnet-owt-7be5be79.pth').to(self.device)
        
        """ Define VGG Perceptual Loss"""
        self.vggPerceptualLoss = VGGPerceptualLoss(model_path=self.vgg_path).to(self.device)

        """ Define Focal Frequency Loss"""
        self.ffl = FFL(loss_weight=1.0, alpha=1.0)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        input_edge = torch.randn([1, 1, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disEdgeA, inputs=(input_edge, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disEdgeA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disEdgeA', macs)
        print('-----------------------------------------------')
        _,_, _, _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')
        macs, params = profile(self.gen2A, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2A', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2A', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters(), self.disEdgeA.parameters(), self.disEdgeB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        # self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)


    def train(self):
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()
        self.disEdgeA.train(), self.disEdgeB.train(), self.pidinet.eval()

        self.start_iter = 1
        if self.resume:
            params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
            self.gen2B.load_state_dict(params['gen2B'])
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])
            self.disEdgeA.load_state_dict(params['disEdgeA'])
            self.disEdgeB.load_state_dict(params['disEdgeB'])
            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])
            self.start_iter = params['start_iter']+1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
            print("Load successful!")
          

        # training loop
        testnum = 4
        for step in range(1, self.start_iter):
            if step % self.print_freq == 0:
                for _ in range(testnum):
                    try:
                        real_A, _ = next(testA_iter)
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = next(testA_iter)

                    try:
                        real_B, _ = next(testB_iter)
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = next(testB_iter)

        iter_in_epoch = round(len(self.testA) / self.batch_size)
        print("self.start_iter",self.start_iter)
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            real_A_edge = get_edge(model=self.pidinet, image=real_A).to(self.device)
            real_B_edge = get_edge(model=self.pidinet, image=real_B).to(self.device)

            # Update D
            self.D_optim.zero_grad()

            real_LA_logit, real_MA_logit, real_GA_logit, _, real_A_z = self.disA(real_A)
            real_LB_logit, real_MB_logit, real_GB_logit, _, real_B_z = self.disB(real_B)
            
            real_A_edge_logit = self.disEdgeA(real_A_edge)
            real_B_edge_logit = self.disEdgeB(real_B_edge)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()
            
            fake_A2B_edge = get_edge(model=self.pidinet, image=fake_A2B)
            fake_B2A_edge = get_edge(model=self.pidinet, image=fake_B2A)
            
            fake_A2B_edge = fake_A2B_edge.detach()
            fake_B2A_edge = fake_B2A_edge.detach()

            fake_LA_logit, fake_MA_logit, fake_GA_logit, _, _ = self.disA(fake_B2A)
            fake_LB_logit, fake_MB_logit, fake_GB_logit, _, _ = self.disB(fake_A2B)
            
            fake_A_edge_logit = self.disEdgeA(fake_B2A_edge)
            fake_B_edge_logit = self.disEdgeA(fake_A2B_edge)


            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_MA = self.MSE_loss(real_MA_logit, torch.ones_like(real_MA_logit).to(self.device)) + self.MSE_loss(fake_MA_logit, torch.zeros_like(fake_MA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_MB = self.MSE_loss(real_MB_logit, torch.ones_like(real_MB_logit).to(self.device)) + self.MSE_loss(fake_MB_logit, torch.zeros_like(fake_MB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))            
            
            """ Define Discriminator adversarial edge loss """
            D_ad_loss_A_edge = self.MSE_loss(real_A_edge_logit, torch.ones_like(real_A_edge_logit).to(self.device)) + self.MSE_loss(fake_A_edge_logit, torch.zeros_like(fake_A_edge_logit).to(self.device))
            D_ad_loss_B_edge = self.MSE_loss(real_B_edge_logit, torch.ones_like(real_B_edge_logit).to(self.device)) + self.MSE_loss(fake_B_edge_logit, torch.zeros_like(fake_B_edge_logit).to(self.device))            

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_loss_MA + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_loss_MB + D_ad_loss_LB)
            
            D_edge_loss = self.edge_weight * (D_ad_loss_A_edge + D_ad_loss_B_edge)

            Discriminator_loss = D_loss_A + D_loss_B + D_edge_loss
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            _,  _,  _, _, real_A_z = self.disA(real_A)
            _,  _,  _, _, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)
            
            fake_A2B_edge = get_edge(model=self.pidinet, image=fake_A2B)
            fake_B2A_edge = get_edge(model=self.pidinet, image=fake_B2A)

            fake_LA_logit, fake_MA_logit, fake_GA_logit, _, fake_A_z = self.disA(fake_B2A)
            fake_LB_logit, fake_MB_logit, fake_GB_logit, _, fake_B_z = self.disB(fake_A2B)
            
            fake_A_edge_logit = self.disEdgeA(fake_B2A_edge)
            fake_B_edge_logit = self.disEdgeA(fake_A2B_edge)
            
            fake_B2A2B = self.gen2B(fake_A_z)
            fake_A2B2A = self.gen2A(fake_B_z)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_MA = self.MSE_loss(fake_MA_logit, torch.ones_like(fake_MA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_MB = self.MSE_loss(fake_MB_logit, torch.ones_like(fake_MB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_edge_loss_A = self.MSE_loss(fake_A_edge_logit, torch.ones_like(fake_A_edge_logit).to(self.device))
            G_edge_loss_B = self.MSE_loss(fake_B_edge_logit, torch.ones_like(fake_B_edge_logit).to(self.device))

            if self.asymmetric_cycle_mapping:
                # fake_A2B2A_edge = get_edge(model=self.pidinet, image=fake_A2B2A)
                # G_edge_cycle_loss_A = (self.lpips.forward(real_A_edge, fake_A2B2A_edge)).mean()
                
                # _,  G_cycle_A_style_loss = self.vggPerceptualLoss(fake_A2B2A, real_A)
                # G_cycle_A_focal_frequency_loss = self.ffl_weight  * self.ffl(fake_A2B2A, real_A)
                
                G_cycle_A_content_loss, G_cycle_A_style_loss = self.vggPerceptualLoss(fake_A2B2A, real_A)
                G_cycle_loss_A = G_cycle_A_content_loss + G_cycle_A_style_loss + self.ffl_weight  * self.ffl(fake_A2B2A, real_A)
                
            else:
                G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A) + self.ffl_weight  * self.ffl(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B) + self.ffl_weight * self.ffl(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_z)
            fake_B2B = self.gen2B(real_B_z)

            if self.asymmetric_cycle_mapping:
#                 fake_A2A_edge = get_edge(model=self.pidinet, image=fake_A2A)
#                 G_recon_loss_A = (self.lpips.forward(real_A_edge, fake_A2A_edge)).mean()

#                 _,  G_recon_A_style_loss = self.vggPerceptualLoss(fake_A2A, real_A)
#                 G_recon_loss_A += G_recon_A_style_loss + self.ffl_weight * self.ffl(fake_A2A, real_A)

                G_recon_A_content_loss, G_recon_A_style_loss = self.vggPerceptualLoss(fake_A2A, real_A)
                G_recon_loss_A = G_recon_A_content_loss + G_recon_A_style_loss + self.ffl_weight * self.ffl(fake_A2A, real_A)
                
            else:
                G_recon_loss_A = self.L1_loss(fake_A2A, real_A) + self.ffl_weight * self.ffl(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B) + self.ffl_weight * self.ffl(fake_B2B, real_B)

            # if self.trunc_flag:
            #     _, _, _, _, fake_trunc_B_z = self.disB(truncate(fake_A2B, self.trunc_a))
            #     fake_trunc_A2B2A = self.gen2A(fake_trunc_B_z)
            #     fake_trunc_A2B2A_edge = get_edge(model=self.pidinet, image=fake_trunc_A2B2A)
            #     G_cycle_loss_A_trunc = (self.lpips.forward(real_A_edge, fake_trunc_A2B2A_edge)).mean()

                # _,  G_cycle_A_trunc_style_loss = self.vggPerceptualLoss(fake_trunc_A2B2A, real_A)
                # G_cycle_loss_A_trunc += G_cycle_A_trunc_style_loss

            if self.asymmetric_cycle_mapping:
                relaxed_cycle_A_weight = self.cycle_weight * ( 1 - (round(step / iter_in_epoch) / round(self.iteration / iter_in_epoch)) * 0.9)
                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_loss_MA + G_ad_loss_LA ) + \
                            self.cycle_weight * G_cycle_loss_A + \
                            self.recon_weight * G_recon_loss_A + \
                            self.edge_weight * G_edge_loss_A
                            # relaxed_cycle_A_weight * G_edge_cycle_loss_A + \
                            # self.cycle_weight * (G_cycle_A_style_loss + G_cycle_A_focal_frequency_loss) + \
                            
            else:
                G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_loss_MA + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A + self.edge_weight * G_edge_loss_A
                  
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_loss_MB + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B + self.edge_weight * G_edge_loss_B

            # if self.trunc_flag:
            #     loss_A_trunc_weight = self.trunc_weight * (round(step / iter_in_epoch) / round(self.iteration / iter_in_epoch)) * 0.9
            #     G_loss_A += loss_A_trunc_weight * G_cycle_loss_A_trunc

            # if self.style_weight > 0:
            #     _, G_style_loss_A = self.vggPerceptualLoss(real_A, fake_B2A)
            #     _, G_style_loss_B = self.vggPerceptualLoss(real_B, fake_A2B)
            #     G_loss_A += self.style_weight * G_style_loss_A
            #     G_loss_B += self.style_weight * G_style_loss_B
                
            
            # if self.content_weight > 0:
            #     G_content_loss_A2B, _ = self.vggPerceptualLoss(real_A, fake_A2B)
            #     G_content_loss_B2A, _ = self.vggPerceptualLoss(real_B, fake_B2A)
            #     G_loss_A += self.content_weight * G_content_loss_A2B
            #     G_loss_B += self.content_weight * G_content_loss_B2A
            
            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            training_time = time.time() - start_time
            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, training_time, Discriminator_loss, Generator_loss))

            if not os.path.isfile(self.log_file):
                with open(self.log_file, 'w') as f:
                    print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, training_time, Discriminator_loss, Generator_loss), file=f)
            else:
                with open(self.log_file, 'a') as f:
                    print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, training_time, Discriminator_loss, Generator_loss), file=f)


            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % self.print_freq == 0:
                print('current D_learning rate:{}'.format(self.D_optim.param_groups[0]['lr']))
                print('current G_learning rate:{}'.format(self.G_optim.param_groups[0]['lr']))
                self.save_path("_params_latest.pt",step)

            if step % self.print_freq == 0:
                train_sample_num = testnum
                test_sample_num = testnum
                A2B = np.zeros((self.img_size * 4, 0, 3))
                B2A = np.zeros((self.img_size * 4, 0, 3))

                self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

                self.gen2B,self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
                self.disA,self.disB = self.disA.to('cpu'), self.disB.to('cpu')
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = next(trainA_iter)
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = next(trainA_iter)

                    try:
                        real_B, _ = next(trainB_iter)
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _ = next(trainB_iter)
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
                    
                    _, _, _,  A_heatmap, real_A_z= self.disA(real_A)
                    _, _, _,  B_heatmap, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A_z)
                    fake_B2A = self.gen2A(real_B_z)

                    _, _, _,  _,  fake_A_z = self.disA(fake_B2A)
                    _, _, _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_A_z)
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)
                    fake_B2B = self.gen2B(real_B_z)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = next(testA_iter)
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = next(testA_iter)

                    try:
                        real_B, _ = next(testB_iter)
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = next(testB_iter)
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')

                    _, _, _,  A_heatmap, real_A_z= self.disA(real_A)
                    _, _, _,  B_heatmap, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A_z)
                    fake_B2A = self.gen2A(real_B_z)

                    _, _, _,  _,  fake_A_z = self.disA(fake_B2A)
                    _, _, _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_A_z)
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)
                    fake_B2B = self.gen2B(real_B_z)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                

                self.gen2B,self.gen2A = self.gen2B.to(self.device), self.gen2A.to(self.device)
                self.disA,self.disB = self.disA.to(self.device), self.disB.to(self.device)
                
                self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['disEdgeA'] = self.disEdgeA.state_dict()
        params['disEdgeB'] = self.disEdgeB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))


    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['disEdgeA'] = self.disEdgeA.state_dict()
        params['disEdgeB'] = self.disEdgeB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.disEdgeA.load_state_dict(params['disEdgeA'])
        self.disEdgeB.load_state_dict(params['disEdgeB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print(self.start_iter)

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(),self.disB.eval()
        for n, (real_A, real_A_path) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            _, _,  _, _, real_A_z= self.disA(real_A)
            fake_A2B = self.gen2B(real_A_z)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
            print(real_A_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

        for n, (real_B, real_B_path) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            _, _,  _, _, real_B_z= self.disB(real_B)
            fake_B2A = self.gen2A(real_B_z)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
            print(real_B_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)
