from venv import create
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import path as m_path
from matplotlib.patches import Ellipse
import matplotlib.colors as colors_mat
import sys  
sys.path.insert(0, "/".join(sys.path[0].split("/")[0:-2])+('/src'))
from sisal.utils import reparametrize,compute_latent,compute_latent_mean,sample_batch,emp_std,compute_loss,compute_latent_synthetic
sys.path.insert(0, "/".join(sys.path[0].split("/")[0:-2])+('/experiments/synthetic_data'))
import imageio
from scipy.stats import norm
import torch.nn.functional as F
import cmasher as cmr    
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from itertools import groupby
from scipy import stats
import statsmodels.api as sm
#from synthetic_data import full_index_normalized_data_synthetic
from synthetic_data import plot_all
#from mouse_pup import full_index_normalized_data_mouse_pup,index_to_image_pos_mouse,load_IMS_mouse_pup,get_image_shape_mouse_pup
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
#from two_points_data import full_small_index_normalized
from pathlib import Path
import pandas as pd
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rcParams, rcParamsDefault
import json
import skimage.io as io
from scipy.ndimage import rotate
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from sisal.kernel_adapted import kernel_adapated,plot_kernel_adapted
#import matplotlib.transforms as mtransforms
import matplotlib.patheffects as patheffects





#csfont = {'fontname':'cmr10'}
random_state_synthetic = 1326

def plot(args,PATH, train_loader, test_loader):
    ### Remove the unlabeled data
    #rcParams.update(rcParamsDefault)
    remove_label0 = True
    
    device = torch.device(args.device)
    model = torch.load(PATH, map_location=torch.device('cpu'))
    ################
    #model = model.float()
    ###############
    p = Plot(model,train_loader, test_loader,args.threshold_collapse)

    if args.dataset == 'kidney':
        loader, mask_to_name = dat.full_index_normalized_data() #Original data to be encode and decode
    elif args.dataset == 'mouse_pup' : 
        loader, mask_to_name = full_index_normalized_data_mouse_pup()
    elif args.dataset == 'synthetic' : 
        print('### Before full index')
        loader, mask_to_name = full_index_normalized_data_synthetic() #Original data to be encode and decode
        print('### After full index ')
        #loader, mask_to_name = full_index_normalized_data_synthetic_with_coeff() #Original data to be encode and decode
    
    #loader,mask_to_name = full_small_index_normalized()

    
    if args.recompute_latent :
        ##USE Following line to recompute latent space
        if args.dataset == 'synthetic' : 
            full_latent,vars,label, coeff = compute_latent_synthetic(loader,model)
        else :
            full_latent , vars , label = compute_latent(loader,model)

    else :
        #USE Following line to reuse already computed dictionary
        with open('saved_data/saved_latent.npy', 'rb') as f:
            print('predefined full latent')
            full_latent = np.load(f)
            vars = np.load(f)
            label = np.load(f)
            print('### Full_latent shape = ', full_latent.shape)

    if remove_label0 :
        ## Removing label 0 :
        mask0 = label!=0
        full_latent = full_latent[mask0,:]
        vars = vars[mask0,:]
        label = label[mask0]

    # print('### Size latent = ', full_latent.shape[0])
    # print('### full_latent = ', full_latent)
    # print('### vars = ', vars)
    if args.plot_latent : 
        print('Ploting Latent train')
        if args.z_dim == 2 :
            #p_temp = 0.2
            #p_temp = 0.7   
            p_temp=1
            #p_temp = 0.7
            p.plot_latent_dim_with_var(full_latent, vars , label,mask_to_name,args.beta,p=p_temp)
            #p.model_evolution(loader,mask_to_name)
            #p.model_evolution_kernel_latent(loader,mask_to_name)
            #p.plot_spatial( [10,50,100,150] ) 
        elif args.z_dim == 3:
            p.plot_latent_dim_3d(full_latent, vars, label, mask_to_name,args.beta)
        

        #print('Plot latent dims pairs')
        #p.plot_latent_dim_pairs(full_latent,vars,label,mask_to_name)
    if args.plot_synthetic_latent_coeff : 
        print('Plot synthetic coefficients latent space')
        #plot_all()
        p.plot_latent_dim_coeff(full_latent, vars, coeff)
    if args.plot_reconstruction_train:
        print('Ploting Reconstruction train')
        p.plot_reconstruction(train_loader, 'train')
    if args.plot_reconstruction_test:
        print('Ploting Reconstruction test')
        p.plot_reconstruction(test_loader, 'test')
    if args.plot_original_vs_recons:
        print('Ploting orginal vs reconstruction')
        p.plot_original_vs_recons(index_center=151)
    if args.latent_traversal : 
        print('Ploting Latent traversal')
        z_min, z_max = limit_latent_space(model,train_loader)
        for i in range(model.z_dim):
            if args.latent_traversal_gifs :
                print('Latent traversal gifs')
                p.latent_traversal(i,train_loader,z_min[i],z_max[i])
            if args.latent_traversal_heatmap : 
                p.latent_traversal_heatmap(i,train_loader,z_min[i],z_max[i])
        if args.latent_traversal_spatial:
            print('Spatial latent traversal')
            # with open('saved_data/saved_latent.npy', 'rb') as f:
            #    full_latent = np.load(f)
            #    vars = np.load(f)
            #    label = np.load(f)
            
            ### Removing Label 0
            # mask0 = label!=0
            # full_latent = full_latent[mask0,:]
            # vars = vars[mask0,:]
            # label = label[mask0]
            
            pa= 0.5
            pa = 1
            if pa != 1 :
                sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0]*pa), replace=False)
                full_latent = full_latent[sub_index]
                vars = vars[sub_index]
                label = label[sub_index]
                #color_train = color_train[sub_index]

            print('Drawing connections')
            for fa in range(model.z_dim):
                print('Factor{}'.format(fa))
                n_p = 1
                _,combined_idx = p.draw_latent_connections(fa, z_min, z_max,full_latent,vars,label,mask_to_name, n_points=n_p)
                
                index_to_line = dict(zip(full_latent[:,0] ,range(full_latent.shape[0])))
                col_dict = p.color_dict()
                print('#### Combined_idx  = ', combined_idx)
                for n in range(n_p):    
                    id = [ index_to_line[i] for i in combined_idx[n] ]
                    p.latent_traversal_app_gif(full_latent[id],label[id],mask_to_name,fa,n)
                
    if args.latent_traversal_subdivision : 
        print('Drawing latent traversal for subdivision')
        ##### Kidney
        pol_limit1 = [(-1.9,-2.1),(-1.3,-3),(-0.4,1.2), (-1.1,1.4),] ### Blue pr1
        pol_limit2 = [(-0.3,-3.4),(0.2,-3.2),(0.2,1.3), (-0.3,1.3),] ### Green pr2

        #### ALL DATA VAN1_41
        # pol_limit1 = [(0.3,-3),(1.1,-3),(0.6,1),(0.4,1.3),(0.1,0.3),(0.1,-1)] #Black
        # pol_limit2 = [(1.7,-3),(2.8,-2),(1,1.3),(0.7,1)] #Red

        #### ALL DATA VAN1_31
        # pol_limit1 = [(-0.1,-3.8),(0.5,-3.8),(0.7,-1.4),(0.7,-0.2),(0.4,-0.1)] #black
        # pol_limit2 = [(1,-3.7),(1.5,-3.7),(1.6,0.5),(1.3,0.3),(0.9,-0.5)] #red

        #### ALL DATA VAN1_35
        # pol_limit2 = [(-2.2,-3),(-1.5,-3.5),(-0.8,-2.1),(-0.2,0),(-0.19,1.55),(-0.21,1.55)] #Black 
        # pol_limit1 =[(-3.4,-1),(-2.4,-1.2),(-1.5,-0.5),(-0.8,0.7), (-0.3,1.55), (-0.7,1.55),(-2,1.2)] #Red
        # pol_limit2 = [(-1.6,-3),(-0.8,-3.5),(-0.6,-1),(-0.3,1),(-0.5,1)] #Black 
        # pol_limit1 =[(-3,-2),(-2.2,-2.2),(-0.9,1),(-1.3,1.2)] #Red
        
        pol_limit2 = [(-1.6,-3),(-0.8,-3.5),(-0.4,-0.8),(-0.3,1.3),(-0.5,1.3)] #Black 
        pol_limit1 =[(-3,-2),(-2.2,-2.2),(-1.2,-0.1),(-0.75,1.3),(-1.3,1.3)] #Red
        
        p.latent_traversal_legs(full_latent,vars,label,mask_to_name,pol_limit1,pol_limit2)

        p.variance_latent_traversal()
    if args.plot_posterior_collapse  : 
        print('Ploting posterior collapse')
        #betas = range(1,10)
        
        betas = [1,4,16,64]
        betas = [1,4]
        p.posterior_collapse(loader, betas,z_dim=10)
        #p.posterior_collapse_qqplot(loader, betas,z_dim=10)
    if args.plot_recons_dis_tradeoff :
        p.plot_recons_dis_trade_of()
    if args.plot_carving : 
        print('Plot Carving')
        p.plot_carving_separate(full_latent,vars,label,mask_to_name,index = 151)
        #p.plot_carving(full_latent,vars,label,mask_to_name,index = 151)
    if args.plot_density : 
        print('Plot kernel density')
        p.plot_kernel_density_estimation(full_latent,vars,label,mask_to_name)
    if args.zoomed_microscopy :
        print('Plot zoomed in microscopy')
        p.plot_zoomed_micro()
    if args.complete_microscopy :
        print('Plot complete microscopy')
        p.plot_complete_micro()
    if args.plot_mz :
        print('Plot spatial view for fixed m/z')
        p.plot_mz()
    if args.plot_spectrum : 
        print('Plot spectrum')
        p.plot_spectrum(train_loader)
        


#Computes largest and smallest value for each latent dim of the training data        
def limit_latent_space( model ,train_loader):
    x = next(iter(train_loader))[0]
    #x = x.to(device)
    z_min = model.forward(x)[0]
    z_min = torch.min(z_min, 0)[0]
    z_max = z_min
    for x,_ in train_loader:
        z = model.forward(x)[0]
        z_min = torch.minimum(z_min,torch.min(z,0)[0])
        z_max = torch.maximum(z_max,torch.max(z,0)[0])
    return (z_min.detach().numpy(),z_max.detach().numpy())




class  Plot():
    def __init__(self,PATH,device,train_loader,test_loader,full_loader) -> None:
        #self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader 
        self.loader = full_loader
        #self.std_threshold = threshold_collapse

        device = torch.device(device)
        model = torch.load(PATH, map_location=torch.device('cpu'))
        full_latent,vars,label, coeff = compute_latent_synthetic(self.loader,model)
        
        self.full_latent = full_latent
        self.vars = vars
        self.label = label
        self.coeff = coeff

    
    def plot_latent_dim(self, loader, title):
        new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
        c_dict = dict(zip(range(len(new_colors)),new_colors))
        latent = np.zeros((len(loader.dataset),self.model.z_dim))  
        label = np.zeros(len(loader.dataset))
        with torch.no_grad():
            prev = 0
            for x,l in loader :
                z_mean,  _ = self.model.forward(x)
                batch = z_mean.size(0)
                latent[prev:prev+batch,:] = z_mean.detach().numpy()
                label[prev:prev+batch] = l
                prev+=batch 
        
        plt.figure(title)
        color_train = np.array([c_dict[e] for e in label])
        plt.scatter(latent[:,0], latent[:,1], c = color_train) 
        plt.title('2 dim latent space for {} points'.format(title))
        plt.savefig('plots/2d_reduction/reduction_{}_points.png'.format(title))

        ############### Subploat only glomerial
        label = np.where(label==1)[0]
        sub_train = latent[label,:]
        c_subplot = color_train[label]
        plt.figure("sub_{}_low_dim".format(title))
        plt.scatter(sub_train[:,0], sub_train[:,1], c = c_subplot) 
        plt.title('subplot for the Glomeruls on {} set'.format(title))
        plt.savefig('plots/2d_reduction/subplot_{}_points.png'.format(title))
    
    
    def get_model_name(self, beta, z_dim):
        return 'model/produce_models/model_weights_zdim_{}_beta_{}.pth'.format(z_dim,beta)


    def posterior_collapse(self,loader,betas,z_dim = 10):
        # Plot the histogram.
        fig, axs = plt.subplots(len(betas), z_dim,figsize=(25, 15))
        L = 1000 #Number of samples used for histogram
        for j in range(len(betas)):
            b=betas[j] 
            #print('### b=  ', b)
            PATH = self.get_model_name(b,z_dim)
            model = torch.load(PATH, map_location=torch.device('cpu'))
            full_latent,vars,_ = compute_latent(loader,model)
            full_latent = full_latent[:,1:]
            
            full_std = emp_std(full_latent)

            n = full_latent.shape[0]
            index = np.random.randint(n,size = L)
            sample_mean = full_latent[index,: ]
            sample_var = vars[index,:]
            samples = sample_batch(sample_mean,sample_var)
            collapse = full_std <= self.std_threshold
            for i in range(z_dim):
                axs[j,i].hist(samples[:,i], bins=40, density=True, alpha=0.6, color='b')
                x = np.linspace(-8, 8, 100)
                p = norm.pdf(x, 0, 1)
                axs[j,i].plot(x, p, 'k', linewidth=2)
                title = "beta = {} s_dim = {}".format(b,i)
                if collapse[i] :
                    title = title + ' c'
                axs[j,i].set_title(title)
                axs[j,i].xaxis.set_visible(False) # Hide only x axis
                axs[j,i].set_ylim([0, 0.5])
        #frame1 = plt.gca()
        #frame1.axes.xaxis.set_ticklabels([])
        #frame1.axes.yaxis.set_ticklabels([])
        plt.savefig('plots/posterior_collapse/posterior_collapse.pdf')
        
    
    def posterior_collapse_qqplot(self,loader,betas,z_dim = 10):
        L=1000 #Number of samples for the histogram
        # Plot the histogram.
        print('## Posterior Collapse qq plot')
        fig, axs = plt.subplots(len(betas), z_dim,figsize=(23, 17))
        for j in range(len(betas)):
            b=betas[j] 
            PATH = self.get_model_name(b,z_dim)
            model = torch.load(PATH, map_location=torch.device('cpu'))
            full_latent,vars,_ = compute_latent(loader,model)

            full_latent = full_latent[:,1:]

            n = full_latent.shape[0]
            index = np.random.randint(n,size = L)
            sample_mean = full_latent[index,: ]
            sample_var = vars[index,:]
            samples = sample_batch(sample_mean,sample_var)
            for i in range(z_dim):
                sm.qqplot(samples[:,i]  , line='s', ax = axs[j,i])
                if j<= len(betas)-2:
                    axs[j,i].set(xlabel=None)
                if i>=1:     
                    axs[j,i].set(ylabel=None)
                title = "beta = {} s_dim = {} ".format(b,i)
                axs[j,i].set_title(title)
            
        plt.savefig('plots/posterior_collapse/qqplot/qqplot_collapse.png')

    def get_cov_ellipse(self,z_mean, z_var, nstd,color, **kwargs): 
        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(z_var)
        # return Ellipse(xy=z_mean, width=width, height=height,fill=False,
        #             color=color,alpha=0.5, **kwargs)
        return Ellipse(xy=z_mean, width=width, height=height,fill=False,
                     color=color,alpha=1,linewidth= 6, **kwargs)
        


    ## Add to ax the scatter points and the covariance matrix around for a proportion p
    def scatter_with_covar(self,ax,full_latent,vars,label, col_dict, mask_to_name):
        #Set the fontsize of xticks and yticks
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        if len(col_dict)==1 : 
            for i in np.unique(label):      
                mask = label ==i
                ax.scatter(x = full_latent[mask,0], y= full_latent[mask,1] ,s =0.01 ,alpha=0.5 )
            for i in range(vars.shape[0]):
                e = self.get_cov_ellipse(full_latent[i,:],vars[i,:],1,color='royalblue')
                ax.add_artist(e)
        else :
            for i in np.unique(label):      
                mask = label ==i
                #alpha=0.5
                ax.scatter(x = full_latent[mask,0], y= full_latent[mask,1] ,s =0.01 ,alpha=0.8, c = col_dict[i], label = mask_to_name[i] )
            for i in range(vars.shape[0]):
                e = self.get_cov_ellipse(full_latent[i,:],vars[i,:],1,color = col_dict[label[i]] )
                ax.add_artist(e)
            #ax.legend(markerscale=50,loc='lower right')
            ax.legend(markerscale=70,loc='upper right', fontsize="20",)


    # p subsample proportion of full _latent
    def plot_latent_dim_with_var(self, mask_to_name,p=0.5):
        print('### Label = ')
        print(np.unique(self.label))
        n_sample = int(self.full_latent.shape[0]*p)
        print('Using {} samples'.format(n_sample))
        
        # target_ROI_arr = ['Glomerulus', 'Proximal_Tubule', 'Distal_Tubule', 'Collecting_Duct', 'Thick_Ascending_Limb']
        #new_colors = ['royalblue','darkorange', 'green' , 'firebrick', 'darkviolet' , 'paletvioletred' ]
        #Blue, Orange , green , red , Turquoise, Pink
        
        # new_colors = ['#1f77b4', 'darkorange', 'green', 'firebrick', 'black', 'darkmagenta']
        # col_dict = dict(zip(range(len(new_colors)),new_colors))
        
        col_dict = self.color_dict(n_col = len(np.unique(self.label)))
        #Subsample a proportion p_loader of the points        
        if p != 1 :
            r = np.random.RandomState(random_state_synthetic)
            sub_index = r.choice(self.full_latent.shape[0], int(self.full_latent.shape[0]*p), replace=False)
            full_latent = self.full_latent[sub_index]
            label = self.label[sub_index]
            vars = self.vars[sub_index]
        
        ## Colored version
        #_, ax = plt.subplots()
        _, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))

        plt.xlim((-4.5, 4.5))
        plt.ylim((-4.5, 4.5))
        #ax.tick_params(axis='x', labelsize=20)
        #ax.tick_params(axis='y', labelsize=20)
        
        #LABELS
        #ax.set_xlabel(r'$z_1$',fontsize="40")
        #ax.set_ylabel(r'$z_2$',fontsize="40", rotation=0,ha='right')
        
        #Synthetic
        # plt.xlim((-3.2, 3.2))
        # plt.ylim((-4.4, 2))

        #ax.set_facecolor('black')
        self.scatter_with_covar(ax,full_latent[:,1:],vars,label, col_dict,mask_to_name)
        #plt.title('2 dim latent space Beta {}'.format(title))

    
        ## Kidney    
        #plt.savefig('plots/2d_reduction/kidney/kidney6/p{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/kidney6/b0_p{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        
        ##All data
        #plt.savefig('plots/2d_reduction/kidney/subsample_{}_colored.png'.format(p), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_41/p_{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_31/p_{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/p_{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        # V6
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/p_{}_sample_{}_colored.png'.format(p,n_sample), bbox_inches='tight',dpi=300)




        ## Synthetic
        # plt.savefig('plots/2d_reduction/synthetic/subsample_{}_colored.png'.format(p), bbox_inches='tight')
        #plt.savefig('plots/2d_reduction/synthetic/complete/v26b4_subsample_{}_colored.png'.format(p), bbox_inches='tight',dpi=300)

        ## Black and White version version
        #_, ax = plt.subplots()
        _, ax = plt.subplots(figsize=(10, 10))
        # plt.xlim((-4, 4))
        # plt.ylim((-4, 4)) 

        plt.xlim((-4.5, 4.5)) 
        plt.ylim((-4.5, 4.5))

        #Synthetic : 
        # plt.xlim((-3.2, 3.2))
        # plt.ylim((-4.4, 2))



        #ax.set_xlabel(r'$z_1$', **csfont,fontsize="40")
        #ax.set_ylabel(r'$z_2$', **csfont,fontsize="40", rotation=0,ha='right')
        
        col_dict={0:'blue'}
        ######################## Black and white
        # Selecting a random index
        ########
        ######## BLOCK TO remove
        n_sample = 2
        idx1 = np.random.choice(full_latent.shape[0],size=n_sample)
        full_latent_temp = full_latent[idx1,1:].reshape(n_sample,-1)
        vars_temp = vars[idx1,:]
        #.reshape(n_sample,-1)
        label_temp = label[idx1]
        self.scatter_with_covar(ax,full_latent_temp[0,:].reshape(1,-1),vars_temp[0,:].reshape(1,-1),np.array([label_temp[0]]), col_dict,mask_to_name)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/one_sample_black_white.png', bbox_inches='tight',dpi=300)
        self.scatter_with_covar(ax,full_latent_temp,vars_temp,label_temp, col_dict,mask_to_name)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/two_samples_black_white.png', bbox_inches='tight',dpi=300)
        ########
        ########
        
        self.scatter_with_covar(ax,full_latent[:,1:],vars,label, col_dict,mask_to_name)
        ########################
        #plt.title('2 dim latent space Beta {}'.format(title))

        #### KIDNEY
        #plt.savefig('plots/2d_reduction/kidney/kidney6/p{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/kidney6/b0_p{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        
        ### Synthetic
        #plt.savefig('plots/2d_reduction/synthetic/complete/v26b4_subsample_{}_black_white.png'.format(p), bbox_inches='tight',dpi=300)
        
        ### ALL DATA
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_41/p_{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        # plt.savefig('plots/2d_reduction/kidney/kidney6/subsample_{}_black_white.pdf'.format(p), bbox_inches='tight')
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney2_5/v7_b1_subsample_{}_black_white.png'.format(p), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney2_5/v1_subsample_{}_black_white.pdf'.format(p), bbox_inches='tight')
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_31/p_{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/p_{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        # V6
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/p_{}_sample_{}_black_white.png'.format(p,n_sample), bbox_inches='tight',dpi=300)
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/one_sample_black_white.png', bbox_inches='tight',dpi=300)


        
        #plt.savefig('plots/2d_reduction/kidney/subsample_{}_epoch_{}.png'.format(p,title), bbox_inches='tight')
        
        plt.close()

###########################################################################################################
###########################################################################################################
###########################################################################################################
    def plot_latent_dim_3d(self, full_latent, vars, label, mask_to_name,title):
        print('Plot latent in 3d')
        p=0.05  ## Proportion of sample to take
        print('Using {} samples'.format(int(full_latent.shape[0]*p)))

        col_dict = self.color_dict()
        if p != 1 :
            sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0]*p), replace=False)
            full_latent = full_latent[sub_index]
            label = label[sub_index]
            vars = vars[sub_index]
        
        _, ax = plt.subplots()

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        stds = np.sqrt(vars)
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)

        for i in range(full_latent.shape[0]):
            center = full_latent[i,1:]
            std = stds[i,:]
            # print('i=',i)
            # print('center :', center)
            # print('std : ' , std)
            x = center[0]+std[0] * np.outer(np.cos(u), np.sin(v))
            y = center[1]+std[1] * np.outer(np.sin(u), np.sin(v))
            z = center[2]+std[2] * np.outer(np.ones_like(u), np.cos(v))
            
            c = col_dict[label[i]]
            ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=c, linewidth=0.1, alpha=0.5, shade=True)

        #ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color='red', linewidth=0.1, alpha=0.7, shade=True)
        lim = 4
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim,lim)
        ax.set_zlim(-lim,lim)
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel('$Z$', fontsize=30)
        plt.title('3 dim latent space {}'.format(title))
        
        #plt.savefig('plots/2d_reduction/3d/pdfs/subsample_{}_b{}.pdf'.format(p,title), bbox_inches='tight')
        plt.savefig('plots/2d_reduction/3d/subsample_{}_b{}.png'.format(p,title), bbox_inches='tight')
        plt.close()

    def plot_latent_dim_coeff(self, alpha_label):
        
        #p = 0.008
        #p= 0.05
        #p=0.2
        p=0.7
        n = self.full_latent.shape[0]
        print('Using {} samples'.format(int(n*p)))
        #col_dict = self.color_dict()
        
        if p != 1 :
            r = np.random.RandomState(random_state_synthetic)
            sub_index = r.choice(n, int(n*p), replace=False)
            #sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0]*p), replace=False)
            full_latent = self.full_latent[sub_index]
            alpha_label = alpha_label[sub_index]
            #vars = self.vars[sub_index]
        
        fig, ax = plt.subplots(figsize = (10,10))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        # plt.xlim((-4, 4))
        # plt.ylim((-4, 4))
        # plt.xlim((-3.5, 5))
        # plt.ylim((-5.5, 1.5))
        # plt.xlim((-3.5, 5))
        # plt.ylim((-5.7, 1.3))
        # plt.xlim((-3.5, 5))
        # plt.ylim((-5.7, 1.6))
        #Synthetic : 
        # plt.xlim((-3.5, 3.5))
        # plt.ylim((-4.5, 2.5))
        plt.xlim((-3.2, 3.2))
        plt.ylim((-4.4, 2))
        
        sc = ax.scatter(x = self.full_latent[:,1], y= self.full_latent[:,2] ,c = alpha_label ,alpha=0.5, cmap='viridis' )
        # for i in range(vars.shape[0]):
        #     width, height = 2 * nstd * np.sqrt(z_var)
        # e=Ellipse(xy=z_mean, width=width, height=height,fill=False,
        #             color=color,alpha=0.5, **kwargs)
            
        # ax.add_artist(e)
        
        # ax.legend(markerscale=50)
        #plt.colorbar(sc)
        
        pos = ax.get_position().get_points().flatten()
        #ax_cbar = fig.add_axes([pos[0]+0.04, 0.86, (pos[2]-pos[0])*0.9, 0.02])
        #ax_cbar = fig.add_axes([pos[0]+0.04, 0.856, (pos[2]-pos[0])*0.9, 0.02])
        ax_cbar = fig.add_axes([pos[0]+0.04, 0.84, (pos[2]-pos[0])*0.9, 0.02])
        cb= plt.colorbar(sc, cax=ax_cbar, orientation='horizontal',aspect=20)
        cb.ax.tick_params(labelsize=20)
        
        #Put ticks on top of the bar
        #ax_cbar.xaxis.set_ticks_position("top")

        #plt.title('Scaling values in latent space')
        #plt.savefig('plots/synthetic_data/latent_dim_coefficients_subsample_{}.pdf'.format(p), bbox_inches='tight')
        #plt.savefig('plots/2d_reduction/synthetic/coefficients/latent26_coefficients_subsample_{}.png'.format(p), bbox_inches='tight',dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize = (10,10))
        sc = ax.scatter(x = full_latent[:,1], y= full_latent[:,2] ,c = alpha_label ,alpha=0.5, cmap='viridis' )
        plt.axis('off')
        #plt.savefig('plots/2d_reduction/synthetic/coefficients/transparant_latent26_coefficients_subsample_{}.png'.format(p), bbox_inches='tight',dpi=300 , transparent=True)


###########################################################################################################
###########################################################################################################
###########################################################################################################

    def plot_latent_dim_pairs(self,full_latent,vars,label,mask_to_name):
        z_dim = full_latent.shape[1]-1 
        for i in range(z_dim) : 
            for j in range(i+1,z_dim):
                title_complement = 'dims_{}_vs_{}'.format(i,j)
                # print('### Label shape = ' , label.shape)
                # print('### mask_to_name = ', mask_to_name)
                # print('### Full latent ', full_latent[:,[i,j]].shape)
                self.plot_latent_dim_with_var(full_latent[:,[0,i+1,j+1]],vars[:,[i,j]], label, mask_to_name, title_complement )
    #Sample one possible reconstruction for an original signal x
    def sample_one_reconstruction(self,x):
        with torch.no_grad():
            z_mean,  z_logvar= self.model.forward(x)        
            z = reparametrize(z_mean,z_logvar)
            #z = z_mean
            decoder_mean = self.model.decoder(z)
        return decoder_mean[:,0,:]


    def reconstruction_loss_plot(self , x, mu_x): 
        batch_size = x.size(0)
        r_loss = 0.5*F.mse_loss(x,mu_x, reduction='sum').div(batch_size)
        return r_loss
    #Plot the reconstruction for a random batch taken from the loader
    def plot_reconstruction(self, loader,title):
        x_p,_ = next(iter(loader))
        _,z_logvar = self.model.forward(x_p)
        decoder_mean = self.sample_one_reconstruction(x_p)
        print(decoder_mean)
        
        print('average recons error = ', self.reconstruction_loss_plot(x_p[:,0,:], decoder_mean))
        for i in range(x_p.shape[0]) :
            #Plot the Original vs reconstruction
            x_or = x_p[i,0,:].detach().numpy()
            x_pos = np.arange(len(x_or))
            x_rec = decoder_mean[i,:].detach().numpy()

            

            plt.figure('train_{}'.format(i),figsize=(20, 10))
            _, stemlines1, _ = plt.stem(x_pos, x_or, 'tab:blue', markerfmt=' ', label='Original')
            plt.setp(stemlines1, 'linewidth', 3)
            _, stemlines2, _ = plt.stem(x_pos+0.3, x_rec, 'tab:orange', markerfmt=' ', label='Reconstruction')
            plt.setp(stemlines2, 'linewidth',3)
            plt.title('reconstruction for a {} sample{}_std_{}'.format(title,i,z_logvar[i].exp().div(2)) )
            plt.legend()
            plt.savefig("plots/reconstruction/reconstruction_{}{}.png".format(title,i))
            plt.close()

            #Plot the original signal
            # plt.figure('train_or_{}'.format(i),figsize=(20, 10))
            # _, stemlines1, _ = plt.stem(x_pos, x_or, 'tab:blue', markerfmt=' ', label='Original')
            # plt.setp(stemlines1, 'linewidth', 3)
            # plt.title('original {} sample{}'.format(title, i))
            # plt.savefig("plots/reconstruction/original/{}_original{}.png".format(title, i))
            # plt.close()

    #Reconstruct the full tensor data 
    def reconstruct_image(self) :
        loader = dat.full_index_data() #Original data to be encode and decode
        full_reconstruction = np.zeros((len(loader.dataset),loader.dataset[0][0].shape[1]))
        with torch.no_grad():
            for x, _ , i in loader :
                full_reconstruction[i,:] = self.sample_one_reconstruction(x)
        return full_reconstruction

    #Plot a seriz of patial images and their reconstruction for mzs in [index_center-3,index_center+3]
    def plot_original_vs_recons(self,index_center=151):
        centroids, _, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        full_reconstruction = self.reconstruct_image()
        
        norms = {
            "standard": 1,
            "normalized": norm
        }
        #norms = [1,norm] #1 corresponds to no normalization
        for index in range(index_center-3,index_center+3) : 
            for norm in norms.keys() :
                fig, axs = plt.subplots(figsize=(10, 10), nrows=2, constrained_layout=True)
                fig.suptitle(f"m/z {mzs[index]:.4f}", fontsize=20)
                # normalization is done by dividing the raw intensity by the normalization data
                path = 'plots/image_reconstruction/'
                ax0title = 'Original Image'
                ax1title = 'Reconstructed Image'
                if norm == 'standard' :
                    path = path + 'standard/'
                else :
                    path = path + 'normalized/'
                    add_title = ' with normalization'
                    ax0title = ax0title + add_title
                    ax1title = ax1title + add_title
                
                axs[0].imshow(dat.reshape_array(centroids[:, index] / norms[norm], image_shape, pixel_index))
                axs[0].set_title(ax0title)
                axs[1].imshow(dat.reshape_array(full_reconstruction[:, index] / norms[norm], image_shape, pixel_index))
                axs[1].set_title(ax1title)  
                plt.savefig(path+'reconstruction_mz_{}.png'.format(mzs[index]))

    #Create the stem plot for the reconstructed signal from z during latent traversal of factor {f_index} at time step {t}
    def create_frame(self, z,f_index,t,interpolation):
        z[f_index] = interpolation[t]
        with torch.no_grad():
            decoder_mean = self.model.decoder(z)[0,0,:]
        intensity = decoder_mean.detach().numpy()
        x_pos = np.arange(len(intensity))
        plt.figure(figsize=(20, 10))

        _, stemlines1, _ = plt.stem(x_pos, intensity, 'tab:blue', markerfmt=' ', label='Original')
        plt.setp(stemlines1, 'linewidth', 3)
        plt.ylim(-6, 8)
        plt.title(f'Latent reconstruction factor {f_index} at step {t}',fontsize=14)
        plt.savefig("plots/latent_traversal/images/latent_factor{}_step_t{}.png".format(f_index,t))
        plt.close()
    
    def create_frame_app(self,z,t,n,f_index,color,labels,col_dict):
        ## Color the stem plot with the desired label
        colors = False
        with torch.no_grad():
            decoder_mean = self.model.decoder(z)[0,0,:]
        intensity = decoder_mean.detach().numpy()
        x_pos = np.arange(len(intensity))
        # Kidney with open('saved_data/traversal/reconstruct_p{}_step_{}.npy'.format(n,t), 'wb') as f:
        #VAN 1_41 with open('saved_data/traversal/van1_41/reconstruct_p{}_step_{}.npy'.format(n,t), 'wb') as f:
        #VAN 1_31 with open('saved_data/traversal/van1_31/reconstruct_p{}_step_{}.npy'.format(n,t), 'wb') as f:
        #VAN 1_35 with open('saved_data/traversal/van1_31/reconstruct_p{}_step_{}.npy'.format(n,t), 'wb') as f:
        with open('saved_data/traversal/van1_35/reconstruct_p{}_step_{}.npy'.format(n,t), 'wb') as f:
            np.save(f,intensity)
        plt.figure(figsize=(20, 10))
        
        if colors :
            _, stemlines1, _ = plt.stem(x_pos, intensity, color, markerfmt=' ')
        else :
            _, stemlines1, _ = plt.stem(x_pos, intensity, markerfmt=' ')
        
        for l in labels: 
            if colors :
                _,s,_  = plt.stem(0, 0,col_dict[l] ,markerfmt=' ',label=l, basefmt=" ")
                plt.setp(s,label = l)
            # else :
            #     _,s,_ = plt.stem(0, 0,col_dict[0] ,markerfmt=' ', basefmt=" ")
            
        

        plt.setp(stemlines1, 'linewidth', 3)
        plt.ylim(-6, 8)
        plt.title(f'Latent reconstruction factor {f_index} at step {t}',fontsize=14)
        plt.savefig("plots/latent_traversal/images/approximate/latent{}_factor{}_step_t{}.png".format(n,f_index,t))
        plt.legend()
        plt.close()

    #Make the latent traversal for factor{f_index} with values ranging from z_min to z_max
    def latent_traversal(self, f_index,loader, z_min, z_max):
        #inter =  10
        inter = (z_max-z_min)/100
        x,_ = next(iter(loader))
        interpolation = torch.arange(z_min, z_max, inter)

        with torch.no_grad():
            z_mean,  z_logvar= self.model.forward(x)
            z_mean = z_mean[0]
            z_logvar = z_logvar[0]
            z_ori = reparametrize(z_mean,z_logvar)
        
        time = range(len(interpolation))
        #create all images
        for t in time:
            self.create_frame(z_ori,f_index,t, interpolation)
        #Save all images in frames
        frames = []
        for t in time:
            image = imageio.v2.imread(f'plots/latent_traversal/images/latent_factor{f_index}_step_t{t}.png')
            frames.append(image)
        imageio.mimsave(f'plots/latent_traversal/gifs/latent_traversal_f{f_index}.gif', # output gif
                    frames,          # array of input frames
                    fps = 5)         # optional: frames per second    
    
    #Make {n_points} different latent traversal between pol_limits1 and pol_limits 2
    def latent_traversal_legs(self,full_latent,vars,label,mask_to_name,pol_limits1, pol_limits2):
        n_points=  50
        p1 = m_path.Path(pol_limits1) 
        p2 = m_path.Path(pol_limits2) 
        flag1 = p1.contains_points(full_latent[:,1:])
        flag2 = p2.contains_points(full_latent[:,1:])

        n_steps = 15
        # print('### min 1 = ', np.min(full_latent[flag2,1]))
        # print('### max 2 = ',np.max(full_latent[flag1,1]) )
        t = np.abs( np.max(full_latent[flag2,1]) - np.min(full_latent[flag1,1]))/ n_steps
        # print('### t = ', t)
        
        # Plot the ellipses of latent space
        _, ax = plt.subplots(figsize=(10,10))
        # plt.xlim((-4, 4))
        # plt.ylim((-4, 4)) 
        plt.xlim((-4.5, 4.5))
        plt.ylim((-4.5, 4.5)) 

        c_dict = self.color_dict()
        c_dict={0:'blue'}
        ## VAN1_41 , VAN1_31,
        poly_colors = ['black', '#a30000']

        #VAN 1_35
        poly_colors = ['#a30000','black']


        #poly_colors = ['#a30000','black']
        self.scatter_with_covar(ax,full_latent[:,1:],vars,label,c_dict,mask_to_name)
        poly1 = Polygon(pol_limits1,alpha = 0.5,ec = "gray",fc = poly_colors[0],visible = True)
        ax.add_patch(poly1)
        poly2 = Polygon(pol_limits2,alpha = 0.5, ec = "gray",fc= poly_colors[1],visible = True)
        ax.add_patch(poly2)
        
        random_start = np.random.choice(np.sum(flag1), size=n_points, replace=False)
        z_start = full_latent[flag1,1: ][random_start]
        
        p_to_steps = {}
        for i in range(n_points):
            print('### Point ', i)
            print('z_start = ', z_start[i,:])
            ax.plot(z_start[i,0],z_start[i,1] , ".")
            z_curr = z_start[i,:]
            #for _ in range(n_steps) :
            j=0
            path = []
            latent_val = [z_curr]
            while not p2.contains_point(z_curr) and j < n_steps :
                plt.quiver(z_curr[0], z_curr[1], t, 0, scale_units='xy', angles='xy', scale=1, alpha=0.8)
                z_curr = z_curr + np.array([t,0])
                latent_val.append(z_curr)
                j+=1
            p_to_steps[i] = len(latent_val)
            #plt.title('Latent traversal sublegs proximal tubule')
            ## KIDNEY
            #plt.savefig('plots/latent_traversal/polygons/latent_traversal.png',bbox_inches='tight',dpi=300)
            ## VAN1_41
            #plt.savefig('plots/latent_traversal/polygons/latent_traversal1_41.png',bbox_inches='tight',dpi=300)
            ## VAN1_31
            #plt.savefig('plots/latent_traversal/polygons/latent_traversal1_31.png',bbox_inches='tight',dpi=300)
            ## VAN1_35
            plt.savefig('plots/latent_traversal/polygons/latent_traversal1_35.png',bbox_inches='tight',dpi=300)
            
            if p2.contains_point(z_curr) :
                print('Gif of current traversal')

                self.latent_traversal_app_gif(np.array(latent_val),np.zeros(len(latent_val)),0,i, index_missing=True)
        #min_d = np.argmin(full_latent[flag,2])

        #### REMOVE van1_41 for base data
        #### For van1_41 : 'saved_data/van1_41/points_to_n_steps.pkl'
        #### ALSO CHANGE PATH IN latent_traversal_app_gif --> create_frame_app --> 
        with open('saved_data/van1_35/points_to_n_steps.pkl', 'wb') as fp:
            pickle.dump(p_to_steps,fp)
        return 0  

    def variance_latent_traversal(self):
        n_traversal = 50
        p_to_n = {0:6,1:6,2:7,3:7,4:4} #Dictionary from points to number of steps in traversal
        
        data_dir = Path(r"/Users/pdelacour/Documents/PL_Ecole/beta_vae/data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
        
        

        data_dir2_5 = Path(r"data/negative/VAN0005-RK-2-5-IMS_lipids_neg_roi=#0_mz=fix")
        data_dir1_31 =  Path(r"data/negative/VAN0042-RK-1-31-IMS_lipids_neg_roi=#1_mz=fix")
        data_dir1_35 = Path(r"data/negative/VAN0049-RK-1-35-IMS_lipids_neg_roi=#1_mz=fix")
        data_dir1_41 = Path(r"data/negative/VAN0063-RK-1-41-IMS_lipids_neg_roi=#1_mz=fix")
        #path_points = 'saved_data/van1_41/points_to_n_steps.pkl'
        path_points = 'saved_data/van1_35/points_to_n_steps.pkl'

        data_dir = data_dir1_35
        ##### ADD or REMOVE van1_41
        ##### For kidney : 'saved_data/points_to_n_steps.pkl'
        ##### For van1_41 : 'saved_data/van1_41/points_to_n_steps.pkl'
        ##### For van1_31 : 'saved_data/van1_31/points_to_n_steps.pkl'
        with open(path_points,'rb') as fp:
            p_to_n = pickle.load(fp)
        combined_p = []
        for p in range(n_traversal):
            n_points = p_to_n[p]
            p_steps = np.zeros((n_points,212))
            for i in range(n_points):
                ### For Kidney : 'saved_data/traversal/reconstruct_p{}_step_{}.npy'
                ### For VAN1_41 'saved_data/traversal/van1_41/reconstruct_p{}_step_{}.npy'
                ### For VAN1_31 'saved_data/traversal/van1_31/reconstruct_p{}_step_{}.npy'
                with open('saved_data/traversal/van1_35/reconstruct_p{}_step_{}.npy'.format(p,i),'rb') as f :
                    p_steps[i,:] = np.load(f)
            combined_p.append(p_steps)
        
        _, mzs = dat.load_centroids(data_dir)
        
        ## Subset of x_ticks that are going to be ploted
        subset =  np.arange(0,combined_p[0].shape[1],step=4)
        
        sign_traversal = np.zeros((len(combined_p),combined_p[0].shape[1]))
        for i,p_steps in enumerate(combined_p) : 
            plotdata = pd.DataFrame({'var':np.var(p_steps, axis = 0)},index =mzs)
            ## If this difference is positive it means the signal was increasing 
            sign_traversal[i,:] = ( p_steps[-1,:] - p_steps[0,:]>= 0)*2-1 ## Casting values to the range [-1,1] 
            plt.figure()
            ax = plotdata.plot(kind="bar",color= '#036512',figsize=(15,5))
            ax.set_xticks(subset)
            ax.set_xticklabels(mzs[subset], rotation = 45,ha='right')
            ax.set_xlabel('$m/z$', fontsize=10)
            ax.set_ylabel('Variance', fontsize = 10)
            plt.savefig('plots/latent_traversal/variance/traversal_p{}.png'.format(i), bbox_inches='tight',dpi=300)
        
        ## For interpretation note that some traversal increase the value and some decrease it
        sign_traversal = np.sum(sign_traversal,axis = 0) >=0
        #print('### Sign traversal = ' , np.sum(sign_traversal,axis = 0))
        
        ## Plot of the stacked traversal
        data = {}
        for i,p_steps in enumerate(combined_p) : 
            data['var_{}'.format(i)] = np.var(p_steps,axis=  0)
        plotdata = pd.DataFrame(data,index = mzs.astype(str))
        plt.figure()
        ax = plotdata.plot(kind="bar",stacked = True,figsize=(15,5),xticks=subset)
        ax.set_xticks(subset)

        #ticks_labels = mzs[subset]
        ticks_labels = np.array([f'{mzs[i]:.2f}' for i in subset ])
        ax.set_xticklabels(ticks_labels, rotation = 45,ha='right')
        #### LEGEND
        ax.set_xlabel('$m/z$', fontsize=10)
        ax.set_ylabel('Variance', fontsize = 10)
        
        ### Remove the legend for 50 points
        ax.get_legend().remove()
        ###
        plt.savefig('plots/latent_traversal/variance/traversal_combined.png', bbox_inches='tight',dpi=300)


        ## Csv file for the variance sorted by order 
        sum_var = np.sum(np.array(list(data.values())), axis = 0)
        col_data = pd.DataFrame({'mzs':mzs, 'variance':sum_var, 'increase':sign_traversal})
        col_data = col_data.sort_values(
            by="variance",
            ascending=False
        )
        col_data.to_csv('saved_data/combined_variance1_35.csv', index=False)

        ## Plot of the variance in order : 
        col_data = col_data.sort_values(
            by="variance",
            ascending=False
        )

        # f, ax = plt.subplots(figsize=(15,5))
        # subset = np.arange(0,212,step=4)
        # ax.set_xticks(subset)
        # ax.set_xticklabels(np.array(col_data['mzs'])[subset], rotation = 45,ha='right')
        # ax.set_xlabel('$m/z$', fontsize=10)
        # ax.set_ylabel('Variance', fontsize = 10)
        # color_bars = {1 : '#050533', 0:'#E34234',} #Darkblue, red
        # label_bars = {1: 'increasing', 0:'decreasing'}
        
        # for l in np.unique(col_data['increase']) :
        #     ind = col_data['increase']==l
        #     ax.scatter(np.arange(len(sum_var))[ind],col_data['variance'][ind], c=color_bars[l], label=label_bars[l])    
        # ax.legend()

        f, ax = plt.subplots(figsize=(15,5))

        subset = np.arange(0,212,step=4)
        ax.set_xticks(subset)
        ticks_labels_mzs = np.array(col_data['mzs'])
        ticks_labels = np.array([f'{ticks_labels_mzs[i]:.3f}' for i in subset ])
        for i in subset : 
            print(f'#### {ticks_labels_mzs[i]} , {ticks_labels_mzs[i]:.3f}')
        ax.set_xticklabels(ticks_labels, rotation = 45,ha='right')
        #ax.set_xticklabels(np.array(col_data['mzs'])[subset], rotation = 45,ha='right')
        
        #ax.set_xlabel('$m/z$', fontsize=10)
        #ax.set_ylabel('Variance', fontsize = 10)
        color_bars = {1 : '#050533', 0:'#E34234',} #Darkblue, red
        color_bars = {1 : '#fb6f92', 0:'#8d99ae',} #Darkblue, red
        color_bars = {1 : '#fe6d73', 0:'#17c3b2',} #Darkblue, red
        label_bars = {1: 'L-to-R increasing', 0:'L-to-R decreasing'}

        #axins = ax.inset_axes((0.5, 0.6, 0.3, 0.3))
        axins = ax.inset_axes((0.3, 0.6, 0.4, 0.3))
        ax.tick_params(axis='y', labelsize=15)
        axins.tick_params(axis='y', labelsize=15)
        axins.tick_params(axis='x', labelsize=15)

        cutx = 14
        eps=0.1
        #axins.set_xlim(-eps,cutx-1+eps)
        X_global =  np.arange(len(sum_var))
        X1 =X_global[:cutx]
        X2 =X_global[cutx:]
        Y1 = col_data['variance'][:cutx]
        Y2 = col_data['variance'][cutx:]

        col_data_sub = col_data[:cutx]['increase']
        for l in np.unique(col_data_sub) :
            ind = col_data_sub==l
            X = X1[ind]
            Y = Y1[ind]
            #ax.scatter(X,Y, c=color_bars[l], label=label_bars[l])  
            ax.scatter(X,Y, c=color_bars[l])  
            axins.scatter(X,Y, c=color_bars[l], label=label_bars[l])    

        subset = np.arange(0,cutx, step=1)
        #ticks_labels_in = np.array([f'{mzs[i]:.2f}' for i in subset ])
        ticks_labels_in = np.array([f'{ticks_labels_mzs[i]:.3f}' for i in subset ])
        axins.set_xticks(subset)

        axins.set_xticklabels(ticks_labels_in, rotation = 45,ha='right')


        #Plot the rest
        col_data_sub = col_data[cutx:]['increase']
        for l in np.unique(col_data_sub) :
            ind = col_data_sub==l
            X = X2[ind]
            Y = Y2[ind]
            ax.scatter(X,Y, c=color_bars[l],label = label_bars[l])  
            
        mark_inset(ax,axins,loc1=1,loc2=3,)
        ax.legend(prop = { "size": 15 })
        #plt.savefig('plots/latent_traversal/variance/ordered_variance_traversal.png', bbox_inches='tight',dpi=300)  
        plt.savefig('plots/latent_traversal/variance/ordered_variance_traversal1_35.png', bbox_inches='tight',dpi=300)  
    
    ## index_missing = Boolean to indicate if latent_val contains the index in first column
    def latent_traversal_app_gif(self,latent_val,label,f_index,n, index_missing = False):
        col_dict = self.color_dict()
        label_unique = np.unique(label)
        colors = [col_dict[l] for l in label]
        for t in range(latent_val.shape[0]):
            if index_missing:
                z=torch.Tensor(latent_val[t,])
                #print('### z = ', z)
            else :
                z=torch.Tensor(latent_val[t,1:])
            self.create_frame_app(z,t,n,f_index,colors[t],label_unique,col_dict)
        frames = []
        for t in range(latent_val.shape[0]):
            image = imageio.v2.imread(f'plots/latent_traversal/images/approximate/latent{n}_factor{f_index}_step_t{t}.png')
            frames.append(image)
        imageio.mimsave(f'plots/latent_traversal/gifs/approximate/latent{n}_traversal_f{f_index}.gif', # output gif
                    frames,          # array of input frames
                    fps = 5)         # optional: frames per second    


    def latent_traversal_heatmap(self,f_index,loader, z_min , z_max):
        #inter =  0.1
        #inter = 10
        inter = (z_max-z_min)/100
        x,_ = next(iter(loader))
        interpolation = torch.arange(z_min, z_max, inter)
        
        with torch.no_grad():
            z_mean,  z_logvar= self.model.forward(x)
            z_mean = z_mean[0]
            z_logvar = z_logvar[0]
            z = reparametrize(z_mean,z_logvar)
        time = range(len(interpolation))
        data = np.zeros((x.shape[2], len(time)))
        for i,t in enumerate(time):
            z[f_index] = interpolation[t]
            decoder_mean = self.model.decoder(z)[0,0,:]
            data[:,i] = decoder_mean.detach().numpy()
        plt.figure()
        v_min = np.min(data)
        v_max = np.max(data)
        c = plt.pcolor(data, edgecolors='k', linewidths=1e-1, cmap='RdBu',vmin=v_min, vmax=v_max)
        plt.colorbar(c)
        plt.xlabel('Time steps')
        plt.ylabel('mz response')
        plt.title(f'Latent traversal for increasing values of factor_{f_index} ')
        plt.savefig("plots/latent_traversal/heatmap/latent_traversal_factor{}_train.pdf".format(f_index))


    # def variance_latent_traversal(self):
    #     p_to_n = {0:6,1:6,2:7,3:7,4:4} #Dictionary from points to number of steps in traversal
    #     with open('saved_data/points_to_n_steps.pkl','rb') as fp:
    #         p_to_n = pickle.load(fp)
    #     combined_p = []
    #     for p in range(50):
    #         n_points = p_to_n[p]
    #         p_steps = np.zeros((n_points,212))
    #         for i in range(n_points):
    #             with open('saved_data/traversal/reconstruct_p{}_step_{}.npy'.format(p,i),'rb') as f :
    #                 p_steps[i,:] = np.load(f)
    #         combined_p.append(p_steps)
    #     data_dir = Path(r"/Users/pdelacour/Documents/PL_Ecole/beta_vae/data/VAN0046-LK-3-45-IMS_lipids_neg_roi=#1_mz=fix")
    #     _, mzs = dat.load_centroids(data_dir)
        
    #     ## Subset of x_ticks that are going to be ploted
    #     subset =  np.arange(0,combined_p[0].shape[1],step=4)
    #     for i,p_steps in enumerate(combined_p) : 
    #         plotdata = pd.DataFrame({'var':np.var(p_steps, axis = 0)},index =mzs)
    #         plt.figure()
    #         ax = plotdata.plot(kind="bar",color= '#036512',figsize=(15,5))
    #         ax.set_xticks(subset)
    #         ax.set_xticklabels(mzs[subset], rotation = 45,ha='right')
    #         ax.set_xlabel('$m/z$', fontsize=10)
    #         ax.set_ylabel('Variance', fontsize = 10)
    #         plt.savefig('plots/latent_traversal/variance/traversal_p{}.png'.format(i), bbox_inches='tight')
        
    #     ## Plot of the stacked traversal
        
    #     data = {}
    #     for i,p_steps in enumerate(combined_p) : 
    #         data['var_{}'.format(i)] = np.var(p_steps,axis=  0)
    #     plotdata = pd.DataFrame(data,index = mzs.astype(str))
    #     plt.figure()
    #     ax = plotdata.plot(kind="bar",stacked = True,figsize=(15,5),xticks=subset)
    #     ax.set_xticks(subset)
    #     ax.set_xticklabels(mzs[subset], rotation = 45,ha='right')
    #     ax.set_xlabel('$m/z$', fontsize=10)
    #     ax.set_ylabel('Variance', fontsize = 10)
    #     ### Remove the legend for 50 points
    #     ax.get_legend().remove()
    #     ###
    #     plt.savefig('plots/latent_traversal/variance/traversal_combined.png', bbox_inches='tight')

    #     ## Csv file for the variance sorted by order 
    #     sum_var = np.sum(np.array(list(data.values())), axis = 0)
    #     col_data = pd.DataFrame({'mzs':mzs, 'variance':sum_var})
    #     col_data = col_data.sort_values(
    #         by="variance",
    #         ascending=False
    #     )
    #     col_data.to_csv('saved_data/combined_variance.csv', index=False)

    #     ## Plot of the variance in order : 
    #     sum_var = np.sum(np.array(list(data.values())), axis = 0)
    #     col_data = pd.DataFrame({'mzs':mzs, 'variance':sum_var})
    #     col_data = col_data.sort_values(
    #         by="variance",
    #         ascending=False
    #     )

    #     f, ax = plt.subplots(figsize=(15,5))
    #     subset = np.arange(0,212,step=4)
    #     ax.set_xticks(subset)
    #     #print('#### ' , col_data['mzs'][subset])
    #     ax.set_xticklabels(np.array(col_data['mzs'])[subset], rotation = 45,ha='right')
    #     ax.set_xlabel('$m/z$', fontsize=10)
    #     ax.set_ylabel('Variance', fontsize = 10)

    #     #plt.figure()
    #     ax.scatter(range(len(sum_var)),col_data['variance'])
    #     plt.savefig('plots/latent_traversal/variance/ordered_variance_traversal.png', bbox_inches='tight')

    




    def color_dict(self,n_col=6):
        new_colors = ['#1f77b4', 'darkorange', 'green', 'firebrick', 'black', 'darkmagenta']
        #new_colors = ['darkgoldenrod', 'darkorange', 'green', 'firebrick', 'black', 'darkmagenta']
        # n_col = 4  
        # new_colors = cmr.take_cmap_colors('viridis', n_col, return_fmt='hex')
        
        ## For SYNTHETIC DATA
        # n_col = 2**3  
        cmap_spa = 'cmr.pride'
        new_colors = cmr.take_cmap_colors(cmap_spa, n_col, return_fmt='hex')
        new_colors[-1] = [0.0, 0.5, 1.0, 1.0]  # Set the last color to blue

        


        c_dict = dict(zip(range(len(new_colors)),new_colors))
        return c_dict

    ## fa : factor on which to do the traversal
    ## z_min : array of min value for the traversal of each factor
    ## z_max : array of max value for the traversal of each factor
    ## n_points : number of points to draw the full trajectory
    def draw_latent_connections(self, fa, z_min, z_max , full_latent, vars, label,mask_to_name,n_points=5) :
        #p= 0.01 ##proportion of subsamples to consider
        # Random selction of point : 
        random = False

        t = (z_max[0]-z_min[0])/100
        #t = t
        
        c_dict = self.color_dict()
        
        #Plot the points and ellipses
        _, ax = plt.subplots()
        self.scatter_with_covar(ax,full_latent[:,1:],vars,label,c_dict,mask_to_name)
    
        #x , _ , idx_start = next(iter(loader))
        n_steps = 40
        
        if random :
            sub_index = np.random.choice(full_latent.shape[0], n_points , replace=False)
            z_start = full_latent[sub_index, 1:]
            idx_start =  full_latent[sub_index, 0]   
        else : #Target specific point 
            n_points=1
            idx_start = [227686]
            print('is in = ', np.sum(full_latent[:,0]==idx_start[0]))
            z_start = full_latent[full_latent[:,0]==idx_start[0],1:]
            print('### Z_start = ', z_start)
            
        #### To target

        #with_color = True
        
        combined_idx = [] 
        combined_arr = []
        for i in range(n_points):
            print('### i= ', i)
            ax.plot(z_start[i,0],z_start[i,1] , ".")
            #z_arr , _ = self.compute_trajectory(fa, z_start[i], idx_start[i],t,n_steps,full_latent)
            z_arr , z_idx = self.compute_full_trajectory(fa, z_start[i], idx_start[i],t,n_steps,full_latent)
            
            combined_idx.append(z_idx)
            combined_arr.append(z_arr)
            for j in range(1, z_arr.shape[0]) : 
                    z_curr = z_arr[j-1,:]
                    z_diff = z_arr[j,:] - z_curr
                    plt.quiver(z_curr[0], z_curr[1], z_diff[0], z_diff[1], scale_units='xy', angles='xy', scale=1, alpha=0.8)
        plt.title('2 dim latent space with var')
        plt.savefig('plots/latent_traversal/2d_traversal/latent_dim_traversal_factor{}.pdf'.format(fa))
        plt.savefig('plots/latent_traversal/2d_traversal/latent_dim_traversal_factor{}.png'.format(fa))
        return combined_arr,combined_idx


    ## For now only works with 2 dimension
    def next_z_traversal(self,fa, z_curr, idx_curr, z_start,t,full_latent, mode='forward') :
        threshold = 0.1 # must be multidimensional when z_dim > 2
        index = np.abs(full_latent[:,2-fa] - z_start[1-fa]) <=  threshold # Eliminate z_2 that are two far
        z_list = full_latent[index,:]
        if len(z_list) > 0 :
            if mode == 'forward':
                step_array = np.where(z_list[:,1+fa] - z_curr[fa] +t >= 0,  z_list[:,1+fa] , np.inf) #z_next >= z_c + t
                idx_closest = step_array.argmin() #index of closest z_1
            else :
                #print('else case')
                step_array = np.where(-z_list[:,1+fa] + z_curr[fa] +t >= 0,  z_list[:,1+fa] , -np.inf) #z_next <= z_c - t
                idx_closest = step_array.argmax() #index of closest z_1
            #idx_closest = step_array.argmin() #index of closest z_1
            if not np.isinf(step_array[idx_closest]):
                z_next = z_list[ idx_closest, 1:]
                idx_next = z_list[ idx_closest, 0]
                return z_next ,idx_next
        return z_curr, idx_curr # We don't traverse

    def compute_full_trajectory(self,fa,z_start,idx_start,t,n_steps, full_latent):
        z_arr1, idx_arr1 = self.compute_trajectory(fa,z_start,idx_start,t,n_steps,full_latent,mode='forward')
        z_arr2, idx_arr2 = self.compute_trajectory(fa,z_start,idx_start,t,n_steps,full_latent,mode='backward')
        
        z_arr = np.concatenate( (np.flip(z_arr2, axis=0) , z_arr1[1:,]) , axis = 0 ) 
        z_idx = np.concatenate( (np.flip(idx_arr2,axis=0),idx_arr1[1:,]), axis = 0 )

        z_idx = [int(key) for key, _ in groupby(z_idx)]
        return z_arr,z_idx

    # z_start : point on which the trajectory starts
    # t : min space until next point on traversal
    # n_steps : number of steps on the trajectory
    # full_latent : 2d matrix containing the full latent space
    def compute_trajectory(self,fa,z_start,idx_start, t , n_steps,full_latent, mode = 'forward'):
        #z_curr = z_start
        #idx_curr = idx_start
        idx_arr = np.zeros(n_steps+1)
        z_arr = np.zeros((n_steps+1, self.model.z_dim))
        idx_arr[0] = idx_start
        z_arr[0,:] = z_start
        for j in range(1,n_steps+1) : 
            z_next,idx_next = self.next_z_traversal(fa,z_arr[j-1,:], idx_arr[j-1] , z_start , t,full_latent,mode)
            idx_arr[j] = idx_next
            z_arr[j,:] = z_next
        
        return z_arr , idx_arr
            


    def index_to_image_pos(self, image_shape, pixel_index):
        reshape_pixel = dat.reshape_array(pixel_index,image_shape,pixel_index)
        index_to_pos = dict()
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if ~np.isnan(reshape_pixel[i,j]):
                    index_to_pos[ reshape_pixel[i,j]] = np.array([i,j])
        return index_to_pos

    def draw_latent_connections_on_image(self, fa , z_min , z_max, full_latent, loader, index = 151) :
        centroids, glomeruls_mask, pixel_index= dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        
        norm = 1
        #norms = [norm]
        
        #for index in range(index_center-3,index_center+3) : 
        #for norm in norms.keys() :
        #fig, axs = plt.subplots(figsize=(10, 10), nrows=1, constrained_layout=True)
        title = 'Spatial_traversal'
        path = 'plots/latent_traversal/spatial_traversal/'
        fig, ax = plt.subplots(figsize=(10, 5), nrows=1, constrained_layout=True)
        
        fig.suptitle(f"m/z {mzs[index]:.4f}", fontsize=20)

        #ax.imshow(dat.reshape_array(centroids[:, index] / norms[norm], image_shape, pixel_index))
        
        ax.imshow(dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index))
        ax.imshow(glomeruls_mask, alpha=0.5)
        
        
        n_steps = 40
        t = (z_max[fa]-z_min[fa])/100 # must be multidimensional when z_dim > 2
        x , _ , idx_start = next(iter(loader))
        index_to_pos = self.index_to_image_pos(image_shape, pixel_index)
        with torch.no_grad():            
            z_mean,  _ = self.model.forward(x)
            z_start = z_mean.detach().numpy()
            for i in np.arange(11,12) :
                #ax.plot(z_start[i,0],z_start[i,1] , ".")
                _ , z_index  = self.compute_trajectory(fa, z_start[i], idx_start[i],t,n_steps,full_latent)
                for j in range(1, n_steps+1) : 
                    z_curr = z_index[j-1]
                    z_next = z_index[j] 
                    x = index_to_pos[z_curr]
                    d_pos = index_to_pos[z_next] - x
                    plt.quiver(x[0], x[1], d_pos[0], d_pos[1], scale_units='xy', angles='xy', scale=1, alpha=0.8)


        plt.savefig(path+'factor_{}_reconstruction_mz_{}.png'.format(fa,mzs[index]))

    def index_to_mask(self):
        loader ,_= dat.full_index_normalized_data()
        id_mask = {}
        for _ , l , id in loader : 
            label = l.detach().numpy()
            index = id.detach().numpy()
            for j in range(len(index)):
                id_mask[index[j]] = label[j]
        return id_mask


    def gif_spatial_latent_traversal(self, fa , z_min , z_max, full_latent,  vars, label,mask_to_name, index=151) :
        centroids, glomeruls_mask, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        
        #p= 0.01 ##proportion of subsamples to consider
        #t = (z_max[fa]-z_min[fa])/100 # must be multidimensional when z_dim > 2
        t = (z_max[0]-z_min[0])/100 # must be multidimensional when z_dim > 2
        #t = 3*t
    
       
        c_dict = self.color_dict()
        #color_train = np.array([c_dict[e] for e in label])

       
        
        #Plot the points and ellipses
        _, ax = plt.subplots(figsize=(10, 10), nrows=2, constrained_layout=True)
        
        self.scatter_with_covar(ax[0],full_latent[:,1:],vars,label,c_dict,mask_to_name)
        ax[1].imshow(dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index))
        ax[1].imshow(glomeruls_mask, alpha=0.5)

        
        n_steps = 40
        index_to_pos = self.index_to_image_pos(image_shape, pixel_index)
        id_mask = self.index_to_mask()

        n_points = 5
        sub_index = np.random.choice(full_latent.shape[0], n_points , replace=False)
        z_start = full_latent[sub_index, 1:]
        idx_start =  full_latent[sub_index, 0]    

        for i in range(n_points) :
            ax[0].plot(z_start[i,0],z_start[i,1] , ".")

            z_arr , z_index = self.compute_trajectory(fa, z_start[i], idx_start[i],t,n_steps,full_latent)
            c_glo = 0
            for j in range(1, n_steps+1) : 
                z_curr = z_arr[j-1,:]
                z_diff = z_arr[j,:] - z_curr
                
                id_curr = z_index[j-1]
                id_next = z_index[j] 
                if id_curr != id_next :
                    ax[0].quiver(z_curr[0], z_curr[1], z_diff[0], z_diff[1], scale_units='xy', angles='xy', scale=1, alpha=0.8)
                    ######################
                    type = id_mask[id_next]
                    c_glo += int(type)
                    ######################
                    x = index_to_pos[id_curr]
                    d_pos = index_to_pos[id_next] - x
                    ax[1].quiver(x[0], x[1], d_pos[0], d_pos[1], scale_units='xy', angles='xy', scale=1, alpha=0.8)
                    plt.title(f'Latent traversal factor {fa} at step {j}. # Hits = {c_glo}',fontsize=14)
                    #plt.savefig("plots/latent_traversal/spatial_image/latent_factor{}_step_t{}.pdf".format(fa,j))
                    plt.savefig("plots/latent_traversal/spatial_image/latent_factor{}_sample{}_step_t{}.png".format(fa,i,j))


            #plt.quiver(z_curr[0], z_curr[1], z_diff[0], z_diff[1], scale_units='xy', angles='xy', scale=1, alpha=0.8)
        plt.savefig('plots/latent_traversal/latent_dim_traversal_factor{}.png'.format(fa))

    
    def plot_recons_dis_trade_of(self):
        _, axs = plt.subplots(3, 1,figsize=(10, 10))
        betas = [1,2,3,4,8]
        colors = ['#1d6bb3','#e8710a','#1b9c10', '#e52592','#9334e6' , '#f9ab00','#12b5cb','#df159a']
        if len(betas)> len(colors):
            print('Not enough colors. Deleting last betas')
            betas = betas[:len(colors)]

        colors = colors[:len(betas)]
        col_beta = dict(zip(betas,colors))
        n=4 #Number of versions
        
        for b in betas :
            train_loss,train_recons,train_kl, test_loss ,dis_metric = compute_loss(b,n)
            
            x_range = range(train_loss.shape[1])
            #print(x_range)
            #print(train_loss[1,:])
            for i in range(n): 
                axs[0].plot(x_range, train_loss[i,:], color = col_beta[b],alpha=0.5)
                axs[1].plot(x_range, train_recons[i,:], color = col_beta[b],alpha=0.5)
            axs[0].plot(x_range, np.mean(train_loss,axis=0),color = col_beta[b],label=b)
            axs[0].title.set_text('Training loss')
            
            axs[1].plot(x_range,np.mean(train_recons,axis=0),col_beta[b],label=b)
            axs[1].title.set_text('Reconstruction error')

            mean_dis = np.mean(dis_metric,axis=0)
            sd_dis = np.std(dis_metric,axis=0)
            axs[2].plot(x_range, mean_dis, color = col_beta[b],label = b)
            axs[2].fill_between(x_range,mean_dis-sd_dis ,mean_dis+sd_dis,alpha=0.3)
            axs[2].title.set_text('Disentangling metric')
            axs[2].set_xlabel('Epochs')

            for j in range(3):
                axs[j].legend().get_frame().set_linewidth(0.0)
                axs[j].legend(title = r"$\beta$",bbox_to_anchor=(1, 1),loc="upper left")
                
            #plt.legend(loc=(1.04, 0))

        plt.savefig("plots/tradeoff/recons_disenangle_tradeoff.pdf", bbox_inches='tight')

    def plot_spatial(self, indices ) : 
        ############# For kidney data
        ## Draw image mask on one mzs in the bottom
        centroids, _, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        #index_to_pos = self.index_to_image_pos(image_shape, pixel_index)

        for i in indices:
            plt.figure(figsize=(10,10))
            plt.imshow(dat.reshape_array(centroids[:, i] / norm, image_shape, pixel_index))
            plt.title(f"m/z {mzs[i]:.4f}",fontsize=20)
            plt.savefig("plots/spatial/mz_{}".format(i), bbox_inches='tight')

    def plot_carving_separate(self,full_latent,vars,label,mask_to_name,index = 151):
        # Specify the font to use
        # rc('font', **{'family': 'serif', 'serif': ['cmr10']})
        # rc('text', usetex=True)

        col_dict = self.color_dict()
        
        ############# For kidney data
        # print('Kidney carving')
        # Draw image mask on one mzs in the bottom
        centroids, _, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        index_to_pos = self.index_to_image_pos(image_shape, pixel_index)
        norm = 1

        #### Polygone kidney6
        ## For bottom right image
        mask = np.zeros(image_shape)
        #pol_limit1 = [(0.4,-3.4),(0.7,-3.2),(0.7,2.2), (0.3,2.2),] ### Orange
        pol_limit1 = [(0.5,-3.4),(0.7,-3.2),(0.7,2.2), (0.2,2.2),] ### Orange
        pol_limit2 = [(-1.9,-2.1),(-1.3,-3),(-0.4,1.2), (-1.1,1.4),] ### Blue
        pol_limit3 = [(-0.3,-3.4),(0.2,-3.2),(0.2,1), (-0.3,1),] ### Green
        pol_limit4= [(0.9,-3),(1.5,-2.9),(1.1,1.2),(0.8,1.2),] ### Black
        pol_limit5= [(2.6,-3),(3.9,-2.8),(3,1.2),(1.4,0.9),(1.3,-0.1),] ### Red
        pol_limit6 = [(0.2,-13),(0.6,-13),(0.6,-3.5), (0,-3.5),] ### Orange extra leg  	#f44e97 #pink
        colors = ['darkorange',"CornflowerBlue", "#6DCC8C", 'black','#a30000','#f44e97','#6a329f']
        colors = ['darkorange',"hotpink", "#6DCC8C", 'black','#a30000','#f44e97','#6a329f']
        colors = ['darkorange','black', "#6DCC8C", "hotpink",'#a30000','#f44e97','#6a329f']
        colors = ['darkorange','black', '#a30000', "hotpink","#6DCC8C",'#f44e97','#6a329f']
        colors = ['darkorange','black', '#a30000', "#573280","#6DCC8C",'#f44e97','#6a329f']
        colors = ['#F96E46','black', '#a30000', "#F9C846","#6DCC8C",'#f44e97','#6a329f'] #Burnt sienna(orange), black , ... , Saffron(yellow)

        #5E548E --> ULtraviolet
        #F0F2A6 --> Vanilla
        #573280 --> Tekhelet (purple)
        #hotpink
        
        # #########
        # ## VAN1_35 OLD
        # #########
        # masks = np.zeros(image_shape)
        # ####### V2
        # pol_limit1 = [(-0.3,-3.5),(0.2,-3.5),(0.3,1.3),(-0.1,1.3)] #Orange
        # # pol_limit2 = [(-2.2,-3),(-1.5,-3.5),(-0.8,-2.1),(-0.2,0),(-0.19,1.55),(-0.21,1.55)] #Black 
        # # pol_limit3 =[(-3.4,-1),(-2.4,-1.2),(-1.5,-0.5),(-0.8,0.7), (-0.3,1.55), (-0.7,1.55),(-2,1.2)] #Red
        # pol_limit2 = [(-2.2,-3),(-1.5,-3.5),(-0.8,-2.1),(-0.2,0),(-0.19,1),(-0.3,1)] #Black 
        # pol_limit3 =[(-3.8,-1.5),(-2.8,-2),(-0.8,0.7), (-0.8,1),(-2,0.9)] #Red
        # pol_limit4 =[(0.6,-3.1),(1.4,-3),(0.7,0.7),(0.4,0.7)]
        # pol_limit5=[(2.6,-3.5),(3,-3.4),(1.4,0.8),(1,0.2),(1.3,-0.9)]

        # pol_limits = [pol_limit1,pol_limit2,pol_limit3,pol_limit4,pol_limit5]
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']


        #########
        ## VAN1_41
        #########
        masks = np.zeros(image_shape)
        ####### V4
        pol_limit1 = [(-0.4,-2.3),(0.1,-2.3),(-0.1,1.7),(-0.5,1.5)] #Orange
        pol_limit2 = [(0.3,-3),(1.1,-3),(0.6,1),(0.4,1.3),(0.1,0.3),(0.1,-1)] #Black
        pol_limit3 = [(1.7,-3),(2.8,-2),(1,1.3),(0.7,1)] #Red
        pol_limit4 =[(-2.4,-2.4),(-1.8,-2.4),(-1.3,-1),(-1.2,0.6),(-2,0.8)] # Pink
        pol_limit5 = [(-1.1,-2.2),(-0.5,-2.3),(-0.6,1),(-1.2,0.9)] #Green
        colors = ['#F96E46','black', '#a30000', "#F9C846","#6DCC8C",'#f44e97','#6a329f'] #Burnt sienna(orange), black , ... , Saffron(yellow)
        names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        pol_limits = [pol_limit1,pol_limit2,pol_limit3,pol_limit4,pol_limit5]

        #########
        ## VAN1_31
        #########
        masks = np.zeros(image_shape)
        ####### V1
        # pol_limit1 =[(-0.2,-2),(-0.1,-1.9),(0.5,1.1),(-0.2,1.1)] #Orange
        # pol_limit2 = [(-0.1,-3.8),(0.5,-3.8),(0.7,-1.4),(0.7,-0.2),(0.4,-0.1)] #black
        # pol_limit3 = [(1,-3.7),(1.5,-3.7),(1.6,0.5),(1.3,0.3),(0.9,-0.5)] #red
        # pol_limit4 = [(0.8,0.8),(4,0.8),(4,2.1),(0.4,2.1)] #pink
        # pol_limit5= [(-1,-2.5),(-0.5,-2.5),(-0.7,0.9),(-1.3,1)] #green
        # #pol_limit6 = [(-1,1.2),(0,1.3),(0,2.2),(-1,2.2)] # Head of pulpo
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        

        #########
        ## VAN1_35 NEW
        #########
        # masks = np.zeros(image_shape)
        # ####### V6
        # pol_limit1 = [(-0.2,-3.8),(0.2,-3.8),(0.2,2),(-0.15,2)] #Orange
        # pol_limit2 = [(-1.6,-3),(-0.8,-3.5),(-0.4,-0.8),(-0.3,1.3),(-0.5,1.3)] #Black 
        # pol_limit3 =[(-3,-2),(-2.2,-2.2),(-1.2,-0.1),(-0.75,1.3),(-1.3,1.3)] #Red
        # pol_limit4 =[(0.55,-3.1),(1.2,-3),(0.7,1),(0.4,1)] #Yellow
        # pol_limit5=[(1.7,-2.9),(2.2,-2.7),(1.3,0.9),(0.9,0.9),(1.2,-0.9)]

        # pol_limits = [pol_limit1,pol_limit2,pol_limit3,pol_limit4,pol_limit5]
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']

        #### IMS_image
        _, ax = plt.subplots(figsize=(10, 10))
        image = dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index)
        #Optional flip to match the microscopy data
        #ax.imshow(np.flip(image , axis = (0,1)))
        
        #VAN1_41
        ax.imshow(image)

        #VAN 1_31
        #ax.imshow(image.T)
        ## Van1_35
        # image_rot = image.T
        # image_rot = image_rot[::-1,:]
        # ax.imshow(image_rot)

        ####
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.set_xlabel('Pixels',fontsize="25")
        # ax.set_ylabel('Pixels',fontsize="25")

        ##### KIDNEY
        #plt.savefig(f'plots/carving/kidney/carving_latent_ims_mz{mzs[index]:.4f}.png',bbox_inches='tight',dpi=300)
        
        ##### ALL DATA
        # 1_41
        plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_41/carving/carving_latent_ims_mz{mzs[index]:.4f}.png',bbox_inches='tight',dpi=300)
        # 1_31
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_31/carving/carving_latent_ims_mz{mzs[index]:.4f}.png',bbox_inches='tight',dpi=300)
        # 1_35
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_35/carving/carving_latent_ims_mz{mzs[index]:.4f}.png',bbox_inches='tight',dpi=300)
        #ax[1,0].set_title(f"m/z {mzs[index]:.4f}")
        print('### IMS ')
        
        #### Latent Space
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)

        ax.set_xlim(-4.5,4.5)
        ax.set_ylim(-4.5,4.5)
        col_dict={0:'blue'} ## To have black and white
        # ax.set_xlabel(r'$z_1$',fontsize="40")
        # ax.set_ylabel(r'$z_2$',fontsize="40", rotation=0)
        self.scatter_with_covar(ax,full_latent[:,1:],vars,label, col_dict, mask_to_name)  

        # add the polygons
        n_poly = 5
        if len(pol_limits)!=0 :
            for i,pol_limit in enumerate(pol_limits):
                if i<n_poly:
                    ## Polygones on top images
                    poly1 = Polygon(pol_limit,alpha = 0.5, ec = "gray", fc = colors[i],visible = True)
                    p = m_path.Path(pol_limit) 
                    ax.add_patch(poly1)
                    #ax[0,1].add_patch(poly2)
                    flag = p.contains_points(full_latent[:,1:])
                    index_mask = full_latent[:,0][flag]
                    ###### Selection for traversal
                    print('##-->## Color = ', colors[i])
                    min_d = np.argmin(full_latent[flag,2])
                    max_d = np.argmax(full_latent[flag,2])
                    print('flag highest = ', full_latent[flag,0][min_d])
                    print('flag lowest = ', full_latent[flag,0][max_d])
                    pos = np.zeros((len(index_mask),2))
                    for j,ind in enumerate(index_mask) :
                        pos[j,:] = index_to_pos[ind]
                ## Plot scatter image bottom right
                if i<n_poly:
                    print('colors {} = {}'.format(colors[i],i+1))
                    for j,ind in enumerate(index_mask) :
                        mask[int(pos[j,0]),int(pos[j,1])] = i+1
        
        ### Save all the mask as binary masks 
        PATH_to_save = 'saved_data/masks/van1_41/'
        names_files = dict(zip(range(1,len(names)+1),names))    
        for l in np.unique(mask) :
            if l !=0 : 
                mask_temp = mask == l
                with open(PATH_to_save+'mask_{}.npy'.format(names_files[l]), 'wb') as f:
                    np.save(f, mask_temp)

        #### KIDNEY
        #plt.savefig('plots/carving/kidney/carving_latent_2d.png',bbox_inches='tight',dpi=300) 
        
        #### ALL DATA
        # 1_41
        plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_41/carving/carving_latent_2d.png',bbox_inches='tight',dpi=300)
        # 1_31
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_31/carving/carving_latent_2d.png',bbox_inches='tight',dpi=300)
        # 1_35
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_35/carving/carving_latent_2d.png',bbox_inches='tight',dpi=300)
             

        #### Masked data
        _, ax = plt.subplots(figsize=(10, 10))
        ## For Mouse pup add .T    ax[1,1].imshow(mask.T, interpolation='nearest')
        cmap_col = colors_mat.ListedColormap(['white'] + colors[:n_poly])
        ############################################################################################ Add .T
        ############################
        ## Optional flip :
        #ax.imshow(np.flip(mask,axis = (0,1)), cmap = cmap_col, interpolation='nearest')
        ###VAN_rest
        #ax.imshow(mask.T, cmap = cmap_col, interpolation='nearest')
        ## VAN1_35 optional rot of maks 
        # mask_rot = mask.T
        # mask_rot = mask_rot[::-1,:]
        # ax.imshow(mask_rot, cmap = cmap_col, interpolation='nearest')
        ### VAN1_41 
        ax.imshow(mask, cmap = cmap_col, interpolation='nearest')
        ##################################
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.set_xlabel('Pixels',fontsize="25")
        # ax.set_ylabel('Pixels',fontsize="25")
        
        #### KIDNEY
        #plt.savefig('plots/carving/kidney/carving_latent_mask.png',bbox_inches='tight',dpi=300)      
        
        #### ALL DATA
        # 1_41
        plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_41/carving/carving_latent_mask.png',bbox_inches='tight',dpi=300)
        # 1_31
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_31/carving/carving_latent_mask.png',bbox_inches='tight',dpi=300)
        # 1_35
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_35/carving/carving_latent_mask.png',bbox_inches='tight',dpi=300)
        
        ###### NEW KDE
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        image = kernel_adapated(full_latent,vars)
        plot_kernel_adapted(ax,fig,image)
        n_poly=5
        if len(pol_limits)!=0 :
            for i,pol_limit in enumerate(pol_limits):
                if i<n_poly:
                    poly1 = Polygon(pol_limit,alpha = 0.7, ec =  "#FF69B4", fc ='none',lw = 8,visible = True) ## Deepink
                    ax.add_patch(poly1)
                    print('Poly {} added'.format(i))
        #plt.savefig('plots/carving/kidney/carving_latent_kde_adapted.png',bbox_inches='tight',dpi=300)
        #### ALL DATA
        # 1_41
        plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_41/carving/carving_latent_kde_adapapted.png',bbox_inches='tight',dpi=300)
        # 1_31
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_31/carving/carving_latent_kde_adapapted.png',bbox_inches='tight',dpi=300)
        # 1_35
        #plt.savefig(f'plots/2d_reduction/kidney/all_data/kidney1_35/carving/carving_latent_kde_adapapted.png',bbox_inches='tight',dpi=300)

        #### KDE data
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.set_xlabel(r'$z_1$',fontsize="40")
        # ax.set_ylabel(r'$z_2$',fontsize="40", rotation=0)
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]   
        grid = np.vstack([X.ravel(), Y.ravel()])
        kernel = stats.gaussian_kde(full_latent[:,1:].T,bw_method='scott')
        #f = 3 #f = 1/3
        self.kernel_density(ax,fig,kernel,kernel.factor,grid,X.shape)
        # add the polygons
        n_poly = 5
        if len(pol_limits)!=0 :
            for i,pol_limit in enumerate(pol_limits):
                if i<n_poly:
                    ## Polygones on top images
                    #poly1 = Polygon(pol_limit,alpha = 0.5, ec = "gray", fc = colors[i],visible = True)
                    poly1 = Polygon(pol_limit,alpha = 0.5, ec =  colors[i], fc ='none',lw = 4,visible = True)
                    #'darkorange','black', '#a30000', "hotpink","#6DCC8C",'#f44e97','#6a329f'
                    ### #FF69B4 -> Deep pink
                    poly1 = Polygon(pol_limit,alpha = 0.7, ec =  "#FF69B4", fc ='none',lw = 8,visible = True)
                    p = m_path.Path(pol_limit) 
                    ax.add_patch(poly1)
                
        #Scott bandwith estimator : {}'.format(factor)

        ##### KIDNEY
        #plt.savefig('plots/carving/kidney/carving_latent_kde_scott_{}.png'.format(kernel.factor),bbox_inches='tight',dpi=300)
        

        #### ALL DATA
        #1_41
        plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_41/carving/carving_latent_kde_scott_{}.png'.format(kernel.factor),bbox_inches='tight',dpi=300)
        # 1_31
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_31/carving/carving_latent_kde_scott_{}.png'.format(kernel.factor),bbox_inches='tight',dpi=300)
        # 1_35
        #plt.savefig('plots/2d_reduction/kidney/all_data/kidney1_35/carving/carving_latent_kde_scott_{}.png'.format(kernel.factor),bbox_inches='tight',dpi=300)
        
       




    def plot_carving(self,full_latent,vars,label,mask_to_name,index = 151):
        #print('### full_latent[0] = ', full_latent[:,0])
        _, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2, constrained_layout=True)

        col_dict = self.color_dict()
        
        ############# For kidney data
        # print('Kidney carving')
        # Draw image mask on one mzs in the bottom
        centroids, _, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        index_to_pos = self.index_to_image_pos(image_shape, pixel_index)

        ############# For mouse_pup
        # print('Mouse_pup carving')
        # centroids ,pixel_index,_= load_IMS_mouse_pup()
        # image_shape = get_image_shape_mouse_pup()
        # #im_array = dat.reshape_array(pixel_index,image_shape,acquisition_mask)
        # index_to_pos = index_to_image_pos_mouse()

        ####################index_to_pos = self.index_to_image_pos(image_shape,pixel_index)
        norm = 1

        ## Bottom left
        #For Mouse pup add .T   ax[1,0].imshow(dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index).T)
        ############################################################################################ Add .T
        ax[1,0].imshow(dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index).T)
        #ax[1,0].set_title(f"m/z {mzs[index]:.4f}")
        
        ax[0,0].set_xlim(-4,4)
        ax[0,0].set_ylim(-4,4)
        self.scatter_with_covar(ax[0,0],full_latent[:,1:],vars,label, col_dict, mask_to_name)        

        ###### Polygon kidney2
        # pol_limit1 = [(-3.5,-2.1),(-0.8,-2.1),(0.5,-2.5), (-0.5,-0.9), (-3,-0.8),]
        # pol_limit2 = [(-2.7,2.2),(0.2,-1.5),(0.5,-1.3),(-1.6,3),]
        # pol_limit3 = [(0.1,-0.4),(0.4,-0.4),(0.3,3.8),(-0.8,3.8),(-0.7,2.6),]
        # pol_limit4 = [(0.6,-1),(1,-1.1),(1.6,2.8),(1,2.9),]
        # pol_limit5 = [(1.2,-0.8),(2,-1.8),(3,1),(2,2.2),(1.7,1.6),]


        #### Polygone kidney6
        ## For bottom right image
        mask = np.zeros(image_shape)
        #pol_limit1 = [(0.4,-3.4),(0.7,-3.2),(0.7,2.2), (0.3,2.2),] ### Orange
        pol_limit1 = [(0.5,-3.4),(0.7,-3.2),(0.7,2.2), (0.2,2.2),] ### Orange
        pol_limit2 = [(-1.9,-2.1),(-1.3,-3),(-0.4,1.2), (-1.1,1.4),] ### Blue
        pol_limit3 = [(-0.3,-3.4),(0.2,-3.2),(0.2,1), (-0.3,1),] ### Green
        #pol_limit4= [(0.9,-3),(1.3,-2.9),(1,-0.4),(0.8,0),] ### Black
        #pol_limit5= [(1.5,-3),(1.8,-2.9),(1.4,-0.9),(1.1,-0.5),] ### Red
        pol_limit4= [(0.9,-3),(1.5,-2.9),(1.1,1.2),(0.8,1.2),] ### Black
        pol_limit5= [(2.6,-3),(3.9,-2.8),(3,1.2),(1.4,0.9),(1.3,-0.1),] ### Red
        pol_limit6 = [(0.2,-13),(0.6,-13),(0.6,-3.5), (0,-3.5),] ### Orange extra leg  	#f44e97 #pink
        colors = ['darkorange',"CornflowerBlue", "#6DCC8C", 'black','#a30000','#f44e97','#6a329f']
        
        #########
        ## VAN2_5
        #########
        masks = np.zeros(image_shape)
        ####### V1
        # pol_limit1 = [(-1.8,-1.6),(-0.5,-1.6),(-0.3,-0.3),(-1.4,-0.1),] # Orange
        # pol_limit2 = [(0.3,-0.1),(3,-0.1),(3,0.5),(0.5,0.7)] # Blue
        # pol_limit3 = [(0.2,-2),(1.8,-1.5),(1,-0.5),(0,-0.1),(-0.1,-1.1),] #Green
        # pol_limit4 = [(-1.2,-3.5),(-0.5,-3.5),(-0.3,-1.9),(-1.1,-1.9)] #Black
        # pol_limit5 =[(-0.9,0.4),(0.1,0.6),(0,1),(-1,2.2),(-1.5,1)] #Red
        #pol_limit6 = [(-1.9,1),(-3.5,1.2),(-3.7,0.2),(-2,-0.2),] #Extra leg on the left

        ### ###### V2
        ### pol_limit1=[(-1.5,-3.8),(-0.5,-4),(0,-1),(-0.8,-1.2)] #Orange
        ### pol_limit2 = [(-3,-3.2),(-2,-3.8),(-1.8,-2.5),(-1.7,-1.7)] #Blue
        ### pol_limit3 =[(0,-0.2),(1.5,0.5),(1,1.2),(0,0.5)] #Green

        # ###### V3
        # pol_limit1 =[(-1,0.6),(0,0.5),(0,1.5),(-1.3,1.2)]
        # pol_limit2 = [(-3.5,0.2),(-1.5,-0.1),(-1.4,0.8),(-3,1.5)]
        # pol_limit3 = [(-1.1,-2),(-0.3,-0.5),(-0.5,0.1),(-1.5,-0.5)]

        #########
        ## VAN1_31
        #########
        masks = np.zeros(image_shape)
        ####### V1
        # pol_limit1 =[(-0.5,-0.5),(0.2,-0.6),(0.5,1.1),(-0.2,1.1)]
        # pol_limit2 = [(-0.1,-2.8),(0.5,-2.8),(0.7,-1.4),(0.7,-0.2),(0.4,-0.1)] #Blue
        # pol_limit3 = [(1,-2.1),(1.5,-2.2),(1.6,0.5),(1.3,0.3),(0.9,-0.5)] #green
        # pol_limit4 = [(0.8,0.8),(4,0.8),(4,2.1),(0.4,2.1)] #Black
        # pol_limit5= [(-1,-2.5),(-0.5,-2.5),(-0.7,0.9),(-1.3,1)] #Red
        # #pol_limit6 = [(-1,1.2),(0,1.3),(0,2.2),(-1,2.2)] # Head of pulpo

        #########
        ## VAN1_35
        #########
        masks = np.zeros(image_shape)
        ####### V2
        # pol_limit1 = [(-0.3,-3.5),(0.2,-3.5),(0.3,1.3),(-0.1,1.3)] #Orange
        # pol_limit3 = [(-3,-3),(-2.1,-3.5),(-0.8,-2.1),(-0.2,0),(-0.3,0.6)] #Blue
        # pol_limit2 =[(-3.5,0.3),(-3,-1),(-0.4,0.5), (-0.2,2), (-0.7,2)] #Green
        # pol_limit4 =[(0.6,-2),(1.4,-1.8),(0.7,0.7),(0.4,0.7)]
        # pol_limit5=[(2.6,-3.5),(3,-3.4),(1.4,0.8),(1,0.2),(1.3,-0.9)]



        #########
        ## VAN1_41
        #########
        masks = np.zeros(image_shape)
        ####### V4
        # pol_limit1 = [(-0.4,-2.3),(0.1,-2.3),(-0.1,1.7),(-0.5,1.5)] #Orange
        # pol_limit2 = [(0.3,-1.8),(1.1,-1.8),(0.6,1),(0.4,1.3),(0.1,0.3),(0.1,-1)] #Blue 
        # pol_limit3 = [(1.4,-1.8),(2.1,-0.8),(1.1,1.2),(0.8,0.8)] #Green
        # pol_limit4 =[(-2,-1),(-1.3,-1),(-1.2,0.6),(-2,0.8)]
        # pol_limit5 = [(-1.2,-1.2),(-0.5,-1.3),(-0.6,1),(-1.2,0.9)]
        
        


        ### Polygone Synthetic1

        ### Polygone Mouse_pup1
        # mask = np.zeros(image_shape)
        # pol_limit1 = [(-1.4,-3.2),(0.3,-3.2),(-0.5,-0.7),(-1.5,-0.8)] ### Orange
        # pol_limit2 = [(0.5,0.8),(2.7,1.9),(2.4,2.7), (0.2,1.4),] ### Red
        # pol_limit3 = [(1.3,-2.4),(2.1,-2.3),(0.5,-0.5), (0.2,-1),] ### Blue
        # pol_limit4 = [(-2,0.6),(-1,0.5),(-2.2,2), (-2.5,1.5),] ### green
        # pol_limit5 = [(0.6,0.2),(3.4,0.7),(3.2,1.7), (0.5,0.7),] ### black




        ## Plot of polygons on top images
        pol_limits = [pol_limit1,pol_limit2,pol_limit3,pol_limit4,pol_limit5,pol_limit6]
        # To draw the base image add following line
        #pol_limits=[]
        ## Colors = [Orange , blue , green,black , red,purple]
        #colors = ['darkorange',"CornflowerBlue", "#6DCC8C", 'black','#a30000','#6a329f']
        #colors = ['darkorange','#a30000',"CornflowerBlue", "#6DCC8C", 'black','#6a329f']
        
        #cmap_col = dict(zip(range(1,len(colors)+1),colors))
        
        #cmap = matplotlib.cm.get_cmap('Spectral')
        #rgba = cmap(0.5)
        
        #'firebrick', 'black', 'darkmagenta']
        
        n_poly = 5
        if len(pol_limits)!=0 :
            for i,pol_limit in enumerate(pol_limits):
                if i<n_poly:
                    ## Polygones on top images
                    poly1 = Polygon(pol_limit,alpha = 0.5, ec = "gray", fc = colors[i],visible = True)
                    poly2 = Polygon(pol_limit,alpha = 0.4, ec = "gray", fc = colors[i],visible = True)
                    p = m_path.Path(pol_limit) 
                    ax[0,0].add_patch(poly1)
                    ax[0,1].add_patch(poly2)
                    flag = p.contains_points(full_latent[:,1:])
                    index_mask = full_latent[:,0][flag]
                    ###### Selection for traversal
                    print('##-->## Color = ', colors[i])
                    min_d = np.argmin(full_latent[flag,2])
                    max_d = np.argmax(full_latent[flag,2])
                    print('flag highest = ', full_latent[flag,0][min_d])
                    print('flag lowest = ', full_latent[flag,0][max_d])
                    ##########################
                    pos = np.zeros((len(index_mask),2))
                    for j,ind in enumerate(index_mask) :
                        #print('#### ind = ', ind)
                        pos[j,:] = index_to_pos[ind]
                ## Plot scatter image bottom right
                if i<n_poly:
                    print('colors {} = {}'.format(colors[i],i+1))
                    for j,ind in enumerate(index_mask) :
                        #mask[int(pos[j,0]),int(pos[j,1])] = i+1 ### before was 1
                        mask[int(pos[j,0]),int(pos[j,1])] = i+1

                    #ax[1,0].scatter(pos[:,1],pos[:,0],alpha=1, c=colors[i],s=0.5)  ## For kidney
                    #ax[1,0].scatter(pos[:,0],pos[:,1],alpha=1, c=colors[i],s=0.1)   ## For Mouse pup 
                
                #ax[1,0].scatter(pos[:,1],pos[:,0],alpha=1, c=colors[i],s=0.5)
        
        #print('## np.unique = ', np.unique(mask))

        ## Plot scatter image bootom right
        ## For Mouse pup add .T    ax[1,1].imshow(mask.T, interpolation='nearest')
        #ax[1,1].imshow(mask, interpolation='nearest')
        cmap_col = colors_mat.ListedColormap(['white'] + colors[:n_poly])
        ############################################################################################ Add .T
        ax[1,1].imshow(mask.T, cmap = cmap_col, interpolation='nearest') ## Add the desired colormap

        ## Create color map for mask
        cmap_left = plt.cm.colors.ListedColormap(['none', colors[5]])
        cmap_left.set_under(color='none')
        #ax[1,0].imshow(mask==6,cmap = cmap_left, interpolation='nearest', alpha=0.5) ## Add the desired colormap
        mask_type = ['Glomerulus','Proximal_tubule','p2' ,'Collecting_duct', 'Distal_tubule' ,'Thick_ascendin_limb']
        mask_type=['','']
        ax[1,1].set_title('Discovered {} mask through carving'.format(mask_type[0]))

        ### Plot kernel top right
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]   
        grid = np.vstack([X.ravel(), Y.ravel()])
        kernel = stats.gaussian_kde(full_latent[:,1:].T,bw_method='scott')
        #f = 3 #f = 1/3
        self.kernel_density(ax[0,1],kernel,kernel.factor,grid,X.shape)

        #plt.savefig('plots/carving/van1_35/carving_latent2.png',bbox_inches='tight')
        #plt.savefig('plots/carving/van1_41/carving_latent4_test4.png',bbox_inches='tight')
        #plt.savefig('plots/carving/van2_5/carving_latent2_5_v1.png',bbox_inches='tight')
        plt.savefig('plots/carving/kidney/carving_latent_v1.png',bbox_inches='tight')


        ### Save all the mask as binary masks 
        PATH_to_save = 'saved_data/masks/van2_5/'
        names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        names_files = dict(zip(range(1,len(names)+1),names))    
        for l in np.unique(mask) :
            if l !=0 : 
                mask_temp = mask == l
                with open(PATH_to_save+'mask_{}.npy'.format(names_files[l]), 'wb') as f:
                    np.save(f, mask_temp)
        
        
        ## To Read
        # PATH_to_save = 'saved_data/masks/'
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        # names_files = dict(zip(range(1,len(names)+1),names))
        # masks = []
        # for l in range(1,len(names)+1):
        #     with open(PATH_to_save+'mask_{}.npy'.format(names_files[l]),'rb') as f:
        #         masks.append(np.load(f)) 
            
        #plt.savefig('plots/carving/carving_latent.pdf',bbox_inches='tight')

    
    
    def plot_kernel_density_estimation(self,full_latent,vars,label,mask_to_name,title=''):
        _ , ax = plt.subplots(nrows=2,ncols=2,figsize=(15, 15))

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        #Create grid to evaluate the kernel on
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]
        grid = np.vstack([X.ravel(), Y.ravel()])
        
        #Define Kernel Scott
        kernel = stats.gaussian_kde(full_latent[:,1:].T,bw_method='scott')
        kernel.set_bandwidth(bw_method='scott')
        factor = kernel.factor
        factors = np.zeros((2,2))
        factors[0,1] = factor
        #factors[1,0] = factor/3
        factors[1,0] = factor/1.5
        #factors[1,1] = factor/10
        factors[1,1] = factor/3

        for i in range(2):
            for j in range(2):
                if i==0 and j == 0 :
                    prop=0.05
                    prop=0.2
                    sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0]*prop), replace=False)
                    self.scatter_with_covar(ax[0,0],full_latent[sub_index,1:],vars[sub_index],label[sub_index], self.color_dict(), mask_to_name)
                    ax[0,0].set_xlim(-4,4)
                    ax[0,0].set_ylim(-4,4)
                else :
                    self.kernel_density(ax[i,j],kernel,factors[i,j],grid,X.shape)
                    # kernel.set_bandwidth(bw_method=factors[i,j])
                    # Z = np.reshape(kernel(grid).T, X.shape)
                    # im = ax[i,j].imshow(np.rot90(Z), cmap=plt.cm.jet,extent=[-4, 4, -4, 4])
                    # ax[i,j].set_title('Scott bandwith estimator : {}'.format(factors[i,j]) )
                    # divider = make_axes_locatable(ax[i,j])
                    # cax = divider.append_axes("bottom", size="5%", pad=0.4)
                    # plt.colorbar(im, cax=cax, orientation="horizontal", format="%.2f")
        plt.savefig("plots/2d_reduction/kernel_density/density_n{}.png".format(title), bbox_inches='tight')

          
    def kernel_density(self,ax,fig, kernel, factor,grid,shape):
        kernel.set_bandwidth(bw_method=factor)
        Z = np.reshape(kernel(grid).T, shape)
        im = ax.imshow(np.rot90(Z), cmap=plt.cm.jet,extent=[-4, 4, -4, 4])
        #ax.set_title('Scott bandwith estimator : {}'.format(factor) )
        ######
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("bottom", size="5%", pad=0.4)
        ######
        #plt.colorbar(im, cax=cax, orientation="horizontal", ticks=MultipleLocator(0.02), format="%.2f")
        #plt.colorbar(im, cax=cax, orientation="horizontal", format="%.2f",location = 'top')

        # add color bar above picture
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size = '5%', pad = 0.5)
        fig.add_axes(cax)
        colorbar = fig.colorbar(im, cax = cax, orientation = 'horizontal')
        # Set the colorbar ticks and label size
        colorbar.ax.tick_params(labelsize=20)


        # Z = np.reshape(kernel(grid).T, X.shape)
        # im1 = ax[1,0].imshow(np.rot90(Z), cmap=plt.cm.jet,extent=[-4, 4, -4, 4])
        
        # ax[1,0].set_title('Scott bandwidth estimator : {}'.format(kernel.factor))
        # divider1 = make_axes_locatable(ax[1,0])
        # cax1 = divider1.append_axes("top", size="10%", pad=0.4)
        # plt.colorbar(im1, cax=cax1, orientation="horizontal", ticks=MultipleLocator(0.02), format="%.2f")


        # kernel.set_bandwidth(bw_method=kernel.factor/3)
        # Z = np.reshape(kernel(grid).T, X.shape)
        # ax[0,1].imshow(np.rot90(Z), cmap=plt.cm.jet,extent=[-4, 4, -4, 4])
        # ax[0,1].set_title('soctt/3 bandwidth estimator : {}'.format(kernel.factor))
        
        # kernel.set_bandwidth(bw_method=kernel.factor/10)
        # Z = np.reshape(kernel(grid).T, X.shape)
        # ax[1,1].imshow(np.rot90(Z), cmap=plt.cm.jet,extent=[-4, 4, -4, 4])
        # ax[1,1].set_title('soctt/10 bandwidth estimator : {}'.format(kernel.factor))
        

        # prop=0.05
        # sub_index = np.random.choice(full_latent.shape[0], int(full_latent.shape[0]*prop), replace=False)
        # full_latent = full_latent[sub_index]
        # label = label[sub_index]
        # vars = vars[sub_index]
        # self.scatter_with_covar(ax[0,0],full_latent[:,1:],vars,label, self.color_dict(), mask_to_name)

        

    def model_evolution(self,loader,mask_to_name):
        #Boolean deciding wether plots for kernel densities are made
        evaluate_density = True
        frames = []
        n = 36
        
        #Create grid to evaluate the kernel on
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]
        grid = np.vstack([X.ravel(), Y.ravel()])

        # for epoch in range(-1,n):
        #     model_path = 'model/model_kidney6/model_weight_n_{}.pth'.format(epoch)
        #     model = torch.load(model_path, map_location=torch.device('cpu'))
        #     full_latent , vars , label = compute_latent(loader,model)
        #     #remove label 0
        #     mask0 = label!=0
        #     full_latent = full_latent[mask0,:]
        #     vars = vars[mask0,:]
        #     label = label[mask0]

        #     ## 2d latent
        #     #self.plot_latent_dim_with_var(full_latent,vars,label,mask_to_name,epoch,p=0.08)
            
        #     ## Kernel density
        #     if evaluate_density :
        #         _, ax = plt.subplots(figsize =(10, 10))
        #         kernel = stats.gaussian_kde(full_latent[:,1:].T,bw_method='scott') 
        #         self.kernel_density(ax,kernel,kernel.factor/3,grid,X.shape)
        #         plt.savefig('plots/2d_reduction/kernel_density/evolution/kernel_density_{}.png'.format(epoch), bbox_inches='tight')


            #self.plot_kernel_density_estimation(full_latent,vars,label,mask_to_name,title=epoch)
        for e in np.arange(-1,n):
            #image = imageio.v2.imread('plots/2d_reduction/kidney/evolution/subsample_0.08_epoch_{}.png'.format(e))
            image = imageio.v2.imread('plots/2d_reduction/kernel_density/evolution/kernel_density_{}.png'.format(e))
            frames.append(image)

        # imageio.mimsave(f'plots/2d_reduction/kidney/evolution/latent_kidney_0.05.gif', # output gif
        #             frames,          # array of input frames
        #             fps = 3)         # optional: frames per second    s
        imageio.mimsave(f'plots/2d_reduction/kidney/kernel_density/kernel_density.gif', # output gif
                    frames,          # array of input frames
                    fps = 3)         # optional: frames per second    
        
    
    def model_evolution_kernel_latent(self,loader,mask_to_name):
        #Boolean deciding wether plots for kernel densities are made
        evaluate_density = True
        frames = []
        n = 36
        
        #Create grid to evaluate the kernel on
        X, Y = np.mgrid[-4:4:200j, -4:4:200j]
        grid = np.vstack([X.ravel(), Y.ravel()])

        
        col_dict = self.color_dict()
        for epoch in range(-1,n):
            model_path = 'model/model_kidney6/model_weight_n_{}.pth'.format(epoch)
            model = torch.load(model_path, map_location=torch.device('cpu'))
            full_latent , vars , label = compute_latent(loader,model)
            #remove label 0
            mask0 = label!=0
            full_latent = full_latent[mask0,:]
            vars = vars[mask0,:]
            label = label[mask0]

            ## 2d latent
            #self.plot_latent_dim_with_var(full_latent,vars,label,mask_to_name,epoch,p=0.08)
            _,ax = plt.subplots(ncols = 2 , figsize = (20,10))
            ax[0].set_xlim(-4,4)
            ax[0].set_ylim(-4,4)
            self.scatter_with_covar(ax[0],full_latent[:,1:],vars,label, col_dict, mask_to_name)
            
            ## Kernel density
            #self.plot_kernel_density_estimation(full_latent,vars,label,mask_to_name,title=epoch)
            kernel = stats.gaussian_kde(full_latent[:,1:].T,bw_method='scott') 
            self.kernel_density(ax[1],kernel,kernel.factor/3,grid,X.shape)

            plt.savefig('plots/2d_reduction/evolution/model_evolution_{}.png'.format(epoch), bbox_inches='tight')

        for e in np.arange(-1,n):
            image = imageio.v2.imread('plots/2d_reduction/evolution/model_evolution_{}.png'.format(e))
            frames.append(image)

        imageio.mimsave(f'plots/2d_reduction/evolution/model_evolution.gif', # output gif
                    frames,          # array of input frames
                    fps = 6)         # optional: frames per second 
    
    ### Function used in zoomed_micro
    def do_plot(self,ax, Z, transform,cmap_col):
        
        im = ax.imshow(Z, interpolation='none',
                    origin='lower',
                    cmap = cmap_col,
                    #extent=[-2, 4, -3, 2], 
                    clip_on=True,
                    alpha=0
                    )

        trans_data = transform + ax.transData
        im.set_transform(trans_data)

        ## VAN 1_31
        #im= im.T
        ##

        # display intended extent of the image
        x1, x2, y1, y2 = im.get_extent()
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
                transform=trans_data,
                linewidth=5)
    def reg(self,r,width,height):
        return [r[0],r[0]+width, r[1], r[1]+height]
    def plot_zoomed_micro(self):
        ### Load microscopy
        ## Kidney
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/VAN0046-LK-3-45-PAS-registered.ome.tiff"
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/2023_05_VAN0046_microscopy_multiple_files_transform_v2.i2m.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/'
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        # colors = ['darkorange',"blue", "#6DCC8C", 'black','#a30000','#f44e97','#6a329f']
        
        ### VAN1_41
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/VAN0063-RK-1-41-PAS_to_postAF_registered.ome.tiff"
        # #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/old_mask/van1_41_transform.i2r.json" 
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/VAN0063-RK-1-41_transform.v2.i2r.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/'
        # names = ['glomerulus','pr1','pr2']

        ### VAN1_31
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/VAN0042-RK-1-31-PAS_to_postAF_registered.ome.tiff"
        # #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/old_mask/van1_31_transform.i2r.json" 
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/VAN0042-RK-1-31_transform.v2.i2r.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/'
        # names = ['glomerulus','pr1','pr2']
        
        ### VAN1_35
        microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/VAN0049-RK-1-35-PAS_to_postAF_registered.ome.tiff"
        #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/old_mask/van1_35_transform.i2r.json" 
        json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/VAN0049-RK-1-35_transform.v2.i2r.json" 
        PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/'
        names = ['glomerulus','pr1','pr2']

        
        #colors = ['darkorange',"blue", "#6DCC8C"]
        #colors = ['#F96E46',"blue", "#6DCC8C"] ## Orange
        
        image = io.imread(microscopy_path)

        ### ADD this to remove black artefacts in VAN1_31
        image[np.sum(image,axis=2)==0,:] = 150
        
        
        
        ## Load the masks
        f = open(json_file)
        data = json.load(f)
        M = data['matrix_xy_px']

        
        
        #colors = ['darkorange',"blue", "blue", 'black','#a30000','#f44e97','#6a329f']
        names_files = dict(zip(range(len(names)),names))
        masks = []
        for l in range(len(names)):
            with open(PATH+'mask_{}.npy'.format(names_files[l]),'rb') as f:
                masks.append(np.load(f)) 
        
        ## Combined the masks
        mask = masks[0]
        for i in range(1,3) : 
            mask = mask + (i+1)*np.copy(masks[i]).astype( 'float32')
        mask[mask==0] = np.nan
        #colors = ['darkorange','black', '#a30000', "hotpink","#6DCC8C",'#f44e97','#6a329f']
        colors = ['#F96E46','black', '#a30000', "hotpink","#6DCC8C",'#f44e97','#6a329f']
        cmap_col = colors_mat.ListedColormap( colors[:3])

        
        #with_label =True
        f, ax = plt.subplots(figsize=(20,15))
        f.patch.set_visible(False)

        ## Plot the image on main axis
        im = ax.imshow(image)
        ax.axis('off')
        
        #### Kidney
        limits = np.array((1900, 13400, 1019, 9000))
        r1 = [6200, 6500,6550,6870]
        r2 = [7230 , 7530, 4050,4380] ## x1 , x2 , y1 , y2
        r3 = [8100,8400,3050,3370]

        ### VAN1_41
        # limits = np.array((511, 13100, 4700, 10300)) # x1 , x2 , y1 , y2
        # width = 280
        # height = 320
        # width = (limits[1]-limits[0])*0.05
        # #/20   
        # #height = limits[3] - limits[2]
        # # Check upper left corner in napari
        # r1 = self.reg([3200,6800],width,height)
        # r2 = self.reg([5300,7250], width,height)## x1 , x2 , y1 , y2
        # r3 = self.reg([8750,8150],width,height )

        # loc = [[1,3],[1,3],[2,4]]
        # loc = [[1,3],[2,4],[2,4]]
        # ax.set_xlim(limits[0],limits[1])
        # ax.set_ylim(limits[3],limits[2])


        ### VAN1_31
        # print('## Before = ' , image.shape)
        # trans_data = Affine2D().rotate(np.pi/2)+ ax.transData
        # im.set_transform(trans_data)
        # print('## After = ', image.shape)
        # limits = np.array((13645, 19037,5972, 18231)) # x1 , x2 , y1 , y2
        # ## Rotate the limit
        # limits = np.array((5972, 18231,13645, 19037))
        # width = 280
        # height = 320
        # width = (limits[1]-limits[0])*0.05
        # offset = 3000

        # # r3 = self.reg([14730,6800],width,height)
        # # r2 = self.reg([16231,9662], width,height)## x1 , x2 , y1 , y2
        # # r1 = self.reg([18317,12980],width,height )

        # r1 = self.reg([10600+offset,18317],width,height)
        # r2 = self.reg([13690+offset,16231], width,height)## x1 , x2 , y1 , y2
        # r3 = self.reg([16480+offset,14630],width,height )

        # r1 = self.reg([10600+offset,18317],width,height)
        # r2 = self.reg([13690+offset,16131], width,height)## x1 , x2 , y1 , y2
        # r3 = self.reg([16680+offset,14830],width,height )
        # loc = [[1,3],[1,3],[1,3]]
        # #loc = [[2,4],[2,4],[2,4]]
        # ax.set_xlim(limits[0]-image.shape[0]+offset,limits[1]-image.shape[0]+offset)
        # ax.set_ylim(limits[3],limits[2])

        ### VAN1_35
        print('## Before = ' , image.shape)
        #image = np.moveaxis(image,[0,1],[1,0])
        trans_data = Affine2D().rotate(np.pi/2)+ ax.transData
        im.set_transform(trans_data)
        print('## After = ', image.shape)
        limits = np.array((11000, 21000,4400, 10950)) # x1 , x2 , y1 , y2
        width = 280
        height = 320
        width = (limits[1]-limits[0])*0.05
        offset =-5400
        #= 3000
        print('### image.shape[0] - offset = ' , image.shape[0]-offset)
        print('### image.shape = ', image.shape)
        r1 = self.reg([6380,9100],width,height)
        r2 = self.reg([9190,7000], width,height)## x1 , x2 , y1 , y2
        r3 = self.reg([12380,4950],width,height )
        loc = [[2,4],[2,4],[2,4]]
        #loc = [[2,4],[2,4],[2,4]]
        ax.set_xlim(limits[0]-image.shape[0]+offset,limits[1]-image.shape[0]+offset)
        ax.set_ylim(limits[3],limits[2])
        
        regions = [r1,r2,r3]
        

        

        #limits = np.array((1039, 14000, 748, 9000))


        # if with_label:
        #     self.do_plot(ax,mask,Affine2D(np.array(M)),cmap_col)

        

        
        axs = []
        inset_map = {1:4,4:1,2:3,3:2}
        for i ,r in enumerate(regions) :
            axins = ax.inset_axes((0+0.34*i, 1.05 , 0.32, 0.4))
            axs.append(axins)
            
            axins.axis('off')

            im_in = axins.imshow(image)
            
            # VAN1_41
            # axins.axis('off')
            # axins.set_xlim(r[0],r[1])
            # axins.set_ylim(r[3],r[2])

            

            ## VAN1_31 & VAN1_35
            ## Rotate images : 
            trans_data = Affine2D().rotate(np.pi/2)+ axins.transData
            im_in.set_transform(trans_data)
            
            axins.set_xlim(r[0]-image.shape[0],r[1]-image.shape[0])
            axins.set_ylim(r[3],r[2])

            #mark_inset(ax,axins,loc1=loc[i][0],loc2=loc[i][1],linewidth=4,ec ='#1f77b4')
            patch, pp1,pp2 = mark_inset(ax,axins,loc1=loc[i][0],loc2=loc[i][1],linewidth=4,ec ='#DE970B') #dark yellow
            pp1.loc1 = loc[i][0]
            pp1.loc2 = inset_map[pp1.loc1]
            pp2.loc1 = loc[i][1]
            pp2.loc2 = inset_map[pp2.loc1]
        
        ### Kidney
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/micro_no_labels.png',bbox_inches='tight', dpi=300)
        
        #### VAN1_41
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/micro_no_labels1_41.png',bbox_inches='tight', dpi=300)

        #### VAN1_31
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/micro_no_labels1_31.png',bbox_inches='tight', dpi=300)

        #### VAN1_35
        plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/micro_no_labels1_35.png',bbox_inches='tight', dpi=300)

        ## Add the masks
        #VAN1_41
        transform_van = Affine2D(np.array(M))
        #VAN1_31 & VAN1_35
        transform_van = Affine2D(np.array(M))+ Affine2D().rotate(np.pi/2)
        
        # rotation = Affine2D().rotate(90)
        # # Create Affine2D with the rotated matrix
        # transform_van = Affine2D(np.array(M)).concatenate(rotation)

        self.do_plot(ax,mask,transform_van,cmap_col)
        for axins in axs :
            self.do_plot(axins,mask,transform_van,cmap_col)
        names = ["","",""]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=names[i]) for i in range(3) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(0, 0), loc=2, borderaxespad=0. , fontsize="40")
        # Add a legend with empty names
        
        ### Kidney
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/micro_with_labels.png',bbox_inches='tight', dpi=300)

        #### VAN1_41
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/micro_with_labels1_41.png',bbox_inches='tight', dpi=300)

        #### VAN1_31
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/micro_with_labels1_31.png',bbox_inches='tight', dpi=300)

        #### VAN1_35
        plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/micro_with_labels1_35.png',bbox_inches='tight', dpi=300)

    def plot_complete_micro(self):
        ### Load microscopy
        ## Kidney
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/VAN0046-LK-3-45-PAS-registered.ome.tiff"
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/2023_05_VAN0046_microscopy_multiple_files_transform_v2.i2m.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/'
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend','funky_gl']
        # colors = ['darkorange',"blue", "#6DCC8C", 'black','#a30000','#f44e97','#6a329f']
        
        ### VAN1_41
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/VAN0063-RK-1-41-PAS_to_postAF_registered.ome.tiff"
        # #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/old_mask/van1_41_transform.i2r.json" 
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/VAN0063-RK-1-41_transform.v2.i2r.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/'
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend']
        # colors = ['#F96E46','black', '#a30000', "#F9C846","#6DCC8C",'#f44e97','#6a329f'] #Burnt sienna(orange), black , ... , Saffron(yellow)

        ### VAN1_31
        # microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/VAN0042-RK-1-31-PAS_to_postAF_registered.ome.tiff"
        # #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/old_mask/van1_31_transform.i2r.json" 
        # json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/VAN0042-RK-1-31_transform.v2.i2r.json" 
        # PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/'
        # #names = ['glomerulus','pr1','pr2']
        # names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend']
        # colors = ['#F96E46','black', '#a30000', "#F9C846","#6DCC8C",'#f44e97','#6a329f'] #Burnt sienna(orange), black , ... , Saffron(yellow)
        
        ### VAN1_35
        microscopy_path = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/VAN0049-RK-1-35-PAS_to_postAF_registered.ome.tiff"
        #json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/old_mask/van1_35_transform.i2r.json" 
        json_file = "/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/VAN0049-RK-1-35_transform.v2.i2r.json" 
        PATH = '/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/'
        #names = ['glomerulus','pr1','pr2',]
        names = ['glomerulus','pr1','pr2','coll_duct','thick_Ascend']
        colors = ['#F96E46','black', '#a30000', "#F9C846","#6DCC8C",'#f44e97','#6a329f'] #Burnt sienna(orange), black , ... , Saffron(yellow)
        
        
        image = io.imread(microscopy_path)

        ### ADD this to remove black artefacts in VAN1_31
        image[np.sum(image,axis=2)==0,:] = 150
        
        ## Load the masks
        f = open(json_file)
        data = json.load(f)
        M = data['matrix_xy_px']

        names_files = dict(zip(range(len(names)),names))
        masks = []
        for l in range(len(names)):
            with open(PATH+'mask_{}.npy'.format(names_files[l]),'rb') as f:
                masks.append(np.load(f)) 
        
        ## Combined the masks
        mask = masks[0]
        for i in range(1,len(names)) : 
            mask = mask + (i+1)*np.copy(masks[i]).astype( 'float32')
        mask[mask==0] = np.nan
        cmap_col = colors_mat.ListedColormap( colors[:len(names)])

        
        
        #with_label =True
        f, ax = plt.subplots(figsize=(20,15))
        f.patch.set_visible(False)

        ## Plot the image on main axis
        im = ax.imshow(image)
        ax.axis('off')

        ### VAN1_35
        trans_data = Affine2D().rotate(np.pi/2)+ ax.transData
        im.set_transform(trans_data)
        limits = np.array((11000, 21000,4400, 10950)) # x1 , x2 , y1 , y2
        # width = 280
        # height = 320
        # width = (limits[1]-limits[0])*0.05
        offset =-5400
        ax.set_xlim(limits[0]-image.shape[0]+offset,limits[1]-image.shape[0]+offset)
        ax.set_ylim(limits[3],limits[2])

        ### VAN1_31
        # print('## Before = ' , image.shape)
        # trans_data = Affine2D().rotate(np.pi/2)+ ax.transData
        # im.set_transform(trans_data)
        # print('## After = ', image.shape)
        # limits = np.array((13645, 19037,5972, 18231)) # x1 , x2 , y1 , y2
        # ## Rotate the limit
        # limits = np.array((5972, 18231,13645, 19037))
        # width = (limits[1]-limits[0])*0.05
        # offset = 3000
        # ax.set_xlim(limits[0]-image.shape[0]+offset,limits[1]-image.shape[0]+offset)
        # ax.set_ylim(limits[3],limits[2])

        ### VAN1_41

        # limits = np.array((511, 13100, 4700, 10300)) # x1 , x2 , y1 , y2
        # #width = 280
        # #height = 320
        # #width = (limits[1]-limits[0])*0.05
        # ax.set_xlim(limits[0],limits[1])
        # ax.set_ylim(limits[3],limits[2])
        
        
        #cmap_col = colors_mat.ListedColormap( colors)
        #cmap_col = colors_mat.ListedColormap(['white'] + colors[:len(names)])

        ## Plot the mask
        #VAN1_41
        transform_van = Affine2D(np.array(M))
        #VAN1_31 & VAN1_35
        transform_van = Affine2D(np.array(M))+ Affine2D().rotate(np.pi/2)
        
        # rotation = Affine2D().rotate(90)
        # # Create Affine2D with the rotated matrix
        # transform_van = Affine2D(np.array(M)).concatenate(rotation)

        self.do_plot(ax,mask,transform_van,cmap_col)
        ### Kidney
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/micro_with_labels.png',bbox_inches='tight', dpi=300)

        #### VAN1_41
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_41/micro_complete_1_41.png',bbox_inches='tight', dpi=300)

        #### VAN1_31
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_31/micro_complete_1_31.png',bbox_inches='tight', dpi=300)

        #### VAN1_35
        #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/micro_complete_1_35.png',bbox_inches='tight', dpi=300)
        plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/micro_complete_extent_1_35.png',bbox_inches='tight', dpi=300)





    #'837.54'
    def plot_mz(self, mz =['837.54','1281','885.54','714.50']) :
        centroids, _, pixel_index = dat.load() #original data
        image_shape, norm, mzs = dat.load_shape_norm_mzs()
        index = 2

        mz = [str(a) for a in mzs]

        #print(mz)
        ## Find the number of the index of the mz
        #mzs2 = np.array([ int(str(a).split('.')[0]) for a in mzs])
        
        for m in mz :
            mzs2 = np.array([ m in str(a) for a in mzs])
            print('Found : {} value for mz {}'.format( np.sum( mzs2), m))
            #index = np.where(mzs2 == mz)[0][0]
            index = np.where(mzs2)[0][0]
            print(index)
            
            ## Plot the desired mz spatial view
            _, ax = plt.subplots(figsize=(10, 10))
            image = dat.reshape_array(centroids[:, index] / norm, image_shape, pixel_index)
            image = np.flip(image,axis =(0,1))

            ## Clip the top 5% largest values:
            image[np.isnan(image)] = 0
            p=  0.97
            q=np.quantile(image,p)
            image = np.clip(image,a_min = 0,a_max=q)

            ### VAN1_41
            ax.imshow(image)
            ### VAN1_35
            # image_rot = image[::-1,:]
            # image_rot = image_rot.T
            # ax.imshow(image_rot)


            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)

            ## Add an arrow for the size 
            ### 1 pixel is 10 micro m
            arrow_size = 100
            ## VAN1_41
            # x_loc = image.shape[1]-120
            # y_loc = image.shape[0]-30 
            # Lower right 
            x_loc = image.shape[1]-110
            y_loc = image.shape[0]-15  
            #Upper right 
            x_loc = image.shape[1]-110
            y_loc = 10  
            ## VAN1_35
            # x_loc = image_rot.shape[1]-120
            # y_loc = image_rot.shape[0]-30
            
            print(x_loc)
            print(y_loc)
            # arr = mpatches.FancyArrowPatch((x_loc, y_loc), (x_loc+arrow_size, y_loc),
            #                             arrowstyle='->,head_width=.2,head_length=0.2', mutation_scale=25,color = 'white',linewidth = 3)
            # ax.add_patch(arr)
            # ax.annotate(u"{}\u03bcm".format(arrow_size*10), (1.2, 1), xycoords=arr, ha='right', va='bottom', fontsize=15, color = 'white')
            rect = mpatches.Rectangle((x_loc, y_loc), arrow_size, 8, linewidth=1.5, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            #ax.annotate(u"{} \u03bcm".format(arrow_size*10), (0.9, 1.2), xycoords=rect, ha='right', va='bottom', fontsize=15, color = 'white')

            #### loc in lower right
            # ax.text(
            # x_loc+8, y_loc-4.5, u"{} \u03bcm".format(arrow_size*10),
            # color='white',
            # fontsize=15,
            # path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
            ###
            #### loc in upper right
            ax.text(
            x_loc+9, y_loc+24, u"{} \u03bcm".format(arrow_size*10),
            color='white',
            fontsize=15,
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
            ###

            plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/mzs/mzs_{}.png'.format(mzs[index]),bbox_inches='tight', dpi=300)
            ### VAN1_35
            #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/mzs/van1_35/mzs_{}.png'.format(mzs[index]),bbox_inches='tight', dpi=300)
            ### VAN1_41
            #plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/mzs/van1_41/mzs_{}.png'.format(mzs[index]),bbox_inches='tight', dpi=300)


    def plot_spectrum(self, loader):
        data_dir2_5 = Path(r"data/negative/VAN0005-RK-2-5-IMS_lipids_neg_roi=#0_mz=fix")
        data_dir1_31 =  Path(r"data/negative/VAN0042-RK-1-31-IMS_lipids_neg_roi=#1_mz=fix")
        data_dir1_35 = Path(r"data/negative/VAN0049-RK-1-35-IMS_lipids_neg_roi=#1_mz=fix")
        data_dir1_41 = Path(r"data/negative/VAN0063-RK-1-41-IMS_lipids_neg_roi=#1_mz=fix")
        #path_points = 'saved_data/van1_41/points_to_n_steps.pkl'
        path_points = 'saved_data/van1_35/points_to_n_steps.pkl'

        data_dir = data_dir1_35

        x,_ = next(iter(loader))
        _, mzs = dat.load_centroids(data_dir)
        print(x[0,0,:].detach().numpy())
        
        ## Subset of x_ticks that are going to be ploted
        subset =  np.arange(0,x.shape[2],step=8)
        for i in range(x.shape[0]):
            plt.figure()
            plotdata = pd.DataFrame({'intensity':x[i,0,:].detach().numpy()},index =mzs)
            ax = plotdata.plot(kind="bar",color= '#036512',figsize=(15,5))
            ax.set_xticks(subset)
            ax.set_xticklabels(mzs[subset], rotation = 45,ha='right')
            ax.tick_params(axis='x', labelsize=17)
            ax.set_xlabel('$m/z$', fontsize=20)
            ax.legend().set_visible(False)
            ax.set_ylabel('Intensity', fontsize = 20)
            plt.savefig('/Users/pdelacour/Documents/PL_Ecole/beta_vae/saved_data/masks/van1_35/spectra/spectrum_{}.png'.format(i), bbox_inches='tight',dpi=300) 
            plt.close()
        # sign_traversal = np.zeros((len(combined_p),combined_p[0].shape[1]))
        # for i,p_steps in enumerate(combined_p) : 
        #     plotdata = pd.DataFrame({'var':np.var(p_steps, axis = 0)},index =mzs)
        #     ## If this difference is positive it means the signal was increasing 
        #     sign_traversal[i,:] = ( p_steps[-1,:] - p_steps[0,:]>= 0)*2-1 ## Casting values to the range [-1,1] 
        #     plt.figure()
        #     ax = plotdata.plot(kind="bar",color= '#036512',figsize=(15,5))
        #     ax.set_xticks(subset)
        #     ax.set_xticklabels(mzs[subset], rotation = 45,ha='right')
        #     ax.set_xlabel('$m/z$', fontsize=10)
        #     ax.set_ylabel('Variance', fontsize = 10)
        #     plt.savefig('plots/latent_traversal/variance/traversal_p{}.png'.format(i), bbox_inches='tight',dpi=300)
        
        # ## For interpretation note that some traversal increase the value and some decrease it
        # sign_traversal = np.sum(sign_traversal,axis = 0) >=0
        # #print('### Sign traversal = ' , np.sum(sign_traversal,axis = 0))
        
        # ## Plot of the stacked traversal
        # data = {}
        # for i,p_steps in enumerate(combined_p) : 
        #     data['var_{}'.format(i)] = np.var(p_steps,axis=  0)
        # plotdata = pd.DataFrame(data,index = mzs.astype(str))
        # plt.figure()
        # ax = plotdata.plot(kind="bar",stacked = True,figsize=(15,5),xticks=subset)
        # ax.set_xticks(subset)

        # #ticks_labels = mzs[subset]
        # ticks_labels = np.array([f'{mzs[i]:.2f}' for i in subset ])
        # ax.set_xticklabels(ticks_labels, rotation = 45,ha='right')
        # #### LEGEND
        # ax.set_xlabel('$m/z$', fontsize=10)
        # ax.set_ylabel('Variance', fontsize = 10)
        
        # ### Remove the legend for 50 points
        # ax.get_legend().remove()
        # ###
        # plt.savefig('plots/latent_traversal/variance/traversal_combined.png', bbox_inches='tight',dpi=300)


        # ## Csv file for the variance sorted by order 
        # sum_var = np.sum(np.array(list(data.values())), axis = 0)
        # col_data = pd.DataFrame({'mzs':mzs, 'variance':sum_var, 'increase':sign_traversal})
        # col_data = col_data.sort_values(
        #     by="variance",
        #     ascending=False
        # )
        # col_data.to_csv('saved_data/combined_variance1_35.csv', index=False)

        # ## Plot of the variance in order : 
        # col_data = col_data.sort_values(
        #     by="variance",
        #     ascending=False
        # )

        # # f, ax = plt.subplots(figsize=(15,5))
        # # subset = np.arange(0,212,step=4)
        # # ax.set_xticks(subset)
        # # ax.set_xticklabels(np.array(col_data['mzs'])[subset], rotation = 45,ha='right')
        # # ax.set_xlabel('$m/z$', fontsize=10)
        # # ax.set_ylabel('Variance', fontsize = 10)
        # # color_bars = {1 : '#050533', 0:'#E34234',} #Darkblue, red
        # # label_bars = {1: 'increasing', 0:'decreasing'}
        
        # # for l in np.unique(col_data['increase']) :
        # #     ind = col_data['increase']==l
        # #     ax.scatter(np.arange(len(sum_var))[ind],col_data['variance'][ind], c=color_bars[l], label=label_bars[l])    
        # # ax.legend()

        # f, ax = plt.subplots(figsize=(15,5))

        # subset = np.arange(0,212,step=4)
        # ax.set_xticks(subset)
        # ticks_labels_mzs = np.array(col_data['mzs'])
        # ticks_labels = np.array([f'{ticks_labels_mzs[i]:.3f}' for i in subset ])
        # for i in subset : 
        #     print(f'#### {ticks_labels_mzs[i]} , {ticks_labels_mzs[i]:.3f}')
        # ax.set_xticklabels(ticks_labels, rotation = 45,ha='right')
        # #ax.set_xticklabels(np.array(col_data['mzs'])[subset], rotation = 45,ha='right')
        
        # #ax.set_xlabel('$m/z$', fontsize=10)
        # #ax.set_ylabel('Variance', fontsize = 10)
        # color_bars = {1 : '#050533', 0:'#E34234',} #Darkblue, red
        # color_bars = {1 : '#fb6f92', 0:'#8d99ae',} #Darkblue, red
        # color_bars = {1 : '#fe6d73', 0:'#17c3b2',} #Darkblue, red
        # label_bars = {1: 'L-to-R increasing', 0:'L-to-R decreasing'}

        # #axins = ax.inset_axes((0.5, 0.6, 0.3, 0.3))
        # axins = ax.inset_axes((0.3, 0.6, 0.4, 0.3))
        # ax.tick_params(axis='y', labelsize=15)
        # axins.tick_params(axis='y', labelsize=15)
        # axins.tick_params(axis='x', labelsize=15)

        # cutx = 14
        # eps=0.1
        # #axins.set_xlim(-eps,cutx-1+eps)
        # X_global =  np.arange(len(sum_var))
        # X1 =X_global[:cutx]
        # X2 =X_global[cutx:]
        # Y1 = col_data['variance'][:cutx]
        # Y2 = col_data['variance'][cutx:]

        # col_data_sub = col_data[:cutx]['increase']
        # for l in np.unique(col_data_sub) :
        #     ind = col_data_sub==l
        #     X = X1[ind]
        #     Y = Y1[ind]
        #     #ax.scatter(X,Y, c=color_bars[l], label=label_bars[l])  
        #     ax.scatter(X,Y, c=color_bars[l])  
        #     axins.scatter(X,Y, c=color_bars[l], label=label_bars[l])    

        # subset = np.arange(0,cutx, step=1)
        # #ticks_labels_in = np.array([f'{mzs[i]:.2f}' for i in subset ])
        # ticks_labels_in = np.array([f'{ticks_labels_mzs[i]:.3f}' for i in subset ])
        # axins.set_xticks(subset)

        # axins.set_xticklabels(ticks_labels_in, rotation = 45,ha='right')


        # #Plot the rest
        # col_data_sub = col_data[cutx:]['increase']
        # for l in np.unique(col_data_sub) :
        #     ind = col_data_sub==l
        #     X = X2[ind]
        #     Y = Y2[ind]
        #     ax.scatter(X,Y, c=color_bars[l],label = label_bars[l])  
            
        # mark_inset(ax,axins,loc1=1,loc2=3,)
        # ax.legend(prop = { "size": 15 })
        # #plt.savefig('plots/latent_traversal/variance/ordered_variance_traversal.png', bbox_inches='tight',dpi=300)  
        # plt.savefig('plots/latent_traversal/variance/ordered_variance_traversal1_35.png', bbox_inches='tight',dpi=300) 







