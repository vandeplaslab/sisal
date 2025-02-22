import numpy as np
import torch
from collections import Counter
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import StandardScaler

def reparametrize(mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

def sample(m, var):
    return m + np.sqrt(var)*np.random.normal(size=m.shape[0])

def sample_batch(m,var):
    return m + np.sqrt(var)*np.random.normal(size=m.shape)

# def sample(m, var):
#     return m + np.sqrt(var)*np.random.normal(size=m.shape[0])

## Compute the full latent space of the data given in loader 
## Return the associate latent mean, variance and label
def compute_latent(loader,model): 
        print('Compute Latent') 
        n = len(loader.dataset)
        latent = np.zeros((n,1+model.z_dim))  
        vars = np.zeros((n,model.z_dim))  
        label = np.zeros(n)
        alpha = np.zeros(n)
        with torch.no_grad():
            prev = 0
            for x, l, *rest, i in loader:
                if len(rest) == 1:  # Means alpha is included
                    alpha_val = rest[0]
                else:
                    alpha_val = None  # Default value if alpha is missing
                z_mean,  z_logvar =model.forward(x)
                batch = z_mean.size(0)
                latent[prev:prev+batch,0] = i
                #print('# i = ', i)
                latent[prev:prev+batch,1:] = z_mean.detach().numpy()
                vars[prev:prev+batch,:] = np.exp(z_logvar.detach().numpy())
                label[prev:prev+batch] = l
                alpha[prev:prev+batch] = alpha_val
                prev+=batch 

            # for x,l , i in loader :
            # #for i, (x,l)  in enumerate(loader) :
            #     #x =x.to(device)
            #     z_mean,  z_logvar =model.forward(x)
            #     batch = z_mean.size(0)
            #     latent[prev:prev+batch,0] = i
            #     #print('# i = ', i)
            #     latent[prev:prev+batch,1:] = z_mean.detach().numpy()
            #     vars[prev:prev+batch,:] = np.exp(z_logvar.detach().numpy())
            #     label[prev:prev+batch] = l
            #     prev+=batch 
                
        
        n_zeros_rows = ~np.all(latent == 0, axis=1)
        latent = latent[n_zeros_rows]
        vars = vars[n_zeros_rows]
        label = label[n_zeros_rows]

        # with open('saved_data/saved_latent.npy', 'wb') as f:
        #     np.save(f, latent)
        #     np.save(f, vars)
        #     np.save(f,label)
        # print('End Compute latent')

        return latent, vars,label,alpha

def compute_latent_synthetic(loader,model): 
        print('Compute Latent') 
        n = len(loader.dataset)
        print('n loader = ', n )
        latent = np.zeros((n,1+model.z_dim))  
        vars = np.zeros((n,model.z_dim))  
        label = np.zeros(n)
        coeff = np.zeros(n)
        with torch.no_grad():
            prev = 0
            for x,l , alpha ,i in loader :
                z_mean,  z_logvar =model.forward(x)
                batch = z_mean.size(0)
                latent[prev:prev+batch,0] = i
                latent[prev:prev+batch,1:] = z_mean.detach().numpy()
                vars[prev:prev+batch,:] = np.exp(z_logvar.detach().numpy())
                label[prev:prev+batch] = l
                coeff[prev:prev+batch] = alpha
                prev+=batch 
        print('latent shape = ' , latent.shape)        
        
        # with open('saved_data/saved_latent.npy', 'wb') as f:
        #     np.save(f, latent)
        #     np.save(f, vars)
        #     np.save(f,label)
        print('End Compute latent')
        
        n_zeros_rows = ~np.all(latent == 0, axis=1)
        latent = latent[n_zeros_rows]
        vars = vars[n_zeros_rows]
        label = label[n_zeros_rows]

        return latent, vars,label, coeff

## Compute the mean of the latent only
def compute_latent_mean(loader,model) :
    print('Compute Latent mean')
    latent = np.zeros((len(loader.dataset),model.z_dim))  
    with torch.no_grad():
        prev = 0
        #for x,l , i in loader :    
        for  x,_  in loader :
            #x =x.to(device)
            z_mean, _ = model.forward(x)
            batch = z_mean.size(0)
            #latent[prev:prev+batch,0] = i
            latent[prev:prev+batch,:] = z_mean.detach().numpy()
            prev+=batch 
    print('End Compute latent mean')
    return latent

# n_b : number of batches on which to compute the estimate
def compute_estimate_std(model,n_b,loader, device):
    batch_size = 32
    latent = np.zeros((n_b*batch_size,model.z_dim))  
    with torch.no_grad():
        prev = 0
        #for x,l , i in loader :    
        for _ in range(n_b):
            x , _ = next(iter(loader))
            x = x.to(device, non_blocking=True)
            #x =x.to(device)
            z_mean, _ = model.forward(x)
            batch = z_mean.size(0)
            emp_std = torch.std(z_mean,axis = 0, unbiased=True)
            prev+=batch 
    return emp_std



# Compute the empirical standard distribution of the full data
def emp_std(full_latent) : 
    return np.std(full_latent,axis=0,ddof=1)    
    #return np.sqrt(np.var(full_latent,axis = 0,ddof=1))

# Compute the limit of the latent space using the full precomputed latent_space
def limit_latent_space_precomp(full_latent):
    z_min = np.min(full_latent, axis = 0)
    z_max = np.max(full_latent, axis = 0)
    return (z_min,z_max)

def accuracy_confusion_matrix(confusion_matrix):
        return np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)


def metric_disentangling(model, z_min,z_max, full_std , std_threshold):
    #std_threshold = 0.2
    L = 100
    #L=32
    M = 800 
    #print('full_std =', full_std)
    #print('len(full_std) = ', len(full_std))
    z_dim = len(full_std)
    
    z_dims = np.arange(z_dim)
    #active_dims = (full_std>=std_threshold).cpu()
    active_dims = full_std>=std_threshold
    if not(all(active_dims)):
        print('Warning some dims have collapsed = ', z_dim-sum(active_dims))
    
    factors = np.random.choice(z_dims[active_dims], M) #Factors that are kept fixed
    metric_data = np.zeros((M,2))
    
    #factors = np.random.choice(z_dim, M) #Factors that are kept fixed
    with torch.no_grad():
        for i in range(M) : 
            f = factors[i]
            random_factors = np.zeros((L,z_dim))
            fixed_value = np.random.uniform(low = z_min[f], high = z_max[f])
            random_factors[:,f] = fixed_value
            not_f = ~np.isin(np.arange(z_dim),[f])
            random_factors[:,not_f] = np.random.uniform(low = z_min[not_f], high = z_max[not_f], size = (L,z_dim-1))
            random_factors = np.float32(random_factors)
            ############
            random_factors = torch.tensor(random_factors).to('cuda', non_blocking=True)
            ############
            #print('## Random_factors = ', random_factors)
            #print(random_factors.shape)

            random_data = model.decoder(random_factors)
            
            latent = model.encoder(random_data).cpu().detach().numpy()
            mu = latent[:,:z_dim]
            logvar = latent[:,z_dim:]
            
            random_latent = sample_batch(mu,np.exp(logvar))
            
            #random_latent = random_latent/full_std
            random_latent = random_latent/full_std.detach().numpy()

            # if i == 0 : 
            #     print('##Factor = ' , f)
            #     print('##Random Latent ')
            #     print(random_latent[:5,:])
            #print('## Random latent = ', random_latent.shape)
            #print('#### full_std = ', full_std.shape)
            random_latent = random_latent
            emp_var = np.var(random_latent , axis=0,ddof=1) ## Unbiased estimate of the variance

            metric_data[i,0] = np.argmin(emp_var)
            metric_data[i,1] = f

    less_var = {}
    for f in range(z_dim):
        c = Counter(metric_data[metric_data[:,0]==f,1])
        pred = c.most_common(1)
        less_var[f] = pred
    
    confusion_matrix = np.zeros((z_dim,z_dim))
    for f in range(z_dim) : 
        # Predicted factors when the true factor was f
        bins = np.bincount(metric_data[metric_data[:,1]==f,0].astype(int))
        for j in range(len(bins)):
            confusion_matrix[f,j] = bins[j]
    confusion_matrix = confusion_matrix[active_dims,:] 

    accuracy = accuracy_confusion_matrix(confusion_matrix)
    
    #accuracy = accuracy_confusion_matrix(confusion_matrix)
    # print('### Confusion Matrix')
    # print(confusion_matrix)
    # print('### Accuracy')
    # print(accuracy)
    return  accuracy , confusion_matrix 



def accuracy_confusion_matrix(confusion_matrix) :
    classifier = np.argmax(confusion_matrix, axis=1)
    accuracy = np.sum(confusion_matrix[np.arange(confusion_matrix.shape[0]),classifier])*1.0/np.sum(confusion_matrix)
    return accuracy
def metric_disentangling_factorising(model,full_latent,vars,z_min,z_max):
        print('## SECOND metric')
        z_dim = len(z_min)
        L = 100 #In the paper L = 100 
        M = 800 #take the majority vote classifier from 800 votes
        full_std = emp_std(full_latent[:,1:])
        

        index = np.random.choice(full_latent.shape[0],M, replace = False) 
        #factors = np.random.binomial(1, 0.5, M)
        factors = np.random.choice(z_dim, M) #Factors that are kept fixed
        
        metric_data = np.zeros((M,2)) #In first pos is that majority variance and in 1 is the label factor
        #for p,i in enumerate(index):
        for p in range(M):
            f = factors[p]
            #z = sample(full_latent[i,1:],vars[i])
            #z=full_latent[i,1:]
            
            random_factors = np.zeros((L,z_dim))
            #random_fac tors[:,f] = z[f] + np.random.normal(scale = 10)
            fixed_value = np.random.uniform(low = z_min[f], high = z_max[f])
            random_factors[:,f] = fixed_value
            not_f = ~np.isin(np.arange(z_dim), [f])            
            random_factors[:,not_f] = np.random.uniform(low = z_min[not_f] , high = z_max[not_f] ,size = (L,z_dim-1))            
            
            random_data = model.decoder(torch.Tensor(random_factors))
            random_latent = model.encoder(random_data).detach().numpy()[:,:z_dim]
            
            #random_latent = random_latent.detach().numpy()[:,:z_dim]
            random_latent = random_latent/full_std

            emp_var = np.var(random_latent , axis=0,ddof=1)
            
            metric_data[p,0] = np.argmin(emp_var)
            metric_data[p,1] = f
        
        less_var = {}
        
        for f in range(z_dim):
            c = Counter(metric_data[metric_data[:,0]==f,1])
            pred = c.most_common(1)
            less_var[f] = pred
        
        confusion_matrix = np.zeros((z_dim,z_dim))

        for f in range(z_dim) : 
            # Predicted factors when the true factor was f
            bins = np.bincount(metric_data[metric_data[:,1]==f,0].astype(int))
            for j in range(len(bins)):
                confusion_matrix[f,j] = bins[j]
        
        classifier = np.argmax(confusion_matrix, axis=1)
        accuracy = np.sum(confusion_matrix[np.arange(confusion_matrix.shape[0]),classifier])*1.0/np.sum(confusion_matrix)
        #accuracy = accuracy_confusion_matrix(confusion_matrix)

        print('##Confusion Matrix')
        print(confusion_matrix)
        print('##Accuracy = ', accuracy)

        return  confusion_matrix

## load a lot of models and take their accuracy, ...
# n = number of versions to load
# b = value of beta
# z_dim dim of latent space 
def load_results(b,n,z_dim):
    #z_dim = 10
    results = []
    for v in range(n):
        with open('saved_data/avg_models/model_z{}_b{}_v{}.npy'.format(z_dim,b,v), 'rb') as f:
            results.append(np.load(f))
    return results

def compute_loss(b,  n=1):
    names = ['epoch' ,'train_loss', 'train_recons', 'train_kl', 'test_loss', 'dis_metric']
    names_to_pos = dict(zip(names, range(len(names))))

    
    z_dim = 10
    results = load_results(b,n,z_dim)

    epochs = len(results[0])
    train_loss = np.zeros((n,epochs))
    train_recons = np.zeros((n,epochs))
    train_kl = np.zeros((n,epochs))
    test_loss = np.zeros((n,epochs))
    dis_metric = np.zeros((n,epochs))
    for i in range(n):
        train_loss[i,:] = np.array(results[i])[:,names_to_pos['train_loss']]
        train_recons[i,:] = np.array(results[i])[:,names_to_pos['train_recons']]
        train_kl[i,:] = np.array(results[i])[:,names_to_pos['train_kl']]
        test_loss[i,:] = np.array(results[i])[:,names_to_pos['test_loss']]
        dis_metric[i,:] = np.array(results[i])[:,names_to_pos['dis_metric']]

    return train_loss,train_recons,train_kl , test_loss,dis_metric

def normalize_train_test_full_loader(centroids: np.ndarray, mask:np.ndarray, batch_size: int = 32 , alpha=None):
    """
    Normalize the given data and return the train, test, and full data loaders.

    Args:
        centroids (np.ndarray): A NumPy array of shape (n_points, n_dim) representing the data points.
        masks (np.ndarray) : A NumPy array of shape (n_points) representing the mask (if unknown set all to 1)
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.
        alpha (np.ndarray) : A NumPy array of shape (n_points) representing the alpha (SNR) value if it is known

    Returns:
            - A list with two DataLoaders (train and test).
            - A full DataLoader including indices of the datapoints

    Notes:
        - This function standardizes the features using StandardScaler.
        - It splits the data into 80% training and 20% testing.
        - It prepares PyTorch DataLoaders for both sets.
        - A full DataLoader is created with additional fields: mask, alpha, and index.
    """

    df = pd.DataFrame(centroids)
    df['mask'] = mask
    df['index'] = range(centroids.shape[0])


    train = df.sample(frac=0.8,random_state=42)
    test = df.drop(train.index)


    val_train = train.drop(['index','mask'], axis=1)
    val_test = test.drop(['index','mask'], axis=1)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    
    scaler = StandardScaler()
    scaler.fit(val_train)
    val_train = scaler.transform(val_train)
    val_test = scaler.transform(val_test)
    

    val=[val_train,val_test]
    d = [train,test]
    loaders = []
    for i in range(len(val)) :
        v = val[i]
        df_temp = d[i]
        v = np.array(v,dtype = 'float32')   [:,np.newaxis,:]
        mask = np.array(df_temp['mask'])
        loader = torch.utils.data.DataLoader([ [v[j],mask[j]] for j in range(v.shape[0])], 
                                            shuffle=False, 
                                            num_workers=3,
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
        loaders.append(loader)
    
    #Compute the full loader 
    val = df.drop(['index','mask'], axis=1)
    val = scaler.transform(val)
    
    val = np.array(val,dtype = 'float32')[:,np.newaxis,:]
    index = np.array(df['index'])
    mask = np.array(df['mask'])

    if alpha is None:
        full_loader = loader = torch.utils.data.DataLoader([ [val[i], mask[i], index[i]] for i in range(val.shape[0])], 
                                            shuffle=False, 
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
    else:
        full_loader = loader = torch.utils.data.DataLoader([ [val[i], mask[i], alpha[i], index[i]] for i in range(val.shape[0])], 
                                            shuffle=False, 
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            drop_last=True)
    
    return loaders[0],loaders[1], full_loader 




