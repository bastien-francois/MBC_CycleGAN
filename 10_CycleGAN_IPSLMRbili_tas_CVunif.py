import numpy as np
import os
from os import makedirs
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from numpy.random import randint
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import randn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import sys


#### Possible choices for this code:
GAN_version="SpatialCycleGAN"

#Physical variable?
var_phys="tas"
#var_phys="pr"

#### Period in the year?
season="winter_79_16"

#### Rank version or minmax version?
rank_version=False

#### QQ data en input?
QQ2B_version=True

#### CV_version?
CV_version="CVunif"
#CV_version="CVchrono"

#### Hyperparameters?: learning rate of disc and gen?
list_lr_gen=[1e-4]
list_lr_disc=[5e-5]

### L-norm
L_norm="L1norm"
#L_norm = "L2norm"


Ref="SAFRAN"
Mod="IPSLMRbili"
#Mod = "SAFRANdetbili"

computation_globalenergy=True

#### Weights for valid, reconstruct and identity
lambda_val=1
lambda_rec=10
lambda_id=1

nb_filters_disc=[64,128]
nb_filters_gen=[64,128,256]

nb_epoch_for_eval = 10
##################################################################
##### Automatized below according to the choices
#
sys.path.insert(1,'/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Code/')
if GAN_version=="SpatialCycleGAN":
    import CycleGAN

### Downscaling (just for rmse plots)
if "SAFRANdet" in Mod:
    is_DS=True
else:
    is_DS=False

if var_phys=="tas":
    PR_version=False
else:
    PR_version=True

### For QQ loading
BC1d="1dQQ"

#### Load Model
os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/" + Mod + "/")
CalibA, ProjA,LON_Paris,LAT_Paris,minCalibA, maxCalibA, IND_Paris,point_max, OriginalCalibA, OriginalProjA = CycleGAN.load_calib_proj_minmaxrank(CV_version, rank_version,"tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",season)

####Load Ref
os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/" + Ref + "/")
CalibB, ProjB, LON_Paris, LAT_Paris, minCalibB, maxCalibB,IND_Paris, point_max, OriginalCalibB, OriginalProjB = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version,"tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",season)

#### Load QQ
os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/" + Ref + "_" + Mod + "/" +  CV_version)
CalibQQ, ProjQQ,_,_,minCalibQQ, maxCalibQQ,_,_,OriginalCalibQQ, OriginalProjQQ = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version, "tas_pr_day_" + CV_version + "_" + BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys + "_day_" + CV_version + "_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",season)

##### Load SpatialR2D2
MBC_output = "SpatialR2D2"
os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/" + Ref + "_" + Mod + "/" +  CV_version)
CalibSpatialR2D2, ProjSpatialR2D2,_,_,minCalibSpatialR2D2, maxCalibSpatialR2D2,_,_,OriginalCalibSpatialR2D2, OriginalProjSpatialR2D2 = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version, "tas_pr_day_" + CV_version + "_" + MBC_output+"_"+Ref+"_" + Mod + "_79_16_Paris",var_phys + "_day_" + CV_version + "_" + MBC_output + "_" + Ref + "_" + Mod + "_79_16_Paris",season)
print("R2D2 loaded")

##### Load SpatialdOTC
MBC_output = "SpatialdOTC"
os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/" + Ref + "_" + Mod + "/" +  CV_version)
CalibSpatialdOTC, ProjSpatialdOTC,_,_,minCalibSpatialdOTC, maxCalibSpatialdOTC,_,_,OriginalCalibSpatialdOTC, OriginalProjSpatialdOTC = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version, "tas_pr_day_" + CV_version + "_" + MBC_output+"_"+Ref+"_" + Mod + "_79_16_Paris",var_phys + "_day_" + CV_version + "_" + MBC_output + "_" + Ref + "_" + Mod + "_79_16_Paris",season)
print("dOTC loaded")

if QQ2B_version==True:
    minCalibX=minCalibQQ
    maxCalibX=maxCalibQQ


#################################################################
# create the discriminator
for lr_disc in list_lr_disc:
    for lr_gen in list_lr_gen:
        print('gen lr' + str(lr_gen))
        print('disc lr' + str(lr_disc))
        discX = CycleGAN.define_discriminator(lr_disc=lr_disc, nb_filters= nb_filters_disc)
        discB = CycleGAN.define_discriminator(lr_disc=lr_disc, nb_filters = nb_filters_disc)
        # create the generator
        genX2B = CycleGAN.define_generator(nb_filters=nb_filters_gen, rank_version=rank_version)
        genB2X = CycleGAN.define_generator(nb_filters=nb_filters_gen, rank_version=rank_version)
        # create the gan
        comb_model = CycleGAN.define_combined(genX2B, genB2X, discX, discB,lr_gen=lr_gen, lambda_valid= lambda_val, lambda_reconstruct = lambda_rec, lambda_identity = lambda_id, L_norm = L_norm)
        #comb_model.summary()
        if rank_version==False:
            name_version="minmax"

        #### Create a new folder
        os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/" + Ref + "_" + Mod + "/" + GAN_version + "/"+ var_phys+"/" +  CV_version + "/"  + season)

        new_folder = var_phys + '_' + name_version + '_' + CV_version + '_lrgen'+str(lr_gen)+'_lrdisc'+str(lr_disc)
        makedirs(new_folder, exist_ok=True)
        makedirs(new_folder + '/models', exist_ok=True)
        makedirs(new_folder + '/calib', exist_ok=True)
        makedirs(new_folder + '/proj', exist_ok=True)
        path_to_save="/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/" + Ref + "_" + Mod + "/" + GAN_version + "/"+var_phys+"/"+  CV_version + "/" + season+"/"+new_folder
        #### Train CycleGAN
        CycleGAN.train_combined_new(rank_version, PR_version,QQ2B_version, CV_version, is_DS, computation_globalenergy, genX2B, genB2X, discX, discB, comb_model, CalibA, CalibB, CalibQQ, ProjA, ProjB, ProjQQ, OriginalCalibA, OriginalCalibB, OriginalCalibQQ, OriginalCalibSpatialR2D2, OriginalCalibSpatialdOTC, OriginalProjA, OriginalProjB, OriginalProjQQ, OriginalProjSpatialR2D2, OriginalProjSpatialdOTC,IND_Paris, LON_Paris, LAT_Paris,point_max, path_to_save, minX=minCalibX, maxX=maxCalibX, minB= minCalibB, maxB = maxCalibB, n_epochs=6000, nb_epoch_for_eval = nb_epoch_for_eval) #####attention n_epochs






