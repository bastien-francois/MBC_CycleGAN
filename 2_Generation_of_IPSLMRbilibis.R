
####  Generate IPSLbis: Two-step procedure
#1. Generation of CDFt
#2. Use of Matrix-recorrelation

### For T2 and PR for CVchrono
#### Step 1. Generation of CDFt
# Ref: IPSLMRbili
# Mod: SAFRANdetbili
rm(list=ls())
gc()
library(fields)
library(CDFt)

#### On JZ
source("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Code/function_1dQQ.R")

load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/IPSLMRbili/tas_pr_day_IPSLMRbili_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/Temporal_and_Random_indices_1979_2016.RData")
###################################################################################################################
###################################################################################################################

CVtype = c("CVchrono")
for(CV in CVtype){
  print(CV)
  assign(paste0('Ind_', CV, '_Calib'),list("winter_79_16"=get(paste0('Ind_', CV, '_Calib_winter_79_16')),
                                           "summer_79_16"=get(paste0('Ind_', CV, '_Calib_summer_79_16')),
                                           "automn_79_16"=get(paste0('Ind_', CV, '_Calib_automn_79_16')),
                                           "spring_79_16"=get(paste0('Ind_', CV, '_Calib_spring_79_16'))))
  assign(paste0('Ind_', CV, '_Proj'),list("winter_79_16"=get(paste0('Ind_', CV, '_Proj_winter_79_16')),
                                          "summer_79_16"=get(paste0('Ind_', CV, '_Proj_summer_79_16')),
                                          "automn_79_16"=get(paste0('Ind_', CV, '_Proj_automn_79_16')),
                                          "spring_79_16"=get(paste0('Ind_', CV, '_Proj_spring_79_16'))))
  ##### TAS
  print("TAS")
  assign(paste0("tas_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_IPSLMRbili_79_16_Paris
  Mod=tas_day_SAFRANdetbili_79_16_Paris
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    npas_=floor(length(get(paste0('Ind_', CV, '_Calib'))[[season]])*2/3)
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        ### QQ for Calib for T2
        tmp_res_calib=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]])$Mch
        eval(parse(text=paste0("tas_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res_calib[,1]")))
        ### CDFt for Proj for T2
        tmp_res_proj=CDFt(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]],npas=npas_)$DS
        eval(parse(text=paste0("tas_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res_proj")))
      }
    }
  }
  
  #### PR #### 
  print("PR")
  assign(paste0("pr_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_IPSLMRbili_79_16_Paris
  Mod=pr_day_SAFRANdetbili_79_16_Paris
  
  th_O=min(Mod[Mod>0])
  th_M=th_O
  print(th_O)
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    npas_=floor(length(get(paste0('Ind_', CV, '_Calib'))[[season]])*2/3)
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res_calib=CDFt_for_PR(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],npas=npas_,th_O=th_O,th_M=th_M)
        eval(parse(text=paste0("pr_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res_calib$DS")))
        tmp_res_proj=CDFt_for_PR(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]],npas=npas_,th_O=th_O,th_M=th_M)
        eval(parse(text=paste0("pr_day_", CV, "_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res_proj$DS")))
      }
    }
  }
}


#### Save CVchrono
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/IPSLMRbili_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris.RData")



### Step 2. Generation of IPSLbis with matrix recorrelation
library(expm)

# Function to transform array in matrix
flatten_array <- function(data_array){ #data_array of type LON x LAT x TIME; output: TIME x VARPHY
  res = matrix(NaN, ncol= dim(data_array)[1] * dim(data_array)[2], nrow = dim(data_array)[3])
  k=0
  for(i in 1:dim(data_array)[1]){
    for(j in 1:dim(data_array)[2]){
      k=k+1
      res[,k]<-data_array[i,j,]
    }
  }
  return(res)
}

# Function to transform matrix in array
convert_matrix_to_array <- function(data_mat,nb_lon = sqrt(ncol(data_mat)), nb_lat = sqrt(ncol(data_mat))){ #data_mat of type TIME x VARPHY; output: LON x LAT x TIME
  res = array(NaN, dim = c(nb_lon, nb_lat, nrow(data_mat)))
  k=0
  for(i in 1:nb_lon){
    for(j in 1:nb_lat){
      k=k+1
      res[i,j,]<-data_mat[,k]
    }
  }
  return(res)
}



CVtype = c("CVchrono")
for(CV in CVtype){
  print(CV)
  assign(paste0('Ind_', CV, '_Calib'),list("winter_79_16"=get(paste0('Ind_', CV, '_Calib_winter_79_16')),
                                           "summer_79_16"=get(paste0('Ind_', CV, '_Calib_summer_79_16')),
                                           "automn_79_16"=get(paste0('Ind_', CV, '_Calib_automn_79_16')),
                                           "spring_79_16"=get(paste0('Ind_', CV, '_Calib_spring_79_16'))))
  assign(paste0('Ind_', CV, '_Proj'),list("winter_79_16"=get(paste0('Ind_', CV, '_Proj_winter_79_16')),
                                          "summer_79_16"=get(paste0('Ind_', CV, '_Proj_summer_79_16')),
                                          "automn_79_16"=get(paste0('Ind_', CV, '_Proj_automn_79_16')),
                                          "spring_79_16"=get(paste0('Ind_', CV, '_Proj_spring_79_16'))))
  ##### TAS
  print("TAS")
  assign(paste0("tas_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_IPSLMRbili_79_16_Paris
  Mod=tas_day_SAFRANdetbili_79_16_Paris
  CDFt=tas_day_CVchrono_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris
  
  nb_dim=ncol(Ref)^2
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef_Calib = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatRef_Proj = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    flatCDFt_Proj = flatten_array(CDFt[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    ## Bias correction
    Z = MRec_for_IPSLbis(Rc=flatRef_Calib,
                         Rp=flatRef_Proj,
                         Mc=flatMod_Calib,
                         Mp=flatMod_Proj,
                         Rp_from_CDFt=flatCDFt_Proj,
                         ratio.seq=rep(FALSE,nb_dim)) #not precipitation
    
    flatSpatialMRecCDFt_Calib = Z$Rch ## Correction in calibration period
    
    flatSpatialMRecCDFt_Proj = Z$Rph ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialMRecCDFt_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialMRecCDFt_Proj)
    
    eval(parse(text=paste0("tas_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("tas_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
  ##### PR
  print("PR")
  assign(paste0("pr_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_IPSLMRbili_79_16_Paris
  Mod=pr_day_SAFRANdetbili_79_16_Paris
  CDFt=pr_day_CVchrono_1dCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris
  
  nb_dim=ncol(Ref)^2
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef_Calib = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatRef_Proj = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    flatCDFt_Proj = flatten_array(CDFt[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    ## Bias correction
    Z = MRec_for_IPSLbis(Rc=flatRef_Calib,
                         Rp=flatRef_Proj,
                         Mc=flatMod_Calib,
                         Mp=flatMod_Proj,
                         Rp_from_CDFt=flatCDFt_Proj,
                         ratio.seq=rep(TRUE,nb_dim)) # precipitation
    
    flatSpatialMRecCDFt_Calib = Z$Rch ## Correction in calibration period
    
    flatSpatialMRecCDFt_Proj = Z$Rph ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialMRecCDFt_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialMRecCDFt_Proj)
    
    eval(parse(text=paste0("pr_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("pr_day_", CV, "_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
}


#### Save CVchrono
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/IPSLMRbili_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_SpatialMRecCDFt_IPSLMRbili_SAFRANdetbili_79_16_Paris.RData")




