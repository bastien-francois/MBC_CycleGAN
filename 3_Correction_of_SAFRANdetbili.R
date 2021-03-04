#### Bias correction of SAFRANdetbili for T2 and PR, CVchrono and CVunif for
#### 1. QQ
#### 2. R2D2
#### 3. dOTC


#### 1. Correction with QQ
rm(list=ls())
gc()
library(fields)

#### On JZ
source("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Code/function_1dQQ.R")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/Temporal_and_Random_indices_1979_2016.RData")
###################################################################################################################
###################################################################################################################

CVtype = c("CVchrono","CVunif")
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
  assign(paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_SAFRAN_79_16_Paris
  Mod=tas_day_SAFRANdetbili_79_16_Paris
  
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
  
  #### PR ####
  print("PR")
  assign(paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  Mod=pr_day_SAFRANdetbili_79_16_Paris
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
}

#### Save CVchrono
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris.RData")

#### Save CVunif
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVunif")
save(list=c("tas_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris.RData")



#### 2. Correction with R2D2
rm(list=ls())
gc()
library(devtools) ### to use with r/3.6 on JZ + faire module load gcc
devtools::install_github("thaos/C3PO")
library(C3PO)

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

#### On JZ
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/Temporal_and_Random_indices_1979_2016.RData")
###################################################################################################################
###################################################################################################################

#icond avec  4 variables centré sur un cadran composé de quatres carrés
# (7,7); (21,7); (7,21); (21,21)
ref_var= c(6*28+7,6*28+21,20*28+7,20*28+21)

search_param = 7
keep_param = 5

CVtype = c("CVchrono","CVunif")
for(CV in CVtype){
  print(CV)
  load(paste0("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/", CV, "/tas_pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris.RData"))
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
  assign(paste0("tas_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_SAFRAN_79_16_Paris
  BC1d=get(paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"))
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatBC1d_Calib = flatten_array(BC1d[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatBC1d_Proj = flatten_array(BC1d[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    flatSpatialR2D2_Calib = r2d2(refdata = flatRef, 
                                 bc1d = flatBC1d_Calib,
                                 icond = c(ref_var),
                                 lag_search = search_param,
                                 lag_keep = keep_param)
    
    flatSpatialR2D2_Proj = r2d2(refdata = flatRef, 
                                bc1d = flatBC1d_Proj,
                                icond = c(ref_var),
                                lag_search = search_param,
                                lag_keep = keep_param)
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialR2D2_Calib$r2d2_bc)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialR2D2_Proj$r2d2_bc)
    
    eval(parse(text=paste0("tas_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("tas_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
  
  
  
  #### PR ####
  print("PR")
  assign(paste0("pr_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  BC1d=get(paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"))
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatBC1d_Calib = flatten_array(BC1d[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatBC1d_Proj = flatten_array(BC1d[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    flatSpatialR2D2_Calib = r2d2(refdata = flatRef, 
                                 bc1d = flatBC1d_Calib,
                                 icond = c(ref_var),
                                 lag_search = search_param,
                                 lag_keep = keep_param)
    
    flatSpatialR2D2_Proj = r2d2(refdata = flatRef, 
                                bc1d = flatBC1d_Proj,
                                icond = c(ref_var),
                                lag_search = search_param,
                                lag_keep = keep_param)
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialR2D2_Calib$r2d2_bc)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialR2D2_Proj$r2d2_bc)
    
    eval(parse(text=paste0("pr_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("pr_day_", CV, "_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
}

#### Save CVchrono
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris.RData")
#
#### Save CVunif
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVunif")
save(list=c("tas_day_CVunif_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVunif_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_SpatialR2D2_SAFRAN_SAFRANdetbili_79_16_Paris.RData")




#### 3. Correction with dOTC
rm(list=ls())
gc()
library(devtools)
# devtools::install_github("yrobink/SBCK/R/SBCK")
library(SBCK)



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

draw_in_bins<-function(data_array,bin_width){ #data_array of dim LON x LAT x TIME
  res=array(NaN,dim=dim(data_array))
  for(i in 1:dim(data_array)[1]){
    print("draw bins")
    for(j in 1:dim(data_array)[2]){
      for(t in 1:dim(data_array)[3]){
        res[i,j,t]<-runif(1,min=(data_array[i,j,t]-bin_width/2),max=(data_array[i,j,t]+bin_width/2))
      }
    }
  }
  return(res)
}


#### On JZ
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.RData")

load("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/Temporal_and_Random_indices_1979_2016.RData")
###################################################################################################################
###################################################################################################################


####
CVtype = c("CVchrono","CVunif")
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
  bin_size = 0.1
  print(bin_size)
  print("TAS")
  assign(paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_SAFRAN_79_16_Paris
  Mod=tas_day_SAFRANdetbili_79_16_Paris
  
  nb_dim=ncol(Ref)^2
  bin_width = rep(bin_size, nb_dim)
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    ## Bias correction
    ## Step 1 : construction of the class dOTC
    dotc = SBCK::dOTC$new(bin_width)
    ## Step 2 : Fit the bias correction model
    dotc$fit( flatRef , flatMod_Calib , flatMod_Proj )
    ## Step 3 : perform the bias correction, Z is a list containing
    ## corrections
    Z = dotc$predict(flatMod_Proj,flatMod_Calib)
    
    flatSpatialdOTC_Calib = Z$Z0 ## Correction in calibration period
    
    flatSpatialdOTC_Proj = Z$Z1 ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialdOTC_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialdOTC_Proj)
    
    final_res_Calib = draw_in_bins(tmp_res_Calib,bin_width = bin_size)
    final_res_Proj = draw_in_bins(tmp_res_Proj,bin_width = bin_size)
    
    eval(parse(text=paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=final_res_Calib")))
    eval(parse(text=paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=final_res_Proj")))
  }
  
  
  
  #### PR ####
  print("PR")
  assign(paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  Mod=pr_day_SAFRANdetbili_79_16_Paris
  
  bin_size = min(Ref[Ref>0])
  print(bin_size)
  nb_dim=ncol(Ref)^2
  bin_width = rep(bin_size, nb_dim)
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    ## Bias correction
    ## Step 1 : construction of the class dOTC
    dotc = SBCK::dOTC$new(bin_width)
    ## Step 2 : Fit the bias correction model
    dotc$fit( flatRef , flatMod_Calib , flatMod_Proj )
    ## Step 3 : perform the bias correction, Z is a list containing
    ## corrections
    Z = dotc$predict(flatMod_Proj,flatMod_Calib)
    
    flatSpatialdOTC_Calib = Z$Z0 ## Correction in calibration period
    
    flatSpatialdOTC_Proj = Z$Z1 ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialdOTC_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialdOTC_Proj)
    
    final_res_Calib = draw_in_bins(tmp_res_Calib,bin_width = bin_size)
    final_res_Proj = draw_in_bins(tmp_res_Proj,bin_width = bin_size)
    
    eval(parse(text=paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=final_res_Calib")))
    eval(parse(text=paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=final_res_Proj")))
  }
}


#### Save CVchrono
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris.RData")
#
#### Save CVunif
setwd("/gpfswork/rech/eal/urq13cl/CycleGAN/Article/Code_and_Data/Data/MBC/SAFRAN_SAFRANdetbili/CVunif")
save(list=c("tas_day_CVunif_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVunif_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_SpatialdOTC_SAFRAN_SAFRANdetbili_79_16_Paris.RData")

