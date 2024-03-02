
library("readxl")
library("tidyverse")
library('ggplot2')

path_dir = 'D:/OneDrive - Chalmers/Git/een210_applied_digital_health/data'
directory = dir(path_dir)

path_label = 'D:/OneDrive - Chalmers/label.xlsx'
label_data <- read_xlsx(path_label)
label_data <- data.frame(label_data)

files = unique(label_data['filename'])

for (i in 1:nrow(files) ){
  dat = read.csv(file.path(path_dir,files[i,]))
  label_ind = label_data[label_data[,'filename']==files[i,],]
  
  for (j in 1:nrow(label_ind)){
    str_idx = label_ind[j,'str_idx']
    end_idx = label_ind[j,'end_idx']
    lab = label_ind[j,'label']
    dat[str_idx:end_idx,'label'] = lab
    
  }
  #write.csv(dat,file.path(path_dir,'label',gsub('.csv','_label.csv',files[i,])))
                      
}

# dat <- data.frame(dat)
dat[,'timestamp'] = as.POSIXct(dat[,'timestamp'])
dat[,'acceleration_x']
ggplot(dat)+
  geom_point(aes(x=timestamp,y=acceleration_x), color='blue')+geom_line(aes(x=timestamp,y=acceleration_x), color='blue')+
  geom_point(aes(x=timestamp,y=acceleration_y), color='red')+geom_line(aes(x=timestamp,y=acceleration_y), color='red')+
  geom_point(aes(x=timestamp,y=acceleration_z), color='green')+ geom_line(aes(x=timestamp,y=acceleration_z), color='green')


# temp <- data.frame(read_xlsx('D:/OneDrive - Chalmers/Git/een210_applied_digital_health/data/fall_back_data_20240129_163134_label.xlsx'))
# write.csv(temp, 'D:/OneDrive - Chalmers/Git/een210_applied_digital_health/data/label/fall_back_data_20240129_163134_label.csv')
