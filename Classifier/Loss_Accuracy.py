import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.dirname(__file__)
parse_log = os.path.join(file_path,"parse_log.sh")
modelname = 'Classifier'
var = 'var9'
file_path = os.path.join(file_path,"..",modelname)

var_log = var.replace('(','\(')
var_log = var_log.replace(')','\)')
command = "bash {0} {1} {2} {3}".format(parse_log,file_path,var_log,"nohup.out")
print command

os.system(command)

fig = plt.figure(figsize=(9,6))
#plt.subplots_adjust(hspace = 0.5)

ax1 = fig.add_subplot(2,1,1)
ax1.set_ylim(0,1.1)
ax1.set_title("Train")
train_pd = pd.read_csv(os.path.join(file_path,var,"data","train","train.csv"),
                        sep=',',
                        index_col='Iteration')
train_pd.plot(ax=ax1, grid=True)

ax2 = fig.add_subplot(2,1,2)
ax2.set_title("Test")

val_pd = pd.read_csv(os.path.join(file_path,var,"data","val","val.csv"),
                       sep=',',
                       index_col='Iteration')

val_pd.plot(ax=ax2, grid=True)

namefig = "LossAccuracy_{0}_1.png".format(modelname)
plt.savefig(os.path.join(file_path,var,namefig))
