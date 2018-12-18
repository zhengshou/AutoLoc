import os
import numpy as np
import yaml

EXP_NAME='AN'
EXP_PATH='exp/'+EXP_NAME+'/'

def rewrite_cfg(tmp,tmpvalue):
	cmd = 'sed -E -i "s/'+tmp+': .*$/'+tmp+': '+str(tmpvalue)+'/g" "'+EXP_PATH+'config.yml"'
	#print cmd
	os.system(cmd)

def rewrite_cfg_str(tmp,tmpvalue):
	cmd = 'sed -E -i "s/'+tmp+': .*$/'+tmp+': \\"'+tmpvalue+'\\"/g" "'+EXP_PATH+'config.yml"'
	#print cmd
	os.system(cmd)

def rewrite_solver(tmp,tmpvalue):
	cmd = 'sed -E -i "s/'+tmp+': .*$/'+tmp+': '+str(tmpvalue)+'/g" "'+EXP_PATH+'solver.tpl"'
	#print cmd
	os.system(cmd)
	
MAX_ITER = 4706
num_process = 24

rewrite_cfg('MAX_ITER',MAX_ITER)
os.system('pwd')

# train on multiple videos
rewrite_cfg_str('DATA_FILE','att_unfused_onlytrain\/train.all.tsv')
rewrite_cfg_str('STAGE','train')
rewrite_solver('snapshot',MAX_ITER)
cmd1 = 'python tools/proc_net.py --phase train --dataset AN --expname '+EXP_NAME+' --rsltname '\
+'"tmp"'
os.system(cmd1)

# test & eval on all testing videos. prll mode.
rewrite_cfg_str('DATA_FILE','att_unfused_onlytrain\/val.all.tsv')
rewrite_cfg_str('STAGE','val')
rewrite_cfg('MAX_ITER',MAX_ITER)
cmd2 = 'python tools/proc_prll_net.py --phase test --dataset AN --expname '+EXP_NAME+' --rsltname '\
+'"test_rslt'+'_trainnms'+str(0.4)+'_fg'+str(0.1)+'_ioc'+str(0.7)\
+'_testnms'+str(0.4)+'"'+' --num_workers '+str(num_process)\
+' --pretrained autoloc_iter_'+str(MAX_ITER)+'.caffemodel'
os.system(cmd2)
# single nms eval
cmd3 = 'python tools/eval_metrics.py --phase test --dataset AN --expname '+EXP_NAME+' --rsltname '\
+'"test_rslt'+'_trainnms'+str(0.4)+'_fg'+str(0.1)+'_ioc'+str(0.7)\
+'_testnms'+str(0.4)+'"'+' --num_workers '+str(num_process)
os.system(cmd3)