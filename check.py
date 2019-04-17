from glob import glob
from sys import argv
from os.path import exists
from os import system

path = argv[1]

folders = glob(path +'/*')
#folders = ['/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001392', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0002240', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0004297', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001396', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0002910', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0000952', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001394', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0031137', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0006561', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001395', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001397', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001080', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0012115', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0005264', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0012437', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001433', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0006707', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0012440', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001409', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001081', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0002612', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001402', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0002605', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0006579', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001404', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001413', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0002613', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0011040', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0012438', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001082', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0006562', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0012334', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0006565', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0031865', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001406', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001407', '/sc/orga/scratch/wangl35/mi_hpo_anno/HP:0001408']

def checkFolder(f):
    for fold in range(5):
	if not exists('%s/predictions-%d.csv.gz' %(f,fold)):
            return False
        if not exists('%s/validation-%d.csv.gz' %(f,fold)):
            return False
    return True

tc = 0
fc = 0
fcf = []
dns = []
for f in folders:
    if checkFolder(f):
        tc += 1
#	print f.split('/')[-1]
    else:
	fc += 1
        fcf.append(f)
        dns.append(f.split('/')[-1])
	print f.split('/')[-1]

print 'Finished: %d.' %tc
print 'Not done: %d...' %fc


if len(argv)==3 and  argv[2] == 'resub':
    print 'Resubmitting...'
    for fn,dn in zip(fcf,dns):
        system('sh run.sh %s %s' %(fn,dn))

