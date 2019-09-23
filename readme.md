# ODE Playground

## Parameters

parameters in `mdmain.py`

```python
group = parser.add_argument_group('Learning  parameters')
parser.add_argument("-folder", default=None,help = "Folder to save and load")
group.add_argument("-epochs", type=int, default=1000, help="Number of epoches to train")
group.add_argument("-batch", type=int, default=1000, help="Batch size of train")
group.add_argument("-cuda", type=int, default=-1, help="If use GPU")
group.add_argument("-lr", type=float, default=0.001, help="Learning rate")
group.add_argument("-save", action='store_true',help="If save or not")
group.add_argument("-load", action='store_true' ,help="If load or not")
group.add_argument("-save_period", type=int, default=10, help="Steps to save in train")
group.add_argument("-K",type=float, default=300, help="Temperature")
group.add_argument("-double", action='store_true',help="Use double or single")

group = parser.add_argument_group('Network parameters')
group.add_argument("-hdim", type=int, default=128, help="Hidden dimension of mlps")
group.add_argument("-numFlow", type=int, default=1, help="Number of flows")
group.add_argument("-nlayers", type=int, default=8, help="Number of mlps in rnvp")
group.add_argument("-nmlp", type=int, default=2, help="Number of layers of mlps")

group = parser.add_argument_group('Target parameters')
group.add_argument("-dataset", default="./alanine-dipeptide-3x250ns-heavy-atom-positions.npz", help="Path to training data")
group.add_argument("-baseDataSet",default=None,help="Known CV data base")
group.add_argument("-miBatch",type=int,default=5, help="Batch size when evaluate MI")
group.add_argument("-miSample",type=int,default=1000, help="Sample when evaluate MI")
group.add_argument("-loadrange",default=3,type=int,help="Array nos to load from npz file")
group.add_argument("-smile", default="CC(=O)NC(C)C(=O)NC",help="smile expression")
group.add_argument("-scaling",default=10,type=float,help = "Scaling factor of npz data, default is for nm to ångströms")
group.add_argument("-fixx",default=0,type=float,help="Offset of x axis")
group.add_argument("-fixy",default=2.3222,type=float,help="Offset of y axis")
group.add_argument("-fixz",default=0,type=float,help="Offset of z axis")

group = parser.add_argument_group("Analysis parameters")
group.add_argument("-interpolation", default=0, type=int, help="Mode except 0,1 to interpolation")

```

## Examples

```python
python ./mdmain.py -dataset ./CLN025.npz -smile NCCCCCCOCCCONCCCCCCOCCCONCCCOOCONCCCCCONCCCCOOCONCCOCCONCCONCCOCCONCCCCNCCCCCCCOCOONCCCCCCOCC -loadrange 1 -cuda 1 -batch 200 -epoch 300 

python ./mdmain.py -load -folder ./opt/Model_NCCCCCCOCCCONCCCCCCOCCCONCCCOOCONCCCCCONCCCCOOCONCCOCCONCCONCCOCCONCCCCNCCCCCCCOCOONCCCCCCOCC_Batch_1000_T_300_depthLevel_1_l32_M10_H512 -dataset ./CLN025.npz -loadrange 1 -fixy 0.0 -smile NCCCCCCOCCCONCCCCCCOCCCONCCCOOCONCCCCCONCCCCOOCONCCOCCONCCONCCOCCONCCCCNCCCCCCCOCOONCCCCCCOCC

python ./mdmain.py -cuda 3 -batch 200 -epoch 500 -fixy 2.3222

python ./mdmain.py -load -folder -fixy 2.3222 ./opt/Model_CCONCCCONC_T_300_depthLevel_1_l8_M2_H128
```

