# RESNET with Attention
Implementation of SE [1], BAM [2] and CBAM [3] Attention Modules

[1] Hu et al., Squeeze-and-Excitation Network (CVPR 2018 Oral), ILSVRC2017 winner
[2] Park et al., BAM: Bottleneck Attention Module, (BMVC 2018 Oral)
[3] Woo et al., CBAM: Convolutional Block Attention Module, ECCV 2018.

| Model         	| Variation         	| Top1 Error 	| Top5 Error 	| Params  	| GFlops 	|
|---------------	|-------------------	|------------	|------------	|---------	|--------	|
| Resnet34      	|                   	| 21.88      	| 5.98       	| 21.328M 	| 1.115  	|
| Resnet34-SE   	|                   	| 20.99      	| 5.58       	| 21.489M 	| 1.115  	|
| Resnet34-BAM  	| Channel-Attention 	| 21.76      	| 5.73       	| 21.339M 	| 1.115  	|
|               	| Spatial-Attention 	| 21.74      	| 5.91       	| 21.339M 	| 1.117  	|
|               	| Joint-Attention   	| 21.12      	| 5.37       	| 21.351M 	| 1.117  	|
| Resnet34-CBAM 	| Channel-Attention 	| 21.01      	| 5.14       	| 21.489M 	| 1.115  	|
|               	| Spatial-Attention 	| 21.21      	| 5.99       	| 21.489M 	| 1.115  	|
|               	| Joint-Attention   	| 20.50      	| 5.10       	| 210491M 	| 1.116  	|
| Resnet50      	|                   	| 21.05      	| 5.25       	| 23.705M 	| 1.147  	|
| Resnet50-SE   	|                   	| 20.34      	| 4.90       	| 26.236M 	| 1.152  	|
| Resnet34-BAM  	| Channel-Attention 	| 20.68      	| 5.03       	| 23.879M 	| 1.147  	|
|               	| Spatial-Attention 	| 20.41      	| 4.87       	| 23.889M 	| 1.149  	|
|               	| Joint-Attention   	| 20.03      	| 4.69       	| 24.063M 	| 1.156  	|
| Resnet34-CBAM 	| Channel-Attention 	| 20.03      	| 4.67       	| 26.236M 	| 1.146  	|
|               	| Spatial-Attention 	| 20.79      	| 4.96       	| 23.707M 	| 1.143  	|
|               	| Joint-Attention   	| 19.82      	| 4.76       	| 26.237M  	| 1.158  	|
