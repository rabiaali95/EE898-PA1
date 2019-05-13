# RESNET with Attention
Implementation of SE, BAM and CBAM attention Modules
| Model         | Variation          | Top1 Error |
| ------------- | ------------------ |----------- |
| Resnet34      |                    | 21.88      | 
| Resnet34-SE   |                    | 20.99      |
| Resnet34-BAM  | Channel-Attention  | 21.76      |
|               | Spatial-Attention  | 21.74      |
|               | Joint-Attention    | 21.12      |             
| Resnet34-CBAM | Channel-Attention  | 21.01      |
|               | Spatial-Attention  | 20.21      |
|               | Joint-Attention    | 21.50      |              
| Resnet50      |                    | 21.05      |
| Resnet50-SE   |                    | 21.34      |
| Resnet50-BAM  | Channel-Attention  | 20.68      |
|               | Spatial-Attention  | 20.41      |
|               | Joint-Attention    | 20.03      |
| Resnet50-CBAM | Channel-Attention  | 20.03      |
|               | Spatial-Attention  | 20.79      |
|               | Joint-Attention    | 19.82      |
