# awesome-dataset-distillation
A curated list of papers on data distillation, including data pruning and coreset-selection.




## Survey Paper
TBU

## Coreset Selection
|📝Title| 📄Paper| 💻Code|🏛Venue|💽 Dataset|
| :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----- |
|Data Pruning by Information Maximization|[🔗](https://openreview.net/forum?id=93XT0lKOct)|| ICLR'25|CIFAR-100, Imagenet-1K, BBH, MMLU, TYDIQA|
|Less is More: Efficient Time Series Dataset Condensation via Two-fold Modal Matching–Extended Version|[🔗](https://arxiv.org/abs/2410.20905)|[🔗](https://github.com/uestc-liuzq/STdistillation)|axriv'24|Weather, Traffic, Electricity, ETT
|Data Pruning via Moving-one-Sample-out|[🔗](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3abe23bf7e295b44369c24465d68987a-Abstract-Conference.html)|[🔗](https://github.com/hrtan/MoSo)|NeurIPS'24|CIFAR-100, Tiny-ImageNet, ImageNet-1K|
|D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning|[🔗](https://arxiv.org/abs/2310.07931)|[🔗](https://github.com/adymaharana/d2pruning)| ICLR'24|CIFAR-10, CIFAR-100, Imagenet-1K, ImDB, ANLI|
|Deep Learning on a Data Diet: Finding Important Examples Early in Training|[🔗](https://arxiv.org/abs/2107.07075)|[🔗](https://github.com/mansheej/data_diet)| ICLR'24|CIFAR-10, CIFAR-100, CINIC-10|
|Data pruning and neural scaling laws: fundamental limitations of score-based algorithms|[🔗](https://arxiv.org/abs/2302.06960)||TMLR'23|CIFAR-10, CIFAR-100
|Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy|[🔗](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ebb6bee50913ba7e1efeb91a1d47a002-Abstract-Conference.html)|[🔗](https://github.com/kaist-dmlab/Prune4Rel)|NeruIPS'23|CIFAR-10N, CIFAR-100N, WebVision, Clothing-1M, ImageNet-1K
|Efficient Coreset Selection with Cluster-based Methods|[🔗](https://dl.acm.org/doi/abs/10.1145/3580305.3599326)||KDD'23|Brazil,CovType,IMDB,IMDB-Large,MNIST|
|Adversarial Coreset Selection for Efficient Robust Training|[🔗](https://arxiv.org/abs/2209.05785)|[🔗](https://github.com/hmdolatabadi/ACS)|IJCV'23
|Dataset Pruning: Reducing Training Data by Examining Generalization Influence|[🔗](https://arxiv.org/abs/2205.09329)||ICRL'23|CIFAR-10, CIFAR-100, Tiny-ImageNet|
|DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning|[🔗](https://arxiv.org/abs/2204.08499)|[🔗](https://github.com/PatrickZH/DeepCore)|DEXA'23|MNIST, QMNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet and ImageNet.
|Coverage-centric Coreset Selection for High Pruning Rates|[🔗](https://arxiv.org/abs/2210.15809)|[🔗](https://github.com/haizhongzheng/Coverage-centric-coreset-selection)| ICLR'23|CIFAR-10, CIFAR-100, SVHN, ImDB, CINIC10, ImageNet ANLI|
|Moderate coreset: A universal method of data selection for real-world data-efficient deep learning|[🔗](https://openreview.net/forum?id=7D5EECbOaf9)|[🔗](https://github.com/tmllab/2023_ICLR_Moderate-DS)| ICLR'23|CIFAR-100, Tiny-ImageNet, ImageNet-1k|
|Probabilistic Bilevel Coreset Selection|[🔗](https://proceedings.mlr.press/v162/zhou22h.html)|[🔗](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection)|ICML'22|MNIST, CIFAR-10
|Beyond neural scaling laws: beating power law scaling via data pruning|[🔗](https://arxiv.org/abs/2206.14486)|[🔗](https://github.com/bsorsch/beyond-neural-scaling-laws)| NeurIPS'22|CIFAR-10, ImageNet, SVHN|
|RETRIEVE: Coreset Selection for Efficient and Robust Semi-Supervised Learning|[🔗](https://proceedings.neurips.cc/paper/2021/hash/793bc52a941b3951dfdb85fb04f9fd06-Abstract.html)|[🔗](https://github.com/decile-team/cords)|NeurIPS'21|CIFAR-10, SVHN, MNIST
|Coresets for time series clustering|[🔗](https://proceedings.neurips.cc/paper/2021/hash/c115ba9e04ab27fbbb664f932112246d-Abstract.html)||NerIPS'21|synthetic time-series data
|Glister: Generalization based data subset selection for efficient and robust learning|[🔗](https://arxiv.org/abs/2012.10630)|[🔗](https://github.com/dssresearch/GLISTER)|AAAI'21|MNIST, CIFAR-10|
|GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training|[🔗](https://arxiv.org/abs/2103.00123)|[🔗](https://github.com/krishnatejakk/GradMatch)| ICML'21|MNIST, CIFAR-10, SVHN, ImageNet-2012|
|Coresets for Data-efficient Training of Machine Learning Models|[🔗](https://arxiv.org/abs/1906.01827)|[🔗](https://github.com/baharanm/craig)| ICML'20|MNIST, CIFAR-10|
|Selection via Proxy: Efficient Data Selection for Deep Learning|[🔗](https://arxiv.org/abs/1812.05159)|[🔗](https://github.com/stanford-futuredata/selection-via-proxy)| ICLR'20|CIFAR-10, CIFAR-100, Imagenet, Amazon Review Full, Amazon Review Polarity|
|An Empirical Study of Example Forgetting during Deep Neural Network Learning|[🔗](https://arxiv.org/abs/2310.07931)|[🔗](https://github.com/adymaharana/d2pruning)| ICLR'19|MNIST, permuted MNIST, CIFAR-10|
|Training gaussian mixture models at scale via coresets|[🔗](https://www.jmlr.org/papers/v18/15-506.html)||JMLR'18|HIGGS, CSN, KDD, MSYP|
|Active Learning for Convolutional Neural Networks: A Core-Set Approach|[🔗](https://arxiv.org/abs/1708.00489)|[🔗](https://github.com/ozansener/active_learning_coreset)| ICLR'18|CIFAR-10, CIFAR-100, SVHN|
|Scalable Training of Mixture Models via Coresets|[🔗](https://proceedings.neurips.cc/paper/2011/hash/2b6d65b9a9445c4271ab9076ead5605a-Abstract.html)||NeurIPS'11|MNIST, Neural tetrode recordings,CSN cell phone accelerometer data|

## Coreset Selection for Continual Learning
|📝Title| 📄Paper| 💻Code|🏛Venue|💽 Dataset|
| :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----- |
|Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm|[🔗](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0251e494a7e75d59e06d37e646f46b7-Abstract-Conference.html)|[🔗](https://github.com/MingruiLiu-ML-Lab/Bilevel-Coreset-Selection-via-Regularization)|NeurIPS'23|CIFAR-100, MNIST, Tiny-ImageNet, Multiple Datasets, Split Food-101
|An efficient dataset condensation plugin and its application to continual learning|[🔗](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d5f34e7e70d80f5037ab16a48e2d186e-Abstract-Conference.html)|[🔗](https://github.com/EnnengYang/An-Efficient-Dataset-Condensation-Plugin)|NeurIPS'23|MNIST, CIFAR-10, CIFAR-100, TinyImageNet
|Regularizing Second-Order Influences for Continual Learning|[🔗](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Regularizing_Second-Order_Influences_for_Continual_Learning_CVPR_2023_paper.html)|[🔗](https://github.com/feifeiobama/InfluenceCL)|CVPR'23|CIFAR-10, CIFAR-100, ImageNet
|Probabilistic Bilevel Coreset Selection|[🔗](https://proceedings.mlr.press/v162/zhou22h.html)|[🔗](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection)|ICML'22|MNIST, CIFAR-10
|GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning|[🔗](https://openaccess.thecvf.com/content/CVPR2022/html/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.html)||CVPR'22|CIFAR-10, CIFAR-100, TinyImageNet
|Online Coreset Selection for Rehearsal-based Continual Learning|[🔗](https://arxiv.org/abs/2106.01085)|[🔗](https://github.com/jaehong31/OCS)|ICLR'22|MNIST, fashion-MNIST, NotMNIST,TrafficSign, SVHN|
|Coresets via Bilevel Optimization for Continual Learning and Streaming|[🔗](https://arxiv.org/abs/2006.03875)|[🔗](https://github.com/zalanborsos/bilevel_coresets)|NeurIPS'20|MNIST, CIFAR-10|
|Gradient based sample selection for online continual learning|[🔗](https://arxiv.org/abs/1903.08671)|[🔗](https://github.com/rahafaljundi/Gradient-based-Sample-Selection)|NeurIPS'19|



## Dataset Condensation/Distillation

TBU

## Resource
[Practical Coreset Constructions for Machine Learning](https://arxiv.org/abs/1703.06476)
