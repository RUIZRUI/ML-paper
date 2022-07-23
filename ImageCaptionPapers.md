# Image Captioning Papers
Image captioning task aims to generate complete and accurate description according to the image content, and involves the multimodal transformation from visual understanding to language generation. Inspired by Machine Translation, the Google team designed an end-to-end model which composed of a visual encoder CNN and a language decoder RNN in 2015. Subsequently, many algorithms and networks are applied to improve the quality of image captioning model, including attention mechanism, Reinforcement Learning, Generative Adversarial Network and so on. Meanwhile, Image Captioning is gradually refined into some sub tasks, including Controllable Image Captioning (CIC), Image Change Captioning (ICC), Image Paragraph Captioning (IPC), etc. This repository will summarize the development of research on Image Captioning in recent years.

## Contributor
Contributed by [Yuxin Liu](), [You Li](https://github.com/a75732410), [Baokun Qi](), [Yuanqiu Liu](https://github.com/Liu-Yuanqiu), [Zhaolong Hu](https://github.com/EkkoWho), [Xiang Zheng](), [Shujing Guo](https://github.com/Liu-Yuanqiu), [Zhiyue Wang]().

Thanks for supports and organization from our advisers [Han Liu](), [Xiaotong Zhang](), [Hong Yu]()!

## Content
[Image Captioning Papers](#image-captioning-papers)

[Visual Question Answering Papers](#visual-question-answering-papers)

[Subtasks about Image Caption](#subtasks-about-image-captioning)

[Datasets and Metrics about Image Captioning](#datasets-and-metrics-about-image-captioning)

## Image Captioning Papers

* 2021
  + AAAI
    - **Dual-Level Collaborative Transformer for Image Captioning.** *Yunpeng Luo, Jiayi Ji, Xiaoshuai Sun, Liujuan Cao, Yongjian Wu, Feiyue Huang, ChiaWen Lin, Rongrong Ji* [[pdf](https://arxiv.org/pdf/2101.06462.pdf)] [[code](https://github.com/luo3300612/image-captioning-DLCT.git)]
    - **Memory-Augmented Image Captioning.** *Zhengcong Fei* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16220/16027)] [[code](https://github.com/feizc/MAIC.git)]
    - **VIVO: Visual Vocabulary Pre-Training for Novel Object Captioning.** *Xiaowei Hu, Xi Yin, Kevin Lin, Lei Zhang, Jianfeng Gao, Lijuan Wang, Zicheng Liu* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16249/16056)] 
    - **Attention Beam An Image Captioning Approach (Student Abstract).** *Anubhav Shrimal, Tanmoy Chakraborty* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17940/17745)] 
    - **Consensus Graph Representation Learning for Better Grounded Image Captioning.** *Wenqiao Zhang, Haochen Shi, Siliang Tang, Jun Xiao, Qiang Yu, Yueting Zhuang* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16452/16259)] 
    - **Image Captioning with Context-Aware Auxiliary Guidance.** *Zeliang Song, Xiaofei Zhou, Zhendong Mao, Jianlong Tan* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16361/16168)] 
    - **Improving Image Captioning by Leveraging Intra- and Inter-layer Global Representation in Transformer Network.** *Jiayi Ji, Yunpeng Luo, Xiaoshuai Sun, Fuhai Chen, Gen Luo, Yongjian Wu, Yue Gao, Rongrong Ji* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16258/16065)] 
    - **Object Relation Attention for Image Paragraph Captioning.** *Zhengcong Fei* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16219/16026)] [[code](https://github.com/feizc/PNAIC.git)]
    - **Text Embedding Bank for Detailed Image Paragraph Captioning.** *Arjun Gupta, Zengming Shen, Thomas S. Huang* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17892/17697)] [[code](https://github.com/arjung128/image-paragraph-captioning.git)]
    - **Confidence-aware Non-repetitive Multimodal Transformers for TextCaps.** *, Renda Bao, Qi Wu, Si Liu* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16389)] 
  + ACMM
    - **A Picture is Worth a Thousand Words： A Unified System for Diverse Captions and Rich Images Generation.** *Yupan Huang, Bei Liu, Jianlong Fu, Yutong Lu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3478561)] [[code](https://github.com/researchmm/generate-it.git)]
    - **Direction Relation Transformer for Image Captioning.** *Zeliang Song, Xiaofei Zhou, Linhua Dong, Jianlong Tan, Li Guo* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475607)] 
    - **Distributed Attention for Grounded Image Captioning.** *Nenglun Chen, Xingjia Pan, Runnan Chen, Lei Yang, Zhiwen Lin, Yuqiang Ren, Haolei Yuan, Xiaowei Guo, Feiyue Huang, Wenping Wang* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475354)] 
    - **Dual Graph Convolutional Networks with Transformer and Curriculum Learning for Image Captioning.** *Xinzhi Dong, Chengjiang Long, Wenju Xu, Chunxia Xiao* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475439)] [[code](https://github.com/Unbear430/DGCN-for-image-captioning.git)]
    - **Group-based Distinctive Image Captioning with Memory Attention.** *Jiuniu Wang, Wenjia Xu, Qingzhong Wang, Antoni B. Chan* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475215)] 
    - **Image Quality Caption with Attentive and Recurrent Semantic Attractor Network.** *Jiuniu Wang, Wenjia Xu, Qingzhong Wang, Antoni B. Chan* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475215)] 
    - **Question-controlled Text-aware Image Captioning.** *Anwen Hu, Shizhe Chen, Qin Jin* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475452)] [[code](https://github.com/HAWLYQ/Qc-TextCap.git)]
    - **Semi-Autoregressive Image Captioning.** *Xu Yan, Zhengcong Fei, Zekang Li, Shuhui Wang, Qingming Huang, Qi Tian* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475179)] [[code](https://github.com/feizc/SAIC.git)]
    - **Similar Scenes arouse Similar Emotions： Parallel Data Augmentation for Stylized Image Captioning.** *Guodun Li, Yuchen Zhai, Zehao Lin, Yin Zhang* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475662)] 
    - **Triangle-Reward Reinforcement Learning：Visual-Linguistic Semantic Alignment for Image Captioning.** *Weizhi Nie, Jiesi Li, Ning Xu, An-An Liu, Xuanya Li, Yongdong Zhang* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3474085.3475604)] 
  + CVPR
    - **Scan2Cap: Context-Aware Dense Captioning in RGB-D Scans.** *Zhenyu Chen, Ali Gholami, Matthias Niessner, Angel X. Chang* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.pdf)] [[code](https://github.com/daveredrum/Scan2Cap.git)]
    - **RSTNet: Captioning With Adaptive Attention on Visual and Non-Visual Words.** *Xuying Zhang, Xiaoshuai Sun, Yunpeng Luo, Jiayi Ji, Yiyi Zhou, Yongjian Wu, Feiyue Huang, Rongrong Ji* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RSTNet_Captioning_With_Adaptive_Attention_on_Visual_and_Non-Visual_Words_CVPR_2021_paper.pdf)] [[code](https://github.com/zhangxuying1004/RSTNet.git)]
    - **Human-Like Controllable Image Captioning With Verb-Specific Semantic Roles.** *Long Chen, Zhihong Jiang, Jun Xiao, Wei Liu* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf)] [[code](https://github.com/mad-red/VSR-guided-CIC.git)]
    - **Image Change Captioning by Learning From an Auxiliary Task.** *Mehrdad Hosseinzadeh, Yang Wang* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.pdf)] 
    - **FAIEr: Fidelity and Adequacy Ensured Image Caption Evaluation.** *Sijin Wang, Ziwei Yao, Ruiping Wang, Zhongqin Wu, Xilin Chen* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_FAIEr_Fidelity_and_Adequacy_Ensured_Image_Caption_Evaluation_CVPR_2021_paper.pdf)] 
    - **Improving OCR-Based Image Captioning by Incorporating Geometrical Relationship.** *Jing Wang, Jinhui Tang, Mingkun Yang, Xiang Bai, Jiebo Luo* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_OCR-Based_Image_Captioning_by_Incorporating_Geometrical_Relationship_CVPR_2021_paper.pdf)] 
    - **Towards Accurate Text-Based Image Captioning With Content Diversity Exploration.** *Guanghui Xu, Shuaicheng Niu, Mingkui Tan, Yucheng Luo, Qing Du, Qi Wu* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Towards_Accurate_Text-Based_Image_Captioning_With_Content_Diversity_Exploration_CVPR_2021_paper.pdf)] [[code](https://github.com/guanghuixu/AnchorCaptioner.git)]
  + CoRR
    - **Macroscopic Control of Text Generation for Image Captioning.** *Zhangzi Zhu,Tianlei Wang,Hong Qu* [[pdf](https://arxiv.org/abs/2101.08000)] 
  + IEEE
    - **Self-Distillation for Few-Shot Image Captioning.** *Xianyu Chen; Ming Jiang; Qi Zhao* [[pdf](https://ieeexplore.ieee.org/document/9423232)] 
  + ACL
    - **Semantic Relation-aware Difference Representation Learning for Change Captioning.** *Yunbin Tu, Tingting Yao, Liang Li, Jiedong Lou, Shengxiang Gao, Zhengtao Yu, Chenggang Yan* [[pdf](https://aclanthology.org/2021.findings-acl.6/)] 
  + EMNLP
    - **CLIPScore: A Reference-free Evaluation Metric for Image Captioning.** *Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi* [[pdf](https://aclanthology.org/2021.emnlp-main.595/)] 
  + ICML
    - **Learning Transferable Visual Models From Natural Language Supervision.** *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever* [[pdf](http://proceedings.mlr.press/v139/radford21a.html)] 
  + IJCAI
    - **Perturb, Predict & Paraphrase: Semi-Supervised Learning using Noisy Student for Image Captioning.** *Arjit Jain, Pranay Reddy Samala, Preethi Jyothi, Deepak Mittal, Maneesh Kumar Singh* [[pdf](https://www.ijcai.org/proceedings/2021/0105.pdf)] 
    - **TCIC: Theme Concepts Learning Cross Language and Vision for Image Captioning.** *Zhihao Fan, Zhongyu Wei, Siyuan Wang, Ruize Wang, Zejun Li, Haijun Shan, Xuanjing Huang* [[pdf](https://www.ijcai.org/proceedings/2021/0091.pdf)] 
    - **Dependent Multi-Task Learning with Causal Intervention for Image Captioning.** *Wenqing Chen, Jidong Tian, Caoyun Fan, Hao He, Yaohui Jin* [[pdf](https://www.ijcai.org/proceedings/2021/0312.pdf)] 

* 2020
  + AAAI
    - **Improving Image Captioning by Leveraging Intra- and Inter-layer Global Representation in Transformer Network.** *Jiayi Ji, Yunpeng Luo, Xiaoshuai Sun, Fuhai Chen, Gen Luo, Yongjian Wu, Yue Gao, Rongrong Ji* [[pdf](https://arxiv.org/pdf/2012.07061.pdf)] 
    - **Feature Deformation Meta-Networks in Image Captioning of Novel Objects.** *Tingjia Cao, Ke Han, Xiaomei Wang, Lin Ma, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6620/6474)] 
    - **Interactive Dual Generative Adversarial Networks for Image Captioning.** *Junhao Liu, Kai Wang, Chunpu Xu, Zhou Zhao, Ruifeng Xu, Ying Shen, Min Yang* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6826/6680)] 
    - **Joint Commonsense and Relation Reasoning for Image and Video Captioning.** *Jingyi Hou, Xinxiao Wu, Xiaoxun Zhang, Yayun Qi, Yunde Jia, Jiebo Luo* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6731/6585)] 
    - **Learning Long- and Short-Term User Literal-Preference with Multimodal Hierarchical Transformer Network for Personalized Image Caption.** *Wei Zhang, Yue Ying, Pan Lu, Hongyuan Zha* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6503/6359)] 
    - **MemCap Memorizing Style Knowledge for Image Captioning.** *Wentian Zhao, Xinxiao Wu, Xiaoxun Zhang* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6998/6852)] [[code](https://github.com/LuoweiZhou/VLP.git)]
    - **Reinforcing an Image Caption Generator Using Off-Line Human Feedback.** *Paul Hongsuck Seo, Piyush Sharma, Tomer Levinboim, Bohyung Han, Radu Soricut* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/5655/5511)] 
    - **Show, Recall, and Tell Image Captioning with Recall Mechanism.** *Li Wang, Zechen Bai, Yonghua Zhang, Hongtao Lu* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6898/6752)] 
    - **Unified Vision-Language Pre-Training for Image Captioning and VQA.** *Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/7005/6859)] [[code](https://github.com/LuoweiZhou/VLP.git)]
  + ACCV
    - **Image Captioning through Image Transformer.** *Sen He, Wentong Liao, Hamed R. Tavakoli, Michael Yang, Bodo Rosenhahn, Nicolas Pugeault* [[pdf](https://arxiv.org/pdf/2004.14231.pdf)] [[code](https://github.com/wtliao/ImageTransformer.git)]
  + ACMM
    - **Attacking Image Captioning Towards Accuracy-Preserving Target Words Removal.** *Jiayi Ji, Xiaoshuai Sun, Yiyi Zhou, Rongrong Ji, Fuhai Chen, Jianzhuang Liu, Qi Tian* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3414009)] 
    - **Bridging the Gap between Vision and Language Domains for Improved Image Captioning.** *Fenglin Liu, Xian Wu, Shen Ge, Xiaoyu Zhang, Wei Fan, Yuexian Zou* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3414004)] 
    - **Cap2Seg： Inferring Semantic and Spatial Context from Captions for Zero-Shot Image Segmentation.** *Guiyu Tian, Shuai Wang, Jie Feng, Li Zhou, Yadong Mu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413990)] 
    - **Hierarchical Scene Graph Encoder-Decoder for Image Paragraph Captioning.** *Xu Yang, Chongyang Gao, Hanwang Zhang, Jianfei Cai* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413859)] 
    - **ICECAP：Information Concentrated Entity-aware Image Captioning.** *Anwen Hu, Shizhe Chen, Qin Jin* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413576)] 
    - **Improving Intra- and Inter-Modality Visual Relation for Image Captioning.** *Yong Wang, Wenkai Zhang, Qing Liu, Zhengyuan Zhang, Xin Gao, Xian Sun* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413877)] 
    - **Iterative Back Modification for Faster Image Captioning.** *Zhengcong Fei* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413901)] 
    - **MemCap: Memorizing Style Knowledge for Image Captioning.** *Wentian Zhao, Xinxiao Wu, Xiaoxun Zhang* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/6998/6852)] [[code](https://github.com/entalent/MemCap.git)]
    - **Multimodal Attention with Image Text Spatial Relationship for OCR-Based Image Captioning.** *Jing Wang, Jinhui Tang, Jiebo Luo* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413753)] [[code](https://github.com/TownWilliam/mma_sr.git)]
    - **Structural Semantic Adversarial Active Learning for Image Captioning.** *Beichen Zhang, Liang Li, Li Su, Shuhui Wang, Jincan Deng, Zheng-Jun Zha, Qingming Huang* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413885)] 
  + CVPR
    - **Meshed-Memory Transformer for Image Captioning.** *Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, Rita Cucchiara* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cornia_Meshed-Memory_Transformer_for_Image_Captioning_CVPR_2020_paper.pdf)] [[code](https://github.com/aimagelab/meshed-memory-transformer.git)]
    - **Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs.** *Shizhe Chen, Qin Jin, Peng Wang, Qi Wu* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9156638)] [[code](https://github.com/cshizhe/asg2cap.git)]
    - **Meshed-Memory Transformer for Image Captioning.** *Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, Rita Cucchiara* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9157222)] [[code](https://github.com/aimagelab/meshed-memory-transformer.git)]
    - **Normalized and Geometry-Aware Self-Attention Network for Image Captioning.** *Longteng Guo, Jing Liu, Xinxin Zhu, Peng Yao, Shichen Lu, Hanqing Lu* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Normalized_and_Geometry-Aware_Self-Attention_Network_for_Image_Captioning_CVPR_2020_paper.pdf)] 
    - **X-Linear Attention Networks for Image Captioning.** *Yingwei Pan, Ting Yao, Yehao Li, Tao Mei* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pan_X-Linear_Attention_Networks_for_Image_Captioning_CVPR_2020_paper.pdf)] [[code](https://github.com/JDAI-CV/image-captioning.git)]
    - **Show, Edit and Tell: A Framework for Editing Image Captions.** *Fawaz Sammani, Luke Melas-Kyriazi* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sammani_Show_Edit_and_Tell_A_Framework_for_Editing_Image_Captions_CVPR_2020_paper.pdf)] [[code](https://github.com/fawazsammani/show-edit-tell.git)]
    - **Transform and Tell: Entity-Aware News Image Captioning.** *Alasdair Tran, Alexander Patrick Mathews, Lexing Xie* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tran_Transform_and_Tell_Entity-Aware_News_Image_Captioning_CVPR_2020_paper.pdf)] [[code](https://github.com/alasdairtran/transform-and-tell.git)]
    - **More Grounded Image Captioning by Distilling Image-Text Matching Model.** *Yuanen Zhou, Meng Wang, Daqing Liu, Zhenzhen Hu, Hanwang Zhang* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_More_Grounded_Image_Captioning_by_Distilling_Image-Text_Matching_Model_CVPR_2020_paper.pdf)] [[code](https://github.com/YuanEZhou/Grounded-Image-Captioning.git)]
  + ECCV
    - **Length-Controllable Image Captioning.** *Chaorui Deng, Ning Ding, Mingkui Tan, Qi Wu* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58601-0.pdf)] [[code](https://github.com/bearcatt/LaBERT.git)]
    - **Captioning Images Taken by People Who Are Blind.** *Danna Gurari, Yinan Zhao, Meng Zhang, Nilavra Bhattacharya* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58520-4.pdf)] 
    - **TextCaps: A Dataset for Image Captioning with Reading Comprehension.** *Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, Amanpreet Singh* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58536-5.pdf)] 
    - **Compare and Reweight: Distinctive Image Captioning Using Similar Images Sets.** *Jiuniu Wang, Wenjia Xu, , Qingzhong Wang, Antoni B. Chan* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58452-8.pdf)] 
    - **Towards Unique and Informative Captioning of Images.** *Zeyu Wang, Berthy Feng, Karthik Narasimhan, Olga Russakovsky* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58571-6.pdf)] [[code](https://github.com/princetonvisualai/SPICE-U.git)]
    - **Comprehensive Image Captioning Via Scene Graph Decomposition.** *Yiwu Zhong, Liwei Wang, Jianshu Chen, Dong Yu, Yin Li* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58568-6.pdf)] [[code](https://github.com/YiwuZhong/Sub-GC.git)]
    - **Connecting Vision and Language with Localized Narratives.** *Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, Vittorio Ferrari* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_38)] 
    - **Finding It at Another Side: A Viewpoint-Adapted Matching Encoder for Change Captioning.** *Xiangxi Shi, Xu Yang, Jiuxiang Gu, Shafiq Joty, Jianfei Ca* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_34)] 
    - **Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks.** *Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_8)][[code](https://github.com/microsoft/Oscar)]
  + COLING
    - **Language-Driven Region Pointer Advancement for Controllable Image Captioning.** *Annika Lindh, Robert Ross, John Kelleher* [[pdf](https://aclanthology.org/2020.coling-main.174/)] 
  + IJCAI
    - **Human Consensus-Oriented Image Captioning.** *Ziwei Wang, Zi Huang, Yadan Luo* [[pdf](https://www.ijcai.org/proceedings/2020/0092.pdf)] 
    - **Recurrent Relational Memory Network for Unsupervised Image Captioning.** *Dan Guo, Yang Wang, Peipei Song, Meng Wang* [[pdf](https://www.ijcai.org/proceedings/2020/0128.pdf)] 
    - **Non-Autoregressive Image Captioning with Counterfactuals-Critical Multi-Agent Learning.** *Longteng Guo, Jing Liu, Xinxin Zhu, Xingjian He, Jie Jiang, Hanqing Lu* [[pdf](https://www.ijcai.org/proceedings/2020/0107.pdf)] 
    - **Recurrent Relational Memory Network for Unsupervised Image Captioning.** *Dan Guo, Yang Wang, Peipei Song, Meng Wang* [[pdf](https://www.ijcai.org/proceedings/2020/0128.pdf)]
  + NeurIPS
    - **Diverse Image Captioning with Context-Object Split Latent Spaces..** *Shweta Mahajan, Stefan Roth* [[pdf](https://proceedings.neurips.cc/paper/2020/file/24bea84d52e6a1f8025e313c2ffff50a-Paper.pdf)] 
    - **RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning.** *Riccardo Del Chiaro, Bartlomiej Twardowski, Andrew D. Bagdanov, Joost van de Weijer* [[pdf](https://proceedings.neurips.cc/paper/2020/file/c2964caac096f26db222cb325aa267cb-Paper.pdf)] 

* 2019
  + AAAI
    - **Connecting Language to Images A Progressive Attention-Guided Network for Simultaneous Image Captioning and Language Grounding.** *Lingyun Song, Jun Liu, Buyue Qian, Yihe Chen* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4916/4789)] 
    - **Deliberate Attention Networks for Image Captioning.** *Lianli Gao, Kaixuan Fan, Jingkuan Song, Xianglong Liu, Xing Xu, Heng Tao Shen* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4845/4718)] 
    - **Hierarchical Attention Network for Image Captioning.** *Weixuan Wang, Zhihong Chen, Haifeng Hu* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4924/4797)] 
    - **Improving Image Captioning with Conditional Generative Adversarial Nets.** *Chen Chen, Shuai Mu, Wanpeng Xiao, Zexiong Ye, Liesi Wu, Qi Ju* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4823/4696)] 
    - **Meta Learning for Image Captioning.** *Nannan Li, Zhenzhong Chen, Shan Liu* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4883/4756)] 
  + ACMM
    - **Aligning Linguistic Words and Visual Semantic Units for Image Captioning.** *Longteng Guo, Jing Liu, Jinhui Tang, Jiangwei Li, Wei Luo, Hanqing Lu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3343031.3350943)] [[code](https://github.com/ltguo19/VSUA-Captioning.git)]
    - **Generating Captions for Images of Ancient Artworks.** *Shurong Sheng, Marie-Francine Moens* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3343031.3350972)] 
    - **Towards Increased Accessibility of Meme Images with the Help of Rich Face Emotion Captions.** *K. R. Prajwal, C. V. Jawahar, Ponnurangam Kumaraguru* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3343031.3350939)] 
    - **Unpaired Cross-lingual Image Caption Generation with Self-Supervised Rewards.** *Yuqing Song, Shizhe Chen, Yida Zhao, Qin Jin* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3343031.3350996)] 
  + CVPR
    - **Good News, Everyone! Context Driven Entity-Aware Captioning for News Images.** *Ali Furkan Biten, Lluís Gómez, Marçal Rusiñol, Dimosthenis Karatzas* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953244)] [[code](https://github.com/furkanbiten/GoodNews.git)]
    - **Fast, Diverse and Accurate Image Captioning Guided by Part-Of-Speech.** *Aditya Deshpande, Jyoti Aneja, Liwei Wang, Alexander G. Schwing, David A. Forsyth* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953300)] 
    - **Adversarial Semantic Alignment for Improved Image Captions.** *Pierre L. Dognin, Igor Melnyk, Youssef Mroueh, Jerret Ross, Tom Sercu* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954255)] 
    - **Unsupervised Image Captioning.** *Yang Feng, Lin Ma, Wei Liu, Jiebo Luo* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953817)] [[code](https://github.com/fengyang0317/unsupervised_captioning.git)]
    - **Self-Critical N-Step Training for Image Captioning.** *Junlong Gao, Shiqi Wang, Shanshe Wang, Siwei Ma, Wen Gao* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953880)] 
    - **MSCap: Multi-Style Image Captioning With Unpaired Stylized Text.** *Longteng Guo, Jing Liu, Peng Yao, Jiangwei Li, Hanqing Lu* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953861)] 
    - **Intention Oriented Image Captions With Guiding Objects.** *Yue Zheng, Yali Li, Shengjin Wang* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954294)] [[code](https://github.com/Inlinebool/cgo-pytorch.git)]
    - **Pointing Novel Objects in Image Captioning.** *Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, Tao Mei* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953517)] 
    - **Look Back and Predict Forward in Image Captioning.** *Yu Qin, Jiajun Du, Yonghua Zhang, Hongtao Lu* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953361)] 
    - **Engaging Image Captioning via Personality.** *Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954085)] 
    - **Describing Like Humans: On Diversity in Image Captioning.** *Qingzhong Wang, Antoni B. Chan* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954161)] [[code](https://github.com/qingzwang/DiversityMetrics.git)]
    - **Exact Adversarial Attack to Image Captioning via Structured Output Learning With Latent Variables.** *Yan Xu, Baoyuan Wu, Fumin Shen, Yanbo Fan, Yong Zhang, Heng Tao Shen, Wei Liu* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954337)] [[code](https://github.com/wubaoyuan/adversarial-attack-to-caption.git)]
    - **Auto-Encoding Scene Graphs for Image Captioning.** *Xu Yang, Kaihua Tang, Hanwang Zhang, Jianfei Cai* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953305)] [[code](https://github.com/yangxuntu/SGAE.git)]
    - **Show,ControlandTell: A Framework for GeneratingControllableandGrounded Captions.** *Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Cornia_Show_Control_and_Tell_A_Framework_for_Generating_Controllable_and_CVPR_2019_paper.html)] 
    - **Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning.** *Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, In So Kweon* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Dense_Relational_Captioning_Triple-Stream_Networks_for_Relationship-Based_Captioning_CVPR_2019_paper.html)] 
  + ICCV
    - **Entangled Transformer for Image Captioning.** *Guang Li, Linchao Zhu, Ping Liu, Yi Yang* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9008532)] 
    - **Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment.** *Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010970)] 
    - **Attention on Attention for Image Captioning.** *Lun Huang, Wenmin Wang, Jie Chen, XiaoYong Wei* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9008770)] [[code](https://github.com/husthuaan/AoANet.git)]
    - **Entangled Transformer for Image Captioning.** *Guang Li, Linchao Zhu, Ping Liu, Yi Yang* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9008532)] 
    - **Exploring Overall Contextual Information for Image Captioning in Human-Like Cognitive Style.** *Hongwei Ge, Zehang Yan, Kai Zhang, Mingde Zhao, Liang Sun* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010625)] 
    - **Generating Diverse and Descriptive Image Captions Using Visual Paraphrases.** *Lixin Liu, Jiajun Tang, Xiaojun Wan, Zongming Guo* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010984)] [[code](https://github.com/pkuliu/visual-paraphrases-captioning.git)]
    - **Hierarchy Parsing for Image Captioning.** *Ting Yao, Yingwei Pan, Yehao Li, Tao Mei* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010299)] 
    - **Human Attention in Image Captioning: Dataset and Analysis.** *Sen He, Hamed Rezazadegan Tavakoli, Ali Borji, Nicolas Pugeault* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010621)] 
    - **Joint Optimization for Cooperative Image Captioning.** *Gilad Vered, Gal Oren, Yuval Atzmon, Gal Chechik* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010870)] [[code](https://github.com/vgilad/CooperativeImageCaptioning.git)]
    - **Learning to Caption Images Through a Lifetime by Asking Questions.** *Tingke Shen, Amlan Kar, Sanja Fidler* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9009050)] [[code](https://github.com/shenkev/Caption-Lifetime-by-Asking-Questions.git)]
    - **Learning to Collocate Neural Modules for Image Captioning.** *Xu Yang, Hanwang Zhang, Jianfei Cai* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010688)] 
    - **Reflective Decoding Network for Image Captioning.** *Lei Ke, Wenjie Pei, Ruiyu Li, Xiaoyong Shen, YuWing Tai_x000D_* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9009778)] 
    - **Sequential Latent Spaces for Modeling the Intention During Diverse Image Captioning.** *Jyoti Aneja, Harsh Agrawal, Dhruv Batra, Alexander Schwing* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010960)] 
    - **Towards Unsupervised Image Captioning With Shared Multimodal Embeddings.** *Iro Laina, Christian Rupprecht, Nassir Navab* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010396)] 
    - **Unpaired Image Captioning Via Scene Graph Alignments.** *Jiuxiang Gu, Shafiq Joty, Jianfei Cai, Handong Zhao, Xu Yang, Gang Wang* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010917)] 
    - **Robust Change Captioning.** *Dong Huk Park, Trevor Darrell,  Anna Rohrbach* [[pdf](https://ieeexplore.ieee.org/document/9008523)] 
  + NeurIPS
    - **Image Captioning: Transforming Objects into Words.** *Simao Herdade, Armin Kappeler, Kofi Boakye, Joao Soares* [[pdf](https://proceedings.neurips.cc/paper/2019/file/680390c55bbd9ce416d1d69a9ab4760d-Paper.pdf)] [[code](https://github.com/yahoo/object_relation_transformer.git)]
    - **Hidden State Guidance: Improving Image Captioning Using an Image Conditioned Autoencoder.** *Jialin Wu, Raymond J. Mooney* [[pdf](https://arxiv.org/pdf/1910.14208.pdf)] 
    - **Can adversarial training learn image captioning.** *Jean-Benoit Delbrouck* [[pdf](https://arxiv.org/pdf/1910.14609v1.pdf)] 
    - **Adaptively Aligned Image Captioning via Adaptive Attention Time.** *Lun Huang, Wenmin Wang, Yaxian Xia, Jie Chen* [[pdf](https://proceedings.neurips.cc/paper/2019/file/fecc3a370a23d13b1cf91ac3c1e1ca92-Paper.pdf)] 
    - **Variational Structured Semantic Inference for Diverse Image Captioning.** *Fuhai Chen, Rongrong Ji, Jiayi Ji, Xiaoshuai Sun, Baochang Zhang, Xuri Ge, Yongjian Wu, Feiyue Huang, Yan Wang* [[pdf](https://proceedings.neurips.cc/paper/2019/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf)] 
  + IEEE
    - **CaptionNet: Automatic End-to-End Siamese Difference Captioning Model With Attention.** *Ariyo Oluwasanmi, Muhammad Umar Aftab,Eatedal Alabdulkreem,  Bulbula Kumeda, Edward Y. Baagyere, Zhiquang Qin* [[pdf](https://ieeexplore.ieee.org/document/8776601)] 
  + EMNLP/IJCNLP
    - **TIGEr: Text-to-Image Grounding for Image Caption Evaluation.** *Ming Jiang, Qiuyuan Huang, Lei Zhang, Xin Wang, Pengchuan Zhang, Zhe Gan, Jana Diesner, Jianfeng Gao* [[pdf](https://aclanthology.org/D19-1220/)] 
  + IJCAI
    - **HorNet: A Hierarchical Offshoot Recurrent Network for Improving Person Re-ID via Image Captioning.** *Shiyang Yan, Jun Xu, Yuai Liu, Lin Xu* [[pdf](https://www.ijcai.org/proceedings/2019/0742.pdf)] 
    - **Swell-and-Shrink: Decomposing Image Captioning by Transformation and Summarization.** *Hanzhang Wang, Hanli Wang, Kaisheng Xu* [[pdf](https://www.ijcai.org/proceedings/2019/0726.pdf)] 
    - **Image Captioning with Compositional Neural Module Networks.** *Junjiao Tian, Jean Oh* [[pdf](https://www.ijcai.org/proceedings/2019/0496.pdf)] 
    - **Exploring and Distilling Cross-Modal Information for Image Captioning.** *Fenglin Liu, Xuancheng Ren, Yuanxin Liu, Kai Lei, Xu Sun* [[pdf](https://www.ijcai.org/proceedings/2019/0708.pdf)] 

* 2018
  + ACMM
    - **Context-Aware Visual Policy Network for Sequence-Level Image Captioning.** *Daqing Liu, Zheng-Jun Zha, Hanwang Zhang, Yongdong Zhang, Feng Wu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3240508.3240632)] [[code](https://github.com/daqingliu/CAVP.git)]
    - **Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering.** *Xuanyi Dong, Linchao Zhu, De Zhang, Yi Yang, Fei Wu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3240508.3240527)] [[code](https://github.com/amirunpri2018/FPAIT.git)]
    - **Look Deeper See Richer Depth-aware Image Paragraph Captioning.** *Ziwei Wang, Yadan Luo, Yang Li, Zi Huang, Hongzhi Yin* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3240508.3240583)] 
  + CVPR
    - **Convolutional Image Captioning.** *Jyoti Aneja, Aditya Deshpande, Alexander G. Schwing* [[pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Aneja_Convolutional_Image_Captioning_CVPR_2018_paper.pdf)] [[code](https://github.com/aditya12agd5/convcap.git)]
    - **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.** *Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578734)] [[code](https://github.com/peteanderson80/bottom-up-attention.git)]
    - **GroupCap: Group-Based Image Captioning With Structured Relevance and Diversity Constraints.** *Fuhai Chen, Rongrong Ji, Xiaoshuai Sun, Yongjian Wu, Jinsong Su* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578244)] 
    - **SemStyle: Learning to Generate Stylised Image Captions Using Unaligned Text.** *Alexander Mathews, Lexing Xie, Xuming He* [[pdf](https://openaccess.thecvf.com/content_cvpr_2018/html/Mathews_SemStyle_Learning_to_CVPR_2018_paper.html)] 
  + ECCV
    - **Boosted Attention: Leveraging Human Attention for Image Captioning.** *Shi Chen, Qi Zhao* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01252-6.pdf)] 
    - **Exploring Visual Relationship for Image Captioning.** *Ting Yao, Yingwei Pan, Yehao Li, Tao Mei* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01264-9.pdf)] 
    - **NNEval Neural Network Based Evaluation Metric for Image Captioning.** *Naeha Sharif, Lyndon White, Mohammed Bennamoun, Syed Afaq Ali Shah* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01237-3.pdf)] 
    - **Recurrent Fusion Network for Image Captioning.** *Wenhao Jiang, Lin Ma, Yu-Gang Jiang, Wei Liu, Tong Zhang* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01216-8.pdf)] [[code](https://github.com/cswhjiang/Recurrent_Fusion_Network.git)]
    - **Rethinking the Form of Latent States in Image Captioning.** *Bo Dai, Deming Ye, Dahua Lin* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01228-1.pdf)] [[code](https://github.com/doubledaibo/2dcaption_eccv2018.git)]
    - **Show, Tell and Discriminate Image Captioning by Self-retrieval with Partially Labeled Data.** *Xihui Liu, Hongsheng Li, Jing Shao, Dapeng Chen, Xiaogang Wang* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01267-0.pdf)] 
    - **Unpaired Image Captioning by Language Pivoting.** *Jiuxiang Gu, Shafiq R. Joty, Jianfei Cai, Gang Wang* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01246-5.pdf)] [[code](https://github.com/gujiuxiang/unpaired_image_captioning.git)]
    - **“Factual” or “Emotional” Stylized Image Captioning with Adaptive Learning and Attention.** *Tianlang Chen, Zhongping Zhang, Quanzeng You, Chen Fang, Zhaowen Wang, Hailin Jin, Jiebo Luo* [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-01249-6.pdf)] 
  + ICCV
    - **An Empirical Study of Language CNN for Image Captioning.** *Jiuxiang Gul, Gang Wang, Jianfei Cai, Tsuhan Chen* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237400)] [[code](https://github.com/showkeyjar/chinese_im2text.pytorch.git)]
    - **Areas of Attention for Image Captioning.** *Marco Pedersoli, Thomas Lucas, Cordelia Schmid, Jakob Verbeek* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237402)] 
    - **Boosting Image Captioning with Attributes.** *Ting Yao, Yingwei Pan, Yehao Li, Zhaofan Qiu, Tao Mei* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237786)] 
    - **Improved Image Captioning Via Policy Gradient Optimization of SPIDEr.** *Siqi Liu, Zhenhai Zhu, Ning Ye, Sergio Guadarrama, Kevin Murphy* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237362)] [[code](https://github.com/peteanderson80/SPICE.git)]
    - **Paying Attention to Descriptions Generated by Image Captioning Models.** *Hamed R. Tavakoliy, Rakshith Shetty, Ali Borji, Jorma Laaksonen* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237534)] [[code](https://github.com/rakshithShetty/captionGAN.git)]
    - **Show, Adapt and Tell Adversarial Training of Cross-Domain Image Captioner.** *TsengHung Chen, YuanHong Liao, ChingYao Chuang, WanTing Hsu, Jianlong Fu, Min Sun* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237326)] [[code](https://github.com/tsenghungchen/show-adapt-and-tell.git)]
  + IJCAI
    - **A Multi-task Learning Approach for Image Captioning.** *Wei Zhao, Benyou Wang, Jianbo Ye, Min Yang, Zhou Zhao, Ruotian Luo, Yu Qiao* [[pdf](https://www.ijcai.org/proceedings/2018/0168.pdf)] 
    - **Multi-Level Policy and Reward Reinforcement Learning for Image Captioning.** *Anan Liu, Ning Xu, Hanwang Zhang, Weizhi Nie, Yuting Su, Yongdong Zhang* [[pdf](https://www.ijcai.org/proceedings/2018/0114.pdf)] 
    - **Show and Tell More Topic-Oriented Multi-Sentence Image Captioning.** *Yuzhao Mao, Chang Zhou, Xiaojie Wang, Ruifan Li* [[pdf](https://www.ijcai.org/proceedings/2018/0592.pdf)] 
    - **Show, Observe and Tell Attribute-driven Attention Model for Image Captioning.** *Hui Chen, Guiguang Ding, Zijia Lin, Sicheng Zhao, Jungong Han* [[pdf](https://www.ijcai.org/proceedings/2018/0084.pdf)] 
  + EMNLP
    - **Training for Diversity in Image Paragraph Captioning.** *Luke Melas-Kyriazi, Alexander Rush, George Han* [[pdf](https://aclanthology.org/D18-1084/)] 
  + ACM
    - **Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering.** *Xuanyi Dong，Linchao Zhu，De Zhang，Yi Yang，Fei Wu* [[pdf](https://dl.acm.org/doi/10.1145/3240508.3240527)] 
  + NeurIPS
    - **A Neural Compositional Paradigm for Image Captioning.** *Bo Dai, Sanja Fidler, Dahua Lin* [[pdf](https://proceedings.neurips.cc/paper/2018/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf)] 
    - **Partially-Supervised Image Captioning.** *Peter Anderson, Stephen Gould, Mark Johnson* [[pdf](https://proceedings.neurips.cc/paper/2018/file/d2ed45a52bc0edfa11c2064e9edee8bf-Paper.pdf)] 

* 2017
  + ACMM
    - **Fluency-Guided Cross-Lingual Image Captioning.** *Weiyu Lan, Xirong Li, Jianfeng Dong* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3123266.3123366)] [[code](https://github.com/weiyuk/fluent-cap.git)]
    - **Image Caption with Synchronous Cross-Attention.** *Yue Wang, Jinlai Liu, Xiaojie Wang* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3126686.3126714)] [[code](https://github.com/sulabhkatiyar/IC_SCA.git)]
    - **StructCap Structured Semantic Embedding for Image Captioning.** *Fuhai Chen, Rongrong Ji, Jinsong Su, Yongjian Wu, Yunsheng Wu* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3123266.3123275)] 
    - **Watch What You Just Said Image Captioning with Text-Conditional Attention.** *Luowei Zhou, Chenliang Xu, Parker A. Koch, Jason J. Corso* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3126686.3126717)] [[code](https://github.com/LuoweiZhou/e2e-gLSTM-sc.git)]
  + CVPR
    - **Bidirectional Beam Search: Forward-Backward Inference in Neural Sequence Models for Fill-in-the-Blank Image Captioning.** *Qing Sun, Stefan Lee, Dhruv Batra* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8100246)] 
    - **Captioning Images with Diverse Objects.** *Subhashini Venugopalan, Lisa Anne Hendricks, Marcus Rohrbach, Raymond J. Mooney, Trevor Darrell, Kate Saenko* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099613)] [[code](https://github.com/willT97/Zero-shot-Image-Captioner.git)]
    - **Deep Reinforcement Learning-Based Image Captioning with Embedding Reward.** *Zhou Ren, Xiaoyu Wang, Ning Zhang, Xutao Lv, Li-Jia Li* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099611)] [[code](https://github.com/bpavankalyan/deep_reinforcement_learning-based_Image_captioning.git)]
    - **Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects.** *Ting Yao, Yingwei Pan, Yehao Li, Tao Mei* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8100042)] 
    - **Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning.** *Jiasen Lu, Caiming Xiong, Devi Parikh, Richard Socher* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099828)] [[code](https://github.com/jiasenlu/AdaptiveAttention.git)]
    - **SCA-CNN: Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning.** *Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, Wei Liu, Tat-Seng Chua* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8100150)] [[code](https://github.com/zjuchenlong/sca-cnn.cvpr17.git)]
    - **Self-Critical Sequence Training for Image Captioning.** *Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, Vaibhava Goel* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099614)] [[code](https://github.com/ruotianluo/ImageCaptioning.pytorch.git)]
    - **StyleNet: Generating Attractive Visual Captions with Styles.** *Chuang Gan; Zhe Gan; Xiaodong He; Jianfeng Gao; Li Deng* [[pdf](https://ieeexplore.ieee.org/document/8099591)] 
    - **Attend to You: Personalized Image Captioning with Context Sequence Memory Networks.** *Cesc Chunseong Park; Byeongchang Kim; Gunhee Kim* [[pdf](https://ieeexplore.ieee.org/document/8100164)] 
    - **A Hierarchical Approach for Generating Descriptive Image Paragraphs.** *Jonathan Krause; Justin Johnson; Ranjay Krishna; Li Fei-Fei* [[pdf](https://ieeexplore.ieee.org/document/8099839)] 
  + IJCAI
    - **MAT: A Multimodal Attentive Translator for Image Captioning.** *Chang Liu, Fuchun Sun, Changhu Wang, Feng Wang, Alan Yuille* [[pdf](https://www.ijcai.org/proceedings/2017/0563.pdf)] 

* 2016
  + AAAI
    - **SentiCap: Generating Image Descriptions with Sentiments.** *Alexander Patrick Mathews, Lexing Xie, Xuming He* [[pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12501)] 
  + CVPR
    - **Rich Image Captioning in the Wild.** *Kenneth Tran, Xiaodong He, Lei Zhang, Jian Sun* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7789551)] 
  + ECCV
    - **SPICE: Semantic Propositional Image Caption Evaluation.** *Peter Anderson, Basura Fernando, Mark Johnson, Stephen Gould* [[pdf](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_24)] 
  + IJCAI
    - **Diverse Image Captioning Via GroupTalk.** *Zhuhao Wang, Fei Wu, Weiming Lu, Jun Xiao, Xi Li, Zitong Zhang, Yueting Zhuang* [[pdf](https://www.ijcai.org/Proceedings/16/Papers/420.pdf)] 

* 2015
  + CVPR
    - **From captions to visual concepts and back.** *Hao Fang, Saurabh Gupta, Forrest N. Iandola, Rupesh Kumar Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, Geoffrey Zweig* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298754)] [[code](https://github.com/s-gupta/visual-concepts.git)]

## Visual Question Answering Papers

* 2021
  + AAAI
    - **Regularizing Attention Networks for Anomaly Detection in Visual Question Answering.** *Doyup Lee, Yeongjae Cheon, Wook-Shin Han:* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16279)] 
  + CVPR
    - **Predicting Human Scanpaths in Visual Question Answering.** *Xianyu Chen, Ming Jiang, Qi Zhao* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Predicting_Human_Scanpaths_in_Visual_Question_Answering_CVPR_2021_paper.html)] 
    - **Found a Reason for me? Weakly-supervised Grounded Visual Question Answering using Capsules.** *Aisha Urooj Khan, Hilde Kuehne, Kevin Duarte, Chuang Gan, Niels da Vitoria Lobo, Mubarak Shah* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/html/Urooj_Found_a_Reason_for_me_Weakly-supervised_Grounded_Visual_Question_Answering_CVPR_2021_paper.html)] [[code](https://github.com/aurooj/WeakGroundedVQA_Capsules)]
    - **Separating Skills and Concepts for Novel Visual Question Answering.** *Spencer Whitehead, Hui Wu, Heng Ji, Rogério Feris, Kate Saenko* [[pdf](https://openaccess.thecvf.com/content/CVPR2021/html/Whitehead_Separating_Skills_and_Concepts_for_Novel_Visual_Question_Answering_CVPR_2021_paper.html)] [[code](https://github.com/SpencerWhitehead/novelvqa)]
  + IJCAI
    - **Chop Chop BERT: Visual Question Answering by Chopping VisualBERT's Heads.** *Chenyu Gao, Qi Zhu, Peng Wang, Qi Wu* [[pdf](https://www.ijcai.org/proceedings/2021/92)] 
  + ACMM
    - **Focal and Composed Vision-semantic Modeling for Visual Question Answering.** *Yudong Han, Yangyang Guo, Jianhua Yin, Meng Liu, Yupeng Hu, Liqiang Nie* [[pdf](https://dl.acm.org/doi/10.1145/3474085.3475609)] 
    - **X-GGM: Graph Generative Modeling for Out-of-distribution Generalization in Visual Question Answering.** *Jingjing Jiang, Ziyi Liu, Yifan Liu, Zhixiong Nan, Nanning Zheng* [[pdf](https://dl.acm.org/doi/10.1145/3474085.3475350)] [[code](https://github.com/jingjing12110/x-ggm)]
    - **From Superficial to Deep: Language Bias driven Curriculum Learning for Visual Question Answering.** *Mingrui Lao, Yanming Guo, Yu Liu, Wei Chen, Nan Pu, Michael S. Lew* [[pdf](https://dl.acm.org/doi/10.1145/3474085.3475492)] 
    - **Towards Reasoning Ability in Scene Text Visual Question Answering.** *Qingqing Wang, Liqiang Xiao, Yue Lu, Yaohui Jin, Hao He* [[pdf](https://dl.acm.org/doi/10.1145/3474085.3475390)] 

* 2020
  + AAAI
    - **Re-Attention for Visual Question Answering.** *Wenya Guo, Ying Zhang, Xiaoping Wu, Jufeng Yang, Xiangrui Cai, Xiaojie Yuan* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16279)] 
    - **Multi-Question Learning for Visual Question Answering.** *Chenyi Lei, Lei Wu, Dong Liu, Zhao Li, Guoxin Wang, Haihong Tang, Houqiang Li* [[pdf](https://ojs.aaai.org//index.php/AAAI/article/view/6794)] 
  + CVPR
    - **On the General Value of Evidence, and Bilingual Scene-Text Visual Question Answering.** *Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, Liangwei Wang* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_On_the_General_Value_of_Evidence_and_Bilingual_Scene-Text_Visual_CVPR_2020_paper.html)] 
    - **Counterfactual Samples Synthesizing for Robust Visual Question Answering.** *Long Chen, Xin Yan, Jun Xiao, Hanwang Zhang, Shiliang Pu, Yueting Zhuang* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.html)] [[code](https://github.com/yanxinzju/CSS-VQA)]
    - **In Defense of Grid Features for Visual Question Answering.** *Huaizu Jiang, Ishan Misra, Marcus Rohrbach, Erik G. Learned-Miller, Xinlei Chen* [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_In_Defense_of_Grid_Features_for_Visual_Question_Answering_CVPR_2020_paper.html)] [[code](https://github.com/facebookresearch/grid-feats-vqa)]
  + ECCV
    - **Visual Question Answering on Image Sets.** *Ankan Bansal, Yuting Zhang, Rama Chellappa* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_4)] 
    - **VQA-LOL: Visual Question Answering Under the Lens of Logic.** *Tejas Gokhale, Pratyay Banerjee, Chitta Baral, Yezhou Yang* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_23)] 
    - **Reducing Language Biases in Visual Question Answering with Visually-Grounded Question Encoder.** *Gouthaman KV, Anurag Mittal* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_2)] 
    - **A Competence-Aware Curriculum for Visual Concepts Learning via Question Answering.** *Qing Li, Siyuan Huang, Yining Hong, Song-Chun Zhu* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_9)] 
    - **Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering.** *Ruixue Tang, Chao Ma, Wei Emma Zhang, Qi Wu, Xiaokang Yang* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_26)] [[code](https://github.com/zaynmi/seada-vqa)]
    - **TRRNet: Tiered Relation Reasoning for Compositional Visual Question Answering.** *Xiaofeng Yang, Guosheng Lin, Fengmao Lv, Fayao Liu* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_25)] 
  + IJCAI
    - **Overcoming Language Priors with Self-supervised Learning for Visual Question Answering.** *Xi Zhu, Zhendong Mao, Chunxiao Liu, Peng Zhang, Bin Wang, Yongdong Zhang* [[pdf](https://www.ijcai.org/proceedings/2020/151)] [[code](https://github.com/CrossmodalGroup/SSL-VQA)]
    - **Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering.** *Zihao Zhu, Jing Yu, Yujing Wang, Yajing Sun, Yue Hu, Qi Wu* [[pdf](https://www.ijcai.org/proceedings/2020/153)] 
  + ACMM
    - **Boosting Visual Question Answering with Context-aware Knowledge Aggregation.** *Guohao Li, Xin Wang, Wenwu Zhu* [[pdf](https://dl.acm.org/doi/10.1145/3394171.3413943)] 
    - **Cascade Reasoning Network for Text-based Visual Question Answering.** *Fen Liu, Guanghui Xu, Qi Wu, Qing Du, Wei Jia, Mingkui Tan* [[pdf](https://dl.acm.org/doi/10.1145/3394171.3413924)] 
    - **Medical Visual Question Answering via Conditional Reasoning.** *Li-Ming Zhan, Bo Liu, Lu Fan, Jiaxin Chen, Xiao-Ming Wu* [[pdf](https://dl.acm.org/doi/10.1145/3394171.3413761)] 
    - **K-armed Bandit based Multi-Modal Network Architecture Search for Visual Question Answering.** *Yiyi Zhou, Rongrong Ji, Xiaoshuai Sun, Gen Luo, Xiaopeng Hong, Jinsong Su, Xinghao Ding, Ling Shao* [[pdf](https://dl.acm.org/doi/10.1145/3394171.3413998)] 

* 2019
  + AAAI
    - **BLOCK: Bilinear Superdiagonal Fusion for Visual Question Answering and Visual Relationship Detection.** *Hedi Ben-younes, Rémi Cadène, Nicolas Thome, Matthieu Cord* [[pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4818)] 
    - **KVQA: Knowledge-Aware Visual Question Answering.** *Sanket Shah, Anand Mishra, Naganand Yadati, Partha Pratim Talukdar* [[pdf]()] 
    - **Differential Networks for Visual Question Answering.** ** [[pdf]()] 
    - **Dynamic Capsule Attention for Visual Question Answering.** *Yiyi Zhou, Rongrong Ji, Jinsong Su, Xiaoshuai Sun, Weiqiu Chen* [[pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4970)] 
  + CVPR
    - **Visual Question Answering as Reading Comprehension.** *Hui Li, Peng Wang, Chunhua Shen, Anton van den Hengel* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Visual_Question_Answering_as_Reading_Comprehension_CVPR_2019_paper.html)] 
    - **MUREL: Multimodal Relational Reasoning for Visual Question Answering.** *Rémi Cadène, Hedi Ben-younes, Matthieu Cord, Nicolas Thome* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Cadene_MUREL_Multimodal_Relational_Reasoning_for_Visual_Question_Answering_CVPR_2019_paper.html)] [[code](https://github.com/Cadene/murel.bootstrap.pytorch)]
    - **Dynamic Fusion With Intra- and Inter-Modality Attention Flow for Visual Question Answering.** *Peng Gao, Zhengkai Jiang, Haoxuan You, Pan Lu, Steven C. H. Hoi, Xiaogang Wang, Hongsheng Li* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Dynamic_Fusion_With_Intra-_and_Inter-Modality_Attention_Flow_for_Visual_CVPR_2019_paper.html)] 
    - **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.** *Drew A. Hudson, Christopher D. Manning* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)] [[code](https://github.com/stanfordnlp/mac-network)]
    - **Explicit Bias Discovery in Visual Question Answering Models.** *Varun Manjunatha, Nirat Saini, Larry S. Davis* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Manjunatha_Explicit_Bias_Discovery_in_Visual_Question_Answering_Models_CVPR_2019_paper.html)] 
    - **OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge.** *Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.html)] 
    - **Transfer Learning via Unsupervised Task Discovery for Visual Question Answering.** *Hyeonwoo Noh, Taehoon Kim, Jonghwan Mun, Bohyung Han* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Noh_Transfer_Learning_via_Unsupervised_Task_Discovery_for_Visual_Question_Answering_CVPR_2019_paper.html)] [[code](https://github.com/HyeonwooNoh/vqa_task_discovery)]
    - **Cycle-Consistency for Robust Visual Question Answering.** *Meet Shah, Xinlei Chen, Marcus Rohrbach, Devi Parikh* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Shah_Cycle-Consistency_for_Robust_Visual_Question_Answering_CVPR_2019_paper.html)] 
    - **Answer Them All! Toward Universal Visual Question Answering Models.** *Robik Shrestha, Kushal Kafle, Christopher Kanan* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Shrestha_Answer_Them_All_Toward_Universal_Visual_Question_Answering_Models_CVPR_2019_paper.html)] [[code](https://github.com/erobic/ramen)]
    - **Deep Modular Co-Attention Networks for Visual Question Answering.** *Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, Qi Tian* [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html)] [[code](https://github.com/MILVLG/mcan-vqa)]
  + ICCV
    - **Scene Text Visual Question Answering.** *Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez i Bigorda, Marçal Rusiñol, C. V. Jawahar, Ernest Valveny, Dimosthenis Karatzas* [[pdf](https://ieeexplore.ieee.org/document/9011031)] [[code](https://github.com/shailzajolly/ICDARVQA)]
    - **Compact Trilinear Interaction for Visual Question Answering.** *Tuong Do, Huy Tran, Thanh-Toan Do, Erman Tjiputra, Quang D. Tran* [[pdf](https://ieeexplore.ieee.org/document/9010363)] [[code](https://github.com/aioz-ai/ICCV19_VQA-CTI)]
    - **Multi-Modality Latent Interaction Network for Visual Question Answering.** *Peng Gao, Haoxuan You, Zhanpeng Zhang, Xiaogang Wang, Hongsheng Li* [[pdf](https://ieeexplore.ieee.org/document/9010837)] 
    - **Relation-Aware Graph Attention Network for Visual Question Answering.** *Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu* [[pdf](https://ieeexplore.ieee.org/document/9010056)] 
    - **SegEQA: Video Segmentation Based Visual Attention for Embodied Question Answering.** *Haonan Luo, Guosheng Lin, Zichuan Liu, Fayao Liu, Zhenmin Tang, Yazhou Yao* [[pdf](https://ieeexplore.ieee.org/document/9009527)] 
  + IJCAI
    - **Densely Connected Attention Flow for Visual Question Answering.** *Fei Liu, Jing Liu, Zhiwei Fang, Richang Hong, Hanqing Lu* [[pdf](https://www.ijcai.org/proceedings/2019/122)] 
  + ACMM
    - **Erasing-based Attention Learning for Visual Question Answering.** *Fei Liu, Jing Liu, Richang Hong, Hanqing Lu* [[pdf](https://dl.acm.org/doi/10.1145/3343031.3350993)] 
    - **CRA-Net: Composed Relation Attention Network for Visual Question Answering.** *Liang Peng, Yang Yang, Zheng Wang, Xiao Wu, Zi Huang* [[pdf](https://dl.acm.org/doi/10.1145/3343031.3350925)] 

* 2018
  + ECCV
    - **Deep Attention Neural Tensor Network for Visual Question Answering.** *Yalong Bai, Jianlong Fu, Tiejun Zhao, Tao Mei* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01258-8_2)] 
    - **Question-Guided Hybrid Convolution for Visual Question Answering.** *Peng Gao, Hongsheng Li, Shuang Li, Pan Lu, Yikang Li, Steven C. H. Hoi, Xiaogang Wang* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01246-5_29)] 
    - **Learning Visual Question Answering by Bootstrapping Hard Attention.** *Mateusz Malinowski, Carl Doersch, Adam Santoro, Peter W. Battaglia* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_1)] 
    - **Straight to the Facts: Learning Knowledge Base Retrieval for Factual Visual Question Answering.** *Medhini Narasimhan, Alexander G. Schwing* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01237-3_28)] 
    - **Question Type Guided Attention in Visual Question Answering.** *Yang Shi, Tommaso Furlanello, Sheng Zha, Animashree Anandkumar* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01225-0_10)] 
    - **Visual Question Answering as a Meta Learning Task.** *Damien Teney, Anton van den Hengel* [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_14)] 

## Subtasks about Image Captioning
### Controllable Image Captioning
  + **Human-Like Controllable Image Captioning With Verb-Specific Semantic Roles.** *Long Chen, Zhihong Jiang, Jun Xiao, Wei Liu* `CVPR` `2021` [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf)] [[code](https://github.com/mad-red/VSR-guided-CIC.git)]
  + **Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs.** *Shizhe Chen, Qin Jin, Peng Wang, Qi Wu* `CVPR` `2020` [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9156638)] [[code](https://github.com/cshizhe/asg2cap.git)]
  + **Connecting Vision and Language with Localized Narratives.** *Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, Vittorio Ferrari* `ECCV` `2020` [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_38)]
  + **Comprehensive Image Captioning Via Scene Graph Decomposition.** *Yiwu Zhong, Liwei Wang, Jianshu Chen, Dong Yu, Yin Li* `ECCV` `2020` [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58568-6.pdf)] [[code](https://github.com/YiwuZhong/Sub-GC.git)]
  + **Dense Relational Captioning: Triple-Stream Networks for Relationship-Based Captioning.** *Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, In So Kweon* `CVPR` `2019` [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Dense_Relational_Captioning_Triple-Stream_Networks_for_Relationship-Based_Captioning_CVPR_2019_paper.html)]
  + **Show Control and Tell: A Framework for Generating Controllable and Grounded Captions.** *Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara* `CVPR` `2019` [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Cornia_Show_Control_and_Tell_A_Framework_for_Generating_Controllable_and_CVPR_2019_paper.html)]
  + **Engaging Image Captioning via Personality.** *Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston* `CVPR` `2019` [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954085)]
  + **Fast, Diverse and Accurate Image Captioning Guided by Part-Of-Speech.** *Aditya Deshpande, Jyoti Aneja, Liwei Wang, Alexander G. Schwing, David A. Forsyth* `CVPR` `2019` [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953300)]
  + **SemStyle: Learning to Generate Stylised Image Captions Using Unaligned Text.** *Alexander Mathews, Lexing Xie, Xuming He* `CVPR` `2018` [[pdf](https://openaccess.thecvf.com/content_cvpr_2018/html/Mathews_SemStyle_Learning_to_CVPR_2018_paper.html)]
  + **Attend to You: Personalized Image Captioning with Context Sequence Memory Networks.** *Cesc Chunseong Park; Byeongchang Kim; Gunhee Kim* `CVPR` `2017` [[pdf](https://ieeexplore.ieee.org/document/8100164)]
  + **StyleNet: Generating Attractive Visual Captions with Styles.** *Chuang Gan; Zhe Gan; Xiaodong He; Jianfeng Gao; Li Deng* `CVPR` `2017` [[pdf](https://ieeexplore.ieee.org/document/8099591)]
  + **SentiCap: Generating Image Descriptions with Sentiments.** *Alexander Patrick Mathews, Lexing Xie, Xuming He* `AAAI` `2016` [[pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12501)]

### Text-Based Image Captioning
  + **Towards Accurate Text-Based Image Captioning With Content Diversity Exploration.** *Guanghui Xu, Shuaicheng Niu, Mingkui Tan, Yucheng Luo, Qing Du, Qi Wu* `CVPR` `2021` [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Towards_Accurate_Text-Based_Image_Captioning_With_Content_Diversity_Exploration_CVPR_2021_paper.pdf)] [[code](https://github.com/guanghuixu/AnchorCaptioner.git)]
  + **Improving OCR-Based Image Captioning by Incorporating Geometrical Relationship.** *Jing Wang, Jinhui Tang, Mingkun Yang, Xiang Bai, Jiebo Luo* `CVPR` `2021` [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_OCR-Based_Image_Captioning_by_Incorporating_Geometrical_Relationship_CVPR_2021_paper.pdf)]
  + **Confidence-aware Non-repetitive Multimodal Transformers for TextCaps.** *Renda Bao, Qi Wu, Si Liu* `AAAI` `2021` [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16389)]
  + **Multimodal Attention with Image Text Spatial Relationship for OCR-Based Image Captioning.** *Jing Wang, Jinhui Tang, Jiebo Luo* `ACMM` `2020` [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413753)] [[code](https://github.com/TownWilliam/mma_sr.git)]
  + **TextCaps: A Dataset for Image Captioning with Reading Comprehension.** *Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, Amanpreet Singh* `ECCV` `2020` [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58536-5.pdf)]

### Image Change Captioning
  + **Image Change Captioning by Learning From an Auxiliary Task.** *Mehrdad Hosseinzadeh, Yang Wang* `CVPR` `2021` [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.pdf)]
  + **Semantic Relation-aware Difference Representation Learning for Change Captioning.** *Yunbin Tu, Tingting Yao, Liang Li, Jiedong Lou, Shengxiang Gao, Zhengtao Yu, Chenggang Yan* `ACL` `2021` [[pdf](https://aclanthology.org/2021.findings-acl.6/)]
  + **Finding It at Another Side: A Viewpoint-Adapted Matching Encoder for Change Captioning.** *Xiangxi Shi, Xu Yang, Jiuxiang Gu, Shafiq Joty, Jianfei Ca* `ECCV` `2020` [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_34)]
  + **Robust Change Captioning.** *Dong Huk Park, Trevor Darrell,  Anna Rohrbach* `ICCV` `2019` [[pdf](https://ieeexplore.ieee.org/document/9008523)]
  + **CaptionNet: Automatic End-to-End Siamese Difference Captioning Model With Attention.** *Ariyo Oluwasanmi, Muhammad Umar Aftab,Eatedal Alabdulkreem,  Bulbula Kumeda, Edward Y. Baagyere, Zhiquang Qin* `IEEE` `2019` [[pdf](https://ieeexplore.ieee.org/document/8776601)]

### Image Paragraph Captioning
  + **Text Embedding Bank for Detailed Image Paragraph Captioning.** *Arjun Gupta, Zengming Shen, Thomas S. Huang* `AAAI` `2021` [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17892/17697)] [[code](https://github.com/arjung128/image-paragraph-captioning.git)]
  + **Object Relation Attention for Image Paragraph Captioning.** *Zhengcong Fei* `AAAI` `2021`  [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16219/16026)] [[code](https://github.com/feizc/PNAIC.git)]
  + **Hierarchical Scene Graph Encoder-Decoder for Image Paragraph Captioning.** *Xu Yang, Chongyang Gao, Hanwang Zhang, Jianfei Cai* `ACMM` `2020`  [[pdf](https://dl.acm.org/doi/pdf/10.1145/3394171.3413859)]
  + **Training for Diversity in Image Paragraph Captioning.** *Luke Melas-Kyriazi, Alexander Rush, George Han* `EMNLP` `2018`  [[pdf](https://aclanthology.org/D18-1084/)]
  + **A Hierarchical Approach for Generating Descriptive Image Paragraphs.** *Jonathan Krause, Justin Johnson, Ranjay Krishna, Li Fei-Fei* `CVPR` `2017`  [[pdf](https://ieeexplore.ieee.org/document/8099839)]

### Few-Shot Image Captioning
  + **Self-Distillation for Few-Shot Image Captioning.** *Xianyu Chen; Ming Jiang; Qi Zhao* `IEEE` `2021`  [[pdf](https://ieeexplore.ieee.org/document/9423232)]
  + **Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering.** *Xuanyi Dong，Linchao Zhu，De Zhang，Yi Yang，Fei Wu* `ACMM` `2018`  [[pdf](https://dl.acm.org/doi/10.1145/3240508.3240527)]
### Unsupervised Image Caption
  + **Recurrent Relational Memory Network for Unsupervised Image Captioning.** *Dan Guo, Yang Wang, Peipei Song, Meng Wang* `IJCAI` `2020`  [[pdf](https://www.ijcai.org/proceedings/2020/0128.pdf)]
  + **Unsupervised Image Captioning.** *Yang Feng, Lin Ma, Wei Liu, Jiebo Luo* `CVPR` `2019`  [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953817)] [[code](https://github.com/fengyang0317/unsupervised_captioning.git)] 
  + **Towards Unsupervised Image Captioning With Shared Multimodal Embeddings.** *Iro Laina, Christian Rupprecht, Nassir Navab* `ICCV` `2019`  [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9010396)]   
  + **Aligning Linguistic Words and Visual Semantic Units for Image Captioning.** *Longteng Guo, Jing Liu, Jinhui Tang, Jiangwei Li, Wei Luo, Hanqing Lu* `ACMM` `2019`  [[pdf](https://dl.acm.org/doi/pdf/10.1145/3343031.3350943)] [[code](https://github.com/ltguo19/VSUA-Captioning.git)]

## Datasets and Metrics about Image Captioning
### Datasets
  + MS COCO
    - **Microsoft COCO: Common Objects in Context.** *Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, C. Lawrence Zitnick* `ECCV` `2014` [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10602-1_48.pdf)] [[link](https://cocodataset.org/#home)]
  + Flickr 8k and Flickr 30k
  + SentiCap
    - **SentiCap: Generating Image Descriptions with Sentiments.** *Alexander Patrick Mathews, Lexing Xie, Xuming He* `AAAI` `2016` [[pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12501)] [[link](http://users.cecs.anu.edu.au/~u4534172/senticap.html)]
  + FlickrStyle
    - **StyleNet: Generating Attractive Visual Captions with Styles.** *Chuang Gan; Zhe Gan; Xiaodong He; Jianfeng Gao; Li Deng* `CVPR` `2017` [[pdf](https://ieeexplore.ieee.org/document/8099591)] [[link](https://paperswithcode.com/dataset/flickrstyle10k)]
  + TextCaps
    - **TextCaps: A Dataset for Image Captioning with Reading Comprehension.** *Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, Amanpreet Singh* `ECCV` `2020` [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58536-5.pdf)] [[link](https://textvqa.org/textcaps/)]
### Metrics
  + BLEU
    - **Bleu: a method for automatic evaluation of machine translation.** *Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu* `ACL` `2002` [[pdf](https://aclanthology.org/P02-1040.pdf)] 
  + METEOR
    - **METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.** *Satanjeev Banerjee, Alon Lavie* `ACL` `2005` [[pdf](https://aclanthology.org/W05-0909.pdf)] 
  + ROUGE
    - **Automatic Evaluation of Summaries Using N-gram Co-occurrence Statistics.** *Chin-Yew Lin, Eduard Hovy* `HLT-NAACL` `2003` [[pdf](https://aclanthology.org/N03-1020.pdf)] 
  + CIDEr
    - **CIDEr: Consensus-based image description evaluation.** *Ramakrishna Vedantam, C. Lawrence Zitnick, Devi Parikh* `CVPR` `2015` [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7299087)] 
    - **CIDEr-R: Robust Consensus-based Image Description Evaluation.** *CIDEr-R: Robust Consensus-based Image Description Evaluation* `W-NUT` `2021` [[pdf](https://aclanthology.org/2021.wnut-1.39.pdf)] 
  + SPICE
    - **SPICE: Semantic Propositional Image Caption Evaluation** *Peter Anderson, Basura Fernando, Mark Johnson, Stephen Gould* `ECCV` `2016` [[pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46454-1_24.pdf)]

