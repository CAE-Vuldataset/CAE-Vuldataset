# SoK: Software Vulnerability Dataset

## Description

Software vulnerabilities are a primary cause of cyberattacks, with new vulnerabilities being discovered on a daily basis.There is a community studying a range of problems related to vulnerability analysis, detection, and automated patching, all of which require competent datasets. However, there is no systematic understanding on the competency of vulnerability datasets for these purposes or application tasks, let alone benchmark datasets. The present SoK aims to fill this void by proposing a set of attributes to characterize software vulnerability datasets. We evaluate the impact of the dataset attributes on the effectiveness of four application tasks. We show that the attributes and their characteristics can not only guide researchers in selecting suitable datasets for their application tasks, but also guide the creation of future datasets that exhibit the desired characteristics.

We first conduct a study of 72 vulnerability datasets, including 62 academic publications and 10 industrial datasets, then we identify and systematize 31 vulnerability datasets that are publicly available.
We also discuss future research directions.

## Vulnerability Datasets

| Dataset | Paper | Venue/Organization | Link |
| --- | --- | --- | --- |
| [benchmark](https://link.springer.com/chapter/10.1007/978-3-030-41579-2_13) | Deep learning-based vulnerable function detection: A benchmark | ICICS | [link](https://github.com/Seahymn2019/Function-level-Vulnerability-Dataset/tree/master/Data) |
| [BigVul](https://dl.acm.org/doi/abs/10.1145/3379597.3387501) | A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries | MSR | [link](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) |
| [CrossVul](https://dl.acm.org/doi/abs/10.1145/3468264.3473122) | CrossVul: a cross-language vulnerability dataset with commit data | ESEC/FSE | [link](https://zenodo.org/record/4734050#.Y_YRtexBy3I) |
| [CVEfixes](https://dl.acm.org/doi/abs/10.1145/3475960.3475985) | CVEfixes:Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software  | PROMISE | [link](https://github.com/secureIT-project/CVEfixes/blob/main/Doc/DataDictionary.md) |
| [D2A](https://ieeexplore.ieee.org/abstract/document/9402126/) | A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis  | ICSE | [link](https://github.com/IBM/D2A#using-the-dataset) |
| [Detect0day](https://ieeexplore.ieee.org/abstract/document/8809499/) | Detecting “0-Day” Vulnerability: An Empirical Study of Secret Security Patch in OSS | DSN | [link](https://github.com/SecretPatch/Dataset) |
| [Devign](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) | Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks | NeurIPS | [link](https://sites.google.com/view/devign) |
| [DiverseVul](https://dl.acm.org/doi/10.1145/3607199.3607242) | DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection | RAID | [link](https://github.com/wagner-group/diversevul) |
| [Funded](https://ieeexplore.ieee.org/abstract/document/9293321/) | Combining Graph-Based Learning With Automated Data Collection for Code Vulnerability Detection | TIFS | [link](https://github.com/HuantWang/FUNDED_NISL) |
| [Le et al.](https://dl.acm.org/doi/abs/10.1145/3524842.3528433) | On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models | ICSE | [link](https://github.com/lhmtriet/Function-level-Vulnerability-Assessment) |
| [Lin et al.](https://ieeexplore.ieee.org/document/8906156) | Software Vulnerability Discovery via Learning Multi-Domain Knowledge Bases | TDSC | [link](https://github.com/DanielLin1986/RepresentationsLearningFromMulti_domain) |
| [Lipp et al.](https://dl.acm.org/doi/abs/10.1145/3533767.3534380) | An Empirical Study on the Effectiveness of Static C Code Analyzers for Vulnerability Detection | ISSTA | [link](https://doi.org/10.5281/zenodo.6515687) |
| [Magma](https://dl.acm.org/doi/abs/10.1145/3410220.3456276) | Magma: A Ground-Truth Fuzzing Benchmark | POMACS | [link](https://github.com/HexHive/magma) |
| [mVulDeePecker](https://ieeexplore.ieee.org/document/8846081) | mVulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection | TDSC | [link](https://github.com/muVulDeePecker/muVulDeePecker) |
| [Pan et al.](https://dl.acm.org/doi/abs/10.1145/3524842.3528433) | On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models | ICSE | [link](https://figshare.com/articles/online_resource/TreeVul_-_Replication_Package/19727050) |
| [PatchDB](https://ieeexplore.ieee.org/abstract/document/9505097/) | PatchDB: A Large-Scale Security Patch Dataset | DSN | [link](https://github.com/SunLab-GMU/PatchDB) |
| [Ponta et al.](https://ieeexplore.ieee.org/abstract/document/8816802/) | A Manually-Curated Dataset of Fixes to Vulnerabilities of Open-Source Software | MSR | [link](https://github.com/SAP/project-kb/tree/main/MSR2019) |
| [REEF](https://ieeexplore.ieee.org/document/10298352) | REEF: A Framework for Collecting Real-World Vulnerabilities and Fixes | ASE | [link](https://github.com/ASE-REEF/REEF-data) |
| [Reis et al.](https://arxiv.org/abs/2110.09635) | A ground-truth dataset of real security patches | NeurIPS | [link](https://github.com/TQRG/security-patches-dataset) |
| [ReposVul](https://dl.acm.org/doi/10.1145/3639478.3647634) | ReposVul: A Repository-Level High-Quality Vulnerability Dataset | ICSE | [link](https://github.com/Eshe0922/ReposVul) |
| [Reveal](https://ieeexplore.ieee.org/abstract/document/9448435/) | Deep Learning based Vulnerability Detection: Are We There Yet? | TSE | [link](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF) [link2](https://github.com/VulDetProject/ReVeal)|
| [SECBENCH](http://ceur-ws.org/Vol-1977/paper6.pdf) | SECBENCH: A Database of Real Security Vulnerabilities | SecSE@ESORICS | [link](https://tqrg.github.io/secbench/ ) |
| [SPI](https://dl.acm.org/doi/abs/10.1145/3468854) | SPI: Automated Identification of Security Patches via Commits | TOSEM | [link](https://sites.google.com/view/du-commits/home) |
| [SySeVR](https://ieeexplore.ieee.org/abstract/document/9321538) | SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities | TDSC | [link](https://github.com/SySeVR/SySeVR) |
| [Lin et al.](https://ieeexplore.ieee.org/document/8329207) | Cross-Project Transfer Representation Learning for Vulnerable Function Discovery | IEEE Trans | [link](https://github.com/DanielLin1986/TransferRepresentationLearning) |
| [VDSimilar](https://www.sciencedirect.com/science/article/pii/S0167404821002418) | VDSimilar: Vulnerability detection based on code similarity of vulnerabilities and patches | Computers & Security | [link](https://github.com/sunhao123456789/siamese_dataset) |
| [VulCNN](https://dl.acm.org/doi/abs/10.1145/3510003.3510229) | VulCNN: An Image-inspired Scalable Vulnerability Detection System | ICSE | [link](https://github.com/CGCL-codes/VulCNN) |
| [VulDeeLocator](https://ieeexplore.ieee.org/abstract/document/9416836/) | VulDeeLocator: A Deep Learning-based Fine-grained Vulnerability Detector | TDSC | [link](https://github.com/VulDeeLocator/VulDeeLocator) |
| [VulDeePecker](https://arxiv.org/abs/1801.01681) | VulDeePecker: A Deep Learning-Based System for Vulnerability Detection | NDSS | [link](https://github.com/CGCL-codes/VulDeePecker) |
| [VulnPatchPairs](https://www.usenix.org/conference/usenixsecurity24/presentation/risse) | Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection | USENIX Security | [link](https://drive.google.com/file/d/11hRh8YlQFgxxpg1JcLRdnblEFLvXxhpK/edit) |
| [VulPecker](https://dl.acm.org/doi/abs/10.1145/2991079.2991102) | VulPecker: An Automated Vulnerability Detection System Based on Code Similarity Analysis | ACSAC | [link](https://github.com/vulpecker/Vulpecker) |

## Folder Description
`./code/`: Script programs used for statistics are stored
