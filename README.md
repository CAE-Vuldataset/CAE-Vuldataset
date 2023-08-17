# SoK: Characterizing and Evaluating Software Vulnerability Datasets

## Description

Software vulnerabilities are the root cause of many cyber attacks. Not surprisingly, vulnerability-related studies, such as vulnerability analysis, detection, and repair, have received much attention. These studies have led to the introduction of many vulnerability datasets.  
However, there is no systematic understanding of the quality of these datasets.
This motivates the present SoK, which proposes a systematic characterization and evaluation of vulnerability datasets. Our systematization is based on four aspects of vulnerability datasets: their quality, purpose, source, and generation method; for measuring their quality, we present seven quantitative metrics, which may be of independent value.
We first conduct a study of 67 vulnerability datasets, including 57 academic publications and 10 industrial datasets, then we identify and systematize 28 vulnerability datasets that are publicly available.
Among other things, we find that there is no single vulnerability dataset or generation method of `` benchmark-quality``. We also discuss future research directions.

## Vulnerability Datasets

| Dataset | Paper | Venue/Organization | Link |
| --- | --- | --- | --- |
| [VulDeePecker](https://arxiv.org/abs/1801.01681) | VulDeePecker: A Deep Learning-Based System for Vulnerability Detection | NDSS | [link](https://github.com/CGCL-codes/VulDeePecker) |
| [SySeVR](https://ieeexplore.ieee.org/abstract/document/9321538) | SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities | TDSC | [link](https://github.com/SySeVR/SySeVR) |
| [VulDeeLocator](https://ieeexplore.ieee.org/abstract/document/9416836/) | VulDeeLocator: A Deep Learning-based Fine-grained Vulnerability Detector | TDSC | [link](https://github.com/VulDeeLocator/VulDeeLocator) |
| [Reveal](https://ieeexplore.ieee.org/abstract/document/9448435/) | Deep Learning based Vulnerability Detection: Are We There Yet? | TSE | [link](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF) [link2](https://github.com/VulDetProject/ReVeal)|
| [Lin et al.](https://link.springer.com/chapter/10.1007/978-3-030-41579-2_13) | Deep learning-based vulnerable function detection: A benchmark | ICICS | [link](https://github.com/Seahymn2019/Function-level-Vulnerability-Dataset/tree/master/Data) |
| [Funded](https://ieeexplore.ieee.org/abstract/document/9293321/) | Combining Graph-Based Learning With Automated Data Collection for Code Vulnerability Detection | TIFS | [link](https://github.com/HuantWang/FUNDED_NISL) |
| [Devign](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) | Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks | NeurIPS | [link](https://sites.google.com/view/devign) |
| [VulCNN](https://dl.acm.org/doi/abs/10.1145/3510003.3510229) | VulCNN: An Image-inspired Scalable Vulnerability Detection System | ICSE | [link](https://github.com/CGCL-codes/VulCNN) |
| [VUDDY](https://ieeexplore.ieee.org/abstract/document/7958600/) | VUDDY: A Scalable Approach for Vulnerable Code Clone Discovery | S&P | [link](https://github.com/squizz617/vuddy) |
| [VulPecker](https://dl.acm.org/doi/abs/10.1145/2991079.2991102) | VulPecker: An Automated Vulnerability Detection System Based on Code Similarity Analysis | ACSAC | [link](https://github.com/vulpecker/Vulpecker) |
| [VDSimilar](https://www.sciencedirect.com/science/article/pii/S0167404821002418) | VDSimilar: Vulnerability detection based on code similarity of vulnerabilities and patches | Computers & Security | [link](https://github.com/sunhao123456789/siamese_dataset) |
| [Magma](https://dl.acm.org/doi/abs/10.1145/3410220.3456276) | Magma: A Ground-Truth Fuzzing Benchmark | POMACS | [link](https://github.com/HexHive/magma) |
| [Lipp et al.](https://dl.acm.org/doi/abs/10.1145/3533767.3534380) | An Empirical Study on the Effectiveness of Static C Code Analyzers for Vulnerability Detection | ISSTA | [link](https://doi.org/10.5281/zenodo.6515687) |
| [SPI](https://dl.acm.org/doi/abs/10.1145/3468854) | SPI: Automated Identification of Security Patches via Commits | TOSEM | [link](https://sites.google.com/view/du-commits/home) |
| [Le et al.](https://dl.acm.org/doi/abs/10.1145/3524842.3528433) | On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models | ICSE | [link](https://github.com/lhmtriet/Function-level-Vulnerability-Assessment) |
| [Pan et al.](https://dl.acm.org/doi/abs/10.1145/3524842.3528433) | On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models | ICSE | [link](https://figshare.com/articles/online_resource/TreeVul_-_Replication_Package/19727050) |
| [Shi et al.](https://ieeexplore.ieee.org/abstract/document/9723619/) | Does OpenBSD and Firefox’s Security Improve with Time? | TII | [link](https://github.com/VulnSet/BetterOrWorse) |
| [Liu et al.](https://dl.acm.org/doi/abs/10.1145/3377811.3380923) | A Large-Scale Empirical Study on Vulnerability Distribution within Projects and the Lessons Learned | ICSE | [link](https://github.com/twelveand0/CarrotsBlender) |
| [Detect0day](https://ieeexplore.ieee.org/abstract/document/8809499/) | Detecting “0-Day” Vulnerability: An Empirical Study of Secret Security Patch in OSS | DSN | [link](https://github.com/SecretPatch/Dataset) |
| [CrossVul](https://dl.acm.org/doi/abs/10.1145/3468264.3473122) | CrossVul: a cross-language vulnerability dataset with commit data | ESEC/FSE | [link](https://zenodo.org/record/4734050#.Y_YRtexBy3I) |
| [PatchDB](https://ieeexplore.ieee.org/abstract/document/9505097/) | PatchDB: A Large-Scale Security Patch Dataset | DSN | [link](https://github.com/SunLab-GMU/PatchDB) |
| [D2A](https://ieeexplore.ieee.org/abstract/document/9402126/) | A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis  | ICSE | [link](https://github.com/IBM/D2A#using-the-dataset) |
| [Ponta et al.](https://ieeexplore.ieee.org/abstract/document/8816802/) | A Manually-Curated Dataset of Fixes to Vulnerabilities of Open-Source Software | MSR | [link](https://github.com/SAP/project-kb/tree/main/MSR2019) |
| [SECBENCH](http://ceur-ws.org/Vol-1977/paper6.pdf) | SECBENCH: A Database of Real Security Vulnerabilities | SecSE@ESORICS | [link](https://tqrg.github.io/secbench/ ) |
| [Reis et al.](https://arxiv.org/abs/2110.09635) | A ground-truth dataset of real security patches | NeurIPS | [link](https://github.com/TQRG/security-patches-dataset) |
| [BigVul](https://dl.acm.org/doi/abs/10.1145/3379597.3387501) | A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries | MSR | [link](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) |
| [CVEfixes](https://dl.acm.org/doi/abs/10.1145/3475960.3475985) | CVEfixes:Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software  | PROMISE | [link](https://github.com/secureIT-project/CVEfixes/blob/main/Doc/DataDictionary.md) |
| [Feng et al.](https://ieeexplore.ieee.org/abstract/document/9163061/) | Efficient Vulnerability Detection based on abstract syntax tree and Deep Learning | INFOCOM | [link](https://samate.nist.gov/SARD) |

## Folder Description
`./code/`: Script programs used for statistics are stored
